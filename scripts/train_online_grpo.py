# scripts/train_online_grpo.py
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

for p in (REPO_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import get_dataset
from sanitize import normalize_entry_point
from pipeline.annotate import execute_wrapped_code


def _sanitize_test_for_exec(t) -> str:
    if isinstance(t, dict):
        t = t.get("test") or t.get("text") or ""
    t = (t or "").rstrip()
    while t.endswith("_"):
        t = t[:-1].rstrip()
    return t + "\n"


def _extract_code_for_entry(text: str, entry: str) -> str:
    if not text:
        return ""
    source = str(text).strip()

    blocks = re.findall(r"```(?:[a-zA-Z]+\n)?(.*?)```", source, re.DOTALL)
    blocks = [b.strip() for b in blocks if b.strip()]

    if not blocks:
        m = re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", source)
        if m:
            return source[m.start():].replace("```", "").strip()
        return source.replace("```", "").strip()

    # Prefer block containing the entry function
    for b in blocks:
        if re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", b):
            return b.strip()

    # Else pick the largest block with a def
    defs = [b for b in blocks if "def " in b]
    if defs:
        return max(defs, key=len).strip()

    return "\n\n".join(blocks).strip()


def build_prompt(example: Dict[str, Any]) -> str:
    # Keep prompts stable; the reward signal drives optimization.
    inp = example.get("input") or ""
    entry = example.get("entry_point") or "solution"
    return (
        "You are an expert Python programmer.\n"
        "Write a fully correct solution that passes the unit tests.\n"
        "Rules:\n"
        f"- Implement exactly one function named `{entry}`.\n"
        "- Use only the Python standard library.\n"
        "- No prints, no comments, no main guard.\n"
        "- Put the solution in one Python code block.\n\n"
        f"Task:\n{inp}\n"
    )


def make_train_rows(dataset_name: str, dataset_path: str) -> List[Dict[str, Any]]:
    ds = get_dataset(dataset_name, dataset_path)
    hf_split = ds.dataset["train"]
    rows = []
    for ex in hf_split:
        rows.append(
            {
                "id": ex.get("id"),
                "entry_point": ex.get("entry_point") or "solution",
                "test": ex.get("test"),
                "prompt": build_prompt(ex),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)

    p.add_argument("--num_generations", type=int, default=6)
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--exec_timeout", type=float, default=2.0)

    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    rows = make_train_rows(args.dataset, args.dataset_path)

    # We keep meta aligned with prompts
    prompts = [r["prompt"] for r in rows]
    meta = [{"id": r["id"], "entry_point": r["entry_point"], "test": r["test"]} for r in rows]

    # Model/tokenizer
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # LoRA
    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    # GRPO (TRL)
    try:
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        raise RuntimeError(
            "Could not import GRPOTrainer from trl. "
            "Your installed TRL version may not include GRPO. "
            "Tell me the output of: python -c \"import trl; print(trl.__version__)\""
        ) from exc

    # Build dataset with meta columns (TRL passes these in kwargs to reward_fn)
    from datasets import Dataset as HFDataset
    train_ds = HFDataset.from_dict({
        "prompt": prompts,
        "id": [m["id"] for m in meta],
        "entry_point": [m["entry_point"] for m in meta],
        "test": [m["test"] for m in meta],
    })

    def _to_text(x, tokenizer) -> str:
        # Most common: already a string
        if x is None:
            return ""
        if isinstance(x, str):
            return x

        # Sometimes a dict-like payload
        if isinstance(x, dict):
            for k in ("text", "content", "completion", "generated_text"):
                if k in x and isinstance(x[k], str):
                    return x[k]
            # fall back
            return str(x)

        # Torch tensor of token ids
        if isinstance(x, torch.Tensor):
            ids = x.detach().cpu().tolist()
            # could be [seq] or [[seq]] depending on batching
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return tokenizer.decode(ids, skip_special_tokens=True)

        # List of token ids (or nested list)
        if isinstance(x, list):
            if not x:
                return ""
            if isinstance(x[0], int):
                return tokenizer.decode(x, skip_special_tokens=True)
            if isinstance(x[0], list) and x[0] and isinstance(x[0][0], int):
                return tokenizer.decode(x[0], skip_special_tokens=True)
            # list of strings etc.
            if isinstance(x[0], str):
                return "\n".join(x)
            return str(x)

        # Fallback
        return str(x)

    def _tolist(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        return list(x) if x is not None else []

    def reward_fn(
        prompts=None,
        completions=None,
        prompts_batch=None,
        completions_batch=None,
        **kwargs
    ) -> List[float]:
        # TRL may pass prompts/completions under different names
        if prompts is None:
            prompts = prompts_batch
        if completions is None:
            completions = completions_batch

        prompts_batch = list(prompts or [])
        completions_batch = list(completions or [])


        # TRL passes extra columns from the dataset in kwargs (same batch order)
        # Coerce to lists to handle tensors or nested structures
        ids = _tolist(kwargs.get("id")) or [None] * len(completions_batch)
        entries = _tolist(kwargs.get("entry_point")) or ["solution"] * len(completions_batch)
        tests = _tolist(kwargs.get("test")) or [""] * len(completions_batch)

        local_meta = [{"id": i, "entry_point": e, "test": t} for i, e, t in zip(ids, entries, tests)]

        # pass fraction (0..1) instead of binary
        rewards: List[float] = []
        for completion, m in zip(completions_batch, local_meta):
            completion_text = _to_text(completion, tokenizer)
            entry = m["entry_point"]
            test_code = _sanitize_test_for_exec(m["test"])
            code = _extract_code_for_entry(completion_text, entry)
            code = normalize_entry_point(code or "", entry)
            wrapped = f"{test_code}\n\n{code}\n\ncheck({entry})"
            ok = execute_wrapped_code(wrapped, timeout=float(args.exec_timeout))
            rewards.append(1.0 if ok else 0.0)   # keep binary first (stable)
        return rewards

    grpo_cfg = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        num_train_epochs=int(args.epochs),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        max_prompt_length=int(args.max_prompt_length),
        max_completion_length=int(args.max_completion_length),
        num_generations=int(args.num_generations),

        generation_batch_size=int(args.num_generations),

        # precision 
        bf16=torch.cuda.is_available(),
        fp16=False,

        # vLLM
        use_vllm=False,
        #vllm_gpu_memory_utilization=0.90,

        report_to=[],
    )


    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    # Save a small manifest (helps debugging)
    manifest = {
        "method": "online_grpo",
        "model_name": args.model_name,
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "lr": args.lr,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "exec_timeout": args.exec_timeout,
        "use_lora": bool(args.use_lora),
    }
    (output_dir / "online_grpo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] Online GRPO done. Saved to: {output_dir}")


if __name__ == "__main__":
    main()
