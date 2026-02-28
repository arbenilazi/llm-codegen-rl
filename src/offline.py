#!/usr/bin/env python3
"""
offline.py — Train from the offline RL dataset (nbest.jsonl + pairs.jsonl).

Usage:
  python -m src.offline \
    --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --dataset mbpp \
    --dataset_path assets/datasets/mbpp \
    --rl_dir assets/rl/mbpp \
    --output_dir results/dpo_mbpp_lora \
    --use_lora \
    --epochs 1 \
    --batch_size 4 \
    --max_pairs_per_task 20
"""

import os, json, argparse, sys
from typing import List, Dict, Any

sys.path.append("src")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

from dataset import get_dataset


# RL data loading utilities

def load_rl_pairs(rl_dir: str) -> Dict[int, Dict[str, Any]]:
    nbest_path = os.path.join(rl_dir, "nbest.jsonl")
    pairs_path = os.path.join(rl_dir, "pairs.jsonl")
    if not os.path.exists(nbest_path):
        raise FileNotFoundError(f"Missing nbest.jsonl at {nbest_path}")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing pairs.jsonl at {pairs_path}")

    id2nbest = {}
    with open(nbest_path) as f:
        for ln in f:
            j = json.loads(ln)
            id2nbest[j["id"]] = {"entry_point": j["entry_point"], "best_list": j["best_list"]}

    id2pairs = {}
    with open(pairs_path) as f:
        for ln in f:
            j = json.loads(ln)
            id2pairs[j["id"]] = j["pairs"]

    out = {}
    for _id, nb in id2nbest.items():
        out[_id] = {**nb, "pairs": id2pairs.get(_id, [])}
    return out


def build_dpo_dataset(
    dataset_name: str,
    dataset_path: str,
    rl_dir: str,
    max_pairs_per_task: int = 20,
    instruction_style: str = "minimal",
) -> Dataset:
    ds = get_dataset(dataset_name, dataset_path)
    split = ds._default_split()
    ref = ds.dataset[split]

    id2prompt = {ex["id"]: ex["input"] for ex in ref}
    id2entry = {ex["id"]: ex["entry_point"] for ex in ref}
    id2data = load_rl_pairs(rl_dir)

    def make_prompt(text: str, entry: str) -> str:
        if instruction_style == "minimal":
            return (
                f"{text}\n\n"
                f"Implement exactly one function named `{entry}`.\n"
                f"Do not write any top-level code (no prints, no input()).\n"
                f"Return only a single Python code block with the function."
            )
        return text

    rows = []
    for _id, pack in id2data.items():
        prompt = id2prompt.get(_id, "")
        entry = id2entry.get(_id, "solution")
        best_list = pack.get("best_list", [])
        pairs = pack.get("pairs", [])[:max_pairs_per_task]
        if not best_list or not pairs:
            continue

        prompt_text = make_prompt(prompt, entry)

        for i, j in pairs:
            if i >= len(best_list) or j >= len(best_list):
                continue
            chosen = best_list[i]["code"]
            rejected = best_list[j]["code"]
            if not chosen or not rejected:
                continue
            rows.append({"prompt": prompt_text, "chosen": chosen, "rejected": rejected})

    if not rows:
        raise RuntimeError("No DPO pairs were materialized. Check your RL artifacts and limits.")
    return Dataset.from_list(rows)


# Model utilities

def wrap_lora(model, use_lora: bool, target_modules: List[str], r: int, alpha: int, dropout: float):
    if not use_lora:
        return model
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    try:
        print("LoRA applied. Trainable parameters:")
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def pick_precision() -> Dict[str, bool]:
    """
    Decide bf16/fp16 based on environment.
    - If CUDA + bf16 supported -> bf16
    - Else if CUDA -> fp16
    - Else CPU (no half)
    """
    has_cuda = torch.cuda.is_available()
    bf16_ok = has_cuda and torch.cuda.is_bf16_supported()
    if bf16_ok:
        return {"bf16": True, "fp16": False, "has_cuda": True}
    if has_cuda:
        return {"bf16": False, "fp16": True, "has_cuda": True}
    return {"bf16": False, "fp16": False, "has_cuda": False}


# Main

def main():
    ap = argparse.ArgumentParser(description="Offline DPO training from RL dataset (nbest + pairs).")
    ap.add_argument("--model_name", required=True, help="e.g., Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--dataset", default="mbpp", help="Adapter name (mbpp or leetcode)")
    ap.add_argument("--dataset_path", default="assets/datasets/mbpp", help="HF saved dataset path")
    ap.add_argument("--rl_dir", default="assets/rl/mbpp", help="Path with nbest.jsonl & pairs.jsonl")
    ap.add_argument("--output_dir", default="results/dpo_mbpp", help="Where to save the model")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--max_pairs_per_task", type=int, default=20)
    ap.add_argument("--instruction_style", type=str, default="minimal")
    # optional quantized loading (needs bitsandbytes)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")
    args = ap.parse_args()

    # 0) precision & device pick
    prec = pick_precision()
    print(f"[i] CUDA visible: {prec['has_cuda']}. Using bf16={prec['bf16']} fp16={prec['fp16']}.")

    # 1) Build DPO dataset
    print("[i] Building DPO dataset from RL artifacts…")
    dpo_ds = build_dpo_dataset(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        rl_dir=args.rl_dir,
        max_pairs_per_task=args.max_pairs_per_task,
        instruction_style=args.instruction_style,
    )
    print(f"[i] DPO rows: {len(dpo_ds)}")

    # 2) Load model/tokenizer
    print("[i] Loading model & tokenizer…")
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    torch_dtype = None
    if prec["bf16"]:
        torch_dtype = torch.bfloat16
    elif prec["fp16"]:
        torch_dtype = torch.float16

    model_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map="auto" if prec["has_cuda"] else None,
    )
    if args.load_in_4bit:
        model_kwargs.update(dict(load_in_4bit=True))
    elif args.load_in_8bit:
        model_kwargs.update(dict(load_in_8bit=True))

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **{k:v for k,v in model_kwargs.items() if v is not None})

    model = wrap_lora(
        model,
        use_lora=args.use_lora,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    if prec["has_cuda"]:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 3) Configure TRL DPO
    print("[i] Configuring DPOTrainer…")
    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=["none"],
        # Precision flags:
        bf16=prec["bf16"],
        fp16=prec["fp16"],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=dpo_ds,
    )

    # 4) Train
    print("[i] Starting DPO training…")
    trainer.train()
    print(f"[ok] DPO training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
