#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

import json
import argparse
import sys
import wandb
import datetime
from typing import List, Dict, Any

# Make src/ importable to reuse your dataset adapter
sys.path.append("src")

from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from dataset import get_dataset


# Load RL artifacts
def load_rl_pairs(rl_dir: str) -> Dict[int, Dict[str, Any]]:
    nbest_path = os.path.join(rl_dir, "nbest.jsonl")
    pairs_path = os.path.join(rl_dir, "pairs.jsonl")

    id2nbest = {}
    with open(nbest_path) as f:
        for ln in f:
            j = json.loads(ln)
            id2nbest[j["id"]] = {
                "entry_point": j["entry_point"],
                "best_list": j["best_list"],
            }

    id2pairs = {}
    with open(pairs_path) as f:
        for ln in f:
            j = json.loads(ln)
            id2pairs[j["id"]] = j["pairs"]

    out = {}
    for _id, nb in id2nbest.items():
        out[_id] = {**nb, "pairs": id2pairs.get(_id, [])}

    return out


# Build DPO dataset
def build_dpo_examples(dataset_name, dataset_path, rl_dir, max_pairs_per_task=6):
    ds = get_dataset(dataset_name, dataset_path)
    split = "train" if "train" in ds.dataset else ds._default_split()
    ref = ds.dataset[split]
    print(f"[dbg] using split={split}, n_ref={len(ref)}")

    # Build maps with *string* ids to avoid int/str mismatch
    id2prompt = {str(ex["id"]): ex["input"] for ex in ref}
    id2entry_ds = {str(ex["id"]): ex.get("entry_point", None) for ex in ref}

    id2data = load_rl_pairs(rl_dir)

    samples = []
    missing_prompt = 0
    missing_entry = 0

    for _id_raw, pack in id2data.items():
        _id = str(_id_raw)  # normalize key

        best_list = pack.get("best_list", [])
        pairs = (pack.get("pairs") or [])[:max_pairs_per_task]
        if not best_list or not pairs:
            continue

        prompt = id2prompt.get(_id)
        if not prompt:
            missing_prompt += 1
            continue  # don't train on empty prompt

        # Prefer RL artifact entry_point (this is what your candidates were normalized to)
        entry = pack.get("entry_point") or id2entry_ds.get(_id) or "solution"
        if not entry:
            missing_entry += 1
            entry = "solution"

        prompt_text = (
            f"{prompt}\n\n"
            f"Implement exactly one function named `{entry}`.\n"
            f"Do not write any top-level code (no prints, no input()).\n"
            f"Return only a single Python code block with the function.\n\n"
            f"### Solution\n"
        )

        for i, j in pairs:
            if i >= len(best_list) or j >= len(best_list):
                continue

            chosen = best_list[i].get("code", "")
            rejected = best_list[j].get("code", "")

            if chosen and rejected:
                samples.append({
                    "prompt": prompt_text,
                    "chosen": chosen,
                    "rejected": rejected,
                })

    if not samples:
        raise RuntimeError("No DPO samples found. Check RL artifacts.")

    print(f"[i] build_dpo_examples: kept={len(samples)} "
          f"(skipped_missing_prompt={missing_prompt}, skipped_missing_entry={missing_entry})")

    return Dataset.from_list(samples)


# Optional LoRA wrapper
def wrap_lora(model, use_lora, target_modules, r, alpha, dropout):
    if not use_lora:
        return model
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora_cfg)


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--dataset", default="mbpp")
    ap.add_argument("--dataset_path", default="assets/datasets/mbpp")
    ap.add_argument("--rl_dir", default="assets/rl/mbpp")
    ap.add_argument("--output_dir", default="results/dpo_mbpp")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--max_pairs_per_task", type=int, default=6)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=50)
    args = ap.parse_args()

    # Initialize Weights & Biases
    args.rl_method = "dpo"
    run_name = f"{args.rl_method}-{args.dataset}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project="bootstrapper-rl",
        name=run_name,
        config=vars(args),
    )

    print("[i] Building DPO dataset…")
    dpo_ds = build_dpo_examples(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        rl_dir=args.rl_dir,
        max_pairs_per_task=args.max_pairs_per_task,
    )
    print(f"[i] DPO rows: {len(dpo_ds)}")

    # Build eval dataset (small subset)
    eval_size = min(128, len(dpo_ds))
    eval_ds = dpo_ds.select(range(eval_size))
    print(f"[i] Eval subset size: {len(eval_ds)}")

    print("[i] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for QLoRA

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load trainable model
    print("[i] Loading trainable model in 4-bit (QLoRA)…")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Load separate frozen reference model for DPO
    print("[i] Loading frozen reference model in 4-bit…")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    ref_model.eval()
    ref_model.config.use_cache = False

    print("[i] Applying LoRA to trainable model…")
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    model = wrap_lora(
        model,
        use_lora=args.use_lora,
        target_modules=target_modules,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    print("[i] Preparing DPOConfig…")
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,

        # W&B logging
        logging_steps=args.logging_steps,
        report_to=["wandb"],

        # Enable evaluation loss logging
        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_steps=args.save_steps,
        save_total_limit=1,
        seed=42,
        max_length=1024,
        max_prompt_length=512,
    )

    print("[i] Initializing DPOTrainer with separate reference model…")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dpo_ds,
        eval_dataset=eval_ds,
    )

    print("[i] Training…")
    trainer.train()
    print(f"[ok] Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
