# pipeline/online_grpo.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def train_online_grpo(cfg: Dict[str, Any]) -> Path:
    """
    Launch online GRPO training via scripts/train_online_grpo.py
    Returns: path to output_dir (LoRA checkpoints live there)
    """
    dataset_cfg = cfg["dataset"]
    gen_cfg = cfg["generation"]

    online_cfg = cfg.get("online_grpo", {})
    # Defaults (safe)
    output_dir = online_cfg.get("output_dir") or f"results/online_grpo_{dataset_cfg['name']}"
    epochs = int(online_cfg.get("epochs", 1))
    lr = float(online_cfg.get("lr", 1e-5))
    batch_size = int(online_cfg.get("batch_size", 1))
    grad_accum = int(online_cfg.get("gradient_accumulation_steps", 8))
    num_generations = int(online_cfg.get("num_generations", 6))
    max_prompt_len = int(online_cfg.get("max_prompt_length", 2048))
    max_completion_len = int(online_cfg.get("max_completion_length", 512))
    exec_timeout = float(online_cfg.get("exec_timeout", 2.0))
    save_steps = int(online_cfg.get("save_steps", 50))
    logging_steps = int(online_cfg.get("logging_steps", 10))

    # Base model for training (we train from base, not from merged)
    base_model_name = online_cfg.get("base_model_name") or gen_cfg["model_name"]

    # LoRA settings (mirror your DPO config style)
    use_lora = bool(online_cfg.get("use_lora", True))
    target_modules = online_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_r = int(online_cfg.get("lora_r", 16))
    lora_alpha = int(online_cfg.get("lora_alpha", 32))
    lora_dropout = float(online_cfg.get("lora_dropout", 0.05))

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_online_grpo.py"),
        "--model_name", str(base_model_name),
        "--dataset", str(dataset_cfg["name"]),
        "--dataset_path", str(REPO_ROOT / dataset_cfg["output_dir"]),
        "--output_dir", str(REPO_ROOT / output_dir),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--num_generations", str(num_generations),
        "--max_prompt_length", str(max_prompt_len),
        "--max_completion_length", str(max_completion_len),
        "--exec_timeout", str(exec_timeout),
        "--save_steps", str(save_steps),
        "--logging_steps", str(logging_steps),
    ]

    if use_lora:
        cmd.append("--use_lora")
        cmd.extend(["--lora_r", str(lora_r)])
        cmd.extend(["--lora_alpha", str(lora_alpha)])
        cmd.extend(["--lora_dropout", str(lora_dropout)])
        cmd.extend(["--target_modules", ",".join(target_modules)])

    # Optional: pass seed from cfg if present
    if cfg.get("seed") is not None:
        cmd.extend(["--seed", str(int(cfg["seed"]))])

    print(f"[i] Launching ONLINE GRPO via: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    return (REPO_ROOT / output_dir).resolve()
