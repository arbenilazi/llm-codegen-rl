# Improving the Capabilities of LLMs for Code Generation Using Reinforcement Learning

This repository contains the implementation and experiment pipeline on reinforcement learning for code generation. The goal is to improve executable correctness of generated Python code on benchmark datasets.

## Project Scope

The pipeline supports:

- baseline evaluation of an instruction-tuned code model,
- candidate generation on train/test splits,
- execution-based annotation of generated candidates,
- RL dataset construction (ranked candidates and preference pairs),
- offline RL fine-tuning with DPO,
- online RL fine-tuning with GRPO,
- post-training evaluation and report generation.

## Repository Structure

```text
.
├── configs/                    # Experiment configurations
├── pipeline/
│   ├── run.py                  # Main orchestrator
│   ├── annotate.py             # Candidate execution + annotation
│   ├── online_grpo.py          # Online GRPO stage wrapper
│   └── analyzer/               # Optional analysis utilities
├── scripts/
│   ├── build_rl_dataset.py     # Build nbest/pairs + manifest
│   ├── train_dpo.py            # Offline DPO training
│   ├── train_online_grpo.py    # Online GRPO training
│   └── merge_lora.py           # LoRA merge models
├── src/
│   ├── dataset.py              # Dataset preparation/loading
│   ├── predictor.py            # Multi-candidate generation/evaluation
│   ├── model.py                # Inference backend abstraction
│   └── adapters/               # Dataset-specific adapters
├── assets/                     # Generated datasets and RL artifacts
├── results/                    # Checkpoints and reports
└── requirements.txt
```

## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Datasets are not bundled in this repository and must be downloaded manually to `assets/datasets/` before running experiments.

Recommended environment:

- Linux with NVIDIA GPU and CUDA for full experiments (we ran experiments on the UniBE Ubelix cluster),
- Python environment compatible with the versions in `requirements.txt`.

## Authentication Variables

Create a local `.env` file (do not commit it):

```bash
HUGGING_FACE_HUB_TOKEN=YOUR_TOKEN_HERE
WANDB_API_KEY=YOUR_TOKEN_HERE
```

## Configuration

The pipeline is fully configuration-driven.

Experiments use:

- `configs/mbpp.json` for MBPP,
- `configs/taco.json` for TACO-verified.

These configuration files define all key experimental parameters, including:

- generation settings (e.g., `num_candidates`, `temperature`, `top_p`, `max_tokens`),
- training settings (e.g., `learning rate`, `batch size`, `epochs`),
- RL dataset construction settings (e.g., `n_best`, pair construction),
- reporting/output paths.

## Pipeline Stages

Run stages with:

```bash
python -m pipeline.run --config configs/mbpp.json --only <stages>
```

Supported values for `--only`:

- `baseline`: baseline generation and pass@k evaluation on test split.
- `train_gen`: candidate generation on train split plus annotation.
- `rl_build`: RL artifact construction from annotated candidates.
- `train_all`: offline DPO training stage in the current orchestrator implementation.
- `test_after`: post-training generation and evaluation on test split.
- `online_grpo`: online GRPO training stage.
- `analyze`: optional analysis stage (candidate diversity, similarity, and summary statistics).

Recommended stage sequences used:

- Offline DPO: `baseline -> train_gen -> rl_build -> train_all -> test_after`
- Online GRPO: `baseline -> online_grpo -> test_after`

## LoRA Merge

For both RL methods, we merge the base model and the LoRA adapter into a single merged model directory before `test_after` so evaluation can run with vLLM again (faster inference). In practice:

- DPO: merge after `train_all`.
- Online GRPO: merge after `online_grpo`.

1. Set `base_model`, `lora_path`, and `output_path` in `scripts/merge_lora.py`.
2. Run:

```bash
python scripts/merge_lora.py
```

3. Update `generation.model_name` in the config to the merged model path.
4. Run:

```bash
python -m pipeline.run --config configs/mbpp.json --only test_after
```

## Output Artifacts

Main generated files:

- `assets/datasets/<dataset>/test/evaluations_baseline.json`
- `assets/datasets/<dataset>/test/evaluations_after.json`

- `assets/datasets/<dataset>/train/candidates.jsonl`
- `assets/datasets/<dataset>/train/candidates_annotated.jsonl`

- `assets/rl/<dataset>/nbest.jsonl`
- `assets/rl/<dataset>/pairs.jsonl`
- `assets/rl/<dataset>/manifest.json`

- `results/reports/<experiment>_summary.json`
- `results/reports/<experiment>_summary.md`

## Datasets

Currently supported:

- **MBPP**: `google-research-datasets/mbpp`
- **TACO-verified**: `likaixin/TACO-verified`

To add a new dataset, create a new adapter under `src/adapters/` and register it through the dataset interface.

## Safety and Limitations

- The pipeline executes generated code for correctness evaluation; run in controlled environments.
- Results can vary with decoding settings, random seeds, software versions, and hardware.
- Large-scale runs are intended for GPU environments.
