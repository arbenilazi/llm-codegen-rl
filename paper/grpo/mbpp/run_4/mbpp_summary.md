# mbpp Summary

- Seed: `42`
- Train/Test split ratio: `0.9`
- RL manifest: `/storage/homefs/ai24v071/llm-codegen-rl/assets/rl/mbpp/manifest.json`
- Checkpoint: `/storage/homefs/ai24v071/llm-codegen-rl/results/dpo_mbpp`

| Metric | Baseline | After | Delta |
| --- | --- | --- | --- |
| pass@1 | 0.468 | 0.523 | +0.055 |
| pass@3 | 0.570 | 0.618 | +0.049 |
| pass@5 | 0.605 | 0.646 | +0.041 |
| pass@10 | 0.650 | 0.668 | +0.018 |
