# mbpp Summary

- Seed: `42`
- Train/Test split ratio: `0.9`
- RL manifest: `/storage/homefs/ai24v071/llm-codegen-rl/assets/rl/mbpp/manifest.json`
- Checkpoint: `/storage/homefs/ai24v071/llm-codegen-rl/results/dpo_mbpp`

| Metric | Baseline | After | Delta |
| --- | --- | --- | --- |
| pass@1 | 0.468 | 0.473 | +0.004 |
| pass@3 | 0.570 | 0.571 | +0.002 |
| pass@5 | 0.605 | 0.603 | -0.002 |
| pass@10 | 0.650 | 0.636 | -0.014 |
