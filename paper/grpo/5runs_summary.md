# GRPO 5-Run Summary

Aggregated from:

- `paper/grpo/mbpp/run_1/mbpp_summary.md` ... `paper/grpo/mbpp/run_5/mbpp_summary.md`
- `paper/grpo/taco_verified/run_1/taco_summary.md` ... `paper/grpo/taco_verified/run_5/taco_summary.md`

Note: artifacts were originally generated in `results/` and later moved to `paper/` for final reporting.

Statistics are computed on the `After` values across 5 runs using sample standard deviation.

## Baseline (Before training)

| Dataset       | pass@1 | pass@3 | pass@5 | pass@10 |
| :------------ | -----: | -----: | -----: | ------: |
| MBPP          |  0.468 |  0.570 |  0.605 |   0.650 |
| TACO Verified |  0.149 |  0.173 |  0.179 |   0.184 |

## MBPP (GRPO, 5 runs)

| Metric  |     Mean |      Std |   Median |      Min |      Max |
| :------ | -------: | -------: | -------: | -------: | -------: |
| pass@1  | 0.519800 | 0.001789 | 0.519000 | 0.519000 | 0.523000 |
| pass@3  | 0.617200 | 0.000447 | 0.617000 | 0.617000 | 0.618000 |
| pass@5  | 0.649200 | 0.001789 | 0.650000 | 0.646000 | 0.650000 |
| pass@10 | 0.680800 | 0.007155 | 0.684000 | 0.668000 | 0.684000 |

## TACO Verified (GRPO, 5 runs)

| Metric  |     Mean |      Std |   Median |      Min |      Max |
| :------ | -------: | -------: | -------: | -------: | -------: |
| pass@1  | 0.172000 | 0.000000 | 0.172000 | 0.172000 | 0.172000 |
| pass@3  | 0.184000 | 0.000000 | 0.184000 | 0.184000 | 0.184000 |
| pass@5  | 0.184000 | 0.000000 | 0.184000 | 0.184000 | 0.184000 |
| pass@10 | 0.184000 | 0.000000 | 0.184000 | 0.184000 | 0.184000 |
