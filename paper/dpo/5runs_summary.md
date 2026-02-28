# DPO 5-Run Summary

Aggregated from:

- `paper/dpo/mbpp/run_1/mbpp_summary.md` ... `paper/dpo/mbpp/run_5/mbpp_summary.md`
- `paper/dpo/taco_verified/run_1/taco_summary.md` ... `paper/dpo/taco_verified/run_5/taco_summary.md`

Note: artifacts were originally generated in `results/` and later moved to `paper/` for final reporting.

Statistics are computed on the `After` values across 5 runs using sample standard deviation.

## Baseline (Before training)

| Dataset       | pass@1 | pass@3 | pass@5 | pass@10 |
| :------------ | -----: | -----: | -----: | ------: |
| MBPP          |  0.468 |  0.570 |  0.605 |   0.650 |
| TACO Verified |  0.149 |  0.173 |  0.179 |   0.184 |

## MBPP (DPO, 5 runs)

| Metric  |     Mean |      Std |   Median |      Min |      Max |
| :------ | -------: | -------: | -------: | -------: | -------: |
| pass@1  | 0.473400 | 0.002608 | 0.472000 | 0.472000 | 0.478000 |
| pass@3  | 0.571400 | 0.000894 | 0.571000 | 0.571000 | 0.573000 |
| pass@5  | 0.602800 | 0.000447 | 0.603000 | 0.602000 | 0.603000 |
| pass@10 | 0.635200 | 0.001789 | 0.636000 | 0.632000 | 0.636000 |

## TACO Verified (DPO, 5 runs)

| Metric  |     Mean |      Std |   Median |      Min |      Max |
| :------ | -------: | -------: | -------: | -------: | -------: |
| pass@1  | 0.153000 | 0.000000 | 0.153000 | 0.153000 | 0.153000 |
| pass@3  | 0.179000 | 0.000000 | 0.179000 | 0.179000 | 0.179000 |
| pass@5  | 0.182000 | 0.000000 | 0.182000 | 0.182000 | 0.182000 |
| pass@10 | 0.184000 | 0.000000 | 0.184000 | 0.184000 | 0.184000 |
