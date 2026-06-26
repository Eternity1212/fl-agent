# Paper Matrix Summary

| run | method | loss | dropout | best macro-F1 | best micro-F1 | bytes | RETFound |
|-----|--------|------|---------|---------------|---------------|-------|----------|
| full46_centralized_mlp_s0 | centralized | balanced_bce | 0.0 | 0.07148399055250683 | 0.11005922678729656 | None | False |
| full46_local_only_mlp_s0 | local_only | balanced_bce | 0.0 | 0.07155896207542753 | 0.08042892242932287 | None | False |
| full46_fedavg_mlp_s0 | fedavg | balanced_bce | 0.0 | 0.07119842022433212 | 0.09533753664511466 | 835296 | False |
| full46_fedprox_mlp_s0 | fedprox | balanced_bce | 0.0 | 0.07160232171400144 | 0.11935451131635035 | 835296 | False |
| full46_robust_fedprox_mlp_dropout01_s0 | robust_fedprox | balanced_bce | 0.1 | 0.07145296787141345 | 0.08238900561430097 | 835296 | False |
| full46_centralized_mlp_s1 | centralized | balanced_bce | 0.0 | 0.07175804665757442 | 0.10864765409383624 | None | False |
| full46_local_only_mlp_s1 | local_only | balanced_bce | 0.0 | 0.07172391434605538 | 0.08198933233220428 | None | False |
| full46_fedavg_mlp_s1 | fedavg | balanced_bce | 0.0 | 0.07039457975541276 | 0.1073135211395804 | 835296 | False |
| full46_fedprox_mlp_s1 | fedprox | balanced_bce | 0.0 | 0.07235227586389262 | 0.09835794350002068 | 835296 | False |
| full46_robust_fedprox_mlp_dropout01_s1 | robust_fedprox | balanced_bce | 0.1 | 0.0721928944720433 | 0.0997010652183066 | 835296 | False |
| full46_centralized_mlp_s2 | centralized | balanced_bce | 0.0 | 0.07153958338292901 | 0.10549332857334107 | None | False |
| full46_local_only_mlp_s2 | local_only | balanced_bce | 0.0 | 0.07164832683058966 | 0.08366978159903139 | None | False |
| full46_fedavg_mlp_s2 | fedavg | balanced_bce | 0.0 | 0.0705172220096279 | 0.11435795205912384 | 835296 | False |
| full46_fedprox_mlp_s2 | fedprox | balanced_bce | 0.0 | 0.07175953511492347 | 0.08591250584133146 | 835296 | False |
| full46_robust_fedprox_mlp_dropout01_s2 | robust_fedprox | balanced_bce | 0.1 | 0.0719234200359774 | 0.08665385591530401 | 835296 | False |

## Method Mean +/- Std

| method | n | best macro-F1 | best micro-F1 |
|--------|---|---------------|---------------|
| centralized | 3 | 0.071594 +/- 0.000145 | 0.108067 +/- 0.002338 |
| fedavg | 3 | 0.070703 +/- 0.000433 | 0.105670 +/- 0.009616 |
| fedprox | 3 | 0.071905 +/- 0.000395 | 0.101208 +/- 0.016902 |
| local_only | 3 | 0.071644 +/- 0.000083 | 0.082029 +/- 0.001621 |
| robust_fedprox | 3 | 0.071856 +/- 0.000374 | 0.089581 +/- 0.009020 |

## Interpretation guardrails

- Rows with `RETFound=False` are fallback/sanity runs and must not be reported as RETFound paper results.
- Validation-calibrated thresholds should be frozen before final test reporting.
