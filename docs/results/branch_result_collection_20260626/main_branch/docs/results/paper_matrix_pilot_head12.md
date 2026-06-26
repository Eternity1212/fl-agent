# Paper Matrix Summary

| run | method | loss | dropout | best macro-F1 | best micro-F1 | bytes | RETFound |
|-----|--------|------|---------|---------------|---------------|-------|----------|
| pilot_head12_centralized_mlp_s0 | centralized | balanced_bce | 0.0 | 0.323076923076923 | 0.2 | None | False |
| pilot_head12_fedavg_mlp_s0 | fedavg | balanced_bce | 0.0 | 0.323076923076923 | 0.2 | 799936 | False |
| pilot_head12_robust_fedprox_mlp_dropout01_s0 | robust_fedprox | balanced_bce | 0.1 | 0.32867132867132864 | 0.20253164556962025 | 799936 | False |

## Method Mean +/- Std

| method | n | best macro-F1 | best micro-F1 |
|--------|---|---------------|---------------|
| centralized | 1 | 0.323077 +/- 0.000000 | 0.200000 +/- 0.000000 |
| fedavg | 1 | 0.323077 +/- 0.000000 | 0.200000 +/- 0.000000 |
| robust_fedprox | 1 | 0.328671 +/- 0.000000 | 0.202532 +/- 0.000000 |

## Interpretation guardrails

- Rows with `RETFound=False` are fallback/sanity runs and must not be reported as RETFound paper results.
- Validation-calibrated thresholds should be frozen before final test reporting.
