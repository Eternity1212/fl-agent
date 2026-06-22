# RFMiD subset smoke matrix

This is a **real RFMiD image subset** smoke experiment, not the final paper-scale run.

Primary smoke metric: validation best-threshold `macro_f1_present` (higher is better).

Baseline: `iid_fedavg_bce_clean`, macro-F1 = 0.134370.

## Key finding

- Best run: `iid_fedprox_balanced_noise_p01` with best macro-F1 0.143479 (+6.78% vs BCE baseline).
- In this smoke setting, the useful signal is **noise/dropout-style robust training under class imbalance**, not FedProx alone.

## Summary table

| run | split | loss | noise | best macro-F1 | t* | macro@0.5 | micro@0.5 | bytes |
|-----|-------|------|-------|---------------|----|-----------|-----------|-------|
| iid_fedavg_balanced_clean | iid | balanced_bce | none | 0.130674 | 0.45 | 0.081391 | 0.124294 | 2399808 |
| iid_fedavg_balanced_noise_p01 | iid | balanced_bce | 0.1 | 0.142158 | 0.45 | 0.044488 | 0.099602 | 2399808 |
| iid_fedavg_bce_clean | iid | bce | none | 0.134370 | 0.10 | 0.000000 | 0.000000 | 2399808 |
| iid_fedprox_balanced_clean | iid | balanced_bce | none | 0.130032 | 0.45 | 0.084633 | 0.127101 | 2399808 |
| iid_fedprox_balanced_noise_p01 | iid | balanced_bce | 0.1 | 0.143479 | 0.45 | 0.054582 | 0.102564 | 2399808 |

## Method comparison: class-balanced loss

- **iid**: BCE best macro-F1 0.134370 -> balanced BCE 0.130674 (-2.75%).

## Method comparison: FedProx on balanced loss

- **iid**: FedAvg best macro-F1 0.130674 vs FedProx 0.130032.

## Robustness / noise ablation

- **iid, balanced FedAvg**: clean best macro-F1 0.130674 -> p=0.1 noise 0.142158.

## Paper-facing interpretation

- This validates the proposed pipeline on real RFMiD images: split generation, validation metrics, label-noise protocol, FL loop, and communication accounting.
- The current positive signal is to treat mild positive-label dropout/noise as a robustness regularizer under severe label imbalance.
- It is still a **smoke-scale** experiment. Paper claims should use the same matrix on full RFMiD plus RETFound/LoRA and stronger baselines.
