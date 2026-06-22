# RFMiD subset smoke matrix

This is a **real RFMiD image subset** smoke experiment, not the final paper-scale run.

Baseline: `iid_fedavg_clean`, final train loss = 0.503173.

## Summary table

| run | split | mu_FedProx | p_noise | L_final | dL vs iid baseline | upload_bytes |
|-----|-------|-------------|---------|---------|--------------------|--------------|
| dirichlet_a0p5_fedavg_clean | dirichlet_a0p5 | 0.0 | none | 0.470864 | -6.42% | 6389184 |
| dirichlet_a0p5_fedavg_noise_p01 | dirichlet_a0p5 | 0.0 | 0.1 | 0.473298 | -5.94% | 6389184 |
| dirichlet_a0p5_fedprox_clean | dirichlet_a0p5 | 0.05 | none | 0.471724 | -6.25% | 6389184 |
| dirichlet_a0p5_fedprox_noise_p01 | dirichlet_a0p5 | 0.05 | 0.1 | 0.474153 | -5.77% | 6389184 |
| iid_fedavg_clean | iid | 0.0 | none | 0.503173 | +0.00% | 6389184 |
| iid_fedavg_noise_p01 | iid | 0.0 | 0.1 | 0.506467 | +0.65% | 6389184 |
| iid_fedprox_clean | iid | 0.05 | none | 0.503615 | +0.09% | 6389184 |
| iid_fedprox_noise_p01 | iid | 0.05 | 0.1 | 0.506908 | +0.74% | 6389184 |

## Method comparison

- **iid, clean**: FedAvg 0.503173 vs FedProx 0.503615 (+0.09%).
- **dirichlet_a0p5, clean**: FedAvg 0.470864 vs FedProx 0.471724 (+0.18%).

## Robustness / noise ablation

- **iid, FedAvg**: clean 0.503173 -> p=0.1 noise 0.506467 (+0.65%).
- **dirichlet_a0p5, FedAvg**: clean 0.470864 -> p=0.1 noise 0.473298 (+0.52%).

## Paper-facing interpretation

- This validates the proposed pipeline on real RFMiD images: split generation, label-noise protocol, FL loop, and communication accounting all run end-to-end.
- It is still a **smoke-scale** experiment. Paper claims should use the same matrix on full RFMiD plus RETFound/LoRA and stronger baselines.
