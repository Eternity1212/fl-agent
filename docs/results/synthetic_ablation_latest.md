# Synthetic ablation & method comparison

Baseline run: **`fedavg_clean`** (final loss = 0.693447).

## Summary table

| run | mu_FedProx | p_noise | L_final | dL vs base | upload_bytes |
|-----|-----------|---------|---------|------------|--------------|
| fedavg_clean | 0.0 | none | 0.693447 | +0.00% | 789536 |
| fedavg_noise_p01 | 0.0 | 0.1 | 0.693478 | +0.00% | 789536 |
| fedavg_noise_p05 | 0.0 | 0.5 | 0.686091 | -1.06% | 789536 |
| fedprox_mu005_clean | 0.05 | none | 0.693447 | +0.00% | 789536 |
| fedprox_mu005_noise_p01 | 0.05 | 0.1 | 0.693478 | +0.00% | 789536 |

## Ablation: FedProx (hold noise fixed)

- **no YAML (no injected label noise)**: compare `fedavg_clean` (mu=0.0) vs `fedprox_mu005_clean` (mu=0.05); loss 0.693447 -> 0.693447
- **p_flip = 0.1**: compare `fedavg_noise_p01` (mu=0.0) vs `fedprox_mu005_noise_p01` (mu=0.05); loss 0.693478 -> 0.693478

## Ablation: label noise (hold FedProx off)

- `fedavg_clean`: noise_p_flip=None, final loss=0.693447
- `fedavg_noise_p01`: noise_p_flip=0.1, final loss=0.693478
- `fedavg_noise_p05`: noise_p_flip=0.5, final loss=0.686091

## Notes

- **Upload bytes** usually match across runs here (same model shape and rounds); differences appear mainly in final loss.
- Synthetic 4-sample setup: for **RFMiD** scale-up, reuse the same spec names with `run_fed_smoke` paths — see `docs/EXPERIMENTS.md`.
