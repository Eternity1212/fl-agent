# RETFound + LoRA optimized full-matrix results summary

Source run directory (gitignored): `runs/paper_matrix/retfound_full_optimized_resume/`

This summary reflects the completed optimized matrix run from `configs/paper_matrix_gpu_full_optimized.yaml`.

## Executive summary

- The optimized matrix completed **29/29 runs** successfully.
- **RETFound + LoRA** is effective on RFMiD in the centralized setting.
- **FedAvg** is the strongest and most stable federated baseline across the clean 3-seed comparison.
- **FedProx** and **Robust-FedProx** did **not** outperform FedAvg in the clean setting.
- Early noise / non-IID ablations also do **not** show Robust-FedProx overtaking FedAvg.

## Clean setting (main comparison)

### Centralized RETFound + LoRA

| Run | micro-F1 | macro-F1-present | macro-AP | macro-AUROC |
|---|---:|---:|---:|---:|
| centralized_s0 | 0.6865 | 0.2273 | 0.2866 | 0.8139 |
| centralized_s1 | 0.6634 | 0.2458 | 0.2764 | 0.8159 |
| centralized_s2 | 0.6695 | 0.2213 | 0.2854 | 0.8092 |

### Local-only RETFound + LoRA

| Run | micro-F1 | macro-F1-present |
|---|---:|---:|
| local_only_s0 | 0.5559 | 0.1580 |
| local_only_s1 | 0.5581 | 0.1581 |
| local_only_s2 | 0.5410 | 0.1536 |

### Federated comparison

| Run | micro-F1 | macro-F1-present | best-micro-F1 | best-macro-F1-present | macro-AP | macro-AUROC |
|---|---:|---:|---:|---:|---:|---:|
| fedavg_s0 | 0.6310 | 0.2259 | 0.6646 | 0.2391 | 0.2679 | 0.8016 |
| fedavg_s1 | 0.6263 | 0.2183 | 0.6549 | 0.2240 | 0.2754 | 0.7847 |
| fedavg_s2 | 0.6270 | 0.2485 | 0.6833 | 0.2562 | 0.2972 | 0.8339 |
| fedprox_s0 | 0.5372 | 0.1586 | 0.6243 | 0.1730 | 0.2034 | 0.7683 |
| fedprox_s1 | 0.5337 | 0.1639 | 0.6145 | 0.1677 | 0.1958 | 0.7581 |
| fedprox_s2 | 0.5368 | 0.1651 | 0.6142 | 0.1813 | 0.1993 | 0.7691 |
| robust_fedprox_s0 | 0.5303 | 0.1563 | 0.6291 | 0.1696 | 0.1932 | 0.7685 |
| robust_fedprox_s1 | 0.5520 | 0.1624 | 0.6242 | 0.1714 | 0.1946 | 0.7484 |
| robust_fedprox_s2 | 0.5506 | 0.1657 | 0.6096 | 0.1759 | 0.1976 | 0.7652 |

## Main finding from clean setting

Across all three seeds, **FedAvg is consistently stronger than both FedProx and Robust-FedProx** on the current RETFound + LoRA RFMiD setup.

This means the strongest supported clean-setting story is:

> RETFound + LoRA is effective for RFMiD, and FedAvg is the strongest stable federated baseline in this setup.

The originally hoped-for clean-setting story — that **Robust-FedProx + positive-label dropout** would outperform FedAvg — is **not supported** by the completed 3-seed main comparison.

## Early ablation readout

### Label noise p=0.2

| Run | micro-F1 | macro-F1-present | macro-AP | macro-AUROC |
|---|---:|---:|---:|---:|
| ablate_fedavg_noise02 | 0.5975 | 0.2244 | 0.2614 | 0.7906 |
| ablate_robust_noise02 | 0.5254 | 0.1613 | 0.1754 | 0.7550 |

### Label noise p=0.4

| Run | micro-F1 | macro-F1-present | macro-AP | macro-AUROC |
|---|---:|---:|---:|---:|
| ablate_fedavg_noise04 | 0.5899 | 0.1964 | 0.2324 | 0.7754 |
| ablate_robust_noise04 | see raw JSON in `runs/paper_matrix/retfound_full_optimized_resume/` |

### Dirichlet non-IID (alpha=0.1)

| Run | micro-F1 | macro-F1-present | macro-AP | macro-AUROC |
|---|---:|---:|---:|---:|
| ablate_fedavg_dir_a0p1 | 0.6799 | 0.2237 | 0.2609 | 0.8082 |
| ablate_robust_dir_a0p1 | 0.5447 | 0.1715 | 0.2042 | 0.7698 |

## Interpretation

### What is supported

1. **RETFound + LoRA works** on RFMiD.
2. **Federated learning is useful** because it outperforms local-only.
3. **FedAvg is the strongest current federated baseline** in the clean main comparison.
4. The optimized training path reproduces the same qualitative conclusions while finishing the full matrix successfully.

### What is not supported

1. Clean-setting superiority of **FedProx** over FedAvg.
2. Clean-setting superiority of **Robust-FedProx** over FedAvg.
3. A strong claim that the current robust regularization is already the best headline method.

## Suggested paper-story adjustment

A defensible story from the completed experiments is:

> We establish a RETFound + LoRA federated benchmark on RFMiD and show that FedAvg is a strong, stable baseline. In contrast, straightforward FedProx and Robust-FedProx variants do not automatically improve over FedAvg in this foundation-model + PEFT regime, indicating that more suitable robust federated objectives remain an open problem.

## Reproducibility notes

- Optimized matrix config: `configs/paper_matrix_gpu_full_optimized.yaml`
- Raw per-run results: `runs/paper_matrix/retfound_full_optimized_resume/*.json` (gitignored)
- Full combined summary: `runs/paper_matrix/retfound_full_optimized_resume/summary.json` (gitignored)
