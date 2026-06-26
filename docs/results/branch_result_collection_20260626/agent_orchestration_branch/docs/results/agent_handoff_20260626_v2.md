# Agent Handoff Results 2026-06-26 v2

## Contents

This snapshot records the latest RETFound + LoRA agent orchestration results imported from
`agent_handoff_bundle_clean_20260626_v2.tar.gz`.

Tracked result files:

- `runs/paper_matrix/agent_stage1/*.json`
- `runs/paper_matrix/agent_stage1/summary.json`
- `runs/paper_matrix/agent/clean_fedavg_s0.json`
- `runs/paper_matrix/agent/clean_agent_s0.json`
- `runs/paper_matrix/agent/het02_fedavg_s0.json`
- `runs/paper_matrix/agent/het02_robust_s0.json`
- `runs/paper_matrix/agent/het02_agent_s0.json`

The handoff status is documented in `docs/AGENT_HANDOFF_STATUS.md`.

## Completion Check

- Stage1 results: 12 / 12 JSON files present, including `summary.json`.
- Full matrix current results: 5 / 5 JSON files present.
- Overall completed matrix progress: 16 / 23 runs.
- Remaining runs: `het04_muonly_s0`, `het04_agentmu_s0`, `het04_agentmu_s1`,
  `het04_agentmu_s2`, `het04_dir_agentmu_s0`, `het04_agent_tau002_s0`,
  `het04_agent_tau005_s0`.

## Key Numbers

Clean control, seed 0:

| run | macro-AUROC | macro-AP | best micro-F1 |
|---|---:|---:|---:|
| `clean_fedavg_s0` | 0.824948 | 0.279640 | 0.659876 |
| `clean_agent_s0` | 0.826474 | 0.290547 | 0.693568 |

Mild heterogeneous noise `het02`, seed 0:

| run | macro-AUROC | macro-AP | best micro-F1 |
|---|---:|---:|---:|
| `het02_fedavg_s0` | 0.721070 | 0.242616 | 0.303548 |
| `het02_robust_s0` | 0.730204 | 0.191450 | 0.595594 |
| `het02_agent_s0` | 0.781214 | 0.266304 | 0.658981 |

Strong heterogeneous noise `het04`, IID, 3 seeds:

| method | macro-AUROC mean | values |
|---|---:|---|
| FedAvg | 0.737991 | 0.737403, 0.737670, 0.738901 |
| Robust-FedProx | 0.668917 | 0.666853, 0.676169, 0.663729 |
| Agent | 0.804335 | 0.790774, 0.810206, 0.812024 |

Non-IID + heterogeneous noise, seed 0:

| run | macro-AUROC |
|---|---:|
| `het04_dir_fedavg_s0` | 0.587643 |
| `het04_dir_agent_s0` | 0.532968 |

## Current Interpretation

The current evidence supports the main story:

- Clean data does not show an agent penalty: `clean_agent_s0` is slightly above
  `clean_fedavg_s0`.
- Under mild heterogeneous noise (`het02`), agent improves macro-AUROC by about
  `+0.0601` over FedAvg and remains clearly above Robust-FedProx.
- Under stronger heterogeneous noise (`het04`, IID, 3 seeds), agent improves mean
  macro-AUROC by about `+0.0663` over FedAvg and beats all FedAvg seeds.
- Static robust regularization is not the right tool in the current setup: it is
  slightly above FedAvg on `het02` macro-AUROC but much worse on `het04`.
- The main open weakness is non-IID + heterogeneous noise, where pure adaptive
  weighting loses to FedAvg. The next decisive run is `het04_dir_agentmu_s0`.

