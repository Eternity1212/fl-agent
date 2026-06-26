# Branch Result Collection 2026-06-26
This folder consolidates result artifacts that were split across Git branches. It is generated from remote branch contents, not from untracked local files.
## Which Branch To Download For The Next Run
Use `feat/agent-orchestration` / `origin/feat/agent-orchestration` for the next agent continuation run. This is the only checked branch that contains all of the following together: agent code, agent configs, `run_agent.sh`, `summarize_agent`, handoff docs, and the latest 16/23 agent result state.
Do not use `main` for the next agent run yet: it has the older baseline docs but not the agent runner/config/result JSON. Do not use `results/retfound-optimized` for agent continuation: it is an archive branch for the 29-run optimized RETFound baseline matrix.
## Branch Map
- `origin/main`: baseline project docs and old `docs/results` summaries. No latest agent JSON.
- `origin/results/retfound-optimized`: 29-run optimized RETFound matrix archived under `results_archive/retfound_optimized/`.
- `origin/feat/agent-orchestration`: latest adaptive agent code/configs plus agent result JSON under `runs/paper_matrix/agent_stage1/` and `runs/paper_matrix/agent/`.
## Copied Contents
- `main_branch/docs/results/`: result summaries copied from `origin/main`.
- `retfound_optimized_branch/results_archive/retfound_optimized/`: optimized RETFound archive copied from `origin/results/retfound-optimized`.
- `agent_orchestration_branch/`: latest agent handoff docs, logs, and JSON copied from `origin/feat/agent-orchestration`.
- `branch_inventory.md`: complete result-related path inventory by branch.
- `metrics_summary.csv`: flat metric table for optimized RETFound and agent runs.
- `paper_readiness_analysis.md`: paper-readiness judgment, evidence strength, missing runs, and recommended next experiments.
## Current Result Status
### Optimized RETFound Baseline Matrix
- Source branch: `origin/results/retfound-optimized`
- Runs archived: 29
- Main 3-seed method means:

| method | macro-AUROC | best micro-F1 | best macro-F1 present |
|---|---:|---:|---:|
| `centralized` | 0.812984 +/- 0.002800 (n=3) | 0.652630 +/- 0.018659 (n=3) | 0.243906 +/- 0.001550 (n=3) |
| `fedavg` | 0.806720 +/- 0.020436 (n=3) | 0.667587 +/- 0.011778 (n=3) | 0.239794 +/- 0.013167 (n=3) |
| `fedprox` | 0.765164 +/- 0.005030 (n=3) | 0.617639 +/- 0.004699 (n=3) | 0.174025 +/- 0.005583 (n=3) |
| `local_only` | NA | 0.567059 +/- 0.020096 (n=3) | 0.162319 +/- 0.003886 (n=3) |
| `robust_fedprox` | 0.760703 +/- 0.008781 (n=3) | 0.620961 +/- 0.008283 (n=3) | 0.172296 +/- 0.002647 (n=3) |

Interpretation: in the optimized baseline matrix, FedAvg is close to centralized and stronger than fixed FedProx / Robust-FedProx. This is the baseline evidence, not the adaptive agent evidence.
### Adaptive Agent Matrix
- Source branch: `origin/feat/agent-orchestration`
- Completed agent matrix state: 16 / 23 runs.
- Copied JSON files: 12 stage1 files including `summary.json`, plus 5 current full-matrix result files.
- Remaining runs: `het04_muonly_s0`, `het04_agentmu_s0`, `het04_agentmu_s1`, `het04_agentmu_s2`, `het04_dir_agentmu_s0`, `het04_agent_tau002_s0`, `het04_agent_tau005_s0`.

| comparison | macro-AUROC result | take-away |
|---|---:|---|
| `clean_fedavg_s0` | 0.824948 |  |
| `clean_agent_s0` | 0.826474 | clean control: no drop versus FedAvg |
| `het02_fedavg_s0` | 0.721070 |  |
| `het02_robust_s0` | 0.730204 |  |
| `het02_agent_s0` | 0.781214 | mild heterogeneous noise: agent strong gain |
| `het04_dir_fedavg_s0` | 0.587643 |  |
| `het04_dir_agent_s0` | 0.532968 | non-IID weakness for pure weighting |
| `het04 FedAvg` | 0.737991 +/- 0.000653 (n=3) | 3-seed IID heterogeneous noise |
| `het04 Robust-FedProx` | 0.668917 +/- 0.005284 (n=3) | 3-seed IID heterogeneous noise |
| `het04 Agent` | 0.804335 +/- 0.009618 (n=3) | 3-seed IID heterogeneous noise |

Interpretation: clean does not drop, `het02` already shows a clear agent gain, and `het04` IID 3-seed results show the strongest positive evidence. The current open problem is non-IID + heterogeneous noise, where pure weighting loses; `het04_dir_agentmu_s0` is the key next run.
## Recommended Next Commands
```bash
git clone git@github.com:Eternity1212/fl-agent.git
cd fl-agent
git checkout feat/agent-orchestration
# then prepare the GPU/RETFound environment as described in docs/AGENT_HANDOFF_STATUS.md
./run_agent.sh full
```
For a precise continuation, run only the remaining missing names listed above with `python3 -m fed_agent.tools.run_paper_matrix --matrix_yaml configs/paper_matrix_agent.yaml --out_dir runs/paper_matrix/agent --only ...`.
