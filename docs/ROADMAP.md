# Roadmap (execution order)

This file tracks the **implementation order** aligned with the project plan (P2 headline, P1 core, P3 optional).

## Done

- **v0.1**: `RFMiDLocalDataset`, optional HF smoke CLI, unit tests, CI (`ruff` + `pytest`).
- **v0.1.1**: Git sync scripts (`publish.sh`, etc.) + SSH workflow docs.
- **v0.2**: Split JSON builders (**IID / Dirichlet-primary / domain-hash**), CLI `python -m fed_agent.tools.build_splits`, optional `RFMiDTorchDataset` (requires `pip install -e ".[torch]"`).
- **v0.3 (current)**: Federated **smoke** loop — FedAvg / FedProx, per-round **upload bytes**, CLI `python -m fed_agent.tools.run_fed_smoke` (see [docs/FEDERATION.md](FEDERATION.md)); `noise_protocol@v1` **parsed and applied** in `RFMiDTorchDataset` when `--noise_protocol_yaml` is set; `summarize_fed_smoke` for reading `--out_json` results.
- **Synthetic minimal experiment**: `python -m fed_agent.tools.minimal_experiment` and `./scripts/run_minimal_baseline.sh` — reproducible 4-sample / 2-client baseline; snapshot under `docs/results/` (see [docs/STATUS.md](STATUS.md), [docs/EXPERIMENTS.md](EXPERIMENTS.md)).
- **Scope note**: software “done” vs grant workstreams — [docs/PROJECT_SCOPE.md](PROJECT_SCOPE.md).

## Next (v0.4+)

- **Stronger noise + baselines**: asymmetric / class-conditional noise; FedDiv-style hooks and ablations.
- **PEFT**: RETFound + LoRA integration (torch required), frozen backbone defaults.
- **Agent (P3)**: `RuleAgent` discrete actions + comm-budget plots.

## Smoke

Local:

```bash
./scripts/smoke.sh
```
