# Roadmap (execution order)

This file tracks the **implementation order** aligned with the project plan (P2 headline, P1 core, P3 optional).

## Done

- **v0.1**: `RFMiDLocalDataset`, optional HF smoke CLI, unit tests, CI (`ruff` + `pytest`).
- **v0.1.1**: Git sync scripts (`publish.sh`, etc.) + SSH workflow docs.
- **v0.2 (current)**: Split JSON builders (**IID / Dirichlet-primary / domain-hash**), CLI `python -m fed_agent.tools.build_splits`, optional `RFMiDTorchDataset` (requires `pip install -e ".[torch]"`).

## Next (v0.3+)

- **Federated simulator skeleton**: `Server`/`Client` loop, FedAvg/FedProx, log comm bytes.
- **Noise protocol**: `noise_protocol@v1` injection + FedDiv-style baseline hooks.
- **PEFT**: RETFound + LoRA integration (torch required), frozen backbone defaults.
- **Agent (P3)**: `RuleAgent` discrete actions + comm-budget plots.

## Smoke

Local:

```bash
./scripts/smoke.sh
```
