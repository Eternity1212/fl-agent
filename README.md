# fl-agent

Federated learning + **RuleAgent / bandit-style orchestration** for **multi-label retinal fundus** tasks (default dataset: **RFMiD**).  
This repo follows the technical plan: **P2 (RETFound + PEFT) headline**, **P1 (noise-robust FL) core**, **P3 (optional agent)**.

## Branches

- `main` — release-ready, tagged (`v0.x`).
- `dev` — integration branch; merge via PR from `feat/*`.
- `feat/*` — short-lived feature branches.

See [docs/BRANCHING.md](docs/BRANCHING.md), [docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md), and **[docs/SYNC.md](docs/SYNC.md)** (daily pull/push).

After SSH key setup, routine publish for the **current branch**:

```bash
./scripts/publish.sh
```

Legacy: push only `main` + `dev`:

```bash
./scripts/push-github.sh
```

## Local clone path (this machine)

`/Users/bytedance/projects/fl-agent`

## Data (not committed)

- Primary dataset: **RFMiD 1.0** (multi-label fundus). Download per [docs/DATA_CARD.md](docs/DATA_CARD.md).
- Raw images stay under `data/raw/` (gitignored).

## Quickstart (skeleton)

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -e ".[dev]"
python3 -m ruff check src tests
python3 -m pytest -q
# Step 1: RFMiDLocalDataset + tests (docs/DATA_CARD.md)
# Next: split JSON builders + torch bridge + RETFound/LoRA
```

## License

Project code: **MIT** (see `LICENSE`). **Dataset licenses are separate** — see `docs/DATA_CARD.md`.
