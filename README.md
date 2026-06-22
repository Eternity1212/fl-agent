# fl-agent

Federated learning + **RuleAgent / bandit-style orchestration** for **multi-label retinal fundus** tasks (default dataset: **RFMiD**).  
This repo follows the technical plan: **P2 (RETFound + PEFT) headline**, **P1 (noise-robust FL) core**, **P3 (optional agent)**.

## Branches

- `main` — release-ready, tagged (`v0.x`).
- `dev` — integration branch; merge via PR from `feat/*`.
- `feat/*` — short-lived feature branches.

See [docs/BRANCHING.md](docs/BRANCHING.md), [docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md), **[docs/SYNC.md](docs/SYNC.md)**, **[docs/STATUS.md](docs/STATUS.md)**（当前做到哪一步、实验是否开始）, **[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)**（环境安装）, **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**（实验设计）, **[docs/ROADMAP.md](docs/ROADMAP.md)**, **[docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md)**, and **[docs/FEDERATION.md](docs/FEDERATION.md)**.

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
./scripts/smoke.sh
```

**当前进度与实验说明**：先读 [docs/STATUS.md](docs/STATUS.md) 与 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)；环境细节见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

**一条命令跑「合成最小 baseline」**（会安装 torch，若尚未安装）：

```bash
./scripts/run_minimal_baseline.sh
```

产物：`runs/minimal/metrics.json`（默认 gitignore）与可提交的摘要 `docs/results/minimal_synthetic_baseline.md`（脚本运行后更新）。

**一条命令跑「合成消融/对比实验」**：

```bash
# 无需 torch，立刻产出 docs/results/numpy_synthetic_ablation_latest.md
./scripts/run_numpy_ablations.sh

# Torch/TinyMLP 版，产出 docs/results/synthetic_ablation_latest.md
./scripts/run_synthetic_ablations.sh
```

```bash
# Full test matrix including torch-only tests (installs torch)
# ./scripts/smoke-torch.sh

# Build split JSON (after you have labels CSV locally)
python3 -m fed_agent.tools.build_splits --labels_csv path/to/RFMiD_Training_Labels.csv

# Federated smoke (requires torch + split JSON + images); see docs/FEDERATION.md
# python3 -m pip install -e ".[torch]"
# python3 -m fed_agent.tools.run_fed_smoke \
#   --labels_csv ... --images_dir ... --split_json ... \
#   --noise_protocol_yaml configs/noise_protocol/example_v1.yaml \
#   --out_json runs/metrics.json
# python3 -m fed_agent.tools.summarize_fed_smoke runs/metrics.json

# Optional torch bridge tests / training (install torch)
# python3 -m pytest -q tests/test_rfmid_torch.py
```

## 使用跑出来的结果（中文）

1. 联邦 smoke 加 `--out_json runs/metrics.json` 会生成一份 JSON，里面有每一轮平均训练损失、各轮上传字节等。
2. 用 `python3 -m fed_agent.tools.summarize_fed_smoke runs/metrics.json` 可以把 JSON **读成人能看懂的摘要**（已在示例文件上演示过）。
3. **「整个课题做完」**和**「这个代码仓库的软件交付」**不是一回事；仓库里能稳定跑通的部分写在 [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md)，论文里还要做的 RETFound 全量训练、多中心数据、临床分析等仍按 [docs/ROADMAP.md](docs/ROADMAP.md) 推进。

## License

Project code: **MIT** (see `LICENSE`). **Dataset licenses are separate** — see `docs/DATA_CARD.md`.
