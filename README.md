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

## 一条命令跑全部（推荐入口）

clone 到任意机器后，直接：

```bash
./run.sh
```

它会自动安装依赖、下载并校验完整 RFMiD、生成联邦划分、（有 `HF_TOKEN`+GPU 时）
下载 RETFound 权重并跑论文矩阵，否则跑 MLP fallback，最后汇总所有结果。

- 有 token：`export HF_TOKEN=hf_xxx && ./run.sh` → 真正 RETFound 结果
- 无 token：`./run.sh` → fallback sanity（结果标记 `RETFound=False`，不能写论文）
- 常用变量：`MODE=auto|retfound|fallback`、`SKIP_INSTALL=1`、`SEEDS="0 1 2"`

> **数据集与模型权重不放进 Git**（许可与体积原因），由 `run.sh` 运行时从官方源
> 自动下载。RETFound 是 gated 仓库，需先在网页同意条款并配置 token，详见
> [docs/RETFOUND_GPU_RUN.md](docs/RETFOUND_GPU_RUN.md)。

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

**一条命令跑「真实 RFMiD 子集 smoke」**（从 Hugging Face 镜像导出 96 张真实图像）：

```bash
python3 -m pip install -e ".[data,torch]"
./scripts/run_rfmid_subset_smoke.sh
```

产物：`docs/results/rfmid_subset_smoke_latest.md`，以及被 gitignore 的
`runs/rfmid_smoke_matrix/latest/summary.{json,csv}`。

**RETFound + LoRA 论文级管线（v0.4）**：

```bash
python3 -m pip install -e ".[data,torch,paper]"

# 下载完整 RFMiD train/validation/test（支持断点跳过已存在图片）
python3 -m fed_agent.tools.export_hf_rfmid_subset \
  --split all --out_dir data/raw/rfmid_full --max_samples 0 --validate

# head-12 快速 sanity：当前会清楚标记 RETFound=False fallback 结果
./scripts/run_paper_pilot.sh

# full-46 MLP fallback 主矩阵；正式 RETFound 主矩阵见 configs/paper_matrix.yaml
./scripts/run_paper_full46_mlp.sh
```

产物：`docs/results/paper_matrix_pilot_head12.md`、
`docs/results/paper_matrix_full46_mlp.md`（运行 full 脚本后生成）和
`runs/paper_matrix/*/summary.{json,csv}`。

**真正的 RETFound 论文主结果（需 GPU + HF token）**：RETFound 权重是 Hugging Face
gated 仓库，需先申请权限并配置 `HF_TOKEN`。完整步骤见
**[docs/RETFOUND_GPU_RUN.md](docs/RETFOUND_GPU_RUN.md)**，迁移到 GPU 机器后：

```bash
export HF_TOKEN=hf_你的token
python3 -m fed_agent.tools.check_env   # 预检 GPU / token / gated 访问
./scripts/run_paper_gpu.sh             # 一键跑 configs/paper_matrix.yaml
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
