# 环境安装

目标：能跑 **lint + 单测**，以及可选的 **Torch + 联邦 smoke / 最小实验**。

## 1. 基础（与 CI 一致）

```bash
cd /path/to/fl-agent
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -e ".[dev]"
./scripts/smoke.sh
```

- **不装 torch** 时：部分测试会 **skipped**（联邦与 Torch 数据集），属正常。
- Python **≥ 3.9**（见 `pyproject.toml`）。

## 2. Torch + 联邦 / 最小实验（推荐本地科研机）

```bash
python3 -m pip install -e ".[torch]"
# 或一次性：见 ./scripts/smoke-torch.sh
./scripts/smoke-torch.sh
```

### Apple Silicon (macOS)

一般直接使用 PyTorch 官方 wheel（`pip install torch` 已包含在 `[torch]` extra）。若需 MPS，在代码里自行设 `device=mps`（当前 smoke 默认 `cpu`）。

### Linux + NVIDIA GPU

安装与 CUDA 版本匹配的 `torch` / `torchvision`（参见 [PyTorch 安装页](https://pytorch.org/get-started/locally/)）。本仓库 smoke **不依赖 GPU**，GPU 留给后续 RETFound 训练。

## 3. 可选：HuggingFace / 数据下载辅助

```bash
python3 -m pip install -e ".[data]"
```

用于 `download_rfmid` 等工具（若存在）；**RFMiD 本体许可与下载**仍以 [DATA_CARD.md](DATA_CARD.md) 为准。

## 4. 产物目录

- `runs/`、`data/raw/`、`configs/splits/generated/` 已在 `.gitignore` 中；实验 JSON、划分产物默认不要提交进 Git，用路径在实验记录里引用即可。
