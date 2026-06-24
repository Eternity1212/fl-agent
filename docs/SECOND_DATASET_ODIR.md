# 第二数据集：ODIR（跨数据集泛化）

为什么要第二个数据集：医学影像审稿人通常要求 ≥2 个数据集才认"泛化"。补一个
ODIR 能把论文从 workshop 档抬到中端档，且**与后续 robust 结果无关、纯加分**。

## 为什么选 ODIR（而不是 APTOS / Messidor）

| 数据集 | 标签类型 | 是否适配本项目 |
|--------|----------|----------------|
| **ODIR-5K** | **多标签 8 类**(N/D/G/C/A/H/M/O) | ✅ 与 RFMiD 多标签方法学一致，最佳 |
| APTOS 2019 | 单标签 DR 分级(0-4) | ❌ 多分类，非多标签 |
| Messidor | 单标签 DR 分级 | ❌ 同上 |

ODIR 标签空间和 RFMiD 有重叠（糖尿病、青光眼、AMD、近视等），适合讲"同一框架
跨数据集成立"。

## 数据来源说明（重要）

- **默认 `--source hf`**：用公开镜像 `bumbledeep/odir`(6392 张，无需任何凭证)。
  注意该镜像是**单标签**版（每张图 8 类里恰好 1 个正例）。它仍然完整走多标签代码
  路径(BCE / per-label F1 / AUROC)，支持噪声与非IID 划分，足以做跨数据集复现。
- **`--source csv`（真多标签升级版）**：若需要"一张图多个正标签"的真多标签，用
  Kaggle `andrewmvd/ocular-disease-recognition-odir5k` 的 `full_df.csv`(含 8 个二值
  列)+ 本地图片目录，命令见下。需要 Kaggle 账号下载。

## 三步走（与 RFMiD 完全相同的本地布局）

### 1. 导出数据

```bash
# 方式 A：HF 公开镜像（推荐，无需凭证）
python3 -m fed_agent.tools.export_odir --source hf --out_dir data/raw/odir

# 方式 B：Kaggle 真多标签（可选升级）
python3 -m fed_agent.tools.export_odir --source csv \
    --csv_path /path/to/full_df.csv \
    --images_dir /path/to/ODIR-5K/preprocessed_images \
    --out_dir data/raw/odir
```

产出布局（与 RFMiD 一致，直接被现有 loader / 训练 runner 读取）：

```
data/raw/odir/train/labels.csv          # ImageID,N,D,G,C,A,H,M,O
data/raw/odir/train/images/<id>.png
data/raw/odir/validation/...
data/raw/odir/test/...
```

### 2. 生成联邦划分

注意 `--out_dir` 用 `generated_odir`，避免和 RFMiD 的 `labels__*.json` 同名冲突：

```bash
for s in 0 1 2; do
  python3 -m fed_agent.tools.build_splits \
    --labels_csv data/raw/odir/train/labels.csv \
    --out_dir configs/splits/generated_odir \
    --seed $s --n_clients 4 --alphas 0.1 0.5 1.0
done
```

### 3. 跑矩阵

```bash
PAPER_MATRIX_YAML=configs/paper_matrix_odir.yaml ./run.sh
```

`configs/paper_matrix_odir.yaml` 含：对比(centralized/local/fedavg/fedprox/robust,
seed 0-2) + 鲁棒性(噪声 0.2/0.4,含解耦版 fedavg+dropout)。

## 出图

把 RFMiD 和 ODIR 的 summary 一起喂给画图脚本，自动出对比表：

```bash
python3 -m fed_agent.tools.make_paper_figures \
    runs/paper_matrix/retfound_full/summary.json \
    runs/paper_matrix/paper_matrix_odir/summary.json \
    --out_dir docs/results/figures --out_md docs/results/paper_figures.md
```

## 验收

- `data/raw/odir/train/labels.csv` 表头应是 `ImageID,N,D,G,C,A,H,M,O`。
- 训练日志里 `RETFound=True`、标签数=8。
- 跨数据集表里 ODIR 的 centralized 应明显高于 local-only（与 RFMiD 同向）。
