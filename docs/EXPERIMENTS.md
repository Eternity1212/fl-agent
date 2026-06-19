# 实验设计（如何从现在走到论文级）

当前代码处于 **v0.3 联邦 smoke + 噪声协议接线**：适合做 **管线验证** 与 **通信字节 / loss 曲线** 的预实验；**尚未**内置 RETFound 训练循环。

## 1. 实验是否「开始」了？

| 类型 | 状态 |
|------|------|
| A. **合成最小 baseline**（4 样本、2 客户端、TinyMLP） | 可随时跑：`minimal_experiment` / `scripts/run_minimal_baseline.sh` |
| B. **真实 RFMiD**（训练集全量或子集、多 seed） | 需先完成数据下载与划分 JSON，再组矩阵 |
| C. **论文主表**（非 IID × 噪声 × 方法 × seed） | 依赖 B + 后续模型与聚合策略扩展 |

## 2. 推荐实验因子（之后扩展 RETFound 仍沿用框架）

### 自变量（建议先冻结文档再跑）

- **划分 / 异质性**：IID vs Dirichlet α vs domain-hash（已有 `build_splits`）。
- **标签噪声**：`noise_protocol@v1` 中 `p_flip` 若干档（如 0, 0.1, 0.2）。
- **联邦算法**：FedAvg vs FedProx（`fedprox_mu` 网格，如 0, 0.01, 0.1）。
- **随机种子**：`label_noise_seed`、划分 `seed`、全局 `torch.manual_seed`（后续统一进配置）。

### 因变量（当前 smoke 已支持）

- `mean_train_loss_clients`（每轮客户端平均训练损失）
- `comm_bytes_upload_per_round`、`total_upload_bytes`
- `noise_protocol` 元数据（若启用 YAML）

### 尚未作为一等公民的（v0.4+）

- 验证集 macro-F1 / mAP、按病种分层指标；留一中心；校准曲线。

## 3. 单次「真实数据」实验流程（建议）

1. 按 [DATA_CARD.md](DATA_CARD.md) 放置 `labels.csv` 与 `images/`。
2. `python -m fed_agent.tools.build_splits --labels_csv ...` → 输出到 `configs/splits/generated/`（勿提交）。
3. `python -m fed_agent.tools.run_fed_smoke --labels_csv ... --images_dir ... --split_json ... --out_json runs/exp001/metrics.json`
4. `python -m fed_agent.tools.summarize_fed_smoke runs/exp001/metrics.json` → 追加到实验笔记（或 `wandb`/表格）。

## 4. 实验矩阵配置

见仓库根目录 `configs/experiment_matrix.yaml`（占位）。正式跑批时建议：**每一行实验**对应一个 shell 片段或 YAML 行，并记录 `git rev-parse HEAD`、pip freeze 片段、数据版本。

## 5. 与「课题完成」的关系

主文实验需要 **B + RETFound + 统计方案**；本文件只约束 **工程侧如何设计、记录、复现**。边界说明见 [PROJECT_SCOPE.md](PROJECT_SCOPE.md)。
