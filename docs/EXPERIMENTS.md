# 实验设计（如何从现在走到论文级）

当前代码处于 **v0.3 联邦 smoke + 噪声协议接线**：适合做 **管线验证** 与 **通信字节 / loss 曲线** 的预实验；**尚未**内置 RETFound 训练循环。

## 1. 实验是否「开始」了？

| 类型 | 状态 |
|------|------|
| A. **合成最小 baseline**（4 样本、2 客户端、TinyMLP） | 已可跑：`minimal_experiment` / `scripts/run_minimal_baseline.sh` |
| A2. **合成消融/对比实验**（FedAvg vs FedProx × 噪声） | 已开始并产出报告：`scripts/run_numpy_ablations.sh`、`scripts/run_synthetic_ablations.sh` |
| B0. **真实 RFMiD 子集 smoke**（HF 96 张子集） | 已开始并产出报告：`scripts/run_rfmid_subset_smoke.sh` |
| B. **真实 RFMiD 全量**（训练/验证/测试、多 seed） | 需扩展为全量矩阵和验证/测试指标 |
| C. **论文主表**（非 IID × 噪声 × 方法 × seed） | 依赖 B + 后续模型与聚合策略扩展 |

## 2. 推荐实验因子（之后扩展 RETFound 仍沿用框架）

### 自变量（建议先冻结文档再跑）

- **划分 / 异质性**：IID vs Dirichlet α vs domain-hash（已有 `build_splits`）。
- **标签噪声**：`noise_protocol@v1` 中 `p_flip` 若干档（如 0, 0.1, 0.2）。
- **联邦算法**：FedAvg vs FedProx（`fedprox_mu` 网格，如 0, 0.01, 0.1）。
- **随机种子**：`label_noise_seed`、划分 `seed`、`FedSmokeConfig.seed`。

### 因变量（当前 smoke 已支持）

- `mean_train_loss_clients`（每轮客户端平均训练损失）
- `comm_bytes_upload_per_round`、`total_upload_bytes`
- `noise_protocol` 元数据（若启用 YAML）

### 当前已生成的合成实验结果

- **NumPy 后端（无需 torch）**：`docs/results/numpy_synthetic_ablation_latest.md`
- **Torch/TinyMLP 后端**：`docs/results/synthetic_ablation_latest.md`
- **真实 RFMiD 子集 smoke**：`docs/results/rfmid_subset_smoke_latest.md`
- 原始 JSON / CSV 默认写入 `runs/`，该目录被 `.gitignore` 忽略；可提交的脱敏摘要放在 `docs/results/`。

### 尚未作为一等公民的（v0.4+）

- 验证集 macro-F1 / mAP、按病种分层指标；留一中心；校准曲线。

## 3. 单次「真实数据」实验流程（建议）

1. 按 [DATA_CARD.md](DATA_CARD.md) 放置 `labels.csv` 与 `images/`；或先运行 `./scripts/run_rfmid_subset_smoke.sh` 导出 HF 子集。
2. `python -m fed_agent.tools.build_splits --labels_csv ...` → 输出到 `configs/splits/generated/`（勿提交）。
3. `python -m fed_agent.tools.run_fed_smoke --labels_csv ... --images_dir ... --split_json ... --out_json runs/exp001/metrics.json`
4. `python -m fed_agent.tools.summarize_fed_smoke runs/exp001/metrics.json` → 追加到实验笔记（或 `wandb`/表格）。

## 4. 论文级 baseline / 消融 / 鲁棒性设计

### 4.1 对比方法（必须有）

- **Centralized RETFound + LoRA**：集中式上限，证明基础模型能力。
- **Local-only RETFound + LoRA**：每个中心单独训练，证明联邦共享的收益。
- **FedAvg + LoRA**：基础联邦 baseline。
- **FedProx + LoRA**：非 IID 常用 baseline。
- **FedBN / SCAFFOLD / FedOpt（至少选 1-2 个）**：增强非 IID 说服力。
- **Noise-robust baseline**：例如 GCE / Co-teaching / FedDiv-style 伪标签或蒸馏模块。

### 4.2 本框架创新点要对应的消融

- **Foundation + PEFT**：RETFound frozen backbone vs LoRA rank {4, 8, 16}。
- **Noise module**：无噪声建模 vs 对称噪声 vs 类条件噪声 / FedDiv-style。
- **Agent scheduler**：固定规则 FL vs RuleAgent / bandit action（客户端采样、聚合权重、通信预算）。
- **Communication accounting**：上传全模型 vs 只上传 LoRA adapter；报告字节数与性能曲线。
- **Non-IID decomposition**：IID、Dirichlet α={0.1, 0.5, 1.0}、domain-hash / 真实设备域。

### 4.3 鲁棒性与有效性

- **标签噪声强度**：p_flip={0, 0.1, 0.2, 0.4}。
- **中心数量**：K={4, 8, 16}。
- **客户端参与率**：C={0.25, 0.5, 1.0}。
- **长尾标签**：macro-F1 / rare-label recall 单独报告。
- **外部验证**：Validation/Test split 或 leave-domain-out（若有设备/中心元数据）。

### 4.4 主指标

- **Primary**：macro-F1、mAP / macro-AUROC（多标签）。
- **Secondary**：per-label F1、rare-label recall、calibration、communication bytes、rounds-to-target。
- **统计**：至少 3 个 seed；报告 mean ± std；主要比较做 paired bootstrap 或 Wilcoxon。

## 5. 实验矩阵配置

见仓库根目录 `configs/experiment_matrix.yaml`（占位）。正式跑批时建议：**每一行实验**对应一个 shell 片段或 YAML 行，并记录 `git rev-parse HEAD`、pip freeze 片段、数据版本。

## 6. 与「课题完成」的关系

主文实验需要 **B + RETFound + 统计方案**；本文件只约束 **工程侧如何设计、记录、复现**。边界说明见 [PROJECT_SCOPE.md](PROJECT_SCOPE.md)。
