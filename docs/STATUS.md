# 当前进度（核对用）

**更新时间**：以仓库内 `git log -1` 为准；本文描述「工程与实验基础设施」阶段。

## 已完成到哪一步

| 层级 | 状态 | 说明 |
|------|------|------|
| 数据接入（RFMiD） | 完成 | 本地 CSV + 图像目录；见 `src/fed_agent/data/` |
| 划分 JSON | 完成 | IID / Dirichlet / domain-hash；`python -m fed_agent.tools.build_splits` |
| 联邦 **smoke**（TinyMLP） | 完成 | FedAvg / FedProx、上传字节、`run_fed_smoke` CLI |
| 噪声协议（对称翻转） | 完成 | YAML → 数据集与指标元数据 |
| 结果摘要工具 | 完成 | `summarize_fed_smoke` |
| 合成消融 / 对比实验 | 已开始 | NumPy 与 Torch 两套 synthetic ablations 已产出 `docs/results/*_ablation_latest.md` |
| 真实 RFMiD 子集 smoke | 已开始 | HF 导出 96 张真实图像；报告见 `docs/results/rfmid_subset_smoke_latest.md` |
| **真实 RFMiD 论文级主实验** | **未开始** | 需全量数据、验证/测试集、RETFound/LoRA 与强 baseline；见 [EXPERIMENTS.md](EXPERIMENTS.md) |
| RETFound + LoRA、FedDiv、Agent | 未开始 | 见 [ROADMAP.md](ROADMAP.md) |

## 实验部分是否已开始

- **严格意义（全量真实数据、多 seed、论文级表格）**：**尚未开始**；还需要全量 RFMiD、验证/测试指标与正式矩阵。
- **工程意义（合成数据、对比与消融）**：**已开始并完成首轮结果**；见 `docs/results/numpy_synthetic_ablation_latest.md` 和 `docs/results/synthetic_ablation_latest.md`。
- **真实数据 smoke**：**已跑通首轮**；见 `docs/results/rfmid_subset_smoke_latest.md`。当前 TinyMLP 结果用于证明框架可运行，不作为最终论文优势结论。

下一步：下载 RFMiD → `build_splits` → `run_fed_smoke` 多组配置 → 汇总进表格（见 [EXPERIMENTS.md](EXPERIMENTS.md)）。

## 相关文档

- [ENVIRONMENT.md](ENVIRONMENT.md) — 环境安装  
- [EXPERIMENTS.md](EXPERIMENTS.md) — 实验设计与日志规范  
- [PROJECT_SCOPE.md](PROJECT_SCOPE.md) — 软件交付 vs 课题边界  
- [FEDERATION.md](FEDERATION.md) — 联邦 CLI 与字段说明  
