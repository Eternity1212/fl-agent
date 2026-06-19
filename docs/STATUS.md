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
| **真实 RFMiD 上的训练实验** | **未开始** | 需本地下载数据 + 划分产物 + 装 torch；见 [EXPERIMENTS.md](EXPERIMENTS.md) |
| RETFound + LoRA、FedDiv、Agent | 未开始 | 见 [ROADMAP.md](ROADMAP.md) |

## 实验部分是否已开始

- **严格意义（真实数据、多 seed、论文级表格）**：**尚未开始**；当前仓库提供的是 **可跑通的最小联邦 smoke**，用于验证管线与通信统计。
- **工程意义（合成 4 张图 + 2 客户端的最小 baseline）**：可通过 `python -m fed_agent.tools.minimal_experiment` 或 `./scripts/run_minimal_baseline.sh` **立即跑出一条 JSON 指标**，作为「实验流水线的占位跑通」。

下一步：下载 RFMiD → `build_splits` → `run_fed_smoke` 多组配置 → 汇总进表格（见 [EXPERIMENTS.md](EXPERIMENTS.md)）。

## 相关文档

- [ENVIRONMENT.md](ENVIRONMENT.md) — 环境安装  
- [EXPERIMENTS.md](EXPERIMENTS.md) — 实验设计与日志规范  
- [PROJECT_SCOPE.md](PROJECT_SCOPE.md) — 软件交付 vs 课题边界  
- [FEDERATION.md](FEDERATION.md) — 联邦 CLI 与字段说明  
