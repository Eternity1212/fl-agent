# 项目范围说明（软件仓库 vs 科研课题）

本仓库 **fl-agent** 交付的是一套 **可复现的工程骨架**：数据接入、划分 JSON、联邦 smoke、噪声协议占位与指标落盘。它可以支撑面上 / 启源的技术路线，但 **不等于** 课题在论文层面的“全部完成”（RETFound 全量训练、多中心真实数据、临床终点与功效分析等仍在课题执行中）。

## 本仓库视为「软件里程碑完成」时包含什么

- **RFMiD**：本地标签表读取、Torch 数据集桥、按 `ImageID` 对齐划分。
- **划分**：IID / Dirichlet / domain-hash → `fl_agent.split.v1` JSON + CLI。
- **联邦 smoke**：FedAvg / FedProx、`state_dict` 上传字节统计、CLI 输出 JSON。
- **噪声**：`noise_protocol@v1` YAML 解析；对称翻转噪声接入 `RFMiDTorchDataset` 与联邦 smoke；指标中带 `noise_protocol` 元数据。
- **结果消费**：`python -m fed_agent.tools.summarize_fed_smoke <metrics.json>`；`docs/examples/fed_smoke_metrics.example.json` 为字段示例；**合成 baseline 快照模板**见 `docs/results/minimal_synthetic_baseline.md`（运行 `./scripts/run_minimal_baseline.sh` 后由脚本覆写为含实测 JSON 的版本）。

## 仍属「课题 / 下一阶段」的工作（不在本仓库一次性做完）

- RETFound + LoRA / 仅上传 adapter 字节的真实通信预算。
- FedDiv 等强基线与系统消融矩阵落地。
- Global / Rule Agent 编排与对照实验。
- 真实多中心数据与临床统计分析。

若需把 **GitHub 上的代码** 与本地对齐，见 [SYNC.md](SYNC.md)；日常验证见 `./scripts/smoke.sh`（无 torch 时部分用例跳过），完整 torch 用例见 `./scripts/smoke-torch.sh`。

## 验收自检（对照上文「软件里程碑」）

| 检查项 | 命令或依据 | 当前环境说明 |
|--------|------------|----------------|
| 静态检查 + 单测 | `./scripts/smoke.sh` | 应 **ruff 通过**；pytest **全通过**；未装 torch 时 **4 个用例 skipped 为预期** |
| 含 torch 的完整单测 | `./scripts/smoke-torch.sh` | 会安装 `torch`；应 **0 skipped** |
| 指标摘要工具 | `python3 -m fed_agent.tools.summarize_fed_smoke docs/examples/fed_smoke_metrics.example.json` | 应打印噪声路径、上传字节、loss 等 |
| 远程协作是否「交付」 | `git status` 应为干净，且已 `push` | **若仍有大量 `??` / `M`，则 GitHub 上尚未包含这些文件，不算协作侧完成** |

