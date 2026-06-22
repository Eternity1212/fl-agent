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
| 真实 RFMiD 子集 smoke | 已开始 | HF 导出真实图像；head-12 疾病子实验报告见 `docs/results/rfmid_subset_smoke_latest.md` |
| **真实 RFMiD 论文级主实验** | 已进入 v0.4 管线 | 完整导出/校验、paper matrix、runner、汇总工具已实现；full-46 fallback 已完成 |
| RETFound + LoRA | 代码已实现，主结果待授权 | 当前 HF RETFound gated 权重未授权时会标记 fallback，不能当论文主结果 |

## 实验部分是否已开始

- **严格意义（全量真实数据、多 seed、论文级表格）**：工程管线已开始；full-46 fallback 主矩阵配置已就绪，正式 RETFound 主结果仍依赖 Hugging Face gated 权重授权。
- **工程意义（合成数据、对比与消融）**：**已开始并完成首轮结果**；见 `docs/results/numpy_synthetic_ablation_latest.md` 和 `docs/results/synthetic_ablation_latest.md`。
- **真实数据 smoke**：**已跑通并出现正向信号**；head-12 子实验中，`balanced_bce + p=0.1 label dropout + FedProx` 的 best macro-F1 为 `0.143479`，相对 BCE baseline `0.134370` 提升约 `+6.78%`。这仍是 smoke-scale 结果，下一步要在全量 RFMiD + RETFound/LoRA 上确认。

当前 RETFound+LoRA 进展：

- 已新增 `src/fed_agent/models/retfound_lora.py`、`src/fed_agent/train/paper_runner.py`、`src/fed_agent/metrics/multilabel.py`。
- 已跑通 head-12 pilot fallback，报告见 `docs/results/paper_matrix_pilot_head12.md`。
- 当前环境检测到 RETFound 权重为 gated 且未授权；需要 `huggingface-cli login` 或设置 `HF_TOKEN` 后才能产出真正 RETFound 主结果。
- full RFMiD 已下载并校验：train=1920、validation=640、test=640。
- full-46 MLP fallback 多 seed 主矩阵已完成，报告见 `docs/results/paper_matrix_full46_mlp.md`；它证明全量数据、split、训练、指标、汇总链路可跑通，但性能较低，不能当论文主结果。
- fallback 聚合结果：FedProx best macro-F1 `0.071905 +/- 0.000395`，Robust-FedProx `0.071856 +/- 0.000374`，FedAvg `0.070703 +/- 0.000433`。

## 相关文档

- [ENVIRONMENT.md](ENVIRONMENT.md) — 环境安装  
- [EXPERIMENTS.md](EXPERIMENTS.md) — 实验设计与日志规范  
- [PROJECT_SCOPE.md](PROJECT_SCOPE.md) — 软件交付 vs 课题边界  
- [FEDERATION.md](FEDERATION.md) — 联邦 CLI 与字段说明  
