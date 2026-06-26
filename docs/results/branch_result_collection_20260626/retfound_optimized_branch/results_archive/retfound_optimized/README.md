# RETFound optimized matrix — archived results

GPU 上跑出的完整 29-run RETFound + LoRA 矩阵的**原始结果归档**(从 GPU 机器
打包上传)。因为仓库 `.gitignore` 忽略了 `runs/`,这里用 `git add -f` 强制
归档,确保原始数据有云端备份。

## 内容

- `runs/paper_matrix/retfound_full_optimized_resume/` — 29 个 run 的原始 JSON
  + `summary.json` + `summary.csv`
- `docs/results/` — 汇总文档(summary / mean±std / results_section / artifacts)
- `configs/` — GPU 上使用的 optimized 配置 + 生成的 splits
- `src/fed_agent/train/paper_runner.py` — 产出这批结果的训练代码快照
  (相对主干只是吞吐优化:DataParallel / num_workers / pin_memory,**算法/配方未变**)
- `HISTORY.txt` — 原 GPU 分支与提交记录

## 关键结论(详见 docs/results/ 与聊天分析)

- FedAvg ≈ centralized,是最强且最稳定的联邦方法(3 seed)。
- FedProx / Robust-FedProx 全面落后;μ 消融显示近端正则强度是主因
  (μ=0.01 几乎追平 FedAvg,μ=0.1 崩);proximal 正则不该直接套到 FM+PEFT。
- 均匀噪声 / 非IID 下 robust 也没翻盘(均匀污染无异质结构可利用)。

> 这是只读归档,不要在此分支继续开发。agent 实验在 `feat/agent-orchestration` 分支。
