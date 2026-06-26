# 实验结果目录说明

- **`runs/`**（仓库根目录，已 gitignore）：默认存放 `run_fed_smoke` 的 `--out_json`、多组实验子目录等。**不要**把含隐私路径或大规模二进制提交进 Git。
- **`docs/results/`**（本目录）：可放 **脱敏后的快照**（例如合成 baseline 的一次 JSON、或从 `summarize_fed_smoke` 粘贴的摘要），便于答辩/协作时「有数可查」。
- **`docs/examples/fed_smoke_metrics.example.json`**：字段示意，非真实跑分。

更新快照时请在文件头注明：`date`、`git commit`、`python/torch 版本`、`数据子集说明`。
