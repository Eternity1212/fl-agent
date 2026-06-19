# Minimal synthetic federated baseline

**说明**：本文件设计为由 `./scripts/run_minimal_baseline.sh` **自动生成**（内含完整指标 JSON）。若你看到的是本模板，说明尚未在本地执行该脚本，或执行环境未完成 PyTorch 下载。

## 在本地一键生成含真实数字的版本

```bash
./scripts/run_minimal_baseline.sh
```

完成后请查看：

- `runs/minimal/metrics.json`（完整 JSON，默认被 `.gitignore` 忽略）
- 本文件 `docs/results/minimal_synthetic_baseline.md`（脚本会覆写为带表格 + JSON 的快照，便于提交或写进材料）

并用：

```bash
python3 -m fed_agent.tools.summarize_fed_smoke runs/minimal/metrics.json
```

打印终端可读摘要。

## 指标里会有哪些字段（与真实跑分一致的结构）

- `rounds`：轮次索引列表  
- `comm_bytes_upload_per_round`：每轮各客户端上传 `state_dict` 的字节之和  
- `mean_train_loss_clients`：每轮参与训练客户端的平均 loss（BCEWithLogits ± FedProx）  
- `total_upload_bytes`：所有轮次上传字节总和  
- `final_state_dict_keys`：聚合后全局模型参数名列表  
- 若传入 `--noise_yaml`，另有 `noise_protocol` 元数据（见 `run_fed_smoke` 文档）

**注意**：上列为 **4 张合成图、2 客户端、TinyMLP** 的占位实验，用于验证管线；**真实 RFMiD 主实验**需按 [EXPERIMENTS.md](../EXPERIMENTS.md) 另行设计。
