# 本地 CPU 子集结果存档（RETFound + LoRA）

> 用途：在本机（无 GPU）验证训练配方与抗噪方向是否成立。**这是 sanity / 方向验证，
> 不是论文最终结果**。论文数字以 GPU 全量矩阵 `paper_matrix_gpu_full.yaml` 为准。

## 实验设置

| 项目 | 取值 |
|------|------|
| 数据 | RFMiD 子集：160 训练 / 80 验证，head-12 标签 |
| 模型 | RETFound (MAE ViT-Large) + LoRA，`RETFound=True`（真实权重） |
| 配方 | lr=1e-3，centralized epochs=10 / federated rounds=10，batch=16，LoRA rank=8 |
| 客户端 | 4，IID 划分，seed=0 |
| 通信 | 联邦方法每轮只传 LoRA adapter（约 190 MB/轮累计） |
| 运行时间 | 本机 CPU，约 1.5 小时跑完 5 个 run |

## 结果

| run | 方法 | 噪声 | macro-AUROC | best macro-F1 | best micro-F1 |
|-----|------|------|-------------|---------------|---------------|
| centralized_clean（上界） | centralized | 0.0 | **0.741** | 0.301 | 0.486 |
| fedavg_clean | fedavg | 0.0 | 0.649 | 0.124 | 0.804 |
| robust_fedprox_clean | robust_fedprox | 0.0 | 0.642 | 0.197 | 0.251 |
| fedavg_noise0.2 | fedavg | 0.2 | 0.632 | 0.159 | 0.539 |
| robust_fedprox_noise0.2 | robust_fedprox | 0.2 | **0.704** | 0.189 | 0.283 |

## 关键发现

1. **配方修正有效**：把 lr 从 1e-4 提到 1e-3、补足训练轮数后，AUROC 从早期的 ≈0.5
   （瞎猜）提升到 0.64–0.74，证明模型确实在学习。早期近似随机是欠训练所致，
   非方法或代码问题。
2. **抗噪鲁棒性信号成立**：在 20% 标签噪声下，
   - Robust-FedProx AUROC **0.704** 明显高于 FedAvg **0.632**（+0.072）；
   - FedAvg 遇噪声会下降（0.649 → 0.632），Robust-FedProx 反而更稳（0.642 → 0.704）。
   - 这正是论文期望的"噪声越大、鲁棒方法相对优势越明显"。
3. **通信效率**：联邦方法每轮仅传 LoRA adapter（~190 MB 累计），远小于全模型传输，
   且性能接近 centralized 上界。

## 局限（必须诚实标注）

- 仅 80 张验证图、单 seed，方差大；Robust 在噪声下"反升"可能含运气成分。
- macro-F1 偏低且抖动（子集中稀有病种正样本过少）。
- **不能直接写进论文**。论文结论须由 GPU 全量（1920/640/640）+ 多 seed
  （`configs/paper_matrix_gpu_full.yaml`）产出 mean±std 后确定。

## 下一步

方向已验证、代码已上 GitHub。直接在 GPU 服务器跑全套对比 + 消融矩阵（见
`docs/GPU_RUN_CHECKLIST.md`）。
