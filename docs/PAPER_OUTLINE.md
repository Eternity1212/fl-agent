# 论文大纲（草案 v1）

> 状态：clean 主对比已三 seed 基本锁定（FedAvg 为最强联邦基线）；噪声 / 非IID /
> μ 等鲁棒性证据待 GPU 跑完。本大纲把"通信高效的联邦基础模型适配"作为**主贡献
> (确定性的)**，把鲁棒性作为**有明确成败标准的实证研究(条件性的)**，因此对后续
> 结果稳健——无论 robust 是否胜出，论文都成立。

## 候选标题

- *Communication-Efficient Federated Adaptation of a Retinal Foundation Model for Multi-Label Fundus Diagnosis*
- *Federating RETFound with LoRA: Near-Centralized Multi-Label Fundus Diagnosis at a Fraction of the Communication*

## 摘要骨架（待填数字）

视网膜多病种筛查需要多中心协作，但隐私与通信成本限制了集中式训练。我们提出对
视网膜基础模型 RETFound 做**参数高效联邦适配**：各中心仅用 LoRA 微调并**只上传
adapter（占全模型 0.39% 参数）**。在 RFMiD 与 ODIR 两个多标签数据集上，联邦
FedAvg+LoRA 达到约 [X]% 的集中式 AUROC、每轮通信减少约 [Y]×，并显著优于单中心。
我们进一步系统评测了标签噪声与非IID 下的鲁棒联邦策略，[结论占位]。

## 贡献（Contributions）

1. **(主, 已坐实)** RETFound+LoRA 的联邦适配框架：近集中式性能 + 极低通信
   （每轮仅传 ~19MB / 全模型 ~1.2GB，约 64×），显著优于 local-only。
2. **(实证研究)** 多标签 + 标签噪声 + 非IID 下，对 FedAvg / FedProx / 抗噪正则的
   系统比较与分析（含 μ / LoRA-rank / dropout 消融）。
3. **(泛化)** 在 RFMiD 与 ODIR 两个数据集上复现，验证框架的跨数据集稳健性。

## 章节结构

1. **Introduction** — 多中心眼底筛查的隐私/通信痛点；基础模型 + PEFT 机会；贡献列表。
2. **Related Work** — 视网膜基础模型(RETFound)；PEFT/LoRA；联邦学习(FedAvg/FedProx)；
   标签噪声鲁棒学习；多标签医学影像。
3. **Method**
   - 3.1 RETFound + LoRA 适配（仅 adapter + 分类头可训练）。
   - 3.2 联邦协议：FedAvg / FedProx / Robust 变体（balanced-BCE + positive-dropout）。
   - 3.3 通信记账（只传 adapter）。
4. **Experimental Setup**
   - 数据集：RFMiD(46 标签) + ODIR(8 标签)。
   - 客户端划分：IID / Dirichlet(α=0.1,0.5,1.0) / domain-hash，seed 0/1/2。
   - 标签噪声协议：对称翻转 p∈{0,0.2,0.4}。
   - 指标：macro-AUROC、macro-AP、best macro/micro-F1(present classes)、每标签 F1。
5. **Results**
   - 5.1 主对比（表1）+ 通信效率（图1）。
   - 5.2 鲁棒性：噪声下降曲线（图2）。
   - 5.3 非IID 异质性（表2）。
   - 5.4 消融：μ / LoRA-rank / dropout（表3）。
   - 5.5 跨数据集复现（表4，ODIR）。
6. **Discussion** — 为何 proximal 在 PEFT-FL 下未必有益（若负结果）；鲁棒正则的适用边界。
7. **Limitations** — 数据规模、单一模态、噪声为模拟等。
8. **Conclusion**。

## 图表 → 数据来源映射

| 图/表 | 内容 | 由哪个 run/脚本产生 | 状态 |
|-------|------|---------------------|------|
| 表1 | RFMiD 主对比 mean±std | `paper_matrix_gpu_full`(对比块) | 🔄 进行中(clean 近完成) |
| 图1 | 通信 vs 性能 | `make_paper_figures` fig1 | ✅ 脚本就绪 |
| 图2 | 噪声下降曲线 | `paper_matrix_robustness` → fig2 | ⏳ 待跑 |
| 表2 | 非IID(Dirichlet) | `paper_matrix_gpu_full`(20–25) | ⏳ 待跑 |
| 表3 | μ/rank/dropout 消融 | `gpu_full`(26–29) + `mu001` | ⏳ 待跑 |
| 表4 | ODIR 跨数据集 | `paper_matrix_odir` | ⏳ 管线就绪待跑 |

## 关键量化卖点（已可写，待最终数字核对）

- LoRA 可训练参数 ≈ 1.19M / 总 304.5M = **0.39%**。
- 每轮上传 ≈ 19MB；全模型 fp32 ≈ 1.22GB → **约 64× 通信节省**。
- FedAvg+LoRA macro-AUROC ≈ 0.80 vs centralized ≈ 0.81 → **约 98% 上限**。
- FedAvg >> local-only（macro-F1 ≈ 0.23 vs 0.16）。

## 叙事决策树（按后续结果选最终主张）

- **若 `fedavg+dropout` 在噪声下显著优于 fedavg(3 seed 一致)** →
  第二贡献升级为正面："一个零成本抗噪正则提升联邦噪声鲁棒性"。
- **若 μ=0.01 显著优于 0.05** → 主对比改用 μ=0.01 的公平超参重述 FedProx。
- **若困难条件下仍无人胜出 fedavg** → 第二贡献写成严谨的反直觉发现
  （配合 μ/rank/dropout sweep，证明"不是没调好"）。

## 目标档位（诚实）

- 主贡献(效率) + 单数据集 = workshop / 低档期刊。
- + 图2 正面结果 或 + ODIR 任一 → 中端会议/期刊有竞争力。
- 两者都有 → 较稳的中端。
- 顶会需额外真方法创新（如噪声自适应聚合），当前组合不足以冲顶。
