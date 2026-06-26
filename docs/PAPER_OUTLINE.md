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
4. **(方法创新, GPU 已坐实 ✅)** **自适应联邦编排 agent**：用每轮"全局模型+客户端
   adapter"在留出探针集上的负 BCE 作为遥测，**自适应聚合权重**(sigmoid 门控压低脏
   客户端) ± **per-client 自适应 μ**。
   **RFMiD RETFound+LoRA, 异质噪声 p=0.4, IID, 3 seed 主结果:**
   - FedAvg 0.738 → **Agent 0.804 (+0.066)**, 3 seed 全胜;
   - 静态 Robust-FedProx **反而更差(0.669)** → 证明"固定正则是错的工具, 只有自适应才行";
   - 机制可解释: 脏客户端权重被压到 ~0, 探针分 clean≈−0.11 vs noisy≈−2.5。
   - 恢复了 noisy→clean 上限(0.825) 76% 的差距。
   - 待补: clean 不掉点对照 + non-IID(纯加权失效, 测 adaptive μ 能否救)。

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
6. **Discussion**
   - 为何 proximal 在 PEFT-FL 下未必有益（若负结果）；鲁棒正则的适用边界。
   - **两根杠杆的关系（设计论证）**：自适应**加权**与自适应 **μ** 并非叠加。当加权
     已把脏客户端压到近 0 权重时，再用 μ 约束它们基本**冗余**（本地 smoke：agentmu
     +0.019 vs agent +0.022，μ 略低且在噪声带内）。μ 的真正用武之地是**不能丢弃
     客户端**的场景——非IID 下脏客户端握有别处没有的标签覆盖，只能"约束"而非"下权重"。
     这解释了我们为何把加权设为主杠杆、μ 设为条件性补充（消融 `het04_dir_agentmu`）。
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
- **✅ 已坐实: 自适应编排 agent 在异质噪声(IID)上 3 seed 显著优于 FedAvg(+0.066)
  且远胜 Robust-FedProx** → 已是真方法创新, **够中端会议/期刊**。
- 进一步上限(可选, 按 full matrix 结果):
  - clean 不掉点对照成立 → 主张更完整("零代价 + 噪声下大增益");
  - non-IID 下 adaptive μ 救活 0.533 → 双杠杆都有效, 冲中高端;
  - + ODIR 跨数据集复现 → 更稳;
  - 顶会仍需理论/收敛分析或多数据集 agent 复现。
