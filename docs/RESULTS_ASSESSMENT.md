# 结果整合与论文可行性评估(截至 2026-06-30)

> 数据来源:GPU(RETFound-MAE ViT-L + LoRA rank8, 40 rounds, K=4, RFMiD 多标签)。
> 主指标用 **macro-AUROC**(稳, GPU run-to-run 抖动 ~0.01);`best_micro_f1` 是
> 挑阈值指标, 在多标签+噪声下方差极大, 仅作参考, 不作结论依据。

---

## 1. 完整结果表

### 表 1 — IID 主线:剂量-响应(核心结果)

| 噪声(脏客户端 2/3) | FedAvg | Agent(ours) | 增益 | seed 数 |
|---|---|---|---|---|
| clean (0%) | 0.848±0.017 | 0.842±0.011 | **−0.006(噪声内, 代价可忽略)** | 3 |
| 0.2 | 0.7630±0.000 | 0.8005±0.011 | **+0.038** | 3 |
| 0.4 | 0.7380±0.001 | 0.8043±0.010 | **+0.066** | 3 |

> het02 3-seed 全方法(macro-AUROC):agent 0.8005 ≈ CCR 0.7869 > FedAvg 0.7630 > **Robust 0.7069(反而最差)**。
> agent 在 ranking(auroc/ap)略优 CCR,CCR 在 micro-F1 略优 agent —— IID 噪声下两者势均力敌。

> 叙事:噪声越大、agent 拉开越多;无噪声时 ≈ FedAvg。
> ⚠️ **精确措辞**:clean 下 agent 比 FedAvg 低 0.006(在 ±0.01 seed 噪声内),是"代价可忽略"
> 而非"打平/更好"。且 agent 在 clean 也更稳(std 0.011 vs 0.017)。clean per-seed:
> fedavg 0.825/0.867/0.853, agent 0.827/0.852/0.848。
> ⚠️ het02 目前 agent 仅 s0, 需补 s1/s2(正在跑)。

### 表 2 — het04(噪声 0.4, IID)全方法消融(3 seed)

| 方法 | macro-AUROC (mean) | per-seed | vs FedAvg |
|---|---|---|---|
| Robust-FedProx (μ=0.05+dropout) | 0.6689 | 0.667/0.676/0.664 | **−0.069(掉分)** |
| FedAvg | 0.7380 | 0.737/0.738/0.739 | — |
| muonly(仅自适应 μ) | ~0.767 | 0.777(s0)/0.767(重跑) | +0.029(1 seed) |
| Agent(仅自适应加权) | 0.8043 | 0.791/0.810/0.812 | **+0.066** |
| agentmu(加权+自适应 μ) | 0.805 | 0.797/0.831/0.787 | +0.067 |

> 消融阶梯:静态正则掉分 < FedAvg < 只μ < 只加权 ≈ 加权+μ。
> **关键:agentmu ≈ agent(IID 下 μ 冗余, 三 seed 均值打平)。** 加权是主力。

### 表 2b — τ(门控软硬度)消融(het04 噪声0.4, seed0)

| τ | macro-AUROC | macro-AP |
|---|---|---|
| 0.02 | 0.8032 | 0.2729 |
| 0.03(默认) | 0.7908(s0) / 0.8043(3-seed 均值) | 0.2705 |
| 0.05 | 0.7933 | 0.2688 |

> **agent 对 τ 不敏感**:0.02–0.05 三档 AUROC 都落在 0.79–0.80,均显著高于 FedAvg(0.738)。
> 说明增益来自机制本身而非精调超参——这是论文里有价值的稳健性证据。

### 表 3 — 非IID(Dirichlet α=0.1 + 噪声 0.4):多 seed 更正版(⚠️ 之前 s0 结论被推翻)

多 seed macro-AUROC(高方差 regime):

| 方法 | macro-AUROC | macro-AP | macro-F1 | micro-F1 | seed 备注 |
|---|---|---|---|---|---|
| **Agent (仅gate)** | **0.7370 ± 0.049** | 0.2092 | 0.1915 | 0.5965 | ⚠️ 疑仅 s1/s2(见下) |
| het04_dir_floor | 0.6791 ± 0.044 | 0.1662 | 0.0955 | 0.0881 | s0/s1/s2 |
| FedAvg | 0.6774 ± 0.042 | 0.1635 | 0.0799 | 0.0744 | s0/s1/s2 |
| agentmu + floor | 0.6128 ± 0.095 | 0.1244 | 0.1184 | 0.2808 | 方差极大, 不稳 |
| CCR (RHFL) | 0.5030(s0) | 0.0566 | 0.000 | 0.000 | 仅 s0, 待 s1/s2 |

> **重大更正**:我此前"`floor` 把 0.533 救到 0.671、修复非IID 塌陷"的结论,**是基于单个 seed(s0)的
> 假象,多 seed 下不成立**。更正后的事实:
> 1. **所谓"非IID 塌陷"是 seed 特异的**:agent 只在 s0 掉到 0.533,s1/s2 恢复到 ~0.74,并非系统性塌陷。
> 2. **floor 不改善均值**:floor(0.679)≈ FedAvg(0.677),且**低于**未加 floor 的 agent(0.737)——
>    floor 反而把 agent 的门控收益稀释了。**"floor 修复非IID"不成立**,应写成 negative/limitation。
> 3. **agentmu+floor 更差且方差极大**(0.613 ± 0.095),不能作为稳定方案。
>
> ⚠️ **seed 匹配陷阱(务必修)**:`het04_dir_agent` 的汇总疑似**只含 s1/s2**(其 s0=0.533 在
> `agent_stage1/` 目录、没进 `agent_supp` 汇总),而 floor/fedavg 含 s0/s1/s2 → **agent 被高估、
> floor 被低估**,不是公平比较。**必须让 fedavg/agent/floor/ccr 用同一组 s0/s1/s2 重算**才能定论。
> 逐 seed 已知:s0 上 floor(0.671)> agent(0.533);s1/s2 上 agent(~0.74)> floor(~0.68)。
> → floor 更像"最坏情况保险(降方差)"而非"提均值",这条要谨慎、待 seed-matched 数据。
>
> **诚实定位**:非IID 是**高方差困难 regime**;稳的结论只有"**FedAvg 的 F1 基本崩(0.07),
> agent 门控在 ranking 上明显更好**";floor/μ 的"修复"目前**不成立**。headline 方法回到**纯 Agent 门控**。

### 表 4 — 效率(确定性卖点)

| 指标 | 值 |
|---|---|
| LoRA 可训练参数 | 1.23M / 304.5M = **0.39%** |
| 每轮上传 | ~19.6 MB |
| vs 全模型 fp32 (~1.22GB) | **≈ 64× 通信节省** |

---

### 表 5 — RHFL/CCR 3-seed 结果(2026-07-01 更新):与 agent 统计打平

het04(噪声0.4)全方法 3-seed(macro-AUROC):

| 方法 | macro-AUROC mean±std | macro-AP mean±std | per-seed(auroc) |
|---|---|---|---|
| Robust-FedProx | 0.6689±0.0053 | — | 0.667/0.676/0.664 |
| FedAvg | 0.7380±0.0006 | — | 0.737/0.738/0.739 |
| **Agent(ours)** | **0.8043±0.0096** | **0.2750±0.0057** | 0.791/0.810/0.812 |
| agentmu | 0.8052±0.0186 | — | 0.797/0.831/0.787 |
| **CCR(RHFL)** | **0.8089±0.0188** | 0.2682±0.0240 | 0.815/0.828/0.784 |

> **更正上一版警报**:补齐 s2(=0.7835)后,CCR 均值从 0.822 回落到 **0.809**,Agent 0.804——
> **两者在 het04 统计打平**(Δauroc=−0.005,远小于各自 std)。逐 seed:CCR 赢 s0/s1、Agent 赢 s2。
>
> **两个有价值的细节**:
> 1. **Agent 更稳**:auroc std 0.0096 vs CCR 0.0188;ap std 0.0057 vs 0.0240。我们的中位数门控比
>    CCR 的激进 softmax **方差小一半**——可复现性是可报告的优点。
> 2. **AP 上 Agent 略高**(0.275 vs 0.268)。
>
> **含义(不变)**:我们**打不出"噪声下比 CCR 更准"**(打平),所以差异化必须靠 CCR 的软肋——
> (a) **clean 零代价**:CCR 即使全干净也过度集中、丢数据,预期掉点;(b) **非IID**:CCR 过度集中预期
> 塌陷。这正是待跑的 `clean_ccr`(3 seed)与 `het04_dir_ccr`(3 seed)要验证的**决定性对照**。
>
> **对 FedAvg/Robust 仍是大幅、稳定的胜出**(+0.066 / +0.135,3 seed),且 Robust 反而掉分——这条
> 结论已 solid,不受 CCR 影响。

## 2. 数据支持的结论(2026-07-02 更正版)

1. **自适应 > 静态(强, solid)**:agent 在 0.2/0.4 噪声下 +0.038/+0.066(3-seed);
   静态 Robust-FedProx 时好时坏、甚至最差(het02 0.707、het04 0.669),**不可靠**。
2. **零代价(强, solid)**:clean 时 agent≈FedAvg(−0.006,噪声内)。
3. **Agent ≈ CCR(IID, 强)**:het02/het04 上 agent 与 RHFL/CCR 打平,agent 在 ranking
   (auroc/ap)略优、方差更小,CCR 在 micro-F1 略优。**不能 claim agent 全面压 CCR**。
4. **非IID:agent 门控明显优于 FedAvg 的 F1(强);但 floor/μ"修复"不成立(更正)**:
   多 seed 下 floor≈FedAvg<纯 agent,agentmu+floor 方差极大。→ **headline = 纯 Agent 门控,
   不含 floor/μ**。非IID 是高方差困难 regime,需 seed-matched 复算(见表3)。
5. **μ 与 floor 都非必需(更正)**:IID 下 μ 冗余、非IID 下 floor/μ 无益甚至有害 →
   **最终方法就是 probe-gated aggregation 本身,简洁**。

---

## 3. 需要补充的实验(按优先级)

> **状态更新(2026-06-30)**:P0 + 部分 P1 已实现并打包成 37-run 一键矩阵,
> 在分支 `feat/agent-supplement`(`configs/paper_matrix_agent_supp.yaml` +
> `run_supplement.sh`),等 GPU 跑出结果。详见 `docs/SUPPLEMENT_RUN.md`。

| 优先级 | 实验 | 目的 | 状态 |
|---|---|---|---|
| 🔴 P0 | **RHFL / 客户端置信度加权(CCR)baseline** | 必须对比同类自适应方法, 否则审稿人判"已被做过" | ✅ 已实现(`agent_aggregation: ccr`), 在补充矩阵 het02/het04 各 3 seed |
| 🔴 P0 | clean / het02 补 **s1/s2** | 主线剂量-响应曲线要 3 seed | ✅ 已加入补充矩阵 |
| 🟠 P1 | 非IID **weight-floor** 修复 + 3 seed | 救活=升档(短板变贡献) | ✅ 已实现(`agent_weight_floor`), 补充矩阵 floor / agentmu+floor 各 3 seed + floor 强度消融 |
| 🟠 P1 | 中等异质 **Dirichlet a=0.5** | 补非极端非IID 一档 | ✅ 已加入补充矩阵 |
| 🟠 P1 | **ODIR 第二数据集**复现主线 | 跨数据集泛化 | ⏳ 管线就绪, 暂未纳入本轮(下一轮) |
| 🟡 P2 | 噪声 sweep 补 **p=0.1, 0.3** | 剂量-响应曲线更密 | ⏳ 未做 |
| 🟡 P2 | **客户端数 K=8/10** + 每客户端异质噪声率 | scalability + 更真实异质 | ⏳ 未做 |
| 🟢 P3 | 非对称/类相关噪声 | 噪声模型更现实(至少讨论) | ⏳ 未做 |

补充矩阵跑完后, 论文主表(FedAvg / Robust / **RHFL-CCR** / Agent / Agent+floor 跨
clean/het02/het04/het04_dir)将**全部 3 seed**;τ / floor 强度 / a=0.5 为单 seed 调参曲线。

---

## 4. 是否足以形成论文 / solid / 创新点(诚实评估)

### 4.1 Solid 吗?

**部分 solid,补充矩阵跑完后即稳。**
- ✅ **已 solid**:het04(0.4)3 seed 主结果 + 全方法消融 + **τ 稳健性**(0.02–0.05 都赢)
  + 机制可解释 + 效率卖点。
- ⚠️ **尚不 solid(但已有解)**:clean / het02 仅 1 seed、非IID 仅 1 seed 且失败、
  缺 RHFL 同类 baseline——这些**已全部在 `feat/agent-supplement` 实现就绪**,等 GPU 结果。
- 一句话:**主结论方向 solid,证据广度的补法已落地**——补充矩阵(RHFL + 多 seed +
  weight-floor)结果回来后即可定稿。

### 4.2 创新点够吗?(关键, 直说)

**中等创新, 不是顶会级"新算法", 但够中端"新regime实证+简单有效方法"。**

- ❌ **不够新的**:核心机制(探针驱动的自适应客户端降权)是已有方向的变体——
  RHFL(CVPR'22 的 CCR)、FedIA、FedGSCA 都做过"按质量给客户端加权"。单论算法,
  审稿人会说增量有限。
- ✅ **真正的增量**(论文该主打这些, 而非"我们发明了自适应加权"):
  1. **基础模型 PEFT 联邦这个 regime 的新发现**:静态鲁棒(FedProx)在 RETFound+LoRA
     联邦下**反而掉分**——与小模型直觉相反, 此前工作没在 FM-PEFT 上验证过。
  2. **PEFT 特有的廉价遥测**:只有 adapter 在动, "全局 backbone+客户端 adapter"探针
     几乎零成本;全模型 FL 做不到这么便宜。
  3. **clean 零代价 + 剂量-响应**的干净刻画(自适应优雅退化为 FedAvg)。
  4. **多标签视网膜基础模型**应用(多数 FLNL 是单标签)。
  5. 非IID 失败边界的诚实刻画。

### 4.3 能投哪、什么条件

| 目标 | 可行性 | 前置条件 |
|---|---|---|
| 顶会主会(NeurIPS/CVPR/ICML) | ❌ | 机制新颖性+规模都不够 |
| MICCAI / MIDL 主会、J-BHI | ⚠️→✅ | 补 RHFL baseline + ODIR + clean/het02 多 seed |
| MICCAI DeCaF / FL workshop(最契合) | ✅ 现在即可 | 现有 + RHFL baseline |
| 中端期刊(J-BHI/CMIG) | ✅ | 同主会条件 |

---

## 5. 结论与建议路径

- **现状**:IID 主线(clean/0.2/0.4)已能讲一个完整、自洽、有反衬(静态 robust)的
  故事 + 可解释机制 + 64× 效率。**够一篇 workshop / 中端期刊。**
- **最值钱的一步**:补 **RHFL/CCR baseline**(P0)——这是从"能发"到"能上主会"的临门
  一脚,且工作量小。
- **第二步**:clean/het02 多 seed + ODIR;非IID 试 weight-floor。
- **创新定位**:不吹"新算法", 主打"**FM-PEFT 联邦下静态鲁棒失效 + 廉价 adapter 探针
  驱动的自适应编排 + 零代价剂量-响应**"这一实证+方法组合。这是诚实且站得住的。
