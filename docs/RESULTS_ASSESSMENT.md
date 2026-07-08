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

### 表 3 — 非IID(Dirichlet α=0.1 + 噪声 0.4):5 方法 3-seed 定稿(2026-07-03, 42 run 全完成)

全部 seed 对齐(同在 `agent_supp`,均 s0/s1/s2):

| 方法 | macro-AUROC | macro-AP | macro-F1 | micro-F1 | F1 崩塌种子 |
|---|---|---|---|---|---|
| dir_agentmu(gate+μ) | 0.6999 ± 0.072 | 0.1787 ± 0.081 | 0.1337 ± 0.098 | 0.4325 ± 0.307 | 仅 s0 |
| dir_floor | 0.6791 ± 0.044 | 0.1662 ± 0.036 | 0.0955 ± 0.028 | 0.0881 ± 0.014 | **s0/s1/s2 全崩** |
| dir_agent(仅gate) | 0.6690 ± 0.104 | 0.1608 ± 0.074 | 0.1276 ± 0.092 | 0.3977 ± 0.285 | 仅 s0 |
| dir_ccr(RHFL) | 0.6457 ± 0.103 | 0.1626 ± 0.077 | 0.1345 ± 0.096 | 0.4163 ± 0.296 | 仅 s0 |
| FedAvg | 0.6475 ± 0.055 | 0.1345 ± 0.047 | 0.0784 ± 0.004 | 0.0757 ± 0.005 | **s0/s1/s2 全崩** |

逐 seed micro-F1(关键证据):

| 方法 | s0 | s1 | s2 |
|---|---|---|---|
| FedAvg | ~0.07 | ~0.07 | ~0.08 |
| dir_floor | ~0.07 | ~0.09 | ~0.10 |
| dir_ccr | **0.000** | 0.585 | 0.664 |
| dir_agent | ~0.00 | ~0.60 | ~0.60 |
| dir_agentmu | **0.000** | ~0.65 | ~0.65 |

> **判决点 1 — μ 没能稳住塌陷**:`het04_dir_agentmu_s0` = auroc 0.606 / **F1 = 0**,仍崩。
> → 不能写"μ 稳住 non-IID";只能写"μ 提升均值但不能跨 seed 稳定防崩"。
>
> **判决点 2 — CCR 非IID 不是全崩**:3-seed = 0.646 ± 0.103,s0 崩(0.503/F1 0)但 s1/s2 正常
> (0.69/0.74)。→ 不能写"CCR 非IID 全崩";只能写"CCR 非IID 显著退化 + 方差大增"。
>
> **★ 统一机制(本表最有价值的发现,你我此前都没点破)**:按"F1 崩塌种子"列分成两类——
> - **稀释型聚合(FedAvg、floor):3 个 seed 上 F1 全崩(≈0.08)**,毫无恢复。
> - **置信度重加权(agent、ccr、agentmu):在 s1/s2 恢复到 F1 0.6+,只在 s0 崩**。
>
> 即:**重加权家族能在多数非IID 划分上救回 F1,朴素/被稀释的聚合永远救不回**。floor 之所以"没用",
> 正是因为 weight_floor=0.5 把门控拉回接近均匀 → **抹掉了让 agent 能恢复的激进降权**(机制自洽,floor 应弃)。
> s0 是一个病态划分:Dirichlet 使部分标签只落在极少数客户端,**任何降权都会丢掉这些标签的唯一载体 → F1 归零**。
> 这把"非IID 开放问题"从"我们的方法不行"精确改写成:**"重加权家族在多数划分上有效,但存在标签覆盖病态划分,
> 目前无解"**——有机制、可辩护,是可发表的 negative finding,而非硬伤。
>
> **可写的强结论**:(1) 稀释型聚合非IID F1 全崩;(2) 重加权家族(含 agent)在多数划分恢复 F1;
> (3) floor 因抹掉降权而失效;(4) agent 与 CCR、agentmu 在非IID 统计打平(方差内)。
> **不能写**:μ 稳住塌陷 ✗;CCR 非IID 全崩 ✗;agent 单点碾压全场 ✗。

### 表 3b — 跨数据集(ODIR, 8类)非IID het04_dir 复现:逐 seed micro-F1

| 方法 | s0 | s1 | s2 | mean ± std |
|---|---|---|---|---|
| CCR (RHFL) | 0.506 | 0.553 | 0.496 | **0.518 ± 0.025** |
| Agent (gate) | 0.405 | 0.541 | 0.504 | 0.483 ± 0.057 |
| Floor | 0.258 | 0.471 | 0.534 | 0.421 ± 0.118 |
| FedAvg | 0.250 | 0.508 | 0.442 | 0.400 ± 0.109 |

> **ODIR 只是"部分复现",要如实区分**:
> - **一致的部分(可写强)**:重加权家族(CCR/agent)在两个数据集上都 **>FedAvg**;且**增益在最难的 s0 最大**
>   (ODIR s0:FedAvg 0.25 → agent 0.41 / CCR 0.51)。→ "adaptive/reweighting family 优于 vanilla,且在
>   非IID 病态最重时救场最明显",这是**跨数据集成立**的核心论点。
> - **不一致的部分(必须承认)**:RFMiD 上"稀释型 F1 全崩(≈0.08)"的**硬塌陷在 ODIR 没出现**——
>   ODIR 上 FedAvg/floor 也有 0.40/0.42,只是 s0 偏低。原因:ODIR 仅 8 类,"某标签只落一个客户端"的病态
>   远轻于 RFMiD ~28 类。→ **正确的跨数据集表述是"重加权家族的增益幅度随标签空间/非IID 病态严重度放大"**,
>   而非"稀释型总是全崩"。标签数是调节变量,两个数据集正好把现象的两端框住。
> - **agent 不是 ODIR 最优**:ODIR 上 CCR(0.518)> agent(0.483),且 CCR 更稳(std 0.025 vs 0.057)。
>   → **"agent 更稳"是数据集/regime 相关的**(RFMiD IID het04 上 agent 比 CCR 稳,ODIR 非IID 上反过来),
>   **不能写成普适结论**。
>
> **★ 结论对论文的影响**:agent 相对 CCR 已**没有"更强/更稳"的普适优势**。agent 唯一可能的普适差异化 =
> **自适应零代价(clean 时门控自动回均匀、不伤;CCR 的激进 softmax 恒集中、预期伤 clean)**。
> → **`clean_ccr` 3-seed 是现在最该看的数据**:若 CCR clean 掉分而 agent 不掉,agent 的"adaptivity"就是
> 干净可辩护的核心贡献;若 CCR clean 也不掉,则 agent 只能定位为"重加权家族的一员,与 CCR 相当"。

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

---

## 6. 最终判词(2026-07-07, 全数据到位后)

### 6.1 数据全景(RFMiD + ODIR)

| 场景 | 赢家 | 关键事实 |
|---|---|---|
| clean (RFMiD) | agent 微胜 | micro-F1 0.704 vs 0.681(更稳);auroc 打平 |
| IID het02 | agent ≈ CCR ≫ fedavg/robust | robust 反噬(最差) |
| IID het04 | CCR ≈ agentmu ≈ agent ≫ fedavg | agent 方差比 CCR 小(此 regime) |
| non-IID (RFMiD) | 重加权家族 > 稀释型 | 家族层面成立, 家族内打平(方差大) |
| non-IID (ODIR) | **CCR > agent** > floor > fedavg | 硬塌陷未复现;agent 非最优、且不如 CCR 稳 |

### 6.2 创新点:诚实分级

- **不成立的强 claim**:❌ "我们发明了更好的聚合方法"——CCR(已有 RHFL)在多数场景 ≥ agent,
  ODIR 上还更强更稳。agent 的聚合机制**没有**普适的性能/稳定性优势。
- **成立的中等 claim**(论文可立):
  1. **实证规律(跨数据集)**:置信度重加权家族在非IID+噪声下优于朴素聚合,**增益随标签空间/
     非IID 病态严重度放大**(RFMiD 28类硬塌陷 ↔ ODIR 8类温和)——标签数是调节变量。这是有机制、
     跨两数据集验证的 empirical finding。
  2. **静态鲁棒反噬**:Robust-FedProx 在异质噪声下反而最差 → 需要自适应。
  3. **系统 benchmark**:RETFound+LoRA 联邦 + 异质标签噪声 + IID/非IID + 多 seed + 同类 baseline(CCR)。
  4. **失败边界诚实刻画**:标签覆盖病态 = 开放问题。
- **已解决的关键 claim(clean_ccr 3-seed 到位, 中间剧本)**:clean 三方法——
  agent auroc 0.8421 / F1 0.7036;fedavg 0.8482 / 0.6810;**ccr 0.8386 / 0.6933**。
  → CCR 有"clean 代价"但**小**:两指标上 CCR 都是自适应/重加权里较低的一个(auroc 全场最低、
  F1 低于 agent 0.01),方向支持"恒集中丢多样性",但非戏剧性崩塌(均在 ~1 std)。
  **结论:adaptivity 是次要支撑点,不是压倒性卖点。** agent 的真正定位 = **全 regime 最佳 all-rounder**:
  clean 上 micro-F1 最高、受污染 IID 匹配 CCR、非IID 属恢复家族——**没有任何单场景明显输**,而 CCR clean 略低、
  robust 噪声反噬、fedavg 噪声崩。这是"跨 regime Pareto 稳健"的诚实主线。

### 6.3 能否支撑论文:结论

**能,但要认清天花板。** 现有数据(即使 clean_ccr 不给力)已支撑一篇**诚实、扎实、有跨数据集实证规律的
medical-FL 论文**,合适出口:MICCAI 卫星 workshop(DeCaF)/ FL workshop / MIDL short / 中端期刊(J-BHI, CMIG)。
**不够**顶会 ML 主会(机制新颖性不足)。

**把它往上抬一档的唯一低成本杠杆 = `clean_ccr` 3-seed**:
- 若确立 adaptivity 优势 → 主线写成"自适应门控:受污染时匹配 CCR,clean 时不付 CCR 的代价" → 冲 MIDL/MICCAI 主会有戏。
- 若不确立 → 主线写成"何时该用客户端置信度重加权?一项跨数据集实证研究" → workshop/期刊稳。
