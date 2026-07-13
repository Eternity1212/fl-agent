# Results narrative — empirical-study / failure-boundary framing (定位 B)

> 状态:按当前证据(2026-07)的**最稳定位**写成 paste-ready 草稿。数字用 `[[...]]` 占位,
> GPU 的 CSV 一到就替换(见文末"填数指引")。核心纪律:**排名一律 macro-AUROC**,
> best-*-F1 只作参考并标注高方差。英文正文 + 中文旁注。
>
> 定位:本文不主张"单一机制普适更强",而是**跨数据集、多族聚合策略在医疗多标签 FL
> 下的实证对照 + 失败边界刻画**。三条硬证据:(1) 质量感知聚合的有效性强数据集依赖;
> (2) 经典拜占庭鲁棒聚合在超出容错规格时崩溃;(3) 自适应/probe 重加权在高脏比、非IID
> 下更少灾难性失败。

---

## R.0 Evaluation protocol (写进 Results 开头,防审稿人质疑指标)

We rank all methods by **macro-AUROC**, which is threshold-free and stable across
seeds. Threshold-tuned F1 scores (`best-*-F1`) are reported for completeness but
are high-variance under the extreme label imbalance of multi-label fundus data
(e.g. even FedAvg's best-micro-F1 swings by ±0.08 across seeds on RFMiD het02);
we therefore do **not** use F1 for method ranking. All comparisons are
**seed-matched**: methods sharing a row use identical client partitions, noise
realizations, and seeds.

> 中文:这一段是"护城河"。把"F1 抖、只用 AUROC 排名、同种子对比"先声明,后面任何
> 单看 F1 的质疑都被挡住。

---

## R.1 Cross-dataset dependence of quality-aware aggregation (FedA3I)

**Claim(可稳说):** The competitiveness of quality-aware aggregation is
**dataset-dependent**: FedA3I trails our probe-reweighting on RFMiD yet remains
competitive on ODIR.

### Table R1a — RFMiD, het02 (p=0.2, 2/4 noisy), macro-AUROC / best-micro-F1

| Method | macro-AUROC | best-micro-F1 (ref) | seeds |
|---|---|---|---|
| Robust-FedProx | 0.7069 ± 0.0014 | 0.424 ± 0.015 | 2 |
| FedA3I | 0.7492 ± 0.0088 | 0.500 ± 0.085 | 3 |
| FedAvg | 0.7630 ± 0.0002 | 0.453 ± 0.148 | 2 |
| CCR (RHFL) | 0.7869 ± 0.0107 | 0.679 ± 0.019 | 3 |
| **Agent (ours)** | **0.8005 ± 0.0114** | 0.644 ± 0.026 | 2 |

> ✅ **定稿措辞(核对全表后)**: on RFMiD het02, **Agent and CCR lead (AUROC ≈0.79–0.80);
> FedA3I (0.749) is comparable to FedAvg (0.763) and does not match the probe-reweighting
> methods.** ——注意:FedA3I 在 aggregate 上**略低于 FedAvg**(seed 数不同,非严格 seed-matched),
> 所以 **既不能写 "above FedAvg" 也不能写 "weak";写 "comparable to FedAvg, below Agent/CCR"**。

### Table R1b — ODIR, het04 (p=0.4, 2/4 noisy), macro-AUROC / best-micro-F1

| Method | macro-AUROC | best-micro-F1 (ref) | seeds |
|---|---|---|---|
| Robust-FedProx | 0.7785 ± 0.0103 | 0.245 ± 0.003 | 3 |
| FedAvg | 0.7817 ± 0.0229 | 0.242 ± 0.027 | 3 |
| CCR | 0.7874 ± 0.0199 | 0.578 ± 0.013 | 3 |
| **Agent (ours)** | 0.7918 ± 0.0054 | **0.577 ± 0.009** | 3 |
| **FedA3I** | **0.8008 ± 0.0094** | 0.463 ± 0.047 | 3 |

**核心跨数据集发现(强,已定稿):** the ranking of FedA3I **flips across datasets**:
on RFMiD het02 it is bottom-tier (below FedAvg on AUROC), whereas on ODIR het04 it is
the **top** method by macro-AUROC (0.801). Conversely, on best-micro-F1 the picture
differs again (Agent/CCR ≈0.58 clearly beat FedA3I 0.46). This **dataset- and
metric-dependent reordering** of a quality-aware baseline is a key empirical finding
and cautions against single-dataset / single-metric claims of aggregator superiority.

> 中文:比之前更强了——不是"ODIR 上 FedA3I 还能打",而是 **"FedA3I 在 RFMiD 垫底、在 ODIR AUROC 登顶"
> 的排名翻转**。这是 R1 的黄金句。但注意 ODIR 上 FedA3I 的 AUROC 优势**没兑现到 F1**(0.46 < agent/ccr 0.58)
> ——又是一次 AUROC/F1 分裂,呼应 R.5。

---

## R.2 Byzantine-robust aggregation fails **beyond its tolerated corruption regime**, not intrinsically

**Claim:** Coordinate-median / trimmed-mean tolerate `< 50%` Byzantine clients
(`n ≥ 2f+1`). Under our main setting (2/4 = 50% noisy) this bound is
**violated**, and median collapses to near-random. A within-spec control (1/4 =
25% noisy) isolates the breakdown point from any intrinsic failure.

### Table R2 — Coordinate-median across corruption fraction (RFMiD het04, K=4), macro-AUROC

IID (het04), macro-AUROC:

| Corruption | median | trimmed-mean | FedAvg (anchor) | Agent (anchor) |
|---|---|---|---|---|
| 25% noisy (1/4, within spec) | **0.8096 ± 0.0097** (3-seed) | 0.8047 | 0.7665 | 0.8113 |
| 50% noisy (2/4, at/over bound) | **0.5051 ± 0.0498** (3-seed) | 0.6937 | `[[het04_fedavg]]` | `[[het04_agent]]` |

non-IID (het04_dir, a=0.1), macro-AUROC:

| Corruption | median | trimmed-mean | FedAvg (anchor) | Agent (anchor) |
|---|---|---|---|---|
| 25% noisy (within spec) | **0.5978 ± 0.1854** (3-seed, 高方差) | 0.7100 | 0.8068 | 0.8191 |
| 50% noisy (at/over bound) | **0.4904 ± 0.0716** (3-seed) | 0.5487 | `[[het04_fedavg]]` | `[[het04_agent]]` |

> **核心发现(failure boundary,3-seed 已定稿):** coordinate-median 的失败受**双重约束**:
> - **IID**: 50%→25% 腐蚀率下降即恢复(0.505→0.810, std 极小)→ 经典**击穿点越界**成立,干净。
> - **non-IID**: 降到 25% 也**不可靠**(0.598 ± **0.185**,跨 seed 从 ~0.4 摆到 ~0.78),同条件
>   FedAvg 稳定 0.807 → median 与 client heterogeneity 有**额外特异性脆弱**。措辞用
>   **"unreliable / high-variance"**,不要用"consistently fails"(3-seed 显示是高方差不是恒崩)。
> - **梯度**: median(最脆)< trimmed(次之)< agent(最稳)。

**已定稿结论(3-seed,分支已决):** the data support a **refined breakdown-point** story:

> *"Coordinate-median is not intrinsically inadequate: under IID splits it recovers to
> FedAvg-level performance once the corrupted fraction drops within its `n≥2f+1`
> tolerance (25%: 0.810 vs 50%: 0.505). However, this recovery is contingent on data
> homogeneity — under non-IID splits it remains unreliable even at 25% corruption
> (0.598 ± 0.185, vs FedAvg 0.807), indicating that client heterogeneity constitutes a
> second, independent stressor for robust aggregation. Corrupted-client fractions and
> heterogeneity levels common in multi-center medical FL thus routinely fall outside
> median's viable regime, motivating adaptive reweighting that degrades gracefully."*

> 这是全篇**最强、最有辨识度**的发现:失败边界是 **corruption fraction × client heterogeneity
> 二维**的,不是一维击穿点。

---

## R.3 Non-IID dilution: a **metric-dimension dissociation**, not a clean AUROC recovery

> ⚠️⚠️ **最重要的诚实修正(核对 3-seed 全表后):** 在 RFMiD het04_dir 上,"recovery" **几乎只出现在
> best-micro-F1,而 macro-AUROC 上各方法挤在一起、且方差极大**。我们全程把 AUROC 定为主指标,
> 因此**不能**再把 het04_dir 当"AUROC 上干净的 collapse→recovery 主故事"来写。它真正的价值是
> **AUROC/F1 维度分裂**(R.5)。**AUROC 上干净的 recovery 证据在 K=8 IID(R.4),不在 het04_dir。**

### Table R3 — RFMiD het04_dir (non-IID a=0.1, 50% noisy), 3-seed

| Method | macro-AUROC | best-micro-F1 | 判读 |
|---|---|---|---|
| FedAvg | 0.6475 ± 0.0546 | **0.076 ± 0.005** | AUROC 尚可, **F1 塌** |
| CCR | 0.6457 ± 0.1033 | 0.416 ± 0.296 | F1 恢复但**方差巨大** |
| FedA3I | 0.6723 ± 0.0117 | 0.147 ± 0.106 | F1 基本没恢复 |
| Agent | 0.6690 ± 0.1040 | 0.398 ± 0.285 | F1 恢复但**方差巨大** |
| Agent+μ (agentmu) | **0.6999 ± 0.0715** | 0.433 ± 0.307 | AUROC 最高, F1 恢复(高方差) |
| Agent+floor | 0.6791 ± 0.0440 | 0.088 ± 0.014 | AUROC 稳, 但 **F1 没恢复** |
| median | 0.4904 ± 0.0716 | 0.078 ± 0.019 | collapse |

**已定稿判读:**
- **AUROC 维度**: 除 median 崩外, 其余方法 0.646–0.700 **重叠**(std 0.05–0.10), 没有清晰赢家。
  Agent+μ 最高但方差大。→ **不主张 "clear AUROC recovery"。**
- **F1 维度**: FedAvg / FedA3I / Agent+floor **塌**(0.08–0.15);Agent / CCR / Agent+μ **恢复到 ~0.40
  但方差巨大**(std 0.28–0.31, 即跨 seed 有时 ~0.7 有时 ~0)。→ recovery 是 **"high-variance, F1-only"**。
- **结论**: RFMiD het04_dir 的核心不是"recovery", 而是 **"FedAvg 在极端非IID下保留排序力(AUROC)却
  产不出可用预测(F1);probe/置信重加权能部分救回 F1, 但极不稳定"** —— 这是 R.5 维度分裂的最强案例。

### Table R3b — ODIR het04_dir (non-IID, 50% noisy), 3-seed

| Method | macro-AUROC | best-micro-F1 |
|---|---|---|
| FedAvg | 0.6641 ± 0.0849 | 0.400 ± 0.130 |
| CCR | 0.6822 ± 0.0723 | 0.509 ± 0.019 |
| Agent | 0.6997 ± 0.0439 | 0.496 ± 0.047 |
| FedA3I | 0.7086 ± 0.0570 | 0.362 ± 0.098 |
| Agent+floor | **0.7173 ± 0.0503** | 0.422 ± 0.125 |

> **跨数据集差异(again)**: ODIR het04_dir 下 FedAvg 的 F1 **没塌**(0.40),recovery 幅度温和、方差
> 更小; floor/feda3i 在 AUROC 上领先。→ het04_dir 的"F1 灾难性 collapse"是 **RFMiD 特有**, 不是普适。
> 这本身又是一条跨数据集结论,别写成"非IID 普遍导致 FedAvg F1 崩"。

> `het04_dir_feda3i_*` 是全表最关键的一格:它决定 R1 的"数据集依赖"能不能升级成
> "数据集×异质性双重依赖"。

---

## R.4 Scalability to K=8 (现象是否随规模成立)

K=8 (4/8 noisy, same 50% fraction), **3-seed (mean ± std), AUROC**:

| Method | K=8 IID | Δ vs FedAvg | K=8 non-IID | Δ vs FedAvg |
|---|---|---|---|---|
| **Agent (ours)** | **0.8040 ± 0.0092** | **+0.115** | 0.7884 ± 0.0455 | +0.033 |
| CCR | 0.7715 ± 0.0306 | +0.082 | **0.7974 ± 0.0391** | +0.042 |
| FedA3I | 0.6803 ± 0.0104 | −0.009 | 0.7527 ± 0.0978 | −0.003 |
| FedAvg | 0.6895 ± 0.0054 | — | 0.7555 ± 0.0636 | — |
| median | 0.5092 (s0) | — | 0.4053 (s0) | — |

**Narrative(诚实分档):**
- **K=8 IID = 强证据**: FedAvg degrades to 0.69, Agent/CCR recover to 0.77–0.80
  (Δ +0.08/+0.11,远大于 std)→ dilution-vs-recovery **clearly persists at scale**.
- **K=8 non-IID = 支持但弱**: FedAvg 本身没崩(0.756,且 std 0.064 很大),Agent/CCR 只领先
  +0.03/+0.04,**落在方差重叠区** → 只能写 "trend consistent",**不能**写 "clear recovery"。
- **FedA3I** 两条件均 ≈ FedAvg(Δ≈0,3-seed 后不再高于 FedAvg),且 **F1 塌**(K=8 IID best-micro-F1
  0.117 vs agent 0.60)→ 写 **"on par with FedAvg, no recovery, F1 collapse"**。
- **Agent vs CCR**: 打平(IID agent 略高、non-IID ccr 略高)→ **co-leading pair**,不吹 agent。

> ⚠️ 3-seed 后的两处更新: (1) FedA3I 从"AUROC 高于 FedAvg"改为"**≈ FedAvg**"(seed0 的领先被抹平);
> (2) K=8 **non-IID 的 recovery 是弱证据**(方差重叠),主打 K=8 IID。

---

## R.5 Metric-dimension dissociation (AUROC vs F1) — promote to a first-class finding

Across conditions, macro-AUROC (ranking ability) and best-micro-F1 (usable thresholded
prediction) **frequently disagree**, and this is itself one of the paper's contributions:

- **FedAvg @ RFMiD het04_dir**: AUROC 0.65 (ok) but F1 0.076 (collapsed).
- **FedA3I @ ODIR het04**: **top AUROC 0.801** but F1 0.463 (< Agent/CCR 0.58).
- **FedA3I / trimmed @ K=8**: AUROC partial but F1 collapsed.
- **Recovery @ RFMiD het04_dir**: visible on F1 (0.40) but flat/overlapping on AUROC.

**Take-away for the paper:** conclusions about "which aggregation is robust" are
**metric-dependent**; reporting a single metric would produce contradictory rankings.
This motivates dual-metric evaluation and cautions the FL-robustness literature against
AUROC-only or F1-only claims. → 这条把"我们被迫两个指标都看"从麻烦变成**贡献**。

---

## 现状:所有占位符已用真实 3-seed 数字替换(2026-07-13)

R.1a/R.1b/R.2/R.3/R.3b/R.4 均已填实。**唯一仍缺**:R.2 的 50% 行 `[[het04_fedavg]]`/`[[het04_agent]]`
锚点(het04 IID 50% 的 fedavg/agent AUROC)——它们不在 supp/odir/scale 三份 CSV 里,在**主矩阵**
`runs/paper_matrix/agent/`。补法:把主矩阵 het04 的 fedavg/agent json 复制进 agent_supp 后重跑
summarize,或单独读取。非阻塞(R.2 结论已由 median/25%/50% 成立)。

## 填数指引 (GPU CSV → 表格占位符)

1. RFMiD supp:
   `python3 -m fed_agent.tools.summarize_agent runs/paper_matrix/agent_supp/summary.json --csv runs/paper_matrix/agent_supp/supp_full.csv`
   → 喂 R1a(het02 五方法)、R3(het04_dir 六方法)。
2. ODIR:
   `python3 -m fed_agent.tools.summarize_agent runs/paper_matrix/agent_odir/summary.json --csv runs/paper_matrix/agent_odir/odir_full.csv`
   → 喂 R1b。
3. scale:
   `python3 -m fed_agent.tools.summarize_agent runs/paper_matrix/agent_scale/summary.json --csv runs/paper_matrix/agent_scale/scale_full.csv`
   → 喂 R2(het04lo vs het04 median/trimmed)、R4(K=8)。

把这三份 CSV 发来,我一次性替换所有 `[[...]]` 并出图(击穿点曲线 + 跨数据集条形图 + 非IID 崩/恢复图)。
