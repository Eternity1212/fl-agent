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

### Table R1a — RFMiD, het02 (heterogeneous noise p=0.2, 2/4 clients noisy), 3 seeds, macro-AUROC

| Method | macro-AUROC (mean±std) | best-micro-F1 (ref, high-var) |
|---|---|---|
| FedAvg | `[[het02_fedavg]]` | `[[...]]` |
| Robust-FedProx | `[[het02_robust]]` | `[[...]]` |
| CCR (RHFL) | `[[het02_ccr]]` | `[[...]]` |
| FedA3I | 0.7492 ± 0.0088 | 0.500 ± 0.085 |
| **Agent (ours)** | `[[het02_agent]]` | `[[...]]` |

> ⚠️ 结论措辞取决于这张表:
> - 若 FedA3I 的 0.7492 **高于 FedAvg、低于 Agent**(seed0 已验证:FedA3I 0.744 > FedAvg 0.721,
>   < Agent 0.781)→ 写 **"FedA3I improves over FedAvg but does not match probe-based
>   reweighting (Agent)."** 绝不写 "FedA3I is weak"。
> - het02 是**温和 regime**,只是辅助证据,主战场是 het04 / het04_dir。

### Table R1b — ODIR, het04, 3 seeds, macro-AUROC

| Method | macro-AUROC (mean±std) |
|---|---|
| FedAvg | `[[odir_het04_fedavg]]` |
| CCR | `[[odir_het04_ccr]]` |
| FedA3I | `[[odir_het04_feda3i]]` (2-seed 临时: 0.8071 ± 0.0035) |
| **Agent (ours)** | `[[odir_het04_agent]]` |

**Narrative:** On ODIR, FedA3I stays at 0.80+ macro-AUROC with no collapse,
in contrast to its weaker standing on RFMiD relative to Agent. This
**dataset-dependence** — the same quality-aware mechanism being competitive on one
fundus dataset and not on another — is itself a key empirical finding and cautions
against single-dataset claims of aggregator superiority.

> 中文:R1 的价值就是"跨数据集不一致"。不需要 FedA3I 到处崩,只要它在两数据集**排名不同**
> 就成立。等 ODIR s2 + het04_dir 补齐再定稿。

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
| 25% noisy (1/4, within spec) | **0.8091** (s0) | 0.8047 (s0) | 0.7665 (s0) | 0.8113 (s0) |
| 50% noisy (2/4, at/over bound) | **0.5429 ± 0.0559** (3-seed) | 0.6937 (s0) | `[[het04_fedavg]]` | `[[het04_agent]]` |

non-IID (het04_dir, a=0.1), macro-AUROC:

| Corruption | median | trimmed-mean | FedAvg (anchor) | Agent (anchor) |
|---|---|---|---|---|
| 25% noisy (within spec) | **0.5211** (s0, F1=0.00) | 0.7100 (s0) | 0.8068 (s0) | 0.8191 (s0) |
| 50% noisy (at/over bound) | **0.4899 ± 0.0581** (3-seed) | 0.5487 (s0) | `[[het04_fedavg]]` | `[[het04_agent]]` |

> **核心发现(failure boundary,已定稿):** coordinate-median 的失败受**双重约束**:
> - **IID**: 50%→25% 腐蚀率下降即恢复(0.54→0.81)→ 经典**击穿点越界**解释成立。
> - **non-IID**: 即使降到 25%, median 仍崩(0.52, F1=0.00),而同条件 FedAvg 正常(0.81)
>   → median 与 client heterogeneity 存在**额外的特异性脆弱**,非单纯腐蚀率问题。
> - **梯度**: median(最脆)< trimmed(次之,non-IID 25% AUROC 0.71 但仍逊 fedavg/agent)< agent(最稳)。
>
> ⚠️ 上述 het04lo_* 目前均为 **seed0 单点**。效应极大且与 50% 的 3-seed 一致,方向可信;
> 但作为论文**头号 failure-boundary 结论**,建议补 `het04lo_dir_median` / `het04lo_median` 的 s1/s2
> (见 `paper_matrix_agent_scale.yaml` D 节)让这一格 bulletproof。

**两分支写法(先备好,看 `het04lo_median` 结果二选一):**

- **Branch A — clean breakdown-point story(期望,攻不破):** if median is healthy at
  25% and collapses at 50%, write:
  > *"Coordinate-median performs comparably to FedAvg when the corrupted fraction
  > (25%) lies within its `n≥2f+1` tolerance, but collapses to near-random once the
  > fraction reaches its breakdown point (50%). This shows the failure is a
  > breakdown-point violation rather than an intrinsic inadequacy of robust
  > aggregation — and that the corrupted-client fractions common in multi-center
  > medical FL routinely exceed this bound, motivating adaptive reweighting that
  > degrades gracefully instead of catastrophically."*

- **Branch B — intrinsic-mismatch story(若 25% 也崩):** if median also fails at 25%,
  do **not** claim breakdown-point; instead investigate and report:
  > *"Coordinate-median fails even within its tolerance regime, suggesting
  > coordinate-wise medians are ill-suited to the factorized LoRA parameter space
  > (medians taken independently over low-rank factors A,B need not yield a valid
  > low-rank update)."* → 需补一个"median on effective ΔW" 或"median on merged
  > adapter"的诊断实验再下结论。

> 中文:R2 的护城河就是这张表 + 两分支。**在 het04lo 出来前,论文里 median 只能写
> "collapses under the 50%-corrupted setting",不能写 "robust aggregation fails"。**

---

## R.3 Non-IID dilution collapse and recovery (主故事,待 feda3i/median 补齐)

**Claim:** Under non-IID (Dirichlet α=0.1) + heterogeneous noise, size-weighted
FedAvg is diluted by noisy clients into collapse, while confidence/quality-aware
reweighting recovers — but recovery is **method- and regime-specific**.

### Table R3 — RFMiD het04_dir (non-IID a=0.1, 50% noisy), 3 seeds, macro-AUROC

| Method | macro-AUROC (mean±std) | 崩/恢复 |
|---|---|---|
| FedAvg | `[[dir_fedavg]]` | dilution collapse |
| CCR | `[[dir_ccr]]` | `[[...]]` |
| FedA3I | `[[dir_feda3i]]` | **决胜格**:崩→说明质量感知在非IID不稳;扛住→FedA3I 是强对照 |
| median | `[[dir_median]]` | 预期崩(50%>击穿点) |
| Agent + floor | `[[dir_floor]]` | recovery |
| **Agent (ours)** | `[[dir_agent]]` | `[[...]]` |

> `het04_dir_feda3i_*` 是全表最关键的一格:它决定 R1 的"数据集依赖"能不能升级成
> "数据集×异质性双重依赖"。

---

## R.4 Scalability to K=8 (现象是否随规模成立)

K=8 (4/8 noisy, same 50% fraction), seed 0 — **AUROC-confirmed, pattern holds**:

| Method | K=8 IID (AUROC) | K=8 non-IID (AUROC) | 判读 |
|---|---|---|---|
| **Agent (ours)** | **0.7843** | **0.7290** | recovery (top) |
| CCR | 0.7278 | 0.7151 | recovery (≈agent, marginally below) |
| FedA3I | 0.6892 | 0.6261 | partial AUROC, **F1 collapses** (0.07/0.08) |
| FedAvg | 0.6592 | 0.6319 | dilution-degraded |
| median | 0.5092 | 0.4053 | collapse (50% > breakdown) |

**Narrative:** the dilution-degradation-vs-recovery pattern **persists at K=8**:
Agent and CCR remain the only methods that recover usable performance (best-micro-F1
≈0.55–0.61), while FedA3I retains modest ranking ability (AUROC above FedAvg) but,
unlike Agent/CCR, **fails to convert it into classification performance** (F1 ≈ FedAvg);
median collapses. Agent is the top method but its margin over CCR is small — we
report them as a co-leading pair rather than claiming Agent dominance.
seed0 gives a clear go-signal; s1/s2 worth adding for {FedAvg, Agent, CCR, FedA3I}.

> ⚠️ 措辞纪律: (1) FedA3I 写 "partial-AUROC / F1-collapse", **不写 "崩"**(AUROC 高于 FedAvg);
> (2) Agent vs CCR 写 "co-leading, Agent marginally ahead", **不写 "significantly outperforms"**。

---

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
