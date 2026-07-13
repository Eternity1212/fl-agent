# Paper draft — Results + Discussion + Contributions/Positioning (DeCaF / MIDL 级初稿)

> 状态:基于当前已落盘的 3-seed 结果(RFMiD supp / ODIR / scale 三份 CSV + het04lo)写成的
> **完整初稿**。数字均来自 `docs/RESULTS_NARRATIVE.md`(已用真实数据填实)。FedNoRo 与
> 非对称噪声两组实验为**进行中**,本稿标注 `[pending]`,结果一到即并入 R.1/R.2/R.6。
>
> 定位:**不主张单一机制普适更强**;而是一篇**跨数据集 × 跨规模 × 跨聚合族的实证对照 +
> 失败边界刻画**论文。四条硬腿:(1) 质量感知聚合有效性强数据集依赖;(2) 经典拜占庭鲁棒
> 聚合在超容错规格 / 非IID 下崩溃;(3) probe/自适应重加权在高脏比、非IID 下更少灾难性失败;
> (4) AUROC 与 F1 维度分裂本身是一项结论。

---

## 0. Working title / abstract skeleton

**Title (候选):**
*"When Does Robust Federated Aggregation Actually Help? An Empirical Study of
Reweighting, Byzantine-Robust, and Quality-Aware Methods for Multi-Label Retinal
Diagnosis under Label Noise."*

**Abstract (draft):**
Federated learning (FL) for multi-label fundus diagnosis must cope with
heterogeneous, noisily-labelled clients, and a growing zoo of "robust"
aggregators promise resilience. We ask a simpler question: *under which
conditions do these methods actually help, and where do they break?* Using
RETFound + LoRA on two public datasets (RFMiD, ODIR-5K), we benchmark five
families — vanilla FedAvg, proximal (FedProx), Byzantine-robust
(coordinate-median, trimmed-mean), quality-aware (FedA3I, FedNoRo), and
probe/confidence reweighting (CCR/RHFL, and an adaptive probe-gated agent) —
across corruption fractions (25–50% noisy clients), symmetric/asymmetric label
noise, non-IID (Dirichlet) partitions, and client scale (K=4, K=8). Ranking by
threshold-free macro-AUROC (with F1 as a secondary usability metric), we find
that (i) the competitiveness of quality-aware aggregation *flips across
datasets* (FedA3I is bottom-tier on RFMiD but top-ranked on ODIR); (ii)
coordinate-median fails along a **two-dimensional boundary** — it recovers when
the corrupted fraction drops within its `n≥2f+1` tolerance under IID splits, but
stays unreliable under non-IID splits even at 25% corruption; (iii)
probe/confidence reweighting degrades most gracefully at high corruption and
scale; and (iv) AUROC and F1 rankings frequently disagree, so single-metric
claims of aggregator superiority are unsafe. We release the full benchmark and
one-command reproduction.

---

## 1. Contributions / positioning (终版)

We frame the contributions as an **empirical study + failure-boundary
characterization**, not a new-SOTA claim:

1. **A unified, seed-matched benchmark** of five aggregation families for
   multi-label medical FL under label noise, on two datasets and two client
   scales, with a foundation-model backbone (RETFound + LoRA) — a setting under
   which most robust-FL claims have *not* been validated.
2. **Cross-dataset non-transferability of quality-aware aggregation:** FedA3I's
   rank *inverts* between RFMiD (bottom) and ODIR (top). This cautions against
   single-dataset aggregator claims.
3. **A two-dimensional failure boundary for Byzantine-robust aggregation:**
   coordinate-median's collapse is governed by *both* corruption fraction and
   client heterogeneity, not corruption fraction alone. We isolate this with a
   within-spec (25%) vs over-bound (50%) control under IID and non-IID splits.
4. **Metric-dimension dissociation as a first-class finding:** macro-AUROC
   (ranking) and best-F1 (usable prediction) routinely disagree; we argue for
   mandatory dual-metric reporting in robust-FL evaluation.
5. **Reproducibility:** one-command pipeline (data download → split → matrix →
   summary/figures), all configs and seeds released.

> 中文:第 2、3、4 条是"卖点"。第 3 条(二维失败边界)是最有辨识度的发现。

---

## 2. Method / experimental design (Results 之前的最短必要交代)

**Backbone & PEFT.** RETFound (MAE-pretrained ViT-Large) with LoRA (rank 8,
α=16) on all attention/MLP linear layers; only LoRA adapters + head are
communicated each round (upload bytes accounted per round).

**Aggregation families evaluated.**
- *FedAvg* — size-proportional averaging (anchor).
- *FedProx* / *robust-FedProx* — proximal term against global model.
- *Byzantine-robust* — coordinate-median, trimmed-mean (trim 0.25).
- *Quality-aware* — **FedA3I** (per-sample loss GMM → quality-weighted
  averaging) and **FedNoRo** (two-stage: FedAvg warmup → client-level loss GMM
  splits clean/noisy → noisy clients trained with a robust GCE loss +
  distance-aware aggregation). `[FedNoRo results pending]`
- *Probe/confidence reweighting* — **CCR/RHFL** (softmax over client confidence)
  and our **adaptive agent** (probe-validation gate centred on the median score;
  reduces to FedAvg when clients are equally good), with optional adaptive
  per-client μ and a weight-floor variant.

**Noise models.** Symmetric label flip (main), and **asymmetric
class-conditional** noise (positives dropped at p, negatives raised at 0.1p;
clinically mimics under-reporting). `[asymmetric results pending]`

**Heterogeneity / scale.** IID and Dirichlet (α=0.1) partitions; K=4 and K=8
clients; corruption fractions 25% (1/4) and 50% (2/4, or 4/8).

**Protocol (护城河段).** We rank by **macro-AUROC** (threshold-free, seed-stable).
best-micro/macro-F1 are reported for usability but *not* used for ranking
(±0.08 seed swings under extreme imbalance). All comparisons are **seed-matched**
(identical partitions, noise realizations, seeds).

---

## 3. Results

### 3.1 Cross-dataset dependence of quality-aware aggregation (R.1)

On **RFMiD het02** (p=0.2, 2/4 noisy), probe-reweighting leads and FedA3I trails:

| Method | macro-AUROC | best-micro-F1 (ref) | seeds |
|---|---|---|---|
| Robust-FedProx | 0.7069 ± 0.0014 | 0.424 ± 0.015 | 2 |
| FedA3I | 0.7492 ± 0.0088 | 0.500 ± 0.085 | 3 |
| FedAvg | 0.7630 ± 0.0002 | 0.453 ± 0.148 | 2 |
| CCR (RHFL) | 0.7869 ± 0.0107 | 0.679 ± 0.019 | 3 |
| **Agent (ours)** | **0.8005 ± 0.0114** | 0.644 ± 0.026 | 2 |
| FedNoRo | `[pending]` | `[pending]` | 3 |

On **ODIR het04** (p=0.4, 2/4 noisy), the ranking **flips** — FedA3I tops AUROC:

| Method | macro-AUROC | best-micro-F1 (ref) | seeds |
|---|---|---|---|
| Robust-FedProx | 0.7785 ± 0.0103 | 0.245 ± 0.003 | 3 |
| FedAvg | 0.7817 ± 0.0229 | 0.242 ± 0.027 | 3 |
| CCR | 0.7874 ± 0.0199 | 0.578 ± 0.013 | 3 |
| **Agent (ours)** | 0.7918 ± 0.0054 | **0.577 ± 0.009** | 3 |
| **FedA3I** | **0.8008 ± 0.0094** | 0.463 ± 0.047 | 3 |

**Finding.** FedA3I is bottom-tier on RFMiD het02 (below FedAvg on AUROC) yet
the top method on ODIR het04. This **dataset- and metric-dependent reordering**
of a quality-aware baseline is a key empirical result and cautions against
single-dataset / single-metric aggregator claims. (Note the AUROC/F1 split:
FedA3I's ODIR AUROC lead does *not* translate to F1 — 0.463 vs Agent/CCR 0.58.)

### 3.2 Byzantine-robust aggregation: a two-dimensional failure boundary (R.2)

Coordinate-median / trimmed-mean tolerate `<50%` corrupted clients (`n≥2f+1`).
We sweep corruption fraction under IID and non-IID splits (RFMiD het04, K=4):

**IID (het04), macro-AUROC:**

| Corruption | median | trimmed | FedAvg | Agent |
|---|---|---|---|---|
| 25% (within spec) | **0.8096 ± 0.0097** | 0.8047 | 0.7665 | 0.8113 |
| 50% (at/over bound) | **0.5051 ± 0.0498** | 0.6937 | `[[het04_fedavg]]` | `[[het04_agent]]` |

**non-IID (het04_dir, α=0.1), macro-AUROC:**

| Corruption | median | trimmed | FedAvg | Agent |
|---|---|---|---|---|
| 25% (within spec) | **0.5978 ± 0.1854** | 0.7100 | 0.8068 | 0.8191 |
| 50% (at/over bound) | **0.4904 ± 0.0716** | 0.5487 | `[[het04_fedavg]]` | `[[het04_agent]]` |

**Finding.** Median's failure is governed by **two** stressors:
- *IID:* dropping corruption 50%→25% restores it (0.505→0.810, tiny std) — a
  clean **breakdown-point** crossover consistent with `n≥2f+1`.
- *non-IID:* even at 25% it is **unreliable** (0.598 ± 0.185, seed-to-seed
  swings ~0.4–0.78, vs stable FedAvg 0.807) — client heterogeneity is a *second,
  independent* stressor.

The severity gradient is **median (most brittle) < trimmed < agent (most
stable)**. Corrupted fractions and heterogeneity common in multi-centre medical
FL thus routinely fall outside median's viable regime, motivating adaptive
reweighting that degrades gracefully.

### 3.3 Non-IID dilution: a metric-dimension dissociation (R.3)

On **RFMiD het04_dir** (non-IID, 50% noisy), 3-seed:

| Method | macro-AUROC | best-micro-F1 | reading |
|---|---|---|---|
| FedAvg | 0.6475 ± 0.0546 | 0.076 ± 0.005 | AUROC ok, **F1 collapsed** |
| CCR | 0.6457 ± 0.1033 | 0.416 ± 0.296 | F1 recovers, huge variance |
| FedA3I | 0.6723 ± 0.0117 | 0.147 ± 0.106 | F1 barely recovers |
| Agent | 0.6690 ± 0.1040 | 0.398 ± 0.285 | F1 recovers, huge variance |
| Agent+μ | **0.6999 ± 0.0715** | 0.433 ± 0.307 | top AUROC, F1 recovers (variance) |
| Agent+floor | 0.6791 ± 0.0440 | 0.088 ± 0.014 | AUROC stable, F1 flat |
| median | 0.4904 ± 0.0716 | 0.078 ± 0.019 | collapse |

**Finding.** On **AUROC**, all non-collapsed methods overlap (0.65–0.70, std
0.05–0.10) — no clean winner. On **F1**, FedAvg/FedA3I/floor stay collapsed
(0.08–0.15) while CCR/Agent/Agent+μ recover to ~0.40 *with huge variance*
(std ~0.29). The story is not "recovery" but **"FedAvg retains ranking power
(AUROC) yet yields unusable thresholded predictions (F1); confidence/probe
reweighting partially restores F1 but unstably."**

On **ODIR het04_dir**, FedAvg's F1 does *not* collapse (0.40), and recovery is
milder with smaller variance — so the catastrophic F1 collapse under non-IID is
**RFMiD-specific**, not universal (another cross-dataset caveat).

### 3.4 Scalability to K=8 (R.4)

K=8 (4/8 noisy, 50% fraction), 3-seed, macro-AUROC:

| Method | K=8 IID | Δ vs FedAvg | K=8 non-IID | Δ vs FedAvg |
|---|---|---|---|---|
| **Agent (ours)** | **0.8040 ± 0.0092** | **+0.115** | 0.7884 ± 0.0455 | +0.033 |
| CCR | 0.7715 ± 0.0306 | +0.082 | **0.7974 ± 0.0391** | +0.042 |
| FedA3I | 0.6803 ± 0.0104 | −0.009 | 0.7527 ± 0.0978 | −0.003 |
| FedAvg | 0.6895 ± 0.0054 | — | 0.7555 ± 0.0636 | — |
| median | 0.5092 (s0) | — | 0.4053 (s0) | — |

**Finding.** The dilution-vs-recovery pattern **persists at scale**: on K=8 IID,
FedAvg degrades to 0.69 while Agent/CCR recover to 0.77–0.80 (Δ well beyond std)
— strong evidence. On K=8 non-IID the trend is consistent but weaker (FedAvg
itself does not collapse; leads fall within variance). FedA3I is ≈FedAvg on
AUROC with collapsed F1 (no recovery); median collapses. Agent and CCR are a
**co-leading pair**.

### 3.5 Metric-dimension dissociation as a finding (R.5)

macro-AUROC and best-micro-F1 disagree across conditions:
- FedAvg @ RFMiD het04_dir: AUROC 0.65 (ok) vs F1 0.076 (collapsed).
- FedA3I @ ODIR het04: **top AUROC 0.801** vs F1 0.463 (< Agent/CCR 0.58).
- FedA3I / trimmed @ K=8: AUROC partial vs F1 collapsed.
- Recovery @ RFMiD het04_dir: visible on F1 (0.40) vs flat on AUROC.

**Take-away.** "Which aggregator is robust" is **metric-dependent**; single-metric
reporting yields contradictory rankings. We recommend mandatory dual-metric
evaluation for robust-FL.

### 3.6 Second noise model & FedNoRo `[pending]`

- **Asymmetric noise** (het04 IID + het04_dir × {fedavg,agent,ccr,feda3i} × 3
  seeds): tests whether the cross-dataset / failure-boundary / metric-split
  findings survive a clinically realistic noise model. `[running]`
- **FedNoRo** (het02/het04/het04_dir × 3 seeds): a two-stage quality-aware SOTA
  distinct from FedA3I (robust GCE loss on identified-noisy clients). Completes
  the noise-FL family coverage. `[running]`

---

## 4. Discussion

**Why do "robust" aggregators disappoint here?** Byzantine-robust rules assume a
bounded fraction of *arbitrary* corruption and IID-like honest updates; medical
FL violates both (label noise is structured, and non-IID heterogeneity makes
honest updates spread out, so coordinate-wise medians discard useful signal).
Quality-aware methods depend on a *separable* loss/quality signal, which is
dataset-dependent (RFMiD's extreme imbalance blurs the clean/noisy loss GMM,
whereas ODIR's is cleaner) — explaining FedA3I's rank inversion.

**Why does probe/confidence reweighting degrade more gracefully?** It is
reference-based (median-centred gate → reduces to FedAvg when clients tie) and
uses a held-out probe signal rather than trusting local losses, so it neither
over-concentrates (unlike CCR's softmax) nor discards heterogeneous-but-honest
updates (unlike median). It still inherits the F1 instability under extreme
non-IID (R.3).

**Limitations.** (1) Two datasets, one backbone; (2) K≤8; (3) F1 variance under
extreme imbalance limits usability conclusions — we mitigate via AUROC-primary
ranking; (4) the adaptive agent is *not* a novel mechanism (kin to FedOUI /
FedVG / FedA3I) — we position it as one member of the reweighting family, not a
contribution in itself.

---

## 5. Submission advice (venue + related work)

**Best-fit venues (empirical/benchmark framing):**
- **MICCAI DeCaF workshop** (Distributed/Collaborative & Federated Learning) —
  ideal fit for a medical-FL empirical study; asymmetric-noise + FedNoRo make it
  competitive here without a novel method.
- **MIDL** (short/full) — welcomes rigorous empirical medical-imaging studies.
- **MICCAI main conference** — feasible *only with* the second noise model
  (asymmetric) + FedNoRo + ideally a third dataset or K=16; borderline on
  novelty since the mechanism is not new. Treat as stretch.
- Journal fallback: *Medical Image Analysis* / *IEEE JBHI* special issues on FL.

**Related work to cite / position against:** FedAvg (McMahan'17), FedProx
(Li'20), Krum / coordinate-median / trimmed-mean (Blanchard'17, Yin'18), RHFL/CCR
(Fang & Ye, CVPR'22), FedA3I (Wu, AAAI'24), FedNoRo (Wu, IJCAI'23), RETFound
(Zhou, Nature'23). **Gap we fill:** none jointly evaluate these families under
*foundation-model + LoRA + multi-label medical + label noise + non-IID + scale*
with dual-metric, seed-matched protocol — and none report the cross-dataset rank
inversion or the two-dimensional median failure boundary.

**Is it competitive?** As a *method* paper: no (mechanism not novel). As an
*empirical study / benchmark with a crisp failure-boundary finding*: yes at
DeCaF/MIDL level, and the two-dimensional median boundary + cross-dataset FedA3I
inversion are genuinely publishable insights.

---

## 6. Remaining to close before submission

1. **R.2 anchors** `[[het04_fedavg]]` / `[[het04_agent]]` (het04 IID 50% AUROC)
   — read from main matrix `runs/paper_matrix/agent/`.
2. **FedNoRo** 3-seed on het02/het04/het04_dir → fill R.1/R.3.
3. **Asymmetric noise** 24 runs → new subsection confirming robustness of
   findings to noise model.
4. Figures: breakdown-point curve (R.2), cross-dataset bar (R.1), non-IID
   collapse/recovery (R.3), K=8 scaling (R.4) — via `make_agent_figures`.
