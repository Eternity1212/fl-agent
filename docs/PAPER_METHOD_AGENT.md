# Paper draft: Method + Results (Adaptive Federated Orchestration)

> 状态:Method 段不依赖具体数字,已可直接粘进论文。Results 段为模板,GPU 结果一到
> 按文末"填数指引"替换 `[[...]]` 占位符即可。语言用英文(论文用),关键处附中文注。

---

## 3. Method

### 3.1 Problem setup

We consider federated multi-label classification of retinal fundus images across
`K` clients (centers). Each client `k` holds a private dataset `D_k`. All clients
share a frozen retinal foundation model (RETFound, a ViT-Large MAE encoder) and
fine-tune only a LoRA adapter `θ` plus a linear multi-label head; the frozen
backbone is never transmitted. For a `C`-label task we minimize a class-balanced
binary cross-entropy

```
L_k(θ) = (1/|D_k|) Σ_(x,y)∈D_k Σ_{c=1}^{C} β_c · BCE(σ(f_θ(x)_c), y_c),
```

where `β_c` reweights positive/negative frequency per label. Only the adapter
parameters (≈0.39% of the model, ~1.2M params) are communicated each round,
giving a ~64× per-round communication reduction versus transmitting the full
model.

### 3.2 Heterogeneous label noise (threat model)

Real multi-center data differ in annotation quality. We model this as
*heterogeneous* client-level label noise: a subset `N ⊂ {1..K}` of "noisy"
clients have their training labels corrupted by symmetric flipping with
probability `p` (each binary label independently flipped w.p. `p`), while the
remaining clients are clean. Crucially, the server does **not** know `N`. Static
federated methods (FedAvg, FedProx with a fixed `μ`, robust regularization
applied uniformly) cannot exploit this structure: they treat every client
identically and therefore either ignore the noise (FedAvg) or over-regularize the
clean clients too (uniform robust), which we show costs accuracy even when data
are clean.

### 3.3 Adaptive federated orchestration

We introduce a lightweight server-side **orchestration agent** that, each round,
observes per-client telemetry and adapts two levers: (i) the aggregation weights
and (ii) the per-client proximal strength. The agent reduces to FedAvg when all
clients look equally good (zero cost on clean data) and selectively suppresses /
constrains clients that look corrupted.

**Telemetry (probe score).** The server holds a small shared *held-out* probe set
`P` (disjoint from the test set, so there is no leakage into reported metrics).
After client `k` returns its locally updated adapter `θ_k^{(t)}` in round `t`, the
server forms the global-backbone-plus-`θ_k` model and scores it on `P` using the
negative validation BCE:

```
s_k^{(t)} = − (1/|P|) Σ_(x,y)∈P Σ_c BCE(σ(f_{θ_k}(x)_c), y_c).
```

We use negative BCE rather than AUROC because it is markedly more sensitive to the
calibration damage caused by label-noise training, so clean clients score clearly
above noisy ones.

**Adaptive aggregation weights.** Let `m^{(t)} = median_k s_k^{(t)}`. The agent
gates each client by how far it sits above/below the round's median:

```
g_k^{(t)} = σ( (s_k^{(t)} − m^{(t)}) / τ ),   w_k^{(t)} = n_k · g_k^{(t)},
```

where `n_k = |D_k|`, `σ` is the logistic function, and `τ` controls gate
hardness. Clients above the median keep ~full size weight; clients below are
smoothly down-weighted. Aggregation is the standard weighted average
`θ^{(t+1)} = Σ_k w_k^{(t)} θ_k^{(t)} / Σ_k w_k^{(t)}`. When scores are
near-identical (clean, homogeneous), all gates ≈ 0.5 and the rule reduces to
size-weighted FedAvg.

**Adaptive per-client proximal strength.** Our FedProx ablation shows that a
*fixed* `μ` is the failure mode: small `μ` ≈ FedAvg while larger `μ` collapses
accuracy. We therefore set `μ` *per client and per round* from the previous
round's probe scores. Using `m^{(t-1)} = median_k s_k^{(t-1)}`,

```
μ_k^{(t)} = μ_max · clip( (m^{(t-1)} − s_k^{(t-1)}) / τ_μ , 0, 1 ),
```

and client `k` optimizes `L_k(θ) + (μ_k^{(t)}/2) ‖θ − θ^{(t)}‖²`. A client at or
above the pack gets `μ_k ≈ 0` (trains freely; clean case reduces to FedAvg); a
client clearly below the pack receives a strong pull toward the global model,
limiting how far its corrupted update can drift.

**Relationship of the two levers (design rationale).** The levers are not simply
additive. When the aggregation weights already drive a noisy client's weight
toward zero, additionally constraining it with `μ` is largely redundant. The
per-client `μ` is intended for settings where a client *cannot* be discarded —
non-IID clients holding label coverage unavailable elsewhere — so it must be
constrained rather than down-weighted. We therefore treat adaptive weighting as
the primary mechanism and adaptive `μ` as a conditional component evaluated in
ablation (Sec. 5.x).

```
Algorithm 1: Adaptive Federated Orchestration (one round t)
  input: global adapter θ^(t), prev scores {s_k^(t-1)}
  for each client k in parallel:
      μ_k ← μ_max · clip((median s^(t-1) − s_k^(t-1))/τ_μ, 0, 1)        # if adaptive-μ
      θ_k ← LocalTrain(θ^(t), D_k; proximal μ_k)
  for each client k:
      s_k ← −BCE_P(global-backbone + θ_k)                              # probe telemetry
  g_k ← σ((s_k − median s)/τ);  w_k ← n_k · g_k                        # adaptive weights
  θ^(t+1) ← Σ_k w_k θ_k / Σ_k w_k                                      # weighted aggregation
```

The agent adds one probe forward-pass per client per round and no extra
communication; the backbone stays frozen and only adapters move.

---

## 5. Results

> 状态:het04 IID 主结果已坐实(GPU, 3 seed)。clean 对照与 non-IID/μ 消融待 full
> matrix 跑完填入(标 [[..]])。

### 5.x Adaptive orchestration under heterogeneous label noise

**Setup.** `K=4` clients, two designated noisy clients (`N={2,3}`) with flip rate
`p=0.4`, IID and Dirichlet(α=0.1) splits, seeds {0,1,2}. RETFound+LoRA (rank 8),
40 rounds. The agent probes on the validation split and we report on the held-out
test split (no leakage). `τ=0.03`.

**Mechanism works (Figure W).** Figure W (`fig_agent_weights.png`) plots the
aggregation weight fraction per client over rounds. The agent progressively
suppresses the two noisy clients (red, dashed) toward zero weight while promoting
the clean clients (blue), confirming it identifies corruption from probe telemetry
alone — without being told which clients are noisy. Quantitatively, by the final
round the mean aggregation weight is ≈0.50/0.50 for the two clean clients and
≈0.00/0.00 for the two noisy clients, driven by a clean separation in probe
scores (clean ≈ −0.11/−0.13 vs noisy ≈ −2.57/−2.41 negative-BCE in a
representative seed).

**Accuracy (Table A / Figure B).** Macro-AUROC, mean±std over 3 seeds, RFMiD,
heterogeneous label noise p=0.4, IID:

| method | macro-AUROC | per-seed |
|---|---|---|
| FedAvg | 0.738 ± 0.001 | 0.7374 / 0.7377 / 0.7389 |
| Robust-FedProx (μ=0.05, dropout=0.1) | 0.669 ± 0.005 | 0.6669 / 0.6762 / 0.6637 |
| **Agent (ours)** | **0.804 ± 0.010** | 0.7908 / 0.8102 / 0.8120 |

Under heterogeneous noise the agent improves macro-AUROC by **+0.066** over FedAvg
and **+0.135** over static Robust-FedProx, consistently across all 3 seeds (every
agent seed exceeds every FedAvg seed). The agent recovers ≈76% of the gap between
the noisy FedAvg baseline (0.738) and the clean ceiling (clean FedAvg 0.825,
seed 0), i.e. it nearly neutralizes the damage of 40% label flipping on half the
clients.

**Static robustness hurts here.** Notably, Robust-FedProx is *worse* than plain
FedAvg (0.669 vs 0.738). Applying a fixed proximal term and dropout uniformly
over-constrains the clean clients while failing to isolate the noisy ones — direct
evidence that, for foundation-model PEFT under heterogeneous noise, the right
intervention is *per-client adaptivity*, not stronger static regularization. This
motivates the agent's selective, telemetry-driven design.

**Clean control.** On clean data the agent matches FedAvg within `[[±Δ]]` AUROC
(clean FedAvg s0 = 0.825; clean agent = [[..]]), i.e. it pays no cost when
adaptation is unnecessary — a property static robust methods lack. [待 clean_agent]

**Non-IID (honest caveat / ablation).** Under Dirichlet(α=0.1) + noise, plain
adaptive *weighting* alone does not beat FedAvg (`het04_dir`: FedAvg = 0.533,
agent = [[..]]): when noisy clients also hold label coverage unavailable
elsewhere, down-weighting them discards needed signal. This is the regime the
adaptive per-client `μ` targets (constrain rather than discard); we evaluate
`het04_dir_agentmu` to test whether it recovers this case. [待 dir_agentmu]

**Ablations (Table C).** (i) `τ` gate hardness (`het04_agent_tau002/005`);
(ii) adaptive-`μ` on/off (`agentmu`) and `μ`-only (`muonly`) to isolate each
lever; (iii) milder noise p=0.2 (`het02_*`).

---

## 6. Discussion

**Static regularization is the wrong tool under heterogeneous noise.** A central,
perhaps counter-intuitive, finding is that the static robust baseline
(Robust-FedProx: fixed proximal `μ=0.05` + positive-dropout, uniformly applied) is
*worse* than plain FedAvg under heterogeneous label noise (0.669 vs 0.738
macro-AUROC, Table A). Two mechanisms explain this. First, the regularization is
applied *indiscriminately*: it constrains the clean clients just as much as the
noisy ones, degrading the very updates we want to keep. Second, a *fixed* strength
cannot match the per-client, per-round severity of corruption — our FedProx `μ`
sweep shows small `μ` ≈ FedAvg while larger `μ` collapses accuracy, so no single
constant is right for all clients simultaneously. This is direct evidence that the
problem is not "insufficient robustness" but "non-adaptive robustness." The agent
resolves both issues: it acts *selectively* (only the clients whose probe scores
fall below the round median are down-weighted/constrained) and *adaptively* (the
intervention strength tracks the telemetry each round), which is precisely why it
turns the −0.069 regression of static robustness into a +0.066 improvement.

**Interpretability as a deployment asset.** Because every decision is grounded in a
scalar probe score per client per round, the agent's behavior is auditable: one
can plot exactly which center was down-weighted and why (Figure W). In a
multi-center clinical setting this transparency is valuable — a silently
mis-weighted site is a liability, whereas a logged, explainable weighting decision
can be reviewed by data-governance stakeholders.

**Two levers, different regimes.** Adaptive weighting and adaptive `μ` are
complementary rather than additive (Sec. 3.3): when a noisy client can be safely
down-weighted (IID), weighting suffices and `μ` is redundant; when it cannot be
discarded without losing unique label coverage (non-IID), constraint via `μ`
becomes the relevant lever. Our non-IID ablation isolates this boundary.

---

## 填数指引 (中文)

GPU 上 stage1/full 跑完后:

1. 出图 + 表(已接进 `run_agent.sh`,也可手动):
   ```
   python -m fed_agent.tools.make_agent_figures \
     --results_dir runs/paper_matrix/agent_stage1 --out_dir docs/figures/stage1
   python -m fed_agent.tools.summarize_agent runs/paper_matrix/agent_stage1/summary.json
   ```
2. `docs/figures/stage1/agent_figures_summary.md` 里的 mean±std 直接替换 Table A 的 `[[..]]`。
3. `fig_agent_weights.png` → Figure W;`fig_agent_auroc.png` → Figure B。
4. `[[+Δ]]` 用 `summarize_agent` 打印的 `delta vs fedavg`。
5. 全部 3 seed 一致 WINS 才把 "consistent across 3 seeds" 保留,否则如实改写。
