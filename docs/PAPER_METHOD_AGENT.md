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

## 5. Results (模板, 待填数)

### 5.x Adaptive orchestration under heterogeneous label noise

**Setup.** `K=4` clients, two designated noisy clients (`N={2,3}`) with flip rate
`p∈{0.2,0.4}`, IID and Dirichlet(α=0.1) splits, seeds {0,1,2}. RETFound+LoRA
(rank 8), 40 rounds. The agent probes on the validation split and we report on the
held-out test split (no leakage).

**Mechanism works (Figure W).** Figure W (`fig_agent_weights.png`) plots the
aggregation weight fraction per client over rounds. The agent progressively
suppresses the two noisy clients (red, dashed) below the uniform `1/K` line while
promoting the clean clients (blue), confirming it identifies corruption from
probe telemetry alone — without being told which clients are noisy.

**Accuracy (Table A / Figure B).**

| condition | method | macro-AUROC | micro-F1@0.5 |
|---|---|---|---|
| het noise p=0.4 (IID) | FedAvg | [[0.7375±0.000]] | [[0.11±..]] |
| het noise p=0.4 (IID) | Robust-FedProx | [[..]] | [[..]] |
| het noise p=0.4 (IID) | **Agent (ours)** | **[[..]]** | **[[..]]** |
| het noise p=0.4 (non-IID) | FedAvg | [[..]] | [[..]] |
| het noise p=0.4 (non-IID) | **Agent (ours)** | **[[..]]** | **[[..]]** |

Under heterogeneous noise the agent improves macro-AUROC by `[[+Δ]]` over FedAvg
and `[[+Δ]]` over static Robust-FedProx (consistent across 3 seeds), and recovers
the default-threshold micro-F1 that collapses for FedAvg due to noise-induced
miscalibration.

**Clean control.** On clean data the agent matches FedAvg within `[[±Δ]]`
AUROC, i.e. it pays no cost when adaptation is unnecessary — a property static
robust methods lack.

**Ablations (Table C).** (i) `τ` gate hardness; (ii) adaptive-`μ` on/off
(`agentmu`) and `μ`-only (`muonly`) to isolate each lever; (iii) the non-IID
condition where down-weighting alone is insufficient.

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
