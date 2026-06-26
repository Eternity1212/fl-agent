# Paper Readiness Analysis

Date: 2026-06-26

This note summarizes whether the current result set is rich enough to support a paper, based on the cross-branch collection in this folder.

## Executive Judgment

The project is now strong enough for a coherent methods paper draft, especially around adaptive federated orchestration under heterogeneous label noise.

The evidence is not yet complete enough for a high-confidence final submission, because the agent matrix is still 16 / 23 runs and the current pure weighting agent fails in the non-IID + heterogeneous-noise setting. The next decisive evidence is whether adaptive per-client `mu` fixes that weakness, especially `het04_dir_agentmu_s0`.

Best current positioning:

- A solid workshop / early conference paper is already plausible.
- A stronger mid-tier conference or journal paper becomes plausible if the remaining agentmu, tau, and non-IID runs confirm the current story.
- A top-tier claim still needs more validation, likely including a second dataset or stronger held-out test/probe separation.

## Evidence Already Strong

### 1. Optimized RETFound baseline matrix is broad

The `results/retfound-optimized` branch contains a 29-run optimized RETFound matrix. The main three-seed comparison shows:

- Centralized macro-AUROC: `0.812984 +/- 0.002800`
- FedAvg macro-AUROC: `0.806720 +/- 0.020436`
- FedProx macro-AUROC: `0.765164 +/- 0.005030`
- Robust-FedProx macro-AUROC: `0.760703 +/- 0.008781`

This establishes an important baseline: in the optimized RETFound + LoRA setup, plain FedAvg is already very strong and fixed regularization can hurt. That makes the later agent result more meaningful, because the agent is not beating a weak baseline.

### 2. Clean control supports "no obvious downside"

The latest agent branch has clean seed-0 controls:

- `clean_fedavg_s0`: macro-AUROC `0.824948`
- `clean_agent_s0`: macro-AUROC `0.826474`

This supports the claim that the agent does not obviously damage clean training. For the paper, this is important because the method should not pay a clean-data tax just to be robust under noise.

Current limitation: this is only seed 0. It is good directional evidence, but the final paper should ideally include 3 clean seeds or clearly label this as a control.

### 3. Mild heterogeneous noise already shows a clear agent gain

For `het02` seed 0:

- FedAvg macro-AUROC: `0.721070`
- Robust-FedProx macro-AUROC: `0.730204`
- Agent macro-AUROC: `0.781214`

Agent improves over FedAvg by about `+0.0601` macro-AUROC. This is strong because it shows the method is not only useful under extreme noise; it also helps under milder heterogeneous client noise.

Current limitation: this is only seed 0, so it should be described as supportive rather than final multi-seed evidence.

### 4. Strong heterogeneous noise is the strongest current result

For `het04` IID heterogeneous noise over 3 seeds:

- FedAvg mean macro-AUROC: `0.737991`
- Robust-FedProx mean macro-AUROC: `0.668917`
- Agent mean macro-AUROC: `0.804335`

Agent improves over FedAvg by about `+0.0663` macro-AUROC and beats Robust-FedProx by a much larger margin. This is currently the paper's strongest experimental anchor.

This result supports the core narrative:

- Fixed robust regularization is not enough.
- Adaptive client weighting can identify and suppress harmful noisy clients.
- The benefit is large enough to be practically meaningful.

### 5. Mechanism evidence exists

The handoff notes report that in `het04_agent_s0/s1/s2`, the agent puts almost all weight on clean clients 0/1 and nearly zero weight on noisy clients 2/3, with clear probe-score separation.

This is valuable because it turns the paper from "metric went up" into an interpretable method story.

## Main Weaknesses

### 1. Non-IID + heterogeneous noise is currently negative

For `het04_dir` seed 0:

- FedAvg macro-AUROC: `0.587643`
- Pure agent weighting macro-AUROC: `0.532968`

This is the biggest scientific risk. The current pure weighting method can over-penalize clients whose data distribution differs, mistaking useful non-IID updates for bad noisy updates.

This does not kill the paper, but it changes the story:

- The agent is strong for heterogeneous label noise under IID splits.
- Non-IID requires a second lever, such as adaptive per-client `mu`, not just adaptive weighting.

The next key run is `het04_dir_agentmu_s0`.

### 2. The latest agent matrix is incomplete

Completed: 16 / 23.

Remaining:

- `het04_muonly_s0`
- `het04_agentmu_s0`
- `het04_agentmu_s1`
- `het04_agentmu_s2`
- `het04_dir_agentmu_s0`
- `het04_agent_tau002_s0`
- `het04_agent_tau005_s0`

The missing runs matter because they answer:

- Is adaptive `mu` independently useful?
- Does adaptive `mu` fix non-IID?
- Is the method sensitive to `tau`?
- Is the main `het04` result robust to a nearby hyperparameter choice?

### 3. Some controls are still single-seed

The strongest `het04` result is three-seed. But clean and `het02` are currently seed 0 only. For a final paper, reviewers may ask whether clean no-drop and mild-noise gain are stable.

Recommended minimum:

- Keep `het04` as the main multi-seed headline.
- Add 3-seed clean controls if compute allows.
- Add `het02` seeds 1/2 if compute allows, or clearly state `het02` is a supporting single-seed result.

### 4. The final evaluation/probe protocol must be presented carefully

The earlier docs mention a v0 caveat about probe/evaluation separation. The current handoff says held-out probe support exists, but the final paper must be explicit:

- What data is used for agent probing?
- What data is used for final evaluation?
- Are thresholds calibrated on validation and reported on test?

This is important because an adaptive agent can be accused of overfitting to the probe set if the protocol is unclear.

## Is It Enough For A Paper?

Yes, enough for a credible paper direction and likely enough for a first complete draft.

Not yet enough for the strongest final version.

The current result set is richer than a simple pilot:

- Real RFMiD pipeline exists.
- RETFound + LoRA optimized baseline matrix exists.
- 29-run baseline archive exists.
- Agent matrix has clean, mild-noise, strong-noise, robust baseline, and non-IID cases.
- Mechanism traces exist through probe scores and client weights.

But the final paper should not overclaim. The safest current claim is:

"Adaptive orchestration improves federated RETFound + LoRA training under heterogeneous client label noise, without obvious clean-data degradation, while fixed robust regularization can underperform. Non-IID remains challenging and motivates adaptive regularization."

The stronger claim becomes possible if `agentmu` fixes non-IID:

"A two-lever adaptive orchestrator, combining client weighting and per-client proximal strength, improves robustness across heterogeneous noise and non-IID shifts."

## Recommended Next Experiments

Priority 1:

- Run `het04_dir_agentmu_s0`.

If this improves over `het04_dir_fedavg_s0 = 0.587643`, it repairs the largest hole.

Priority 2:

- Run `het04_agentmu_s0/s1/s2`.

This shows whether adaptive `mu` helps or hurts in the main setting.

Priority 3:

- Run `het04_agent_tau002_s0` and `het04_agent_tau005_s0`.

This tests whether the agent result is too sensitive to `tau = 0.03`.

Priority 4:

- Add clean seeds 1/2 and `het02` seeds 1/2 if compute allows.

These are not as urgent as non-IID, but they strengthen the paper.

## Recommended Paper Structure

1. Problem: federated retinal multi-label learning with heterogeneous client label quality.
2. Baseline finding: optimized FedAvg is strong; fixed FedProx / Robust-FedProx can hurt.
3. Method: adaptive orchestration using probe telemetry to weight client updates.
4. Main result: `het04` 3-seed agent gain over FedAvg and Robust-FedProx.
5. Supporting result: clean control and `het02` mild-noise gain.
6. Mechanism: client weights suppress noisy clients and preserve clean clients.
7. Limitation: pure weighting fails under non-IID; adaptive `mu` is the next lever.
8. Final ablation: `agentmu` and `tau` sensitivity.

## Current Bottom Line

The experimental package is already meaningful and paper-shaped. The main story is strong, but the paper's final strength depends on the remaining seven agent runs, especially `het04_dir_agentmu_s0`.

