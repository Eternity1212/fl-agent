"""Telemetry-driven adaptive aggregation policy (Strategy A).

The orchestrator turns per-client telemetry collected each federated round into
aggregation weights. Design goals:

  * **Clean -> FedAvg**: when all clients look equally good, weights reduce to the
    size-proportional FedAvg weights (no penalty), which static robust methods
    fail to do.
  * **Noisy/heterogeneous -> selective**: clients with low probe validation score
    and/or outlier updates are down-weighted, protecting the global model.

Signals supported:
  * ``probe_scores`` (higher = better, e.g. macro-AUROC on a shared probe set)
  * ``update_vectors`` (optional) trainable-param deltas; low cosine to the
    consensus direction marks an outlier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentDecision:
    weights: list[float]
    probe_component: list[float]
    geometry_component: list[float]


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _sigmoid_gate(scores: list[float], tau: float) -> list[float]:
    """Suppress outlier-low clients while keeping the good ones balanced.

    A sigmoid centred on the *median* probe score means clients at or above the
    pack keep a gate near 1 (so equally-good clients stay balanced and the model
    does not collapse onto a single client), whereas clearly-worse clients
    (e.g. label-noise corrupted) are gated toward 0. ``tau`` is the soft margin
    in probe-score units.
    """
    if tau <= 0:
        return [1.0 for _ in scores]
    center = _median(scores)
    return [1.0 / (1.0 + math.exp(-(s - center) / tau)) for s in scores]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def decide_client_mu(
    *,
    probe_scores: list[float],
    mu_max: float,
    tau: float,
) -> list[float]:
    """Per-client proximal strength from telemetry (joint adaptive orchestration).

    Motivated by the empirical finding that a *fixed* FedProx mu is the failure
    mode (small mu ~ FedAvg, large mu collapses). Here mu is set per client and
    per round from the previous round's probe scores:

        mu_i = mu_max * clamp((median_s - s_i) / tau, 0, 1)

    A client at or above the pack (s_i >= median) gets mu_i = 0 -> trains freely
    (clean case reduces to FedAvg, which is best). A client clearly below the
    pack (noisy) gets a strong proximal pull toward the global model, limiting
    how far its corrupted update can drift.
    """
    n = len(probe_scores)
    if n == 0:
        return []
    if tau <= 0:
        return [0.0] * n
    center = _median(probe_scores)
    out = []
    for s in probe_scores:
        frac = (center - s) / tau
        frac = max(0.0, min(1.0, frac))
        out.append(float(mu_max) * frac)
    return out


def decide_weights(
    *,
    probe_scores: list[float],
    sizes: list[float],
    tau: float = 0.05,
    update_vectors: list[list[float]] | None = None,
    geometry_floor: float = 0.0,
) -> AgentDecision:
    """Return adaptive aggregation weights from per-client telemetry.

    ``w_i  ∝  size_i * gate(probe_i) * geometry_i``

    With identical probe scores and no geometry signal, every gate is 0.5 and
    this collapses to size-proportional FedAvg.
    """
    n = len(probe_scores)
    if n == 0:
        return AgentDecision([], [], [])
    if len(sizes) != n:
        raise ValueError("sizes length must match probe_scores")

    probe_w = _sigmoid_gate(probe_scores, tau)

    geom_w = [1.0] * n
    if update_vectors is not None and n > 1:
        dim = len(update_vectors[0])
        consensus = [
            sum(uv[d] for uv in update_vectors) / n for d in range(dim)
        ]
        for i, uv in enumerate(update_vectors):
            c = _cosine(uv, consensus)
            # map cosine in [-1,1] -> [geometry_floor, 1]
            geom_w[i] = geometry_floor + (1.0 - geometry_floor) * max(0.0, c)

    raw = [sizes[i] * probe_w[i] * geom_w[i] for i in range(n)]
    z = sum(raw)
    if z <= 0:
        # fall back to size-proportional
        zs = sum(sizes) or 1.0
        raw = [s / zs for s in sizes]
    else:
        raw = [r / z for r in raw]
    return AgentDecision(weights=raw, probe_component=probe_w, geometry_component=geom_w)
