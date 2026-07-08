from __future__ import annotations

import math
from collections import OrderedDict
from typing import Iterable

import torch


def fedavg_state_dict(
    state_dicts: Iterable[OrderedDict[str, torch.Tensor]],
    weights: list[float],
) -> OrderedDict[str, torch.Tensor]:
    """Weighted average of PyTorch state_dicts (FedAvg)."""

    sd_list = list(state_dicts)
    if not sd_list:
        raise ValueError("state_dicts is empty")
    if len(sd_list) != len(weights):
        raise ValueError("weights length must match state_dicts")
    wsum = float(sum(weights))
    if wsum <= 0:
        raise ValueError("sum(weights) must be positive")

    keys = sd_list[0].keys()
    for sd in sd_list[1:]:
        if sd.keys() != keys:
            raise ValueError("All state_dicts must have identical keys")

    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k in keys:
        base = sd_list[0][k]
        acc = torch.zeros_like(base, dtype=torch.float32)
        for sd, w in zip(sd_list, weights):
            acc = acc + sd[k].detach().float() * (float(w) / wsum)
        out[k] = acc.to(dtype=base.dtype)
    return out


def coordinate_median_state_dict(
    state_dicts: Iterable[OrderedDict[str, torch.Tensor]],
) -> OrderedDict[str, torch.Tensor]:
    """Coordinate-wise median aggregation (Yin et al., ICML'18).

    A classic Byzantine-robust aggregation baseline: for every parameter
    coordinate, take the median across clients instead of the (weighted) mean.
    This is *not* expressible as per-client scalar weights, so it needs its own
    aggregator rather than going through :func:`fedavg_state_dict`.
    """

    sd_list = list(state_dicts)
    if not sd_list:
        raise ValueError("state_dicts is empty")
    keys = sd_list[0].keys()
    for sd in sd_list[1:]:
        if sd.keys() != keys:
            raise ValueError("All state_dicts must have identical keys")

    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k in keys:
        base = sd_list[0][k]
        stacked = torch.stack([sd[k].detach().float() for sd in sd_list], dim=0)
        med = torch.median(stacked, dim=0).values
        out[k] = med.to(dtype=base.dtype)
    return out


def trimmed_mean_state_dict(
    state_dicts: Iterable[OrderedDict[str, torch.Tensor]],
    trim_ratio: float = 0.1,
) -> OrderedDict[str, torch.Tensor]:
    """Coordinate-wise trimmed-mean aggregation (Yin et al., ICML'18).

    For each coordinate, drop the ``floor(trim_ratio * n)`` largest and smallest
    client values, then average the rest. ``trim_ratio=0`` reduces to a plain
    (unweighted) mean; the trim is skipped automatically when it would remove
    every client.
    """

    sd_list = list(state_dicts)
    n = len(sd_list)
    if n == 0:
        raise ValueError("state_dicts is empty")
    keys = sd_list[0].keys()
    for sd in sd_list[1:]:
        if sd.keys() != keys:
            raise ValueError("All state_dicts must have identical keys")

    k_trim = int(math.floor(float(trim_ratio) * n))
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in keys:
        base = sd_list[0][key]
        stacked = torch.stack([sd[key].detach().float() for sd in sd_list], dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        if k_trim > 0 and (n - 2 * k_trim) >= 1:
            sorted_vals = sorted_vals[k_trim : n - k_trim]
        out[key] = sorted_vals.mean(dim=0).to(dtype=base.dtype)
    return out


def state_dict_nbytes(sd: OrderedDict[str, torch.Tensor]) -> int:
    return int(sum(int(v.numel()) * int(v.element_size()) for v in sd.values()))
