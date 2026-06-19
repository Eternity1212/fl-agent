from __future__ import annotations

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


def state_dict_nbytes(sd: OrderedDict[str, torch.Tensor]) -> int:
    return int(sum(int(v.numel()) * int(v.element_size()) for v in sd.values()))
