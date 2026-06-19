from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

NOISE_PROTOCOL_VERSION = "noise_protocol@v1"


@dataclass(frozen=True)
class NoiseProtocolV1:
    """Validated subset of ``noise_protocol@v1`` used by training / FL smoke."""

    symmetric_flip_p_flip: float


def parse_noise_protocol_v1(payload: dict[str, Any]) -> NoiseProtocolV1:
    if payload.get("version") != NOISE_PROTOCOL_VERSION:
        raise ValueError(
            f"noise protocol: expected version {NOISE_PROTOCOL_VERSION!r}, "
            f"got {payload.get('version')!r}",
        )
    sym = payload.get("symmetric_flip_on_positives")
    if sym is None:
        sym = {}
    if not isinstance(sym, dict):
        raise ValueError("symmetric_flip_on_positives must be a mapping or omitted")
    p = float(sym.get("p_flip", 0.0))
    if p < 0 or p > 1:
        raise ValueError("p_flip must be in [0, 1]")
    return NoiseProtocolV1(symmetric_flip_p_flip=p)


def load_noise_protocol_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("noise protocol YAML must be a mapping")
    return payload


def apply_symmetric_label_noise(
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    p_flip: float,
) -> np.ndarray:
    """Independently flip each *positive* label bit with probability ``p_flip``.

    Notes
    -----
    This is a simplified stand-in for more realistic asymmetric / class-conditional
    noise used in later milestones.
    """

    if y.ndim != 2:
        raise ValueError("y must be 2D float/bool matrix [N, K]")
    p = float(p_flip)
    if p < 0 or p > 1:
        raise ValueError("p_flip must be in [0, 1]")

    y2 = y.astype(np.float32, copy=True)
    n, k = y2.shape
    for i in range(n):
        for j in range(k):
            if y2[i, j] <= 0.5:
                continue
            if rng.random() < p:
                y2[i, j] = 0.0
            else:
                y2[i, j] = 1.0
    return y2
