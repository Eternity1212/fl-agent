"""Dependency-light synthetic FL ablations (NumPy only, no Torch download needed)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NumpyExperimentSpec:
    """One row in the NumPy synthetic ablation matrix."""

    name: str
    fedprox_mu: float
    noise_p_flip: float = 0.0
    seed: int = 0
    rounds: int = 8
    local_epochs: int = 4
    lr: float = 0.4


def default_numpy_specs() -> list[NumpyExperimentSpec]:
    """FedAvg/FedProx and label-noise ablations for the synthetic fixture."""

    return [
        NumpyExperimentSpec(name="fedavg_clean", fedprox_mu=0.0, noise_p_flip=0.0),
        NumpyExperimentSpec(name="fedprox_mu005_clean", fedprox_mu=0.05, noise_p_flip=0.0),
        NumpyExperimentSpec(name="fedavg_noise_p01", fedprox_mu=0.0, noise_p_flip=0.1),
        NumpyExperimentSpec(name="fedprox_mu005_noise_p01", fedprox_mu=0.05, noise_p_flip=0.1),
        NumpyExperimentSpec(name="fedavg_noise_p05", fedprox_mu=0.0, noise_p_flip=0.5),
    ]


def _fixture() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    # Two separable labels, split over two clients with mild feature heterogeneity.
    x = np.array(
        [
            [1.0, 0.1, 0.0],
            [0.9, 0.2, 0.1],
            [0.0, 1.0, 0.2],
            [0.1, 0.8, 0.3],
            [0.8, 0.0, 0.4],
            [0.2, 0.7, 0.5],
        ],
        dtype=np.float32,
    )
    y = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    clients = {
        "0": np.array([0, 1, 4], dtype=np.int64),
        "1": np.array([2, 3, 5], dtype=np.int64),
    }
    return x, y, clients


def _apply_positive_flip(y: np.ndarray, *, p_flip: float, seed: int) -> np.ndarray:
    if p_flip <= 0.0:
        return y.astype(np.float32, copy=True)
    rng = np.random.default_rng(int(seed))
    out = y.astype(np.float32, copy=True)
    pos = np.argwhere(out > 0.5)
    if len(pos) == 0:
        return out
    n_flip = int(round(float(p_flip) * float(len(pos))))
    if n_flip <= 0:
        return out
    order = rng.permutation(len(pos))[: min(n_flip, len(pos))]
    for i in order:
        r, c = pos[int(i)]
        out[int(r), int(c)] = 0.0
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _bce_loss(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> float:
    p = _sigmoid(x @ w + b)
    eps = 1e-7
    return float(-(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)).mean())


def _local_train(
    x: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    *,
    w_global: np.ndarray,
    b_global: np.ndarray,
    spec: NumpyExperimentSpec,
) -> tuple[np.ndarray, np.ndarray, float]:
    w = w_global.copy()
    b = b_global.copy()
    x_c = x[idx]
    y_c = y[idx]
    n = float(len(idx))

    for _ in range(int(spec.local_epochs)):
        p = _sigmoid(x_c @ w + b)
        err = (p - y_c) / n
        grad_w = x_c.T @ err
        grad_b = err.sum(axis=0)
        if spec.fedprox_mu > 0.0:
            grad_w = grad_w + float(spec.fedprox_mu) * (w - w_global)
            grad_b = grad_b + float(spec.fedprox_mu) * (b - b_global)
        w -= float(spec.lr) * grad_w
        b -= float(spec.lr) * grad_b

    return w, b, _bce_loss(x_c, y_c, w, b)


def run_numpy_synthetic_one(spec: NumpyExperimentSpec) -> dict[str, Any]:
    """Run one deterministic NumPy FL experiment and return fed-smoke-like metrics."""

    x, y_clean, clients = _fixture()
    y = _apply_positive_flip(y_clean, p_flip=spec.noise_p_flip, seed=spec.seed)

    rng = np.random.default_rng(int(spec.seed))
    w = rng.normal(loc=0.0, scale=0.05, size=(x.shape[1], y.shape[1])).astype(np.float32)
    b = np.zeros((y.shape[1],), dtype=np.float32)

    upload_bytes_one_client = int(w.nbytes + b.nbytes)
    losses: list[float] = []
    clean_eval_losses: list[float] = []
    comm: list[int] = []

    for _round in range(int(spec.rounds)):
        local_ws: list[np.ndarray] = []
        local_bs: list[np.ndarray] = []
        weights: list[float] = []
        round_losses: list[float] = []
        for idx in clients.values():
            lw, lb, loss = _local_train(x, y, idx, w_global=w, b_global=b, spec=spec)
            local_ws.append(lw)
            local_bs.append(lb)
            weights.append(float(len(idx)))
            round_losses.append(loss)
        alpha = np.asarray(weights, dtype=np.float32) / float(sum(weights))
        w = sum(a * lw for a, lw in zip(alpha, local_ws))
        b = sum(a * lb for a, lb in zip(alpha, local_bs))
        losses.append(float(sum(round_losses) / len(round_losses)))
        clean_eval_losses.append(_bce_loss(x, y_clean, w, b))
        comm.append(upload_bytes_one_client * len(clients))

    metrics: dict[str, Any] = {
        "rounds": list(range(int(spec.rounds))),
        "comm_bytes_upload_per_round": comm,
        "mean_train_loss_clients": losses,
        "clean_eval_loss": clean_eval_losses,
        "final_state_dict_keys": ["linear.weight", "linear.bias"],
        "total_upload_bytes": int(sum(comm)),
        "backend": "numpy_logistic_synthetic",
    }
    if spec.noise_p_flip > 0.0:
        metrics["noise_protocol"] = {
            "path": "(inline numpy synthetic)",
            "symmetric_flip_p_flip": float(spec.noise_p_flip),
        }
    return metrics


def run_numpy_synthetic_suite(
    *,
    out_dir: Path,
    specs: list[NumpyExperimentSpec] | None = None,
) -> dict[str, Any]:
    """Run all NumPy synthetic ablations and write per-run JSON plus ``summary.json``."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = specs or default_numpy_specs()

    runs_out: list[dict[str, Any]] = []
    for spec in specs:
        metrics = run_numpy_synthetic_one(spec)
        payload = {
            "spec": {
                "name": spec.name,
                "fedprox_mu": spec.fedprox_mu,
                "noise_protocol_yaml": None,
                "label_noise_seed": spec.seed,
                "rounds": spec.rounds,
                "lr": spec.lr,
                "local_epochs": spec.local_epochs,
                "batch_size": 0,
                "image_size": [],
            },
            "metrics": metrics,
        }
        runs_out.append(payload)
        (out / f"{spec.name}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "numpy_logistic_synthetic",
        "out_dir": str(out.resolve()),
        "n_runs": len(runs_out),
        "runs": runs_out,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
