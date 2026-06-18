from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = "fl_agent.split.v1"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _primary_labels(y: np.ndarray) -> np.ndarray:
    """Return an integer primary label per row (first argmax). All-zero rows -> -1."""

    if y.ndim != 2:
        raise ValueError("y must be 2D")
    prim = np.argmax(y, axis=1).astype(np.int64)
    row_sum = y.sum(axis=1)
    prim[row_sum <= 0] = -1
    return prim


def build_iid_split(image_ids: Iterable[str], n_clients: int, seed: int) -> dict[str, list[str]]:
    ids = list(image_ids)
    rng = _rng(seed)
    rng.shuffle(ids)
    buckets = [[] for _ in range(int(n_clients))]
    for i, sid in enumerate(ids):
        buckets[i % n_clients].append(sid)
    return {str(k): v for k, v in enumerate(buckets)}


def build_dirichlet_split_primary(
    image_ids: tuple[str, ...],
    y: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int,
) -> dict[str, list[str]]:
    """Label-skew split: for each primary class, partition its samples across clients.

    Each sample is assigned exactly once (by its primary argmax class; ties use first index).
    """

    if len(image_ids) != int(y.shape[0]):
        raise ValueError("image_ids length must match y.shape[0]")
    n_clients = int(n_clients)
    if n_clients <= 0:
        raise ValueError("n_clients must be positive")

    rng = _rng(seed)
    prim = _primary_labels(y)

    client_ids: list[list[str]] = [[] for _ in range(n_clients)]

    n_classes = int(y.shape[1])
    for c in range(n_classes):
        idx = np.where(prim == c)[0]
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        props = rng.dirichlet(np.repeat(float(alpha), n_clients))
        props = props / props.sum()
        counts = rng.multinomial(int(idx.size), props)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        for k in range(n_clients):
            part = idx[int(offsets[k]) : int(offsets[k + 1])]
            for j in part.tolist():
                client_ids[k].append(image_ids[int(j)])

    assigned = {sid for bucket in client_ids for sid in bucket}
    leftover = [sid for sid in image_ids if sid not in assigned]
    if leftover:
        for j, sid in enumerate(leftover):
            client_ids[j % n_clients].append(sid)

    return {str(k): sorted(set(bucket)) for k, bucket in enumerate(client_ids)}


def build_domain_hash_split(
    image_ids: Iterable[str],
    n_clients: int,
    seed: int,
) -> dict[str, list[str]]:
    """Synthetic domain split: deterministic hash bucket per image_id."""

    n_clients = int(n_clients)
    if n_clients <= 0:
        raise ValueError("n_clients must be positive")

    buckets: list[list[str]] = [[] for _ in range(n_clients)]
    for sid in image_ids:
        h = hashlib.sha256(f"{seed}:{sid}".encode("utf-8")).digest()
        k = int.from_bytes(h[:4], "little", signed=False) % n_clients
        buckets[k].append(sid)
    return {str(k): sorted(set(bucket)) for k, bucket in enumerate(buckets)}


def split_payload(
    *,
    split: str,
    n_clients: int,
    seed: int,
    alpha: float | None,
    label_names: tuple[str, ...],
    clients: dict[str, list[str]],
    y_lookup: dict[str, np.ndarray],
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for ck, ids in clients.items():
        if not ids:
            stats[ck] = {"n": 0, "label_mean": [0.0] * len(label_names)}
            continue
        m = np.stack([y_lookup[s] for s in ids], axis=0).mean(axis=0)
        stats[ck] = {"n": len(ids), "label_mean": m.astype(float).tolist()}

    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "split": split,
        "n_clients": int(n_clients),
        "seed": int(seed),
        "label_names": list(label_names),
        "clients": clients,
        "stats": stats,
    }
    if alpha is not None:
        payload["alpha"] = float(alpha)
    return payload


def write_split_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_split_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
