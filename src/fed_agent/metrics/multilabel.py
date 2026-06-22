"""Multi-label metrics with validation-threshold calibration."""

from __future__ import annotations

from typing import Any

import numpy as np


def _binarize(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(y_prob) >= float(threshold)).astype(np.int64)


def _f1_from_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = (np.asarray(y_true) > 0.5).astype(np.int64)
    yp = np.asarray(y_pred).astype(np.int64)
    tp = int((yt * yp).sum())
    fp = int(((1 - yt) * yp).sum())
    fn = int((yt * (1 - yp)).sum())
    micro = float((2 * tp) / max((2 * tp + fp + fn), 1))

    per_label: list[float] = []
    for j in range(int(yt.shape[1])):
        yt_j = yt[:, j]
        if int(yt_j.sum()) == 0:
            continue
        yp_j = yp[:, j]
        tp_j = int((yt_j * yp_j).sum())
        fp_j = int(((1 - yt_j) * yp_j).sum())
        fn_j = int((yt_j * (1 - yp_j)).sum())
        per_label.append(float((2 * tp_j) / max((2 * tp_j + fp_j + fn_j), 1)))
    macro_present = float(sum(per_label) / len(per_label)) if per_label else 0.0
    return {"micro_f1": micro, "macro_f1_present": macro_present}


def calibrate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    grid: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Find the scalar threshold maximizing macro-F1 over labels present in ``y_true``."""

    grid = grid or tuple(i / 100.0 for i in range(5, 96, 5))
    best = {"threshold": 0.5, "macro_f1_present": -1.0, "micro_f1": 0.0}
    for t in grid:
        scores = _f1_from_binary(y_true, _binarize(y_prob, float(t)))
        if scores["macro_f1_present"] > best["macro_f1_present"]:
            best = {
                "threshold": float(t),
                "macro_f1_present": scores["macro_f1_present"],
                "micro_f1": scores["micro_f1"],
            }
    return best


def _ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:  # pragma: no cover - optional dependency
        return {"macro_ap": None, "macro_auroc": None}

    yt = (np.asarray(y_true) > 0.5).astype(np.int64)
    yp = np.asarray(y_prob, dtype=np.float32)
    ap_values: list[float] = []
    auc_values: list[float] = []
    for j in range(int(yt.shape[1])):
        if len(set(yt[:, j].tolist())) < 2:
            continue
        ap_values.append(float(average_precision_score(yt[:, j], yp[:, j])))
        auc_values.append(float(roc_auc_score(yt[:, j], yp[:, j])))
    return {
        "macro_ap": float(sum(ap_values) / len(ap_values)) if ap_values else None,
        "macro_auroc": float(sum(auc_values) / len(auc_values)) if auc_values else None,
    }


def multilabel_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    calibrate: bool = True,
) -> dict[str, Any]:
    """Return F1 and optional ranking metrics for multi-label classification."""

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fixed = _f1_from_binary(y_true, _binarize(y_prob, float(threshold)))
    out: dict[str, Any] = {
        "threshold": float(threshold),
        "micro_f1": fixed["micro_f1"],
        "macro_f1_present": fixed["macro_f1_present"],
    }
    if calibrate:
        best = calibrate_threshold(y_true, y_prob)
        out.update(
            {
                "best_threshold": best["threshold"],
                "best_micro_f1": best["micro_f1"],
                "best_macro_f1_present": best["macro_f1_present"],
            },
        )
    out.update(_ranking_metrics(y_true, y_prob))
    return out
