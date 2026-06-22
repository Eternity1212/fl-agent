from __future__ import annotations

import numpy as np

from fed_agent.metrics.multilabel import calibrate_threshold, multilabel_classification_metrics


def test_calibrate_threshold_prefers_better_macro_f1() -> None:
    y = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    p = np.array([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3]], dtype=np.float32)
    best = calibrate_threshold(y, p, grid=(0.3, 0.5, 0.7))
    assert best["macro_f1_present"] > 0.8
    assert best["threshold"] in {0.3, 0.5, 0.7}


def test_multilabel_classification_metrics_contains_required_keys() -> None:
    y = np.array([[1, 0], [0, 1]], dtype=np.float32)
    p = np.array([[0.9, 0.1], [0.1, 0.8]], dtype=np.float32)
    out = multilabel_classification_metrics(y, p)
    assert out["micro_f1"] == 1.0
    assert out["macro_f1_present"] == 1.0
    assert out["best_macro_f1_present"] == 1.0
    assert "macro_ap" in out
