from __future__ import annotations

from fed_agent.experiments.report import build_ablation_markdown, extract_run_row


def test_extract_run_row() -> None:
    payload = {
        "spec": {
            "name": "x",
            "fedprox_mu": 0.01,
            "noise_protocol_yaml": None,
            "rounds": 2,
        },
        "metrics": {
            "mean_train_loss_clients": [0.8, 0.7],
            "total_upload_bytes": 100,
            "noise_protocol": {"symmetric_flip_p_flip": 0.1},
        },
    }
    r = extract_run_row(payload)
    assert r["train_loss_final"] == 0.7
    assert r["noise_p_flip"] == 0.1


def test_build_ablation_markdown() -> None:
    rows = [
        {
            "name": "fedavg_clean",
            "fedprox_mu": 0.0,
            "noise_p_flip": None,
            "train_loss_final": 1.0,
            "train_loss_start": 1.1,
            "total_upload_bytes": 100,
            "rounds": 2,
        },
        {
            "name": "fedprox",
            "fedprox_mu": 0.1,
            "noise_p_flip": None,
            "train_loss_final": 0.9,
            "train_loss_start": 1.0,
            "total_upload_bytes": 100,
            "rounds": 2,
        },
    ]
    md = build_ablation_markdown(rows, baseline_name="fedavg_clean")
    assert "fedavg_clean" in md
    assert "dL vs base" in md
