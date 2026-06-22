from __future__ import annotations

from pathlib import Path

from fed_agent.experiments.numpy_suite import (
    NumpyExperimentSpec,
    run_numpy_synthetic_one,
    run_numpy_synthetic_suite,
)
from fed_agent.experiments.report import build_ablation_markdown, extract_run_row


def test_run_numpy_synthetic_one() -> None:
    metrics = run_numpy_synthetic_one(
        NumpyExperimentSpec(name="x", fedprox_mu=0.0, noise_p_flip=0.0, rounds=2),
    )
    assert metrics["backend"] == "numpy_logistic_synthetic"
    assert len(metrics["mean_train_loss_clients"]) == 2
    assert metrics["total_upload_bytes"] > 0


def test_run_numpy_synthetic_suite(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run_numpy_synthetic_suite(
        out_dir=out,
        specs=[
            NumpyExperimentSpec(name="fedavg_clean", fedprox_mu=0.0, rounds=2),
            NumpyExperimentSpec(name="fedavg_noise_p01", fedprox_mu=0.0, noise_p_flip=0.1),
        ],
    )
    assert summary["n_runs"] == 2
    assert (out / "summary.json").is_file()
    rows = [extract_run_row(r) for r in summary["runs"]]
    md = build_ablation_markdown(rows, baseline_name="fedavg_clean")
    assert "fedavg_noise_p01" in md
