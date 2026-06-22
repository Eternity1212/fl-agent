from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from fed_agent.experiments.report import build_ablation_markdown, extract_run_row
from fed_agent.experiments.synthetic_suite import SyntheticExperimentSpec, run_synthetic_suite


def test_run_synthetic_suite_two_runs(tmp_path: Path) -> None:
    fixture = tmp_path / "fx"
    out = tmp_path / "out"
    specs = [
        SyntheticExperimentSpec(name="a", fedprox_mu=0.0, noise_protocol_yaml=None, rounds=1),
        SyntheticExperimentSpec(
            name="b",
            fedprox_mu=0.0,
            noise_protocol_yaml=None,
            rounds=1,
            lr=0.05,
        ),
    ]
    summary = run_synthetic_suite(fixture_dir=fixture, out_dir=out, specs=specs)
    assert summary["n_runs"] == 2
    assert (out / "summary.json").is_file()
    rows = [extract_run_row(r) for r in summary["runs"]]
    md = build_ablation_markdown(rows, baseline_name="a")
    assert "a" in md and "b" in md
