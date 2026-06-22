"""Experiment harnesses (synthetic baselines first, then real-data sweeps)."""

from fed_agent.experiments.numpy_suite import (
    NumpyExperimentSpec,
    run_numpy_synthetic_suite,
)
from fed_agent.experiments.report import build_ablation_markdown, extract_run_row

__all__ = [
    "NumpyExperimentSpec",
    "build_ablation_markdown",
    "extract_run_row",
    "run_numpy_synthetic_suite",
]
