"""Run multiple federated smoke configurations on a shared synthetic fixture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fed_agent.fed.simulator import FedSmokeConfig, run_multilabel_fed_smoke
from fed_agent.tools.minimal_experiment import write_minimal_fixture


@dataclass(frozen=True)
class SyntheticExperimentSpec:
    """One row in an ablation / comparison matrix."""

    name: str
    fedprox_mu: float
    noise_protocol_yaml: Path | None
    label_noise_seed: int = 0
    rounds: int = 2
    lr: float = 0.1
    local_epochs: int = 1
    batch_size: int = 2
    image_size: tuple[int, int] = (16, 16)


def default_synthetic_specs(repo_root: Path) -> list[SyntheticExperimentSpec]:
    """Default comparison + ablation grid (synthetic 4-sample setup)."""

    nd = Path(repo_root) / "configs" / "noise_protocol"
    return [
        SyntheticExperimentSpec(
            name="fedavg_clean",
            fedprox_mu=0.0,
            noise_protocol_yaml=None,
        ),
        SyntheticExperimentSpec(
            name="fedprox_mu005_clean",
            fedprox_mu=0.05,
            noise_protocol_yaml=None,
        ),
        SyntheticExperimentSpec(
            name="fedavg_noise_p01",
            fedprox_mu=0.0,
            noise_protocol_yaml=nd / "ablation_noise_01.yaml",
        ),
        SyntheticExperimentSpec(
            name="fedprox_mu005_noise_p01",
            fedprox_mu=0.05,
            noise_protocol_yaml=nd / "ablation_noise_01.yaml",
        ),
        SyntheticExperimentSpec(
            name="fedavg_noise_p05",
            fedprox_mu=0.0,
            noise_protocol_yaml=nd / "ablation_noise_05.yaml",
        ),
    ]


def run_synthetic_suite(
    *,
    fixture_dir: Path,
    out_dir: Path,
    specs: list[SyntheticExperimentSpec],
) -> dict[str, Any]:
    """Materialize fixture if needed, run each spec, write per-run JSON + ``summary.json``."""

    fixture_dir = Path(fixture_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_csv, images_dir, split_json = write_minimal_fixture(fixture_dir)

    runs_out: list[dict[str, Any]] = []
    for spec in specs:
        cfg = FedSmokeConfig(
            rounds=int(spec.rounds),
            local_epochs=int(spec.local_epochs),
            batch_size=int(spec.batch_size),
            lr=float(spec.lr),
            fedprox_mu=float(spec.fedprox_mu),
            device="cpu",
            seed=int(spec.label_noise_seed),
        )
        metrics = run_multilabel_fed_smoke(
            labels_csv=labels_csv,
            images_dir=images_dir,
            split_json=split_json,
            image_size=spec.image_size,
            cfg=cfg,
            noise_protocol_yaml=spec.noise_protocol_yaml,
            label_noise_seed=int(spec.label_noise_seed),
        )
        np_path = spec.noise_protocol_yaml
        yaml_s = str(np_path) if np_path else None
        spec_dict = {
            "name": spec.name,
            "fedprox_mu": spec.fedprox_mu,
            "noise_protocol_yaml": yaml_s,
            "label_noise_seed": spec.label_noise_seed,
            "rounds": spec.rounds,
            "lr": spec.lr,
            "local_epochs": spec.local_epochs,
            "batch_size": spec.batch_size,
            "image_size": list(spec.image_size),
        }
        payload = {
            "spec": spec_dict,
            "metrics": metrics,
        }
        runs_out.append(payload)
        safe = spec.name.replace("/", "_")
        (out_dir / f"{safe}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_dir": str(fixture_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "n_runs": len(runs_out),
        "runs": runs_out,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
