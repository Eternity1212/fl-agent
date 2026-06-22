"""Run real RFMiD-subset smoke matrix: splits × FedAvg/FedProx × label noise."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fed_agent.data.rfmid import load_rfmid_label_table
from fed_agent.fed.simulator import FedSmokeConfig, run_multilabel_fed_smoke
from fed_agent.splits.partition import (
    build_dirichlet_split_primary,
    build_iid_split,
    split_payload,
    write_split_json,
)


@dataclass(frozen=True)
class RFMiDSmokeSpec:
    name: str
    split: str
    fedprox_mu: float
    noise_yaml: Path | None
    noise_p: float | None


def default_specs(noise_dir: Path) -> list[RFMiDSmokeSpec]:
    return [
        RFMiDSmokeSpec("iid_fedavg_clean", "iid", 0.0, None, None),
        RFMiDSmokeSpec("iid_fedprox_clean", "iid", 0.05, None, None),
        RFMiDSmokeSpec(
            "iid_fedavg_noise_p01",
            "iid",
            0.0,
            noise_dir / "ablation_noise_01.yaml",
            0.1,
        ),
        RFMiDSmokeSpec(
            "iid_fedprox_noise_p01",
            "iid",
            0.05,
            noise_dir / "ablation_noise_01.yaml",
            0.1,
        ),
        RFMiDSmokeSpec("dirichlet_a0p5_fedavg_clean", "dirichlet_a0p5", 0.0, None, None),
        RFMiDSmokeSpec("dirichlet_a0p5_fedprox_clean", "dirichlet_a0p5", 0.05, None, None),
        RFMiDSmokeSpec(
            "dirichlet_a0p5_fedavg_noise_p01",
            "dirichlet_a0p5",
            0.0,
            noise_dir / "ablation_noise_01.yaml",
            0.1,
        ),
        RFMiDSmokeSpec(
            "dirichlet_a0p5_fedprox_noise_p01",
            "dirichlet_a0p5",
            0.05,
            noise_dir / "ablation_noise_01.yaml",
            0.1,
        ),
    ]


def _build_splits(labels_csv: Path, out_dir: Path, *, n_clients: int, seed: int) -> dict[str, Path]:
    image_ids, label_names, y = load_rfmid_label_table(labels_csv)
    y_lookup = {sid: y[i] for i, sid in enumerate(image_ids)}
    out_dir.mkdir(parents=True, exist_ok=True)

    iid = build_iid_split(image_ids, n_clients=n_clients, seed=seed)
    iid_payload = split_payload(
        split="iid",
        n_clients=n_clients,
        seed=seed,
        alpha=None,
        label_names=label_names,
        clients=iid,
        y_lookup=y_lookup,
    )
    iid_path = out_dir / f"rfmid_subset__iid_K{n_clients}_S{seed}.json"
    write_split_json(iid_path, iid_payload)

    dirichlet = build_dirichlet_split_primary(
        image_ids,
        y,
        n_clients=n_clients,
        alpha=0.5,
        seed=seed,
    )
    dir_payload = split_payload(
        split="dirichlet_primary",
        n_clients=n_clients,
        seed=seed,
        alpha=0.5,
        label_names=label_names,
        clients=dirichlet,
        y_lookup=y_lookup,
    )
    dir_path = out_dir / f"rfmid_subset__dirichlet_a0p5_K{n_clients}_S{seed}.json"
    write_split_json(dir_path, dir_payload)

    return {"iid": iid_path, "dirichlet_a0p5": dir_path}


def _row(spec: RFMiDSmokeSpec, metrics: dict[str, Any]) -> dict[str, Any]:
    losses = metrics.get("mean_train_loss_clients") or []
    return {
        "name": spec.name,
        "split": spec.split,
        "fedprox_mu": spec.fedprox_mu,
        "noise_p": spec.noise_p,
        "final_train_loss": float(losses[-1]) if losses else float("nan"),
        "start_train_loss": float(losses[0]) if losses else float("nan"),
        "total_upload_bytes": int(metrics.get("total_upload_bytes", 0)),
    }


def _pct(new: float, base: float) -> str:
    if base == 0 or base != base:
        return "n/a"
    return f"{100.0 * (new - base) / base:+.2f}%"


def _report(rows: list[dict[str, Any]]) -> str:
    by_name = {r["name"]: r for r in rows}
    baseline = by_name["iid_fedavg_clean"]
    lines = [
        "# RFMiD subset smoke matrix",
        "",
        "This is a **real RFMiD image subset** smoke experiment, not the final paper-scale run.",
        "",
        f"Baseline: `iid_fedavg_clean`, final train loss = {baseline['final_train_loss']:.6f}.",
        "",
        "## Summary table",
        "",
        "| run | split | mu_FedProx | p_noise | L_final | dL vs iid baseline | upload_bytes |",
        "|-----|-------|-------------|---------|---------|--------------------|--------------|",
    ]
    for r in sorted(rows, key=lambda x: (x["split"], x["name"])):
        np_s = "none" if r["noise_p"] is None else str(r["noise_p"])
        lines.append(
            f"| {r['name']} | {r['split']} | {r['fedprox_mu']} | {np_s} | "
            f"{r['final_train_loss']:.6f} | "
            f"{_pct(r['final_train_loss'], baseline['final_train_loss'])} | "
            f"{r['total_upload_bytes']} |",
        )

    lines.extend(["", "## Method comparison", ""])
    for split in ["iid", "dirichlet_a0p5"]:
        fa = by_name[f"{split}_fedavg_clean"]
        fp = by_name[f"{split}_fedprox_clean"]
        lines.append(
            f"- **{split}, clean**: FedAvg {fa['final_train_loss']:.6f} vs "
            f"FedProx {fp['final_train_loss']:.6f} "
            f"({_pct(fp['final_train_loss'], fa['final_train_loss'])}).",
        )

    lines.extend(["", "## Robustness / noise ablation", ""])
    for split in ["iid", "dirichlet_a0p5"]:
        clean = by_name[f"{split}_fedavg_clean"]
        noisy = by_name[f"{split}_fedavg_noise_p01"]
        lines.append(
            f"- **{split}, FedAvg**: clean {clean['final_train_loss']:.6f} -> "
            f"p=0.1 noise {noisy['final_train_loss']:.6f} "
            f"({_pct(noisy['final_train_loss'], clean['final_train_loss'])}).",
        )

    lines.extend(
        [
            "",
            "## Paper-facing interpretation",
            "",
            "- This validates the proposed pipeline on real RFMiD images: split generation, "
            "label-noise protocol, FL loop, and communication accounting all run end-to-end.",
            "- It is still a **smoke-scale** experiment. Paper claims should use the same matrix "
            "on full RFMiD plus RETFound/LoRA and stronger baselines.",
            "",
        ],
    )
    return "\n".join(lines)


def run_matrix(
    *,
    labels_csv: Path,
    images_dir: Path,
    out_dir: Path,
    n_clients: int,
    seed: int,
    rounds: int,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_paths = _build_splits(
        labels_csv,
        out_dir / "splits",
        n_clients=int(n_clients),
        seed=int(seed),
    )

    specs = default_specs(Path("configs/noise_protocol"))
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for spec in specs:
        cfg = FedSmokeConfig(
            rounds=int(rounds),
            local_epochs=1,
            batch_size=8,
            lr=0.05,
            fedprox_mu=float(spec.fedprox_mu),
            device="cpu",
            seed=int(seed),
        )
        metrics = run_multilabel_fed_smoke(
            labels_csv=labels_csv,
            images_dir=images_dir,
            split_json=split_paths[spec.split],
            image_size=image_size,
            cfg=cfg,
            noise_protocol_yaml=spec.noise_yaml,
            label_noise_seed=int(seed),
        )
        spec_payload = {
            "name": spec.name,
            "split": spec.split,
            "fedprox_mu": spec.fedprox_mu,
            "noise_yaml": str(spec.noise_yaml) if spec.noise_yaml else None,
            "noise_p": spec.noise_p,
        }
        payload = {"spec": spec_payload, "metrics": metrics}
        runs.append(payload)
        rows.append(_row(spec, metrics))
        (out_dir / f"{spec.name}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels_csv": str(Path(labels_csv)),
        "images_dir": str(Path(images_dir)),
        "n_clients": int(n_clients),
        "seed": int(seed),
        "rounds": int(rounds),
        "runs": runs,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = _report(rows)
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run RFMiD real-subset smoke matrix.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--images_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("runs/rfmid_smoke_matrix/latest"))
    p.add_argument("--n_clients", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--image_size", type=int, nargs=2, default=[32, 32])
    p.add_argument("--publish_docs", action="store_true")
    args = p.parse_args(argv)

    summary = run_matrix(
        labels_csv=args.labels_csv,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        n_clients=int(args.n_clients),
        seed=int(args.seed),
        rounds=int(args.rounds),
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
    )
    report_path = Path(args.out_dir) / "REPORT.md"
    print(report_path.read_text(encoding="utf-8"))
    print(f"Wrote: {Path(args.out_dir) / 'summary.json'}")
    print(f"Wrote: {Path(args.out_dir) / 'summary.csv'}")

    if args.publish_docs:
        dest = Path("docs/results/rfmid_subset_smoke_latest.md")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report_path, dest)
        print(f"Published: {dest}")

    print(f"Completed {len(summary['rows'])} RFMiD smoke runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
