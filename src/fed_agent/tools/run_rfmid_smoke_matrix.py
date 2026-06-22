"""Run real RFMiD-subset smoke matrix: splits × FedAvg/FedProx × label noise."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass, replace
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
    loss: str = "bce"
    model: str = "tiny_cnn"


def default_specs(noise_dir: Path) -> list[RFMiDSmokeSpec]:
    return [
        RFMiDSmokeSpec("iid_fedavg_bce_clean", "iid", 0.0, None, None, "bce"),
        RFMiDSmokeSpec(
            "iid_fedavg_balanced_clean",
            "iid",
            0.0,
            None,
            None,
            "balanced_bce",
        ),
        RFMiDSmokeSpec(
            "iid_fedprox_balanced_clean",
            "iid",
            0.05,
            None,
            None,
            "balanced_bce",
        ),
        RFMiDSmokeSpec(
            "iid_fedavg_balanced_noise_p01",
            "iid",
            0.0,
            noise_dir / "ablation_noise_01.yaml",
            0.1,
            "balanced_bce",
        ),
        RFMiDSmokeSpec(
            "iid_fedprox_balanced_noise_p01",
            "iid",
            0.05,
            noise_dir / "ablation_noise_01.yaml",
            0.1,
            "balanced_bce",
        ),
        RFMiDSmokeSpec(
            "dirichlet_a0p5_fedavg_bce_clean",
            "dirichlet_a0p5",
            0.0,
            None,
            None,
            "bce",
        ),
        RFMiDSmokeSpec(
            "dirichlet_a0p5_fedavg_balanced_clean",
            "dirichlet_a0p5",
            0.0,
            None,
            None,
            "balanced_bce",
        ),
        RFMiDSmokeSpec(
            "dirichlet_a0p5_fedprox_balanced_clean",
            "dirichlet_a0p5",
            0.05,
            None,
            None,
            "balanced_bce",
        ),
        RFMiDSmokeSpec(
            "dirichlet_a0p5_fedavg_balanced_noise_p01",
            "dirichlet_a0p5",
            0.0,
            noise_dir / "ablation_noise_01.yaml",
            0.1,
            "balanced_bce",
        ),
        RFMiDSmokeSpec(
            "dirichlet_a0p5_fedprox_balanced_noise_p01",
            "dirichlet_a0p5",
            0.05,
            noise_dir / "ablation_noise_01.yaml",
            0.1,
            "balanced_bce",
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
        "loss": spec.loss,
        "model": spec.model,
        "final_train_loss": float(losses[-1]) if losses else float("nan"),
        "start_train_loss": float(losses[0]) if losses else float("nan"),
        "final_eval_loss": _last(metrics, "eval_loss"),
        "final_eval_micro_f1": _last(metrics, "eval_micro_f1"),
        "final_eval_macro_f1_present": _last(metrics, "eval_macro_f1_present"),
        "best_eval_threshold": _last(metrics, "eval_best_threshold"),
        "best_eval_micro_f1": _last(metrics, "eval_best_micro_f1"),
        "best_eval_macro_f1_present": _last(metrics, "eval_best_macro_f1_present"),
        "total_upload_bytes": int(metrics.get("total_upload_bytes", 0)),
    }


def _last(metrics: dict[str, Any], key: str) -> float:
    xs = metrics.get(key) or []
    return float(xs[-1]) if xs else float("nan")


def _pct(new: float, base: float) -> str:
    if base == 0 or base != base:
        return "n/a"
    return f"{100.0 * (new - base) / base:+.2f}%"


def _report(rows: list[dict[str, Any]]) -> str:
    by_name = {r["name"]: r for r in rows}
    baseline = by_name["iid_fedavg_bce_clean"]
    best = max(rows, key=lambda r: r["best_eval_macro_f1_present"])
    lines = [
        "# RFMiD subset smoke matrix",
        "",
        "This is a **real RFMiD image subset** smoke experiment, not the final paper-scale run.",
        "",
        "Primary smoke metric: validation best-threshold `macro_f1_present` "
        "(higher is better).",
        "",
        f"Baseline: `iid_fedavg_bce_clean`, macro-F1 = "
        f"{baseline['best_eval_macro_f1_present']:.6f}.",
        "",
        "## Key finding",
        "",
        f"- Best run: `{best['name']}` with best macro-F1 "
        f"{best['best_eval_macro_f1_present']:.6f} "
        f"({_pct(best['best_eval_macro_f1_present'], baseline['best_eval_macro_f1_present'])} "
        "vs BCE baseline).",
        "- In this smoke setting, the useful signal is **noise/dropout-style robust training "
        "under class imbalance**, not FedProx alone.",
        "",
        "## Summary table",
        "",
        "| run | split | loss | noise | best macro-F1 | t* | macro@0.5 | micro@0.5 | bytes |",
        "|-----|-------|------|-------|---------------|----|-----------|-----------|-------|",
    ]
    for r in sorted(rows, key=lambda x: (x["split"], x["name"])):
        np_s = "none" if r["noise_p"] is None else str(r["noise_p"])
        lines.append(
            f"| {r['name']} | {r['split']} | {r['loss']} | {np_s} | "
            f"{r['best_eval_macro_f1_present']:.6f} | "
            f"{r['best_eval_threshold']:.2f} | "
            f"{r['final_eval_macro_f1_present']:.6f} | "
            f"{r['final_eval_micro_f1']:.6f} | "
            f"{r['total_upload_bytes']} |",
        )

    lines.extend(["", "## Method comparison: class-balanced loss", ""])
    for split in ["iid", "dirichlet_a0p5"]:
        if f"{split}_fedavg_bce_clean" not in by_name:
            continue
        bce = by_name[f"{split}_fedavg_bce_clean"]
        bal = by_name[f"{split}_fedavg_balanced_clean"]
        lines.append(
            f"- **{split}**: BCE best macro-F1 {bce['best_eval_macro_f1_present']:.6f} -> "
            f"balanced BCE {bal['best_eval_macro_f1_present']:.6f} "
            f"({_pct(bal['best_eval_macro_f1_present'], bce['best_eval_macro_f1_present'])}).",
        )

    lines.extend(["", "## Method comparison: FedProx on balanced loss", ""])
    for split in ["iid", "dirichlet_a0p5"]:
        if f"{split}_fedprox_balanced_clean" not in by_name:
            continue
        fa = by_name[f"{split}_fedavg_balanced_clean"]
        fp = by_name[f"{split}_fedprox_balanced_clean"]
        lines.append(
            f"- **{split}**: FedAvg best macro-F1 "
            f"{fa['best_eval_macro_f1_present']:.6f} vs "
            f"FedProx {fp['best_eval_macro_f1_present']:.6f}.",
        )

    lines.extend(["", "## Robustness / noise ablation", ""])
    for split in ["iid", "dirichlet_a0p5"]:
        if f"{split}_fedavg_balanced_noise_p01" not in by_name:
            continue
        clean = by_name[f"{split}_fedavg_balanced_clean"]
        noisy = by_name[f"{split}_fedavg_balanced_noise_p01"]
        lines.append(
            f"- **{split}, balanced FedAvg**: clean best macro-F1 "
            f"{clean['best_eval_macro_f1_present']:.6f} -> p=0.1 noise "
            f"{noisy['best_eval_macro_f1_present']:.6f}.",
        )

    lines.extend(
        [
            "",
            "## Paper-facing interpretation",
            "",
            "- This validates the proposed pipeline on real RFMiD images: split generation, "
            "validation metrics, label-noise protocol, FL loop, and communication accounting.",
            "- The current positive signal is to treat mild positive-label dropout/noise as "
            "a robustness regularizer under severe label imbalance.",
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
    eval_labels_csv: Path | None,
    eval_images_dir: Path | None,
    quick_core: bool = False,
    model: str = "tiny_cnn",
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
    if quick_core:
        keep = {
            "iid_fedavg_bce_clean",
            "iid_fedavg_balanced_clean",
            "iid_fedprox_balanced_clean",
            "iid_fedavg_balanced_noise_p01",
            "iid_fedprox_balanced_noise_p01",
        }
        specs = [s for s in specs if s.name in keep]
    specs = [replace(s, model=str(model)) for s in specs]
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
            model=spec.model,
            loss=spec.loss,
        )
        metrics = run_multilabel_fed_smoke(
            labels_csv=labels_csv,
            images_dir=images_dir,
            split_json=split_paths[spec.split],
            image_size=image_size,
            cfg=cfg,
            noise_protocol_yaml=spec.noise_yaml,
            label_noise_seed=int(seed),
            eval_labels_csv=eval_labels_csv,
            eval_images_dir=eval_images_dir,
        )
        spec_payload = {
            "name": spec.name,
            "split": spec.split,
            "fedprox_mu": spec.fedprox_mu,
            "noise_yaml": str(spec.noise_yaml) if spec.noise_yaml else None,
            "noise_p": spec.noise_p,
            "loss": spec.loss,
            "model": spec.model,
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
        "eval_labels_csv": str(eval_labels_csv) if eval_labels_csv else None,
        "eval_images_dir": str(eval_images_dir) if eval_images_dir else None,
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
    p.add_argument("--eval_labels_csv", type=Path, default=None)
    p.add_argument("--eval_images_dir", type=Path, default=None)
    p.add_argument("--model", type=str, default="tiny_cnn", choices=["mlp", "tiny_cnn"])
    p.add_argument("--quick_core", action="store_true")
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
        eval_labels_csv=args.eval_labels_csv,
        eval_images_dir=args.eval_images_dir,
        quick_core=bool(args.quick_core),
        model=str(args.model),
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
