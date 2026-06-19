"""Synthetic 4-sample, 2-client federated smoke — baseline pipeline without real RFMiD paths."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fed_agent.data.rfmid import load_rfmid_label_table
from fed_agent.splits.partition import split_payload, write_split_json


def write_minimal_fixture(root: Path) -> tuple[Path, Path, Path]:
    """Write labels CSV, four PNGs, and a 2-client split JSON under ``root``."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    img_dir = root / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    for sid in ["s0", "s1", "s2", "s3"]:
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_dir / f"{sid}.png")

    csv_path = root / "labels.csv"
    csv_path.write_text(
        "ImageID,A,B\ns0,1,0\ns1,1,0\ns2,0,1\ns3,0,1\n",
        encoding="utf-8",
    )

    image_ids, label_names, y = load_rfmid_label_table(csv_path)
    y_lookup = {sid: y[i] for i, sid in enumerate(image_ids)}
    clients = {"0": ["s0", "s1"], "1": ["s2", "s3"]}
    payload = split_payload(
        split="minimal_synthetic",
        n_clients=2,
        seed=0,
        alpha=None,
        label_names=label_names,
        clients=clients,
        y_lookup=y_lookup,
    )
    split_path = root / "split.json"
    write_split_json(split_path, payload)
    return csv_path, img_dir, split_path


def _metrics_snapshot_md(metrics: dict[str, Any], *, meta: dict[str, str]) -> str:
    lines = [
        "# Minimal synthetic federated baseline",
        "",
        "| Field | Value |",
        "|-------|-------|",
    ]
    for k, v in meta.items():
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            "## Metrics (subset)",
            "",
            f"- **total_upload_bytes**: `{metrics.get('total_upload_bytes')}`",
            f"- **comm_bytes_upload_per_round**: `{metrics.get('comm_bytes_upload_per_round')}`",
            f"- **mean_train_loss_clients**: `{metrics.get('mean_train_loss_clients')}`",
        ],
    )
    if "noise_protocol" in metrics:
        lines.append(f"- **noise_protocol**: `{metrics['noise_protocol']}`")
    full_json = json.dumps(metrics, indent=2, ensure_ascii=False)
    lines.extend(["", "## Full JSON", "", "```json", full_json, "```", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        msg = "ERROR: torch is required. Install: python3 -m pip install -e '.[torch]'"
        print(msg, file=sys.stderr)
        return 2

    from fed_agent.fed.simulator import FedSmokeConfig, run_multilabel_fed_smoke

    p = argparse.ArgumentParser(description="Synthetic 4-sample federated smoke baseline.")
    p.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Write/read fixture under this directory (persistent). If omitted, uses a temp dir.",
    )
    p.add_argument("--out_json", type=Path, default=None)
    p.add_argument("--write_docs_snapshot", type=Path, default=None)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--fedprox_mu", type=float, default=0.01)
    p.add_argument("--image_size", type=int, nargs=2, default=[16, 16])
    p.add_argument("--noise_yaml", type=Path, default=None)
    p.add_argument("--label_noise_seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = FedSmokeConfig(
        rounds=int(args.rounds),
        local_epochs=int(args.local_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        fedprox_mu=float(args.fedprox_mu),
        device="cpu",
    )

    if args.workdir is not None:
        work = Path(args.workdir)
        labels_csv, images_dir, split_json = write_minimal_fixture(work)
        metrics = run_multilabel_fed_smoke(
            labels_csv=labels_csv,
            images_dir=images_dir,
            split_json=split_json,
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
            cfg=cfg,
            noise_protocol_yaml=args.noise_yaml,
            label_noise_seed=int(args.label_noise_seed),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="fl_agent_minexp_") as td:
            work = Path(td)
            labels_csv, images_dir, split_json = write_minimal_fixture(work)
            metrics = run_multilabel_fed_smoke(
                labels_csv=labels_csv,
                images_dir=images_dir,
                split_json=split_json,
                image_size=(int(args.image_size[0]), int(args.image_size[1])),
                cfg=cfg,
                noise_protocol_yaml=args.noise_yaml,
                label_noise_seed=int(args.label_noise_seed),
            )

    text = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    print(text)
    if args.out_json is not None:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text, encoding="utf-8")

    if args.write_docs_snapshot is not None:
        import platform

        meta = {
            "kind": "synthetic_minimal_fed_smoke",
            "python": platform.python_version(),
            "workdir": str(args.workdir) if args.workdir else "(temp)",
        }
        snap = _metrics_snapshot_md(metrics, meta=meta)
        Path(args.write_docs_snapshot).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_docs_snapshot).write_text(snap, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
