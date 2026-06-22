"""Run paper-scale RFMiD experiment matrix from YAML config."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from fed_agent.train.paper_runner import PaperRunConfig, run_paper_experiment


def _merge(defaults: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    out = dict(defaults)
    out.update(row)
    return out


def _cfg_from_row(defaults: dict[str, Any], row: dict[str, Any]) -> PaperRunConfig:
    payload = _merge(defaults, row)
    if "image_size" in payload:
        payload["image_size"] = tuple(int(x) for x in payload["image_size"])
    payload.pop("name", None)
    payload.pop("split_json", None)
    return PaperRunConfig(**payload)


def run_matrix(
    *,
    matrix_yaml: Path,
    out_dir: Path,
    max_runs: int | None = None,
    only: set[str] | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(matrix_yaml).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("paper matrix YAML must be a mapping")

    dataset = config["dataset"]
    defaults = config.get("defaults", {})
    rows = list(config.get("runs", []))
    if only:
        rows = [r for r in rows if str(r["name"]) in only]
    if max_runs is not None:
        rows = rows[: int(max_runs)]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_out: list[dict[str, Any]] = []
    for row in rows:
        name = str(row["name"])
        run_path = out_dir / f"{name}.json"
        if run_path.is_file():
            runs_out.append(json.loads(run_path.read_text(encoding="utf-8")))
            continue
        cfg = _cfg_from_row(defaults, row)
        split_json_value = row.get("split_json", dataset.get("split_json"))
        split_json = Path(split_json_value) if cfg.method != "centralized" else None
        result = run_paper_experiment(
            train_labels_csv=Path(dataset["train_labels_csv"]),
            train_images_dir=Path(dataset["train_images_dir"]),
            eval_labels_csv=Path(dataset["validation_labels_csv"]),
            eval_images_dir=Path(dataset["validation_images_dir"]),
            cfg=cfg,
            split_json=split_json,
        )
        payload = {"name": name, "config": asdict(cfg), "result": result}
        runs_out.append(payload)
        run_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_yaml": str(matrix_yaml),
        "out_dir": str(out_dir),
        "n_runs": len(runs_out),
        "runs": runs_out,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run RFMiD paper matrix YAML.")
    p.add_argument("--matrix_yaml", type=Path, default=Path("configs/paper_matrix.yaml"))
    p.add_argument("--out_dir", type=Path, default=Path("runs/paper_matrix/latest"))
    p.add_argument("--max_runs", type=int, default=None)
    p.add_argument("--only", type=str, nargs="*", default=None)
    args = p.parse_args(argv)

    summary = run_matrix(
        matrix_yaml=args.matrix_yaml,
        out_dir=args.out_dir,
        max_runs=args.max_runs,
        only=set(args.only) if args.only else None,
    )
    print(f"Wrote: {Path(args.out_dir) / 'summary.json'}")
    print(f"Completed {summary['n_runs']} paper-matrix runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
