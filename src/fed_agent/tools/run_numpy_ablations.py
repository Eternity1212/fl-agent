"""CLI: run dependency-light NumPy synthetic FL ablations and publish reports."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fed_agent.experiments.numpy_suite import run_numpy_synthetic_suite
from fed_agent.experiments.report import (
    build_ablation_markdown,
    extract_run_row,
    rows_to_csv,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NumPy synthetic FedAvg/FedProx ablations.")
    p.add_argument("--repo_root", type=Path, default=Path.cwd())
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/numpy_ablations/<UTC timestamp>).",
    )
    p.add_argument("--baseline", type=str, default="fedavg_clean")
    p.add_argument(
        "--publish_docs",
        action="store_true",
        help="Copy REPORT.md to docs/results/numpy_synthetic_ablation_latest.md",
    )
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    if args.out_dir is not None:
        out = Path(args.out_dir).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = repo / "runs" / "numpy_ablations" / stamp
    out.mkdir(parents=True, exist_ok=True)

    summary = run_numpy_synthetic_suite(out_dir=out)
    rows = [extract_run_row(r) for r in summary["runs"]]
    md = build_ablation_markdown(rows, baseline_name=str(args.baseline))

    (out / "REPORT.md").write_text(md, encoding="utf-8")
    (out / "summary.csv").write_text(rows_to_csv(rows), encoding="utf-8")

    print(md)
    print(f"\nWrote: {out / 'REPORT.md'}")
    print(f"Wrote: {out / 'summary.csv'}")
    print(f"Wrote: {out / 'summary.json'}")

    if args.publish_docs:
        dest = repo / "docs" / "results" / "numpy_synthetic_ablation_latest.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "REPORT.md", dest)
        print(f"Published: {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
