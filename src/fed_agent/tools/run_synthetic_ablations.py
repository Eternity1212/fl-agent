"""CLI: synthetic ablation suite + Markdown / CSV report."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from fed_agent.experiments.report import (
    build_ablation_markdown,
    extract_run_row,
    rows_to_csv,
)


def main(argv: list[str] | None = None) -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: install torch first: python3 -m pip install -e '.[torch]'", file=sys.stderr)
        return 2

    from fed_agent.experiments.synthetic_suite import default_synthetic_specs, run_synthetic_suite

    p = argparse.ArgumentParser(description="Synthetic FedAvg/FedProx + label-noise ablations.")
    p.add_argument("--repo_root", type=Path, default=Path.cwd())
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/synthetic_ablations/<UTC timestamp>).",
    )
    p.add_argument(
        "--baseline",
        type=str,
        default="fedavg_clean",
        help="Run name used as baseline for relative loss column.",
    )
    p.add_argument(
        "--publish_docs",
        action="store_true",
        help="Copy REPORT.md to docs/results/synthetic_ablation_latest.md",
    )
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    if args.out_dir is not None:
        out = Path(args.out_dir).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = repo / "runs" / "synthetic_ablations" / stamp
    out.mkdir(parents=True, exist_ok=True)
    fixture_dir = out / "fixture"

    specs = default_synthetic_specs(repo)
    summary = run_synthetic_suite(fixture_dir=fixture_dir, out_dir=out, specs=specs)
    rows = [extract_run_row(r) for r in summary["runs"]]
    md = build_ablation_markdown(rows, baseline_name=str(args.baseline))
    (out / "REPORT.md").write_text(md, encoding="utf-8")
    (out / "summary.csv").write_text(rows_to_csv(rows), encoding="utf-8")

    print(md)
    print(f"\nWrote: {out / 'REPORT.md'}")
    print(f"Wrote: {out / 'summary.csv'}")
    print(f"Wrote: {out / 'summary.json'}")

    if args.publish_docs:
        dest = repo / "docs" / "results" / "synthetic_ablation_latest.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "REPORT.md", dest)
        print(f"Published: {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
