"""Summarize paper matrix JSON into Markdown and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _row(run: dict[str, Any]) -> dict[str, Any]:
    cfg = run["config"]
    result = run["result"]
    ev = result["eval"]
    return {
        "name": run["name"],
        "method": cfg["method"],
        "backbone": cfg["backbone"],
        "loss": cfg["loss"],
        "positive_dropout": cfg["positive_dropout"],
        "train_label_noise": cfg.get("train_label_noise", 0.0),
        "best_macro_f1_present": ev.get("best_macro_f1_present"),
        "best_micro_f1": ev.get("best_micro_f1"),
        "macro_ap": ev.get("macro_ap"),
        "macro_auroc": ev.get("macro_auroc"),
        "total_upload_bytes": result.get("total_upload_bytes"),
        "is_retfound": result.get("model_info", {}).get("is_retfound"),
    }


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Paper Matrix Summary",
        "",
        "| run | method | loss | dropout | noise | best macro-F1 | best micro-F1 | "
        "macro-AUROC | bytes | RETFound |",
        "|-----|--------|------|---------|-------|---------------|---------------|"
        "-------------|-------|----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['method']} | {r['loss']} | {r['positive_dropout']} | "
            f"{r['train_label_noise']} | "
            f"{r['best_macro_f1_present']} | {r['best_micro_f1']} | "
            f"{r['macro_auroc']} | "
            f"{r['total_upload_bytes']} | {r['is_retfound']} |",
        )
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    if by_method:
        lines.extend(
            [
                "",
                "## Method Mean +/- Std",
                "",
                "| method | n | best macro-F1 | best micro-F1 |",
                "|--------|---|---------------|---------------|",
            ],
        )
        for method in sorted(by_method.keys()):
            group = by_method[method]
            macro = [float(r["best_macro_f1_present"]) for r in group]
            micro = [float(r["best_micro_f1"]) for r in group]
            macro_std = statistics.stdev(macro) if len(macro) > 1 else 0.0
            micro_std = statistics.stdev(micro) if len(micro) > 1 else 0.0
            lines.append(
                f"| {method} | {len(group)} | "
                f"{statistics.mean(macro):.6f} +/- {macro_std:.6f} | "
                f"{statistics.mean(micro):.6f} +/- {micro_std:.6f} |",
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Rows with `RETFound=False` are fallback/sanity runs and must not be reported "
            "as RETFound paper results.",
            "- Validation-calibrated thresholds should be frozen before final test reporting.",
            "",
        ],
    )
    return "\n".join(lines)


def summarize(summary_json: Path, *, out_md: Path, out_csv: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    rows = [_row(r) for r in payload.get("runs", [])]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(rows), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize paper matrix summary.json.")
    p.add_argument("summary_json", type=Path)
    p.add_argument("--out_md", type=Path, default=Path("docs/results/paper_matrix_latest.md"))
    p.add_argument("--out_csv", type=Path, default=Path("runs/paper_matrix/latest/summary.csv"))
    args = p.parse_args(argv)
    rows = summarize(args.summary_json, out_md=args.out_md, out_csv=args.out_csv)
    print(f"Wrote: {args.out_md}")
    print(f"Wrote: {args.out_csv}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
