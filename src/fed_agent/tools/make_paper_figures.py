"""Aggregate paper-matrix summary.json files into publication tables and figures.

Reads one or more ``summary.json`` files (from ``run_paper_matrix``), groups runs
by method-variant and label-noise level, and emits:

  * a main comparison table (mean +/- std over seeds)  -> markdown
  * Figure 1: per-method performance bar with communication annotation
  * Figure 2: noise-degradation curves (best macro-F1 / macro-AUROC vs noise)

The script is robust to partial data: it plots whatever runs are present, so it
can be run on an in-progress matrix.

Usage:
    python3 -m fed_agent.tools.make_paper_figures \
        runs/paper_matrix/retfound_full/summary.json \
        runs/paper_matrix/paper_matrix_robustness/summary.json \
        --out_dir docs/results/figures \
        --out_md docs/results/paper_figures.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Stable display order + colors for variants.
VARIANT_ORDER = [
    "Centralized",
    "Local-only",
    "FedAvg",
    "FedAvg+Dropout",
    "FedProx",
    "Robust-FedProx",
]
VARIANT_COLOR = {
    "Centralized": "#444444",
    "Local-only": "#9e9e9e",
    "FedAvg": "#1f77b4",
    "FedAvg+Dropout": "#2ca02c",
    "FedProx": "#ff7f0e",
    "Robust-FedProx": "#d62728",
}


def _variant_name(cfg: dict[str, Any]) -> str:
    method = cfg.get("method", "?")
    pd = float(cfg.get("positive_dropout", 0.0) or 0.0)
    if method == "centralized":
        return "Centralized"
    if method == "local_only":
        return "Local-only"
    if method == "fedavg":
        return "FedAvg+Dropout" if pd > 0 else "FedAvg"
    if method == "fedprox":
        return "FedProx"
    if method == "robust_fedprox":
        return "Robust-FedProx"
    return method


def _load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        for r in data.get("runs", []):
            name = str(r.get("name"))
            if name in seen:
                continue
            seen.add(name)
            runs.append(r)
    return runs


def _agg(values: list[float]) -> tuple[float, float, int]:
    if not values:
        return float("nan"), float("nan"), 0
    return mean(values), (pstdev(values) if len(values) > 1 else 0.0), len(values)


def _fmt(m: float, s: float) -> str:
    if math.isnan(m):
        return "-"
    return f"{m:.3f} +/- {s:.3f}"


def build_main_table(runs: list[dict[str, Any]]) -> str:
    """Mean +/- std over seeds for clean (noise=0) runs, grouped by variant."""
    buckets: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        cfg = r["config"]
        if float(cfg.get("train_label_noise", 0.0) or 0.0) != 0.0:
            continue
        v = _variant_name(cfg)
        ev = r["result"]["eval"]
        b = buckets.setdefault(v, {"f1": [], "micro": [], "auroc": [], "ap": []})
        b["f1"].append(float(ev["best_macro_f1_present"]))
        b["micro"].append(float(ev["best_micro_f1"]))
        b["auroc"].append(float(ev["macro_auroc"]))
        b["ap"].append(float(ev["macro_ap"]))

    lines = [
        "## Main comparison (clean, mean +/- std over seeds)",
        "",
        "| variant | n | best macro-F1 | best micro-F1 | macro-AUROC | macro-AP |",
        "|---------|---|---------------|---------------|-------------|----------|",
    ]
    for v in VARIANT_ORDER:
        if v not in buckets:
            continue
        b = buckets[v]
        n = len(b["f1"])
        f1 = _agg(b["f1"])
        mi = _agg(b["micro"])
        au = _agg(b["auroc"])
        ap = _agg(b["ap"])
        lines.append(
            f"| {v} | {n} | {_fmt(f1[0], f1[1])} | {_fmt(mi[0], mi[1])} "
            f"| {_fmt(au[0], au[1])} | {_fmt(ap[0], ap[1])} |"
        )
    lines.append("")
    return "\n".join(lines)


def fig_main_bar(runs: list[dict[str, Any]], out_path: Path) -> bool:
    """Figure 1: per-variant best macro-F1 (clean) with communication annotation."""
    buckets: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        cfg = r["config"]
        if float(cfg.get("train_label_noise", 0.0) or 0.0) != 0.0:
            continue
        v = _variant_name(cfg)
        ev = r["result"]["eval"]
        upload = r["result"].get("total_upload_bytes")
        b = buckets.setdefault(v, {"f1": [], "mb": []})
        b["f1"].append(float(ev["best_macro_f1_present"]))
        if upload:
            b["mb"].append(float(upload) / 1e6)

    variants = [v for v in VARIANT_ORDER if v in buckets]
    if not variants:
        return False

    means = [_agg(buckets[v]["f1"])[0] for v in variants]
    stds = [_agg(buckets[v]["f1"])[1] for v in variants]
    colors = [VARIANT_COLOR.get(v, "#777") for v in variants]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(variants, means, yerr=stds, capsize=4, color=colors, alpha=0.9)

    # Centralized as dashed upper-bound line, if present.
    if "Centralized" in buckets:
        cval = _agg(buckets["Centralized"]["f1"])[0]
        ax.axhline(cval, ls="--", color="#444", lw=1, alpha=0.7)
        ax.text(
            len(variants) - 0.5, cval, " centralized upper bound",
            va="bottom", ha="right", fontsize=8, color="#444",
        )

    for bar, v in zip(bars, variants):
        mb = buckets[v]["mb"]
        if mb:
            ax.annotate(
                f"{mean(mb):.0f} MB",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=8, color="#333",
            )

    ax.set_ylabel("best macro-F1 (present classes)")
    ax.set_title("Federated adaptation: performance vs upload cost (clean)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def fig_noise_curves(runs: list[dict[str, Any]], out_path: Path, metric: str) -> bool:
    """Figure 2: metric vs label-noise level, one line per variant (mean+/-std)."""
    # variant -> noise -> [values]
    data: dict[str, dict[float, list[float]]] = {}
    for r in runs:
        cfg = r["config"]
        v = _variant_name(cfg)
        noise = float(cfg.get("train_label_noise", 0.0) or 0.0)
        ev = r["result"]["eval"]
        data.setdefault(v, {}).setdefault(noise, []).append(float(ev[metric]))

    # Only plot variants that actually span >1 noise level.
    plottable = {v: d for v, d in data.items() if len(d) > 1}
    if not plottable:
        return False

    fig, ax = plt.subplots(figsize=(7, 5))
    for v in VARIANT_ORDER:
        if v not in plottable:
            continue
        d = plottable[v]
        xs = sorted(d.keys())
        ys = [mean(d[x]) for x in xs]
        es = [pstdev(d[x]) if len(d[x]) > 1 else 0.0 for x in xs]
        ax.errorbar(
            xs, ys, yerr=es, marker="o", capsize=3,
            label=v, color=VARIANT_COLOR.get(v, "#777"),
        )

    ax.set_xlabel("training label-noise rate")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Robustness: performance degradation under label noise")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build paper tables and figures.")
    p.add_argument("summaries", type=Path, nargs="+", help="summary.json path(s)")
    p.add_argument("--out_dir", type=Path, default=Path("docs/results/figures"))
    p.add_argument("--out_md", type=Path, default=Path("docs/results/paper_figures.md"))
    args = p.parse_args(argv)

    runs = _load_runs(list(args.summaries))
    if not runs:
        print("No runs found in the given summaries.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig1 = args.out_dir / "fig1_comm_vs_perf.png"
    fig2_f1 = args.out_dir / "fig2_noise_macro_f1.png"
    fig2_au = args.out_dir / "fig2_noise_macro_auroc.png"

    has_fig1 = fig_main_bar(runs, fig1)
    has_f1 = fig_noise_curves(runs, fig2_f1, "best_macro_f1_present")
    has_au = fig_noise_curves(runs, fig2_au, "macro_auroc")

    parts = [
        "# Paper figures and tables",
        "",
        f"Aggregated from {len(runs)} runs across {len(args.summaries)} summary file(s).",
        "",
        build_main_table(runs),
    ]
    if has_fig1:
        parts += ["## Figure 1: performance vs communication (clean)", "",
                  f"![comm-vs-perf]({fig1.name})", ""]
    if has_f1 or has_au:
        parts += ["## Figure 2: robustness under label noise", ""]
        if has_f1:
            parts += [f"![noise-macro-f1]({fig2_f1.name})", ""]
        if has_au:
            parts += [f"![noise-macro-auroc]({fig2_au.name})", ""]
    else:
        parts += ["> Noise-sweep figures will appear once noise runs are present.", ""]

    args.out_md.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote: {args.out_md}")
    print(
        f"Figures dir: {args.out_dir} "
        f"(fig1={has_fig1}, noise_f1={has_f1}, noise_auroc={has_au})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
