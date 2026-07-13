#!/usr/bin/env python3
"""Generate the 5 core paper figures from the verified 3-seed summary numbers.

Self-contained: the aggregated (mean ± std) results are embedded below, taken
from docs/RESULTS_NARRATIVE.md / docs/PAPER_DRAFT.md (RFMiD supp / ODIR / scale
3-seed CSVs + agent_stage1 het04-50% anchors). This lets us render the
summary-based figures WITHOUT the raw GPU JSONs on this machine.

Figures produced (-> docs/figures/paper/):
  fig2_failure_boundary.png  corruption x heterogeneity heatmap, per method
  fig3_auroc_f1_scatter.png  AUROC vs best-micro-F1, all setting-method points
  fig4_rank_flip.png         RFMiD het02 vs ODIR het04 rank reversal
  fig5_k_scaling.png         K=4 -> K=8 dilution/recovery
  fig6_conclusion_matrix.png method x condition qualitative verdict grid

Figures requiring RAW per-round / per-sample data (training curves, client
weight distributions, per-label F1 boxplots) must be produced on the GPU box
from runs/paper_matrix/*/*.json; see make_agent_figures.py.

Usage:
  python3 scripts/make_core_figures.py [--out_dir docs/figures/paper]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Consistent colour per method across all figures.
COLORS = {
    "FedAvg": "#7f7f7f",
    "Robust-FedProx": "#8c564b",
    "median": "#d62728",
    "trimmed": "#ff7f0e",
    "FedA3I": "#9467bd",
    "CCR": "#1f77b4",
    "Agent": "#2ca02c",
    "Agent+μ": "#17becf",
    "Agent+floor": "#bcbd22",
}

# --- Fig 2: failure-boundary (RFMiD het04, K=4), macro-AUROC ------------------
# rows = heterogeneity (IID / non-IID), cols = corruption (25% / 50%)
FAILURE = {
    #            IID-25%  IID-50%  nonIID-25% nonIID-50%
    "median":  [0.8096, 0.5051, 0.5978, 0.4904],
    "trimmed": [0.8047, 0.6937, 0.7100, 0.5487],
    "FedAvg":  [0.7665, 0.7380, 0.8068, 0.6475],
    "Agent":   [0.8113, 0.8043, 0.8191, 0.6690],
}

# --- Fig 3: AUROC vs best-micro-F1 scatter (all conditions with both) ---------
# (method, condition_label, AUROC, bestmicroF1)
SCATTER = [
    ("Robust-FedProx", "RFMiD het02", 0.7069, 0.424),
    ("FedA3I", "RFMiD het02", 0.7492, 0.500),
    ("FedAvg", "RFMiD het02", 0.7630, 0.453),
    ("CCR", "RFMiD het02", 0.7869, 0.679),
    ("Agent", "RFMiD het02", 0.8005, 0.644),
    ("Robust-FedProx", "ODIR het04", 0.7785, 0.245),
    ("FedAvg", "ODIR het04", 0.7817, 0.242),
    ("CCR", "ODIR het04", 0.7874, 0.578),
    ("Agent", "ODIR het04", 0.7918, 0.577),
    ("FedA3I", "ODIR het04", 0.8008, 0.463),
    ("FedAvg", "RFMiD het04_dir", 0.6475, 0.076),
    ("CCR", "RFMiD het04_dir", 0.6457, 0.416),
    ("FedA3I", "RFMiD het04_dir", 0.6723, 0.147),
    ("Agent", "RFMiD het04_dir", 0.6690, 0.398),
    ("Agent+μ", "RFMiD het04_dir", 0.6999, 0.433),
    ("Agent+floor", "RFMiD het04_dir", 0.6791, 0.088),
    ("median", "RFMiD het04_dir", 0.4904, 0.078),
    ("FedAvg", "ODIR het04_dir", 0.6641, 0.400),
    ("CCR", "ODIR het04_dir", 0.6822, 0.509),
    ("Agent", "ODIR het04_dir", 0.6997, 0.496),
    ("FedA3I", "ODIR het04_dir", 0.7086, 0.362),
    ("Agent+floor", "ODIR het04_dir", 0.7173, 0.422),
]

# --- Fig 4: rank flip RFMiD het02 vs ODIR het04 (macro-AUROC) -----------------
RANKFLIP = {
    #            RFMiD het02, ODIR het04
    "Robust-FedProx": [0.7069, 0.7785],
    "FedA3I":         [0.7492, 0.8008],
    "FedAvg":         [0.7630, 0.7817],
    "CCR":            [0.7869, 0.7874],
    "Agent":          [0.8005, 0.7918],
}

# --- Fig 5: K=4 -> K=8 (het04, 50% noisy), macro-AUROC ------------------------
# mean per K; err = std (None where single-seed).
KSCALE = {
    "IID": {
        "FedAvg": {"K": [4, 8], "auroc": [0.7380, 0.6895], "err": [0.0007, 0.0054]},
        "Agent":  {"K": [4, 8], "auroc": [0.8043, 0.8040], "err": [0.0096, 0.0092]},
        "median": {"K": [4, 8], "auroc": [0.5051, 0.5092], "err": [0.0498, None]},
    },
    "non-IID": {
        "FedAvg": {"K": [4, 8], "auroc": [0.6475, 0.7555], "err": [0.0546, 0.0636]},
        "Agent":  {"K": [4, 8], "auroc": [0.6690, 0.7884], "err": [0.1040, 0.0455]},
        "median": {"K": [4, 8], "auroc": [0.4904, 0.4053], "err": [0.0716, None]},
    },
}

# --- Fig 6: conclusion matrix (method x condition -> qualitative verdict) -----
COND_COLS = [
    "IID high\ncorruption",
    "non-IID moderate\ncorruption",
    "severe dir\nshift",
    "K=8\ndilution",
    "metric\nconsistency",
]
VERDICT = {
    "FedAvg":  ["dilutes", "brittle", "F1 collapse", "dilutes", "split"],
    "median":  ["collapse", "collapse", "collapse", "collapse", "collapse"],
    "trimmed": ["brittle", "brittle", "collapse", "n/a", "split"],
    "FedA3I":  ["rank-flip", "rank-flip", "F1 collapse", "no recovery", "split"],
    "CCR":     ["stable", "stable*", "F1 recover*", "recovers", "split"],
    "Agent":   ["stable", "stable*", "F1 recover*", "recovers", "split"],
}
VERDICT_COLOR = {
    "stable": "#2ca02c", "stable*": "#98df8a", "recovers": "#2ca02c",
    "F1 recover*": "#98df8a", "rank-flip": "#1f77b4", "dilutes": "#ff7f0e",
    "brittle": "#ffbb78", "no recovery": "#ff7f0e", "split": "#c7c7c7",
    "F1 collapse": "#d62728", "collapse": "#d62728", "n/a": "#eeeeee",
}


def fig2_failure_boundary(out: Path) -> None:
    methods = list(FAILURE)
    fig, axes = plt.subplots(1, len(methods), figsize=(13, 3.4))
    vmin, vmax = 0.45, 0.83
    for ax, m in zip(axes, methods):
        grid = np.array(FAILURE[m]).reshape(2, 2)  # [IID/nonIID][25/50]
        im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")
        ax.set_title(m, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["25%", "50%"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["IID", "non-IID"])
        ax.set_xlabel("corruption fraction")
        for i in range(2):
            for j in range(2):
                v = grid[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("client heterogeneity")
    fig.suptitle("Fig. 2  Failure boundary: macro-AUROC over corruption × heterogeneity "
                 "(RFMiD het04, K=4)", fontsize=12)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("macro-AUROC")
    fig.savefig(out / "fig2_failure_boundary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3_scatter(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    markers = {"RFMiD het02": "o", "ODIR het04": "s",
               "RFMiD het04_dir": "^", "ODIR het04_dir": "D"}
    seen_m: set[str] = set()
    for method, cond, auroc, f1 in SCATTER:
        c = COLORS.get(method, "#333333")
        ax.scatter(auroc, f1, c=c, marker=markers[cond], s=90,
                   edgecolors="black", linewidths=0.5, alpha=0.9,
                   label=method if method not in seen_m else None)
        seen_m.add(method)
    ax.set_xlabel("macro-AUROC (ranking ability)")
    ax.set_ylabel("best-micro-F1 (usable prediction)")
    ax.set_title("Fig. 3  Metric dissociation: high AUROC ≠ high F1\n"
                 "(colour = method, marker = condition)")
    ax.grid(True, alpha=0.3)
    # annotate the flagship dissociation: FedA3I @ ODIR het04
    ax.annotate("FedA3I @ ODIR het04\n(top AUROC, mid F1)",
                xy=(0.8008, 0.463), xytext=(0.72, 0.30),
                arrowprops=dict(arrowstyle="->", color="#9467bd"), fontsize=8)
    ax.annotate("FedAvg @ RFMiD het04_dir\n(ok AUROC, F1 collapse)",
                xy=(0.6475, 0.076), xytext=(0.66, 0.20),
                arrowprops=dict(arrowstyle="->", color="#7f7f7f"), fontsize=8)
    # method legend + marker legend
    ax.legend(loc="upper left", fontsize=8, title="method", framealpha=0.9)
    from matplotlib.lines import Line2D
    mleg = [Line2D([0], [0], marker=mk, color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=9, label=cond)
            for cond, mk in markers.items()]
    leg2 = ax.legend(handles=mleg, loc="lower right", fontsize=8, title="condition")
    ax.add_artist(leg2)
    fig.savefig(out / "fig3_auroc_f1_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig4_rank_flip(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    x = [0, 1]
    for m, vals in RANKFLIP.items():
        c = COLORS.get(m, "#333333")
        lw = 3.0 if m == "FedA3I" else 1.6
        ax.plot(x, vals, "-o", color=c, linewidth=lw, markersize=8, label=m,
                zorder=3 if m == "FedA3I" else 2)
        ax.text(-0.03, vals[0], m, ha="right", va="center", fontsize=9, color=c)
        ax.text(1.03, vals[1], m, ha="left", va="center", fontsize=9, color=c)
    ax.set_xticks(x); ax.set_xticklabels(["RFMiD het02", "ODIR het04"], fontsize=11)
    ax.set_ylabel("macro-AUROC")
    ax.set_xlim(-0.5, 1.5)
    ax.set_title("Fig. 4  Cross-dataset rank reversal\n"
                 "(FedA3I: bottom on RFMiD → top on ODIR)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out / "fig4_rank_flip.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig5_k_scaling(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (cond, data) in zip(axes, KSCALE.items()):
        for m, d in data.items():
            c = COLORS.get(m, "#333333")
            err = [e if e is not None else 0 for e in d["err"]]
            ax.errorbar(d["K"], d["auroc"], yerr=err, fmt="-o", color=c,
                        capsize=4, markersize=8, linewidth=2, label=m)
        ax.set_title(f"{cond} (het04, 50% noisy)")
        ax.set_xlabel("number of clients K")
        ax.set_xticks([4, 8])
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7)
    axes[0].set_ylabel("macro-AUROC")
    axes[0].legend(fontsize=9)
    fig.suptitle("Fig. 5  Dilution vs. recovery from K=4 to K=8", fontsize=12)
    fig.savefig(out / "fig5_k_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig6_conclusion_matrix(out: Path) -> None:
    methods = list(VERDICT)
    nrow, ncol = len(methods), len(COND_COLS)
    fig, ax = plt.subplots(figsize=(10, 0.72 * nrow + 1.6))
    for i, m in enumerate(methods):
        for j, verdict in enumerate(VERDICT[m]):
            color = VERDICT_COLOR.get(verdict, "#dddddd")
            ax.add_patch(plt.Rectangle((j, nrow - 1 - i), 1, 1,
                                       facecolor=color, edgecolor="white", lw=2))
            ax.text(j + 0.5, nrow - 1 - i + 0.5, verdict, ha="center",
                    va="center", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, ncol); ax.set_ylim(0, nrow)
    ax.set_xticks([j + 0.5 for j in range(ncol)])
    ax.set_xticklabels(COND_COLS, fontsize=9)
    ax.set_yticks([nrow - 1 - i + 0.5 for i in range(nrow)])
    ax.set_yticklabels(methods, fontsize=10, fontweight="bold")
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Fig. 6  Conclusion matrix (qualitative verdict; * = high variance)",
                 pad=28, fontsize=12)
    fig.savefig(out / "fig6_conclusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="docs/figures/paper")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig2_failure_boundary(out)
    fig3_scatter(out)
    fig4_rank_flip(out)
    fig5_k_scaling(out)
    fig6_conclusion_matrix(out)
    print(f"Wrote 5 core figures + conclusion matrix to {out}/")
    for p in sorted(out.glob("fig*.png")):
        print("  ", p)


if __name__ == "__main__":
    main()
