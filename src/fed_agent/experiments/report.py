"""Turn federated smoke summaries into comparison / ablation tables."""

from __future__ import annotations

import csv
import io
from typing import Any


def extract_run_row(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten one ``run`` entry from :func:`run_synthetic_suite` output."""

    spec = payload["spec"]
    metrics = payload["metrics"]
    losses = metrics.get("clean_eval_loss") or metrics.get("mean_train_loss_clients") or []
    noise_p = None
    if "noise_protocol" in metrics:
        noise_p = metrics["noise_protocol"].get("symmetric_flip_p_flip")
    return {
        "name": spec["name"],
        "fedprox_mu": float(spec["fedprox_mu"]),
        "noise_p_flip": noise_p,
        "train_loss_final": float(losses[-1]) if losses else float("nan"),
        "train_loss_start": float(losses[0]) if losses else float("nan"),
        "total_upload_bytes": int(metrics.get("total_upload_bytes", 0)),
        "rounds": int(spec["rounds"]),
    }


def _rel_pct(new: float, base: float) -> str:
    if base == 0 or base != base:
        return "n/a"
    return f"{100.0 * (new - base) / base:+.2f}%"


def build_ablation_markdown(rows: list[dict[str, Any]], *, baseline_name: str) -> str:
    """Markdown table + short ablation bullets vs ``baseline_name``."""

    by_name = {r["name"]: r for r in rows}
    if baseline_name not in by_name:
        raise ValueError(f"baseline {baseline_name!r} not in rows")
    b = by_name[baseline_name]

    lines = [
        "# Synthetic ablation & method comparison",
        "",
        f"Baseline run: **`{baseline_name}`** (final loss = {b['train_loss_final']:.6f}).",
        "",
        "## Summary table",
        "",
        "| run | mu_FedProx | p_noise | L_final | dL vs base | upload_bytes |",
        "|-----|-----------|---------|---------|------------|--------------|",
    ]
    for r in sorted(rows, key=lambda x: x["name"]):
        dloss = _rel_pct(r["train_loss_final"], b["train_loss_final"])
        npv = r["noise_p_flip"]
        np_s = "none" if npv is None else str(npv)
        lines.append(
            f"| {r['name']} | {r['fedprox_mu']} | {np_s} | "
            f"{r['train_loss_final']:.6f} | {dloss} | "
            f"{r['total_upload_bytes']} |",
        )

    # Paired ablations (same noise, toggle FedProx)
    lines.extend(["", "## Ablation: FedProx (hold noise fixed)", ""])
    for noise_label, subset in [
        ("no YAML (no injected label noise)", [r for r in rows if r["noise_p_flip"] is None]),
        ("p_flip = 0.1", [r for r in rows if r["noise_p_flip"] == 0.1]),
    ]:
        if len(subset) < 2:
            continue
        subset = sorted(subset, key=lambda x: x["fedprox_mu"])
        a0, a1 = subset[0], subset[-1]
        lines.append(
            f"- **{noise_label}**: compare `{a0['name']}` (mu={a0['fedprox_mu']}) vs "
            f"`{a1['name']}` (mu={a1['fedprox_mu']}); "
            f"loss {a0['train_loss_final']:.6f} -> {a1['train_loss_final']:.6f}",
        )

    lines.extend(["", "## Ablation: label noise (hold FedProx off)", ""])
    clean_mu0 = [r for r in rows if r["fedprox_mu"] == 0.0]
    def sort_key(x: dict[str, Any]) -> tuple[bool, float]:
        return (x["noise_p_flip"] is not None, float(x["noise_p_flip"] or 0))

    for r in sorted(clean_mu0, key=sort_key):
        lines.append(
            f"- `{r['name']}`: noise_p_flip={r['noise_p_flip']!r}, "
            f"final loss={r['train_loss_final']:.6f}",
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- **Upload bytes** usually match across runs here "
            "(same model shape and rounds); differences appear mainly in final loss.",
            "- Synthetic 4-sample setup: for **RFMiD** scale-up, reuse the same spec names with "
            "`run_fed_smoke` paths — see `docs/EXPERIMENTS.md`.",
            "",
        ],
    )
    return "\n".join(lines)


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    """CSV string for all scalar columns in ``rows``."""

    if not rows:
        return ""
    buf = io.StringIO()
    keys = sorted(rows[0].keys())
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in keys})
    return buf.getvalue()
