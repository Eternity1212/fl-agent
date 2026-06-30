"""Summarize an agent matrix summary.json into a fedavg/robust/agent comparison.

Groups runs by condition (clean / het02 / het04 / het04_dir / het04_tau*) and
method variant, averages over seeds, and prints whether the agent beats FedAvg.

Usage:
    python3 -m fed_agent.tools.summarize_agent runs/paper_matrix/agent/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

METRICS = ["macro_auroc", "macro_ap", "best_macro_f1_present", "best_micro_f1"]


# Canonical method tokens (order matters for combined names like agentmu+floor).
KNOWN_METHODS = [
    "fedavg",
    "robust",
    "ccr",
    "muonly",
    "agentmu",
    "agent",
    "floor03",
    "floor",
]


def _variant(name: str, cfg: dict[str, Any]) -> tuple[str, str]:
    """Return (condition, method) parsed from run name like 'het04_agentmu_floor_s1'.

    Method may combine multiple tokens (e.g. 'agentmu+floor'); the leftover tokens
    form the condition (e.g. 'het04_dir').
    """
    base = name.rsplit("_s", 1)[0]  # strip trailing _s<seed>
    parts = base.split("_")
    found = [m for m in KNOWN_METHODS if m in parts]
    method = "+".join(found) if found else "?"
    cond = "_".join(p for p in parts if p not in found)
    return cond or "main", method


def _agg(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    m = mean(xs)
    s = pstdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.4f}±{s:.4f}" if len(xs) > 1 else f"{m:.4f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize agent matrix results.")
    p.add_argument("summary", type=Path)
    args = p.parse_args(argv)

    data = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    runs = data.get("runs", [])

    # (condition, method) -> metric -> [values]
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    heldout = set()
    for r in runs:
        name = str(r.get("name"))
        cfg = r.get("config", {})
        res = r.get("result", {})
        ev = res.get("eval", {})
        cond, method = _variant(name, cfg)
        b = buckets.setdefault((cond, method), {k: [] for k in METRICS})
        for k in METRICS:
            if k in ev and ev[k] is not None:
                b[k].append(float(ev[k]))
        if res.get("agent_probe_heldout"):
            heldout.add(name)

    conds = sorted({c for c, _ in buckets})
    print(f"\n=== Agent matrix comparison ({len(runs)} runs) ===")
    print(f"held-out probe used by: {len(heldout)} agent runs\n")
    for cond in conds:
        print(f"-- condition: {cond} --")
        header = f"{'method':<14} | " + " | ".join(f"{m:<16}" for m in METRICS)
        print(header)
        print("-" * len(header))
        # stable display order: baselines first, then agent variants, then others
        order = ["fedavg", "robust", "ccr", "muonly", "agent", "agentmu",
                 "agent+floor03", "agent+floor", "agentmu+floor"]
        methods_here = [m for (c, m) in buckets if c == cond]
        ordered = [m for m in order if m in methods_here]
        ordered += sorted(m for m in methods_here if m not in order)
        fed_auroc = None
        for method in ordered:
            b = buckets.get((cond, method))
            if not b:
                continue
            row = f"{method:<14} | " + " | ".join(f"{_agg(b[m]):<16}" for m in METRICS)
            print(row)
            if method == "fedavg" and b["macro_auroc"]:
                fed_auroc = mean(b["macro_auroc"])
        for variant in ordered:
            if variant == "fedavg":
                continue
            ab = buckets.get((cond, variant))
            if ab and ab["macro_auroc"] and fed_auroc is not None:
                delta = mean(ab["macro_auroc"]) - fed_auroc
                if delta > 0.005:
                    verdict = "WINS"
                elif delta >= -0.005:
                    verdict = "~tie"
                else:
                    verdict = "loses"
                print(f">>> {variant} vs fedavg macro_auroc delta = {delta:+.4f}  [{verdict}]")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
