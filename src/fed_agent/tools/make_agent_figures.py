"""把 agent 实验结果渲染成论文级图表 (matplotlib)。

产出 (默认写到 docs/figures/):
  1. fig_agent_weights.png  —— 聚合权重逐轮轨迹 (干净 vs 脏客户端),
     直观展示 agent 自适应压低脏客户端。这是方法创新最有说服力的一张图。
  2. fig_agent_auroc.png    —— 各条件下 fedavg / robust / agent(/agentmu) 的
     macro_auroc 柱状图 + 跨 seed 误差棒。
  3. agent_figures_summary.md —— 配套数字表 (mean±std)。

用法:
  python -m fed_agent.tools.make_agent_figures \
    --results_dir runs/paper_matrix/agent_stage1 \
    --out_dir docs/figures

依赖: matplotlib (pip install -e '.[paper]')。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_ORDER = ["fedavg", "robust", "agent", "agentmu", "muonly"]
METHOD_COLOR = {
    "fedavg": "#888888",
    "robust": "#4C72B0",
    "agent": "#C44E52",
    "agentmu": "#DD8452",
    "muonly": "#55A868",
}


def _iter_runs(obj):
    if isinstance(obj, dict) and "result" in obj:
        yield obj
        return
    if isinstance(obj, list):
        for x in obj:
            yield from _iter_runs(x)
        return
    if isinstance(obj, dict):
        if isinstance(obj.get("runs"), list):
            for x in obj["runs"]:
                yield from _iter_runs(x)
            return
        for v in obj.values():
            if isinstance(v, dict) and "result" in v:
                yield v


def _load(results_dir: str):
    runs = []
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    # Prefer individual run files; if only summary.json exists, use it.
    indiv = [f for f in files if os.path.basename(f) != "summary.json"]
    use = indiv or files
    for f in use:
        try:
            obj = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        runs.extend(_iter_runs(obj))
    return runs


def _variant(name: str):
    base = name.rsplit("_s", 1)[0]
    parts = base.split("_")
    known = set(METHOD_ORDER)
    method = next((p for p in parts if p in known), "?")
    cond = "_".join(p for p in parts if p not in known) or "base"
    return cond, method


def _weight_fracs_per_round(run):
    """返回 {client: [frac_r0, frac_r1, ...]}。"""
    out = defaultdict(list)
    for rnd in run["result"].get("agent_weight_history", []):
        total = sum(float(v) for v in rnd.values()) or 1.0
        for ck, w in rnd.items():
            out[ck].append(float(w) / total)
    return out


def fig_weights(runs, out_path):
    """对 agent runs 取 seed 平均, 画各客户端权重占比逐轮曲线。"""
    agent_runs = [
        r for r in runs
        if r["result"].get("config", {}).get("agent_aggregation") == "agent"
        and r["result"].get("agent_weight_history")
    ]
    if not agent_runs:
        print("[weights] 没有 agent run (agg=agent 且有权重历史), 跳过。")
        return False

    noisy = set(str(c) for c in agent_runs[0]["result"].get("agent_noisy_clients", []))
    # 跨 seed 平均 (按最短轮数对齐)
    per_client_rounds = defaultdict(list)  # client -> list of per-seed [fracs]
    for r in agent_runs:
        fr = _weight_fracs_per_round(r)
        for ck, seq in fr.items():
            per_client_rounds[ck].append(seq)
    if not per_client_rounds:
        return False
    min_rounds = min(len(s) for seqs in per_client_rounds.values() for s in seqs)

    plt.figure(figsize=(7, 4.5))
    for ck in sorted(per_client_rounds, key=lambda x: int(x)):
        seqs = [s[:min_rounds] for s in per_client_rounds[ck]]
        avg = [mean(vals) for vals in zip(*seqs)]
        is_noisy = ck in noisy
        plt.plot(
            range(1, min_rounds + 1), avg,
            label=f"client {ck} ({'noisy' if is_noisy else 'clean'})",
            color="#C44E52" if is_noisy else "#4C72B0",
            linestyle="--" if is_noisy else "-",
            linewidth=2.0, alpha=0.9,
        )
    n_clients = len(per_client_rounds)
    plt.axhline(1.0 / n_clients, color="gray", linestyle=":", linewidth=1,
                label=f"uniform (1/{n_clients})")
    plt.xlabel("Federated round")
    plt.ylabel("Aggregation weight fraction")
    plt.title("Agent adaptively suppresses noisy clients")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[weights] -> {out_path}")
    return True


def _auroc_buckets(runs):
    buckets = defaultdict(list)  # (cond, method) -> [auroc,...]
    for r in runs:
        ev = r["result"].get("eval", {})
        a = ev.get("macro_auroc")
        if a is None:
            continue
        cond, method = _variant(r.get("name", "?"))
        buckets[(cond, method)].append(float(a))
    return buckets


def fig_auroc(runs, out_path):
    buckets = _auroc_buckets(runs)
    if not buckets:
        print("[auroc] 无可用数据, 跳过。")
        return False
    conds = sorted({c for c, _ in buckets})
    methods = [m for m in METHOD_ORDER if any((c, m) in buckets for c in conds)]
    if not methods:
        return False

    x = range(len(conds))
    n_m = len(methods)
    width = 0.8 / n_m
    plt.figure(figsize=(max(6, 1.6 * len(conds) + 2), 4.5))
    for i, m in enumerate(methods):
        means, stds = [], []
        for c in conds:
            vals = buckets.get((c, m), [])
            means.append(mean(vals) if vals else 0.0)
            stds.append(pstdev(vals) if len(vals) > 1 else 0.0)
        offs = [xi + (i - (n_m - 1) / 2) * width for xi in x]
        plt.bar(offs, means, width=width, yerr=stds, capsize=3,
                label=m, color=METHOD_COLOR.get(m, None), alpha=0.9)
    plt.xticks(list(x), conds, rotation=0)
    plt.ylabel("macro-AUROC")
    plt.title("Macro-AUROC by method (mean ± std over seeds)")
    lo = min(min(v) for v in buckets.values())
    plt.ylim(max(0.0, lo - 0.05), 1.0)
    plt.legend(fontsize=9)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[auroc] -> {out_path}")
    return True


def write_summary_md(runs, out_path):
    buckets = _auroc_buckets(runs)
    # 默认阈值 micro_f1 也汇总 (展示校准恢复)
    f1b = defaultdict(list)
    for r in runs:
        ev = r["result"].get("eval", {})
        if ev.get("micro_f1") is not None:
            cond, method = _variant(r.get("name", "?"))
            f1b[(cond, method)].append(float(ev["micro_f1"]))
    conds = sorted({c for c, _ in buckets})
    lines = ["# Agent figures summary\n",
             "macro-AUROC 与默认阈值 micro-F1 (mean±std over seeds)。\n"]
    for cond in conds:
        lines.append(f"\n## condition: {cond}\n")
        lines.append("| method | macro_auroc | micro_f1@0.5 | n |")
        lines.append("|---|---|---|---|")
        base = buckets.get((cond, "fedavg"))
        base_m = mean(base) if base else None
        for m in METHOD_ORDER:
            vals = buckets.get((cond, m))
            if not vals:
                continue
            am = mean(vals)
            asd = pstdev(vals) if len(vals) > 1 else 0.0
            fv = f1b.get((cond, m), [])
            fm = mean(fv) if fv else float("nan")
            fsd = pstdev(fv) if len(fv) > 1 else 0.0
            delta = ""
            if base_m is not None and m != "fedavg":
                d = am - base_m
                tag = "WINS" if d > 0.005 else ("~tie" if d >= -0.005 else "loses")
                delta = f" ({d:+.4f} {tag})"
            lines.append(
                f"| {m} | {am:.4f}±{asd:.4f}{delta} | {fm:.4f}±{fsd:.4f} | {len(vals)} |"
            )
    open(out_path, "w").write("\n".join(lines) + "\n")
    print(f"[summary] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="含 *.json 的结果目录")
    ap.add_argument("--out_dir", default="docs/figures")
    args = ap.parse_args()

    runs = _load(args.results_dir)
    if not runs:
        print(f"在 {args.results_dir} 没找到可解析的 run。")
        return 1
    print(f"加载 {len(runs)} 个 run。")
    os.makedirs(args.out_dir, exist_ok=True)
    fig_weights(runs, os.path.join(args.out_dir, "fig_agent_weights.png"))
    fig_auroc(runs, os.path.join(args.out_dir, "fig_agent_auroc.png"))
    write_summary_md(runs, os.path.join(args.out_dir, "agent_figures_summary.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
