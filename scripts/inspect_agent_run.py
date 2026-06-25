#!/usr/bin/env python3
"""单文件 agent 结果速读器 (纯标准库, 可直接拷到 GPU 机器上运行)。

用途: 把 agent stage1 / full 产出的 json 一眼看懂——
  1. 三大指标 (macro_auroc / 默认阈值 micro_f1 / best_micro_f1)
  2. 聚合权重轨迹: 脏客户端是否被压低 (机制是否生效的直接证据)
  3. 探针分 / 自适应 mu (若启用)
  4. 多文件时自动给出 agent - fedavg 的 macro_auroc 差值与 verdict

用法:
  python inspect_agent_run.py runs/paper_matrix/agent_stage1/het04_agent_s0.json
  python inspect_agent_run.py runs/paper_matrix/agent_stage1/*.json
  python inspect_agent_run.py runs/paper_matrix/agent_stage1/summary.json

无需任何依赖 (只用 Python 标准库)。
"""
from __future__ import annotations

import json
import sys
from statistics import mean


def _iter_runs(obj):
    """从单个 run / summary(list 或 dict) 中统一抽出 run 列表。"""
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


def _load_runs(paths):
    runs = []
    for p in paths:
        try:
            with open(p) as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"!! 跳过 {p}: {e}")
            continue
        runs.extend(_iter_runs(obj))
    return runs


def _bar(frac, width=24):
    n = int(round(frac * width))
    return "#" * n + "-" * (width - n)


def _weight_fractions(weight_history):
    """每轮权重 dict -> 每客户端"平均占比"。"""
    if not weight_history:
        return {}
    per_client_frac: dict[str, list[float]] = {}
    for rnd in weight_history:
        total = sum(float(v) for v in rnd.values()) or 1.0
        for ck, w in rnd.items():
            per_client_frac.setdefault(ck, []).append(float(w) / total)
    return {ck: mean(v) for ck, v in per_client_frac.items()}


def _variant(name: str):
    """从 run name 拆出 (condition, method)，与 summarize_agent 一致。"""
    base = name.rsplit("_s", 1)[0]
    parts = base.split("_")
    known = {"fedavg", "robust", "agent", "agentmu", "muonly"}
    method = next((p for p in parts if p in known), "?")
    cond = "_".join(p for p in parts if p not in known) or "(base)"
    return cond, method


def inspect_one(run: dict):
    name = run.get("name", "?")
    res = run.get("result", {})
    cfg = res.get("config", run.get("config", {}))
    ev = res.get("eval", {})
    agg = cfg.get("agent_aggregation", "?")
    adaptive_mu = bool(cfg.get("agent_adaptive_mu", False))
    noisy = set(str(c) for c in res.get("agent_noisy_clients", cfg.get("agent_noisy_clients", [])))
    rounds = len(res.get("train_loss", []))

    print("=" * 64)
    print(f"{name}   [method={cfg.get('method','?')} | agg={agg} | adaptive_mu={adaptive_mu}]")
    print(f"  noisy clients: {sorted(noisy)}   rounds: {rounds}   "
          f"probe_heldout: {res.get('agent_probe_heldout')}")
    print("-" * 64)

    print("EVAL:")
    print(f"  macro_auroc        : {ev.get('macro_auroc', float('nan')):.4f}")
    print(f"  macro_ap           : {ev.get('macro_ap', float('nan')):.4f}")
    print(f"  micro_f1@{ev.get('threshold', 0.5)}       : {ev.get('micro_f1', float('nan')):.4f}")
    print(f"  best_micro_f1      : {ev.get('best_micro_f1', float('nan')):.4f} "
          f"@thr={ev.get('best_threshold')}")
    print(f"  best_macro_f1_pres : {ev.get('best_macro_f1_present', float('nan')):.4f}")
    print("-" * 64)

    fracs = _weight_fractions(res.get("agent_weight_history", []))
    if fracs:
        print("AGGREGATION WEIGHTS (mean fraction across rounds, 想看到 noisy < clean):")
        clean_v, noisy_v = [], []
        for ck in sorted(fracs, key=lambda x: int(x)):
            tag = "NOISY" if ck in noisy else "clean"
            (noisy_v if ck in noisy else clean_v).append(fracs[ck])
            print(f"  client {ck} ({tag:>5}): {fracs[ck]:.3f}  {_bar(fracs[ck])}")
        if clean_v and noisy_v:
            ca, na = mean(clean_v), mean(noisy_v)
            ratio = na / ca if ca else float("nan")
            if agg == "size":
                verdict = "(对照组, 不分化)"
            elif ratio < 0.85:
                verdict = "[脏客户端被压低 ✓ 机制生效]"
            elif ratio > 1.15:
                verdict = "[!! 反了: 脏客户端权重更高, 需排查]"
            else:
                verdict = "[~未分化: 探针没区分出脏/干净, 考虑调小 agent_tau]"
            print(f"  -> clean avg {ca:.3f} | noisy avg {na:.3f} | "
                  f"ratio noisy/clean = {ratio:.2f}  {verdict}")
        print("-" * 64)

    probe = res.get("agent_probe_history", [])
    if probe:
        last = probe[-1]
        print("PROBE SCORES (last round, 越高越干净):")
        for ck in sorted(last, key=lambda x: int(x)):
            tag = "NOISY" if ck in noisy else "clean"
            print(f"  client {ck} ({tag:>5}): {float(last[ck]):+.4f}")
        print("-" * 64)

    if adaptive_mu:
        mu_hist = res.get("agent_mu_history", [])
        mu_frac = {}
        for rnd in mu_hist:
            for ck, m in rnd.items():
                mu_frac.setdefault(ck, []).append(float(m))
        if mu_frac:
            print("ADAPTIVE MU (mean across rounds, 脏客户端应更大):")
            for ck in sorted(mu_frac, key=lambda x: int(x)):
                tag = "NOISY" if ck in noisy else "clean"
                print(f"  client {ck} ({tag:>5}): {mean(mu_frac[ck]):.4f}")
            print("-" * 64)
    print()


def compare(runs):
    """多 run 时: 按 condition 分组, 给出 agent/robust 相对 fedavg 的 macro_auroc 差。"""
    print("#" * 64)
    print("# 汇总对比 (按 condition 分组, 以 fedavg 为基准)")
    print("#" * 64)
    buckets: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        name = r.get("name", "?")
        auroc = r.get("result", {}).get("eval", {}).get("macro_auroc")
        if auroc is None:
            continue
        cond, method = _variant(name)
        buckets.setdefault((cond, method), []).append(float(auroc))

    conds = sorted({c for c, _ in buckets})
    for cond in conds:
        print(f"\n-- condition: {cond} --")
        base = buckets.get((cond, "fedavg"))
        base_m = mean(base) if base else None
        for method in ["fedavg", "robust", "agent", "agentmu", "muonly"]:
            vals = buckets.get((cond, method))
            if not vals:
                continue
            m = mean(vals)
            line = f"  {method:8s} macro_auroc = {m:.4f}  (n={len(vals)})"
            if base_m is not None and method != "fedavg":
                d = m - base_m
                if d > 0.005:
                    tag = "WINS"
                elif d >= -0.005:
                    tag = "~tie"
                else:
                    tag = "loses"
                line += f"   delta vs fedavg = {d:+.4f}  [{tag}]"
            print(line)
    print()


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    runs = _load_runs(argv)
    if not runs:
        print("没有可解析的 run。")
        return 1
    for r in runs:
        inspect_one(r)
    if len(runs) > 1:
        compare(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
