"""Smoke test for the adaptive orchestration agent.

Setup: RFMiD subset (head-12 labels), 4 IID clients, TinyCNN @ 64px (fast, CPU).
Clients 2 and 3 get heavy per-client symmetric label noise (flip prob 0.4);
clients 0 and 1 stay clean.

We run the *same* federated training twice under identical noise:
  * aggregation = "size"  -> FedAvg baseline (equal/size weight to all clients)
  * aggregation = "agent" -> probe-driven adaptive weights

If the agent works, it should (a) down-weight the noisy clients 2/3 and
(b) match or beat FedAvg on the clean validation set.
"""

from __future__ import annotations

from pathlib import Path

from fed_agent.train.paper_runner import PaperRunConfig, run_paper_experiment

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data/raw/rfmid_sub/train/head12_labels.csv"
TRAIN_IMG = ROOT / "data/raw/rfmid_full/train/images"
VAL_CSV = ROOT / "data/raw/rfmid_sub/validation/head12_labels.csv"
VAL_IMG = ROOT / "data/raw/rfmid_full/validation/images"
SPLIT = ROOT / "configs/splits/generated/rfmid_sub_head12_iid_K4_S0.json"

COMMON = dict(
    method="agent_fed",
    backbone="tiny_cnn",
    seed=0,
    rounds=10,
    local_epochs=4,
    batch_size=16,
    lr=1e-3,
    loss="balanced_bce",
    image_size=(64, 64),
    device="cpu",
    require_retfound=False,
    agent_noisy_clients=(2, 3),
    agent_client_noise=0.6,
    agent_tau=0.03,
)


def _run(aggregation: str, *, noise: float | None = None) -> dict:
    common = dict(COMMON)
    if noise is not None:
        common["agent_client_noise"] = noise
    cfg = PaperRunConfig(agent_aggregation=aggregation, **common)
    return run_paper_experiment(
        train_labels_csv=TRAIN_CSV,
        train_images_dir=TRAIN_IMG,
        eval_labels_csv=VAL_CSV,
        eval_images_dir=VAL_IMG,
        cfg=cfg,
        split_json=SPLIT,
    )


def _fmt_eval(ev: dict) -> str:
    keys = ["macro_auroc", "macro_ap", "best_macro_f1_present", "best_micro_f1"]
    return "  ".join(f"{k}={float(ev[k]):.4f}" for k in keys)


def main() -> int:
    print("=== Ceiling: FedAvg, ALL clients clean (noise=0) ===")
    ceil = _run("size", noise=0.0)
    print(_fmt_eval(ceil["eval"]))

    print("\n=== FedAvg baseline (size weights, clients 2/3 noisy@0.6) ===")
    base = _run("size")
    print(_fmt_eval(base["eval"]))

    print("\n=== Agent (probe-adaptive weights, same noise) ===")
    agent = _run("agent")
    print(_fmt_eval(agent["eval"]))

    print("\n=== Agent probe scores (-val BCE; higher=better) ===")
    for i, s in enumerate(agent["agent_probe_history"]):
        row = "  ".join(f"c{k}={s[k]:+.3f}" for k in sorted(s, key=int))
        print(f"round {i:2d}: {row}")

    print("\n=== Agent per-round weights (clients 0,1 clean / 2,3 noisy) ===")
    for i, w in enumerate(agent["agent_weight_history"]):
        row = "  ".join(f"c{k}={w[k]:.3f}" for k in sorted(w, key=int))
        print(f"round {i:2d}: {row}")

    a = float(agent["eval"]["macro_auroc"])
    b = float(base["eval"]["macro_auroc"])
    c = float(ceil["eval"]["macro_auroc"])
    print(f"\nmacro_auroc: clean-ceiling={c:.4f}  fedavg-noisy={b:.4f}  agent-noisy={a:.4f}")
    print(f"  noise cost to FedAvg  = {c - b:+.4f}  (how much room exists)")
    print(f"  agent recovery vs FedAvg = {a - b:+.4f}")
    last = agent["agent_weight_history"][-1]
    clean = last["0"] + last["1"]
    dirty = last["2"] + last["3"]
    print(f"final weight mass: clean(0,1)={clean:.3f}  noisy(2,3)={dirty:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
