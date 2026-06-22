"""Run a tiny federated smoke simulation (Torch required)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fed_agent.fed.simulator import FedSmokeConfig, run_multilabel_fed_smoke


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a tiny FedAvg/FedProx smoke simulation.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--images_dir", type=Path, required=True)
    p.add_argument("--split_json", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--fedprox_mu", type=float, default=0.0)
    p.add_argument("--image_size", type=int, nargs=2, default=[32, 32])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default="mlp", choices=["mlp", "tiny_cnn"])
    p.add_argument("--loss", type=str, default="bce", choices=["bce", "balanced_bce"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--noise_protocol_yaml", type=Path, default=None)
    p.add_argument("--label_noise_seed", type=int, default=0)
    p.add_argument("--eval_labels_csv", type=Path, default=None)
    p.add_argument("--eval_images_dir", type=Path, default=None)
    p.add_argument("--out_json", type=Path, default=None)
    args = p.parse_args(argv)

    cfg = FedSmokeConfig(
        rounds=int(args.rounds),
        local_epochs=int(args.local_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        fedprox_mu=float(args.fedprox_mu),
        device=str(args.device),
        seed=int(args.seed),
        model=str(args.model),
        loss=str(args.loss),
        threshold=float(args.threshold),
    )

    metrics = run_multilabel_fed_smoke(
        labels_csv=args.labels_csv,
        images_dir=args.images_dir,
        split_json=args.split_json,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        cfg=cfg,
        noise_protocol_yaml=args.noise_protocol_yaml,
        label_noise_seed=int(args.label_noise_seed),
        eval_labels_csv=args.eval_labels_csv,
        eval_images_dir=args.eval_images_dir,
    )
    text = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    print(text)
    if args.out_json is not None:
        Path(args.out_json).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
