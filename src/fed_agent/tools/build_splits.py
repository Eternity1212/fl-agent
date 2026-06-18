"""Build federated client splits from an RFMiD-style labels CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from fed_agent.data.rfmid import load_rfmid_label_table
from fed_agent.splits.partition import (
    build_dirichlet_split_primary,
    build_domain_hash_split,
    build_iid_split,
    split_payload,
    write_split_json,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build IID / Dirichlet / domain-hash split JSON files.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("configs/splits/generated"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_clients", type=int, default=8)
    p.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[0.1, 0.5, 1.0],
        help="Dirichlet alpha values (primary-label skew).",
    )
    args = p.parse_args(argv)

    image_ids, label_names, y = load_rfmid_label_table(args.labels_csv)
    y_lookup = {sid: y[i] for i, sid in enumerate(image_ids)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.labels_csv.stem

    clients_iid = build_iid_split(image_ids, args.n_clients, args.seed)
    payload_iid = split_payload(
        split="iid",
        n_clients=args.n_clients,
        seed=args.seed,
        alpha=None,
        label_names=label_names,
        clients=clients_iid,
        y_lookup=y_lookup,
    )
    write_split_json(out_dir / f"{stem}__iid_K{args.n_clients}_S{args.seed}.json", payload_iid)

    clients_dom = build_domain_hash_split(image_ids, args.n_clients, args.seed)
    payload_dom = split_payload(
        split="domain_hash",
        n_clients=args.n_clients,
        seed=args.seed,
        alpha=None,
        label_names=label_names,
        clients=clients_dom,
        y_lookup=y_lookup,
    )
    write_split_json(
        out_dir / f"{stem}__domain_hash_K{args.n_clients}_S{args.seed}.json",
        payload_dom,
    )

    for alpha in args.alphas:
        clients_dir = build_dirichlet_split_primary(
            image_ids,
            y,
            n_clients=args.n_clients,
            alpha=float(alpha),
            seed=args.seed,
        )
        payload_dir = split_payload(
            split="dirichlet_primary",
            n_clients=args.n_clients,
            seed=args.seed,
            alpha=float(alpha),
            label_names=label_names,
            clients=clients_dir,
            y_lookup=y_lookup,
        )
        a_tag = str(alpha).replace(".", "p")
        write_split_json(
            out_dir / f"{stem}__dirichlet_a{a_tag}_K{args.n_clients}_S{args.seed}.json",
            payload_dir,
        )

    print(f"Wrote split JSON files under: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
