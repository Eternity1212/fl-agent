from __future__ import annotations

from pathlib import Path
from typing import Any

from fed_agent.data.rfmid import load_rfmid_label_table
from fed_agent.splits.partition import (
    build_dirichlet_split_primary,
    build_domain_hash_split,
    build_iid_split,
    read_split_json,
    split_payload,
    write_split_json,
)


def _tiny_csv(tmp_path: Path) -> Path:
    p = tmp_path / "labels.csv"
    rows = [
        "ImageID,A,B",
        "s0,1,0",
        "s1,1,0",
        "s2,0,1",
        "s3,0,1",
        "s4,1,1",
        "s5,0,0",
    ]
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_partition_no_duplicates_and_covers_all(tmp_path: Path) -> None:
    csv_path = _tiny_csv(tmp_path)
    image_ids, label_names, y = load_rfmid_label_table(csv_path)
    y_lookup = {sid: y[i] for i, sid in enumerate(image_ids)}

    for split_name, clients in [
        ("iid", build_iid_split(image_ids, n_clients=3, seed=0)),
        ("dom", build_domain_hash_split(image_ids, n_clients=3, seed=0)),
        ("dir", build_dirichlet_split_primary(image_ids, y, n_clients=3, alpha=0.5, seed=0)),
    ]:
        flat: list[str] = []
        for ids in clients.values():
            flat.extend(ids)
        assert len(flat) == len(set(flat)), split_name
        assert set(flat) == set(image_ids), split_name

        payload = split_payload(
            split=split_name,
            n_clients=3,
            seed=0,
            alpha=0.5 if split_name == "dir" else None,
            label_names=label_names,
            clients=clients,
            y_lookup=y_lookup,
        )
        out = tmp_path / f"{split_name}.json"
        write_split_json(out, payload)
        loaded: dict[str, Any] = read_split_json(out)
        assert loaded["schema"].startswith("fl_agent.split")
        assert loaded["n_clients"] == 3

