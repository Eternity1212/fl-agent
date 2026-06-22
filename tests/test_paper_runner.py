from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from fed_agent.data.rfmid import load_rfmid_label_table
from fed_agent.splits.partition import split_payload, write_split_json
from fed_agent.train.paper_runner import PaperRunConfig, run_paper_experiment


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for sid in ["1", "2", "3", "4"]:
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_dir / f"{sid}.png")
    labels = tmp_path / "labels.csv"
    labels.write_text("ImageID,A,B\n1,1,0\n2,1,0\n3,0,1\n4,0,1\n", encoding="utf-8")
    ids, label_names, y = load_rfmid_label_table(labels)
    payload = split_payload(
        split="iid",
        n_clients=2,
        seed=0,
        alpha=None,
        label_names=label_names,
        clients={"0": ["1", "2"], "1": ["3", "4"]},
        y_lookup={sid: y[i] for i, sid in enumerate(ids)},
    )
    split_json = tmp_path / "split.json"
    write_split_json(split_json, payload)
    return labels, img_dir, split_json


def test_run_paper_experiment_centralized(tmp_path: Path) -> None:
    labels, img_dir, _split = _fixture(tmp_path)
    cfg = PaperRunConfig(
        method="centralized",
        backbone="mlp",
        epochs=1,
        batch_size=2,
        image_size=(8, 8),
        device="cpu",
    )
    out = run_paper_experiment(
        train_labels_csv=labels,
        train_images_dir=img_dir,
        eval_labels_csv=labels,
        eval_images_dir=img_dir,
        cfg=cfg,
    )
    assert "best_macro_f1_present" in out["eval"]


def test_run_paper_experiment_fedavg(tmp_path: Path) -> None:
    labels, img_dir, split_json = _fixture(tmp_path)
    cfg = PaperRunConfig(
        method="fedavg",
        backbone="mlp",
        rounds=1,
        local_epochs=1,
        batch_size=2,
        image_size=(8, 8),
        device="cpu",
    )
    out = run_paper_experiment(
        train_labels_csv=labels,
        train_images_dir=img_dir,
        eval_labels_csv=labels,
        eval_images_dir=img_dir,
        cfg=cfg,
        split_json=split_json,
    )
    assert out["total_upload_bytes"] > 0
