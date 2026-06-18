from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fed_agent.data.rfmid import RFMiDLocalDataset


def _write_tiny_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[:, :, 0] = 255
    Image.fromarray(arr).save(path)


def test_rfmid_local_dataset_roundtrip(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    _write_tiny_png(img_dir / "i1.png")
    _write_tiny_png(img_dir / "i2.png")

    csv_path = tmp_path / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ImageID", "A", "B"])
        w.writerow(["i1", 0, 1])
        w.writerow(["i2", 1, 0])

    ds = RFMiDLocalDataset(labels_csv=csv_path, images_dir=img_dir)
    assert len(ds) == 2
    assert ds.label_names == ("A", "B")

    s0 = ds[0]
    assert s0["image_id"] == "i1"
    assert tuple(s0["label_names"]) == ("A", "B")
    assert s0["label"].tolist() == [0.0, 1.0]
    assert s0["image"].size == (8, 8)


def test_rfmid_missing_image_raises(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ImageID", "A"])
        w.writerow(["missing", 0])

    ds = RFMiDLocalDataset(labels_csv=csv_path, images_dir=img_dir)
    with pytest.raises(FileNotFoundError):
        _ = ds[0]
