from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_rfmid_torch_dataset_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from PIL import Image

    from fed_agent.data.rfmid_torch import RFMiDTorchDataset

    img_dir = tmp_path / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    for sid in ["s0", "s1"]:
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(img_dir / f"{sid}.png")

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("ImageID,A\ns0,0\ns1,1\n", encoding="utf-8")

    ds = RFMiDTorchDataset(labels_csv=csv_path, images_dir=img_dir, image_size=(8, 8))
    x, y, meta = ds[0]
    assert tuple(x.shape) == (3, 8, 8)
    assert y.shape == (1,)
    assert meta["image_id"] == "s0"
