from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from fed_agent.data.rfmid import RFMiDLocalDataset


class RFMiDTorchDataset(torch.utils.data.Dataset):
    """Torch bridge around :class:`RFMiDLocalDataset` with fixed resize + tensor labels."""

    def __init__(
        self,
        labels_csv: Path,
        images_dir: Path,
        *,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        self._base = RFMiDLocalDataset(labels_csv=labels_csv, images_dir=images_dir)
        self._size = (int(image_size[0]), int(image_size[1]))

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        sample = self._base[index]
        pil: Image.Image = sample["image"]  # type: ignore[assignment]
        try:
            resample = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
        except AttributeError:  # Pillow < 9
            resample = Image.BILINEAR  # type: ignore[attr-defined]
        pil = pil.resize(self._size, resample=resample)

        arr = np.asarray(pil, dtype=np.float32) / 255.0  # HWC
        chw = np.transpose(arr, (2, 0, 1))
        x = torch.from_numpy(chw)

        y_np = sample["label"]  # type: ignore[assignment]
        y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))

        meta = {"image_id": sample["image_id"], "label_names": sample["label_names"]}
        return x, y, meta
