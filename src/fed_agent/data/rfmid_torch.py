from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from fed_agent.data.rfmid import RFMiDLocalDataset
from fed_agent.noise.protocol import apply_symmetric_label_noise


class RFMiDTorchDataset(torch.utils.data.Dataset):
    """Torch bridge around :class:`RFMiDLocalDataset` with fixed resize + tensor labels."""

    def __init__(
        self,
        labels_csv: Path,
        images_dir: Path,
        *,
        image_size: tuple[int, int] = (224, 224),
        label_noise_p_flip: float | None = None,
        label_noise_seed: int = 0,
    ) -> None:
        self._base = RFMiDLocalDataset(labels_csv=labels_csv, images_dir=images_dir)
        self._size = (int(image_size[0]), int(image_size[1]))
        self._label_noise_p_flip = (
            None if label_noise_p_flip is None else float(label_noise_p_flip)
        )
        self._label_noise_seed = int(label_noise_seed)

    def __len__(self) -> int:
        return len(self._base)

    @property
    def image_ids(self) -> tuple[str, ...]:
        return self._base.image_ids

    def indices_for_ids(self, ids: list[str]) -> list[int]:
        id_to_idx = {sid: i for i, sid in enumerate(self.image_ids)}
        out: list[int] = []
        for sid in ids:
            if sid in id_to_idx:
                out.append(id_to_idx[sid])
        return out

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

        y_np = np.asarray(sample["label"], dtype=np.float32)
        if self._label_noise_p_flip is not None and self._label_noise_p_flip > 0.0:
            rng = np.random.default_rng(
                (int(self._label_noise_seed) * 1_000_003 + int(index) * 97_482_607)
                & 0xFFFFFFFF,
            )
            y_np = apply_symmetric_label_noise(
                y_np.reshape(1, -1),
                rng=rng,
                p_flip=float(self._label_noise_p_flip),
            )[0]
        y = torch.from_numpy(y_np)

        meta = {"image_id": sample["image_id"], "label_names": sample["label_names"]}
        return x, y, meta
