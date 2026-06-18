from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image


def _norm_col(name: str) -> str:
    return name.strip().lower().replace(" ", "")


class RFMiDLocalDataset:
    """Load RFMiD-style multi-label tables from a local folder (no network I/O).

    Expected layout (flexible filenames):

    - ``labels_csv``: CSV with an ``ImageID`` column (case-insensitive) and one
      binary column per label (0/1).
    - ``images_dir``: directory containing ``{ImageID}.png`` / ``.jpg`` / ``.jpeg``.

    This is intentionally minimal for **federated simulation bootstrap**; you
    can point it at the official RFMiD unzip layout by passing explicit paths.
    """

    def __init__(self, labels_csv: Path, images_dir: Path) -> None:
        self.labels_csv = Path(labels_csv)
        self.images_dir = Path(images_dir)
        if not self.labels_csv.is_file():
            raise FileNotFoundError(f"Missing labels CSV: {self.labels_csv}")
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing images dir: {self.images_dir}")

        image_ids: list[str] = []
        label_matrix: list[list[float]] = []
        label_names: list[str]

        with self.labels_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row")

            fields = list(reader.fieldnames)
            norm_map = {_norm_col(h): h for h in fields}
            if "imageid" not in norm_map:
                raise ValueError("CSV must include an ImageID column")
            id_key = norm_map["imageid"]

            label_keys = [h for h in fields if h != id_key]
            label_names = label_keys

            for row in reader:
                iid = str(row[id_key]).strip()
                if not iid:
                    continue
                vec: list[float] = []
                for key in label_keys:
                    raw = row.get(key, "")
                    if raw is None or str(raw).strip() == "":
                        vec.append(0.0)
                    else:
                        vec.append(float(int(float(str(raw).strip()))))
                image_ids.append(iid)
                label_matrix.append(vec)

        if not image_ids:
            raise ValueError("No rows parsed from labels CSV")

        self.image_ids = tuple(image_ids)
        self.label_names = tuple(label_names)
        self.labels = np.asarray(label_matrix, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _resolve_image_path(self, image_id: str) -> Path:
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            p = self.images_dir / f"{image_id}{ext}"
            if p.is_file():
                return p
        raise FileNotFoundError(f"No image file for id={image_id} under {self.images_dir}")

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        image_id = self.image_ids[index]
        path = self._resolve_image_path(image_id)
        image = Image.open(path).convert("RGB")
        y = self.labels[index].copy()
        return {"image_id": image_id, "image": image, "label": y, "label_names": self.label_names}

    def iter_label_names(self) -> Iterator[str]:
        yield from self.label_names

    @staticmethod
    def suggest_label_columns(fieldnames: Sequence[str]) -> tuple[str, list[str]]:
        """Utility for CLI / notebooks: split ImageID vs label columns."""
        fields = list(fieldnames)
        norm_map = {_norm_col(h): h for h in fields}
        if "imageid" not in norm_map:
            raise ValueError("CSV must include an ImageID column")
        id_key = norm_map["imageid"]
        label_keys = [h for h in fields if h != id_key]
        return id_key, label_keys
