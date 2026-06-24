"""Export the ODIR fundus dataset into the RFMiD-compatible local layout.

The goal is a *second* multi-label fundus dataset that plugs into the exact same
pipeline (build_splits, RFMiDTorchDataset, paper_runner) with zero code changes.
It writes, under ``out_dir``:

    <out_dir>/train/labels.csv        (ImageID + one binary column per label)
    <out_dir>/train/images/<id>.png
    <out_dir>/validation/...
    <out_dir>/test/...

Two sources are supported:

  * ``--source hf`` (default): the public ``bumbledeep/odir`` Hugging Face mirror.
    NOTE: this mirror is single-label per image (one positive among the 8 ODIR
    classes), so each CSV row has exactly one positive. It still exercises the
    full multi-label code path (BCE / per-label F1 / AUROC), and supports noise
    and non-IID splits. No credentials required.

  * ``--source csv``: a user-provided ODIR ``full_df.csv`` (Kaggle
    ``andrewmvd/ocular-disease-recognition-odir5k``) with true 8-way multi-label
    columns + a local images directory. Use this for genuine multi-label rows.

Examples:
    # quick smoke test (10 samples)
    python3 -m fed_agent.tools.export_odir --source hf --out_dir data/raw/odir \
        --max_samples 10

    # full HF export
    python3 -m fed_agent.tools.export_odir --source hf --out_dir data/raw/odir
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

# Canonical ODIR 8-class order (used as label columns).
ODIR_LETTERS = ["N", "D", "G", "C", "A", "H", "M", "O"]
# Map common ODIR label strings (from the HF mirror) to canonical letters.
ODIR_NAME_TO_LETTER = {
    "normal": "N",
    "diabetes": "D",
    "diabetic": "D",
    "glaucoma": "G",
    "cataract": "C",
    "amd": "A",
    "age related macular degeneration": "A",
    "hypertension": "H",
    "hypertensive": "H",
    "myopia": "M",
    "pathological myopia": "M",
    "others": "O",
    "other": "O",
    "abnormality": "O",
    "other diseases/abnormalities": "O",
}


def _letter_for(label: str) -> str:
    key = str(label).strip().lower()
    if key in ODIR_NAME_TO_LETTER:
        return ODIR_NAME_TO_LETTER[key]
    # Fall back: first char upper-cased, so unknown labels still get a column.
    return key[:1].upper() if key else "O"


def _write_split(
    rows: list[tuple[str, dict[str, int], Any]],
    split_dir: Path,
    label_cols: list[str],
) -> None:
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = split_dir / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageID", *label_cols])
        for image_id, label_vec, image in rows:
            image.convert("RGB").save(images_dir / f"{image_id}.png")
            writer.writerow([image_id, *[label_vec.get(c, 0) for c in label_cols]])


def _split_indices(
    n: int, val_frac: float, test_frac: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = idx[:n_test]
    val = idx[n_test : n_test + n_val]
    train = idx[n_test + n_val :]
    return train, val, test


def export_from_hf(
    out_dir: Path,
    *,
    max_samples: int = 0,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> dict[str, int]:
    from datasets import load_dataset

    ds = load_dataset("bumbledeep/odir", split="train")
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    records: list[tuple[str, dict[str, int], Any]] = []
    for i, ex in enumerate(ds):
        letter = _letter_for(ex.get("label", "O"))
        if letter not in ODIR_LETTERS:
            letter = "O"
        vec = {c: 0 for c in ODIR_LETTERS}
        vec[letter] = 1
        records.append((str(i), vec, ex["image"]))

    tr, va, te = _split_indices(len(records), val_frac, test_frac, seed)
    out_dir = Path(out_dir)
    _write_split([records[i] for i in tr], out_dir / "train", ODIR_LETTERS)
    _write_split([records[i] for i in va], out_dir / "validation", ODIR_LETTERS)
    _write_split([records[i] for i in te], out_dir / "test", ODIR_LETTERS)
    return {"train": len(tr), "validation": len(va), "test": len(te)}


def export_from_csv(
    out_dir: Path,
    csv_path: Path,
    images_dir: Path,
    *,
    id_col: str = "filename",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> dict[str, int]:
    from PIL import Image

    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        label_cols = [c for c in fields if c in ODIR_LETTERS]
        if not label_cols:
            raise ValueError(
                f"No ODIR label columns {ODIR_LETTERS} found in {csv_path} "
                f"(columns: {fields})"
            )
        if id_col not in fields:
            raise ValueError(f"id_col '{id_col}' not in CSV columns: {fields}")

        records: list[tuple[str, dict[str, int], Any]] = []
        for i, row in enumerate(reader):
            fname = str(row[id_col]).strip()
            src = images_dir / fname
            if not src.is_file():
                continue
            vec = {c: int(float(row.get(c, 0) or 0)) for c in label_cols}
            records.append((str(i), vec, Image.open(src)))

    if not records:
        raise ValueError("No usable rows (check images_dir and id_col).")
    tr, va, te = _split_indices(len(records), val_frac, test_frac, seed)
    out_dir = Path(out_dir)
    _write_split([records[i] for i in tr], out_dir / "train", label_cols)
    _write_split([records[i] for i in va], out_dir / "validation", label_cols)
    _write_split([records[i] for i in te], out_dir / "test", label_cols)
    return {"train": len(tr), "validation": len(va), "test": len(te)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export ODIR into RFMiD-compatible layout.")
    p.add_argument("--source", choices=["hf", "csv"], default="hf")
    p.add_argument("--out_dir", type=Path, default=Path("data/raw/odir"))
    p.add_argument("--max_samples", type=int, default=0, help="0 = all (hf only)")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv_path", type=Path, default=None, help="ODIR full_df.csv (csv source)")
    p.add_argument("--images_dir", type=Path, default=None, help="images dir (csv source)")
    args = p.parse_args(argv)

    if args.source == "hf":
        counts = export_from_hf(
            args.out_dir,
            max_samples=args.max_samples,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
    else:
        if not args.csv_path or not args.images_dir:
            p.error("--source csv requires --csv_path and --images_dir")
        counts = export_from_csv(
            args.out_dir,
            args.csv_path,
            args.images_dir,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )

    print(f"Exported ODIR to {args.out_dir}: {counts}")
    print(f"Labels CSV: {args.out_dir}/train/labels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
