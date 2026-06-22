"""Export RFMiD from the Hugging Face mirror to the local RFMiD layout."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

HF_REPO_ID = "ctmedtech/RFMID"
TRAIN_LABELS = "Training_Set/Training_Set/RFMiD_Training_Labels.csv"
TRAIN_IMAGE_PREFIX = "Training_Set/Training_Set/Training"
VAL_LABELS = "Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv"
VAL_IMAGE_PREFIX = "Evaluation_Set/Evaluation_Set/Validation"
TEST_LABELS = "Test_Set/Test_Set/RFMiD_Testing_Labels.csv"
TEST_IMAGE_PREFIX = "Test_Set/Test_Set/Test"

SPLIT_META = {
    "train": (TRAIN_LABELS, TRAIN_IMAGE_PREFIX),
    "validation": (VAL_LABELS, VAL_IMAGE_PREFIX),
    "test": (TEST_LABELS, TEST_IMAGE_PREFIX),
}


def _clean_header(name: str) -> str:
    return str(name).replace("\ufeff", "").strip()


def _select_diverse_rows(
    rows: list[dict[str, str]],
    *,
    id_key: str,
    label_keys: list[str],
    max_samples: int,
) -> list[dict[str, str]]:
    disease_keys = [k for k in label_keys if k.lower() != "disease_risk"]
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        primary = "all_zero"
        for key in disease_keys:
            if str(row.get(key, "0")).strip() == "1":
                primary = key
                break
        buckets.setdefault(primary, []).append(row)

    selected: list[dict[str, str]] = []
    keys = sorted(buckets.keys())
    offset = 0
    while len(selected) < int(max_samples):
        added = False
        for key in keys:
            bucket = buckets[key]
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if len(selected) >= int(max_samples):
                    break
        if not added:
            break
        offset += 1

    return sorted(selected, key=lambda r: int(str(r[id_key]).strip()))


def _split_paths(split: str) -> tuple[str, str]:
    if split not in SPLIT_META:
        raise ValueError("split must be one of: train, validation, test")
    return SPLIT_META[split]


def export_hf_rfmid_subset(
    *,
    out_dir: Path,
    max_samples: int | None = 96,
    split: str = "train",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Download labels + selected images and write ``labels.csv`` / ``images`` folder."""

    from huggingface_hub import hf_hub_download

    labels_name, image_prefix = _split_paths(split)

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=labels_name)
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("RFMiD training labels CSV has no header")
        raw_fields = [_clean_header(x) for x in reader.fieldnames]
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({_clean_header(k): str(v).strip() for k, v in row.items() if k is not None})

    id_key = "ID" if "ID" in raw_fields else "ImageID"
    if id_key not in raw_fields:
        raise ValueError(f"Could not find ID column in RFMiD CSV: {raw_fields[:5]}")
    label_keys = [k for k in raw_fields if k != id_key]
    if max_samples is None or int(max_samples) <= 0:
        selected = sorted(rows, key=lambda r: int(str(r[id_key]).strip()))
    else:
        selected = _select_diverse_rows(
            rows,
            id_key=id_key,
            label_keys=label_keys,
            max_samples=int(max_samples),
        )

    labels_out = out_dir / "labels.csv"
    with labels_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ImageID", *label_keys])
        writer.writeheader()
        for row in selected:
            writer.writerow({"ImageID": row[id_key], **{k: row.get(k, "0") for k in label_keys}})

    for row in selected:
        image_id = str(row[id_key]).strip()
        src = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=f"{image_prefix}/{image_id}.png",
        )
        dst = images_dir / f"{image_id}.png"
        if dst.is_file() and not overwrite:
            continue
        shutil.copyfile(src, dst)

    return labels_out, images_dir


def export_hf_rfmid_all(
    *,
    out_dir: Path,
    max_samples: int | None = None,
    overwrite: bool = False,
) -> dict[str, tuple[Path, Path]]:
    """Export train/validation/test into ``out_dir/<split>/labels.csv`` + ``images``."""

    root = Path(out_dir)
    out: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "validation", "test"):
        out[split] = export_hf_rfmid_subset(
            out_dir=root / split,
            max_samples=max_samples,
            split=split,
            overwrite=overwrite,
        )
    return out


def validate_rfmid_layout(root: Path) -> dict[str, int]:
    """Return row counts for a local full RFMiD layout, raising if files are missing."""

    from fed_agent.data.rfmid import load_rfmid_label_table

    root = Path(root)
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        labels_csv = root / split / "labels.csv"
        images_dir = root / split / "images"
        if not labels_csv.is_file():
            raise FileNotFoundError(f"Missing {split} labels: {labels_csv}")
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} images dir: {images_dir}")
        image_ids, _labels, _y = load_rfmid_label_table(labels_csv)
        missing = [sid for sid in image_ids if not (images_dir / f"{sid}.png").is_file()]
        if missing:
            msg = f"{split}: missing {len(missing)} image files, first={missing[0]}"
            raise FileNotFoundError(msg)
        counts[split] = len(image_ids)
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export RFMiD from HF mirror into local layout.")
    p.add_argument("--out_dir", type=Path, default=Path("data/raw/rfmid_hf_subset"))
    p.add_argument(
        "--max_samples",
        type=int,
        default=96,
        help="Per-split sample cap. Use 0 for full split.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test", "all"],
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate", action="store_true")
    args = p.parse_args(argv)

    max_samples = None if int(args.max_samples) <= 0 else int(args.max_samples)
    if args.split == "all":
        payload = export_hf_rfmid_all(
            out_dir=args.out_dir,
            max_samples=max_samples,
            overwrite=bool(args.overwrite),
        )
        for split, (labels_csv, images_dir) in payload.items():
            print(f"{split}: labels_csv={labels_csv} images_dir={images_dir}")
        if args.validate:
            print("counts:", validate_rfmid_layout(args.out_dir))
    else:
        labels_csv, images_dir = export_hf_rfmid_subset(
            out_dir=args.out_dir,
            max_samples=max_samples,
            split=str(args.split),
            overwrite=bool(args.overwrite),
        )
        print(f"Wrote labels_csv: {labels_csv}")
        print(f"Wrote images_dir: {images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
