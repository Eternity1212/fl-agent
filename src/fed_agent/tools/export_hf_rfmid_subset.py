"""Export a small real RFMiD subset from the Hugging Face mirror to local layout."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

HF_REPO_ID = "ctmedtech/RFMID"
TRAIN_LABELS = "Training_Set/Training_Set/RFMiD_Training_Labels.csv"
TRAIN_IMAGE_PREFIX = "Training_Set/Training_Set/Training"


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


def export_hf_rfmid_subset(*, out_dir: Path, max_samples: int = 96) -> tuple[Path, Path]:
    """Download labels + selected images and write ``labels.csv`` / ``images`` folder."""

    from huggingface_hub import hf_hub_download

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=TRAIN_LABELS)
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
            filename=f"{TRAIN_IMAGE_PREFIX}/{image_id}.png",
        )
        shutil.copyfile(src, images_dir / f"{image_id}.png")

    return labels_out, images_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export a small real RFMiD subset from HF mirror.")
    p.add_argument("--out_dir", type=Path, default=Path("data/raw/rfmid_hf_subset"))
    p.add_argument("--max_samples", type=int, default=96)
    args = p.parse_args(argv)

    labels_csv, images_dir = export_hf_rfmid_subset(
        out_dir=args.out_dir,
        max_samples=int(args.max_samples),
    )
    print(f"Wrote labels_csv: {labels_csv}")
    print(f"Wrote images_dir: {images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
