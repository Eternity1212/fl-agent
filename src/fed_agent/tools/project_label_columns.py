"""Project an RFMiD-style labels CSV to a selected subset of label columns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def project_label_columns(*, labels_csv: Path, out_csv: Path, labels: list[str]) -> Path:
    labels_csv = Path(labels_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with labels_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("labels CSV has no header")
        if "ImageID" not in reader.fieldnames:
            raise ValueError("labels CSV must contain ImageID")
        missing = [x for x in labels if x not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing label columns: {missing}")
        rows = list(reader)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ImageID", *labels])
        writer.writeheader()
        for row in rows:
            writer.writerow({"ImageID": row["ImageID"], **{k: row.get(k, "0") for k in labels}})

    return out_csv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Project RFMiD labels CSV to selected labels.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--labels", type=str, nargs="+", required=True)
    args = p.parse_args(argv)

    out = project_label_columns(
        labels_csv=args.labels_csv,
        out_csv=args.out_csv,
        labels=list(args.labels),
    )
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
