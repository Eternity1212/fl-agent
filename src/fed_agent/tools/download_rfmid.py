from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optional RFMiD download / probe via Hugging Face.",
    )
    parser.add_argument(
        "--smoke-stream",
        action="store_true",
        help=(
            "Load one streaming sample from ctmedtech/RFMID "
            '(requires `pip install -e ".[data]"`).'
        ),
    )
    args = parser.parse_args(argv)

    if not args.smoke_stream:
        print("No action specified. Typical local workflow:")
        print("  1) Download RFMiD (see docs/DATA_CARD.md).")
        print("  2) Unzip under data/raw/rfmid/ ...")
        print("  3) Point RFMiDLocalDataset at labels_csv + images_dir.")
        print()
        print("Optional online smoke (small network download):")
        print("  python -m fed_agent.tools.download_rfmid --smoke-stream")
        return 0

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        print("Missing dependency. Install with: pip install -e \".[data]\"", file=sys.stderr)
        raise SystemExit(2) from exc

    ds = load_dataset("ctmedtech/RFMID", split="train", streaming=True)
    sample = next(iter(ds))
    print("OK: received one streaming training sample.")
    print("Keys:", sorted(sample.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
