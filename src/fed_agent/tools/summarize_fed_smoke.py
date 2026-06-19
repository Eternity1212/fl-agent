"""Print a short human-readable summary of ``run_fed_smoke`` JSON metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize run_fed_smoke metrics JSON.")
    p.add_argument("json_path", type=Path)
    args = p.parse_args(argv)

    payload = json.loads(Path(args.json_path).read_text(encoding="utf-8"))

    print("=== Federated smoke summary ===")
    if "noise_protocol" in payload:
        npi = payload["noise_protocol"]
        print("noise_protocol.path:", npi.get("path"))
        print("noise_protocol.symmetric_flip_p_flip:", npi.get("symmetric_flip_p_flip"))
    print("total_upload_bytes:", payload.get("total_upload_bytes"))
    losses = payload.get("mean_train_loss_clients") or []
    if losses:
        print("mean_train_loss_clients (per round):", ", ".join(f"{x:.6f}" for x in losses))
        print("mean_train_loss_clients (final):", f"{losses[-1]:.6f}")
    comm = payload.get("comm_bytes_upload_per_round") or []
    if comm:
        print("comm_bytes_upload_per_round:", ", ".join(str(int(x)) for x in comm))
    keys = payload.get("final_state_dict_keys") or []
    print("final_state_dict_keys (count):", len(keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
