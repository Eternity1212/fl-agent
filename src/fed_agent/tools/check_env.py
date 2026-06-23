"""Preflight environment check before running the RETFound + LoRA paper matrix.

Run this on the GPU machine first. It verifies GPU, optional deps, the Hugging
Face token, and gated RETFound checkpoint access, then prints a clear summary.
"""

from __future__ import annotations

import argparse
import os


def _check_torch_gpu() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:
        return False, f"torch not importable: {exc}"

    if not torch.cuda.is_available():
        return False, "torch installed but CUDA GPU not available (device=cpu)"

    names = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        names.append(f"{props.name} ({vram_gb:.1f} GB)")
    return True, "CUDA GPUs: " + "; ".join(names)


def _check_timm() -> tuple[bool, str]:
    try:
        import timm

        return True, f"timm {timm.__version__}"
    except Exception as exc:
        return False, f"timm not importable: {exc}"


def _check_token() -> tuple[bool, str]:
    from fed_agent.models.retfound_lora import resolve_hf_token

    token = resolve_hf_token()
    if not token:
        return False, "no HF token in env (set HF_TOKEN)"
    return True, f"HF token present (***{token[-4:]})"


def _check_retfound() -> tuple[bool, str]:
    from fed_agent.models.retfound_lora import (
        RETFOUND_CKPT_FILENAME,
        RETFOUND_HF_REPO,
        check_retfound_access,
    )

    ok, msg = check_retfound_access()
    if ok:
        return True, f"RETFound checkpoint downloadable: {msg}"
    target = f"{RETFOUND_HF_REPO}/{RETFOUND_CKPT_FILENAME}"
    return False, f"RETFound gated access failed for {target}: {msg}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Preflight env check for RETFound paper runs.")
    p.add_argument(
        "--skip-retfound",
        action="store_true",
        help="Skip the gated RETFound download probe (still checks GPU/token).",
    )
    args = p.parse_args(argv)

    checks: list[tuple[str, bool, str]] = []
    ok_gpu, msg_gpu = _check_torch_gpu()
    checks.append(("GPU", ok_gpu, msg_gpu))
    ok_timm, msg_timm = _check_timm()
    checks.append(("timm", ok_timm, msg_timm))
    ok_token, msg_token = _check_token()
    # A local checkpoint (RETFOUND_CKPT_PATH) removes the need for an HF token.
    local_ckpt = os.environ.get("RETFOUND_CKPT_PATH")
    if not ok_token and local_ckpt:
        ok_token = True
        msg_token = f"no HF token, but RETFOUND_CKPT_PATH set ({local_ckpt})"
    checks.append(("HF token", ok_token, msg_token))

    ok_retfound = True
    if not args.skip_retfound:
        ok_retfound, msg_retfound = _check_retfound()
        checks.append(("RETFound access", ok_retfound, msg_retfound))

    print("Preflight environment check")
    print("=" * 60)
    for name, ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
    print("=" * 60)

    blocking = not (ok_timm and ok_token and ok_retfound)
    if not ok_gpu:
        print("WARNING: no CUDA GPU; RETFound ViT-Large will be very slow on CPU.")
    if blocking:
        print("RESULT: NOT READY. Resolve the FAIL items above before the paper run.")
        return 1
    print("RESULT: READY. You can run scripts/run_paper_gpu.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
