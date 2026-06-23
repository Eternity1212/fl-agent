#!/usr/bin/env bash
# Medium-scale RETFound+LoRA verification run (seed 0) on a GPU machine.
#
# Goal: quickly check the relative-advantage story (noise robustness + LoRA
# communication efficiency) BEFORE committing to the full multi-seed matrix.
# Estimated wall time on one A100: ~1-1.5 hours.
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   ./scripts/run_paper_medium.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export MODE="${MODE:-retfound}"
export PAPER_MATRIX_YAML="configs/paper_matrix_retfound_medium.yaml"

exec ./run.sh
