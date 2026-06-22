#!/usr/bin/env bash
# One-command RETFound + LoRA paper run for a GPU machine.
#
# Prerequisites (see docs/RETFOUND_GPU_RUN.md):
#   1. You have requested + been granted access to the gated RETFound repo:
#      https://huggingface.co/YukunZhou/RETFound_mae_natureCFP
#   2. export HF_TOKEN=hf_xxx   (a read token from https://huggingface.co/settings/tokens)
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   ./scripts/run_paper_gpu.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="${RFMID_DATA_DIR:-data/raw/rfmid_full}"
MATRIX_YAML="${PAPER_MATRIX_YAML:-configs/paper_matrix.yaml}"
OUT_DIR="${PAPER_OUT_DIR:-runs/paper_matrix/retfound}"

echo "==> [1/6] Installing dependencies (data, torch, paper)"
python3 -m pip install -e ".[data,torch,paper]"

echo "==> [2/6] Preflight environment check (GPU + token + gated access)"
python3 -m fed_agent.tools.check_env

echo "==> [3/6] Downloading + validating full RFMiD (skips files already present)"
python3 -m fed_agent.tools.export_hf_rfmid_subset \
  --split all --out_dir "${DATA_DIR}" --max_samples 0 --validate

echo "==> [4/6] Building IID / Dirichlet / domain-hash splits for seeds 0,1,2"
for seed in 0 1 2; do
  python3 -m fed_agent.tools.build_splits \
    --labels_csv "${DATA_DIR}/train/labels.csv" \
    --out_dir configs/splits/generated \
    --seed "${seed}" \
    --n_clients 4 \
    --alphas 0.1 0.5 1.0
done

echo "==> [5/6] Running paper matrix: ${MATRIX_YAML}"
python3 -m fed_agent.tools.run_paper_matrix \
  --matrix_yaml "${MATRIX_YAML}" \
  --out_dir "${OUT_DIR}"

echo "==> [6/6] Summarizing results"
python3 -m fed_agent.tools.summarize_paper_matrix \
  "${OUT_DIR}/summary.json" \
  --out_md docs/results/paper_matrix_retfound.md \
  --out_csv "${OUT_DIR}/summary.csv"

echo ""
echo "Done. Key outputs:"
echo "  - docs/results/paper_matrix_retfound.md   (committed summary table)"
echo "  - ${OUT_DIR}/summary.{json,csv}           (raw, gitignored)"
echo ""
echo "Sanity: every row should show RETFound=True. If any row is False, stop and"
echo "re-check HF_TOKEN and gated access (do NOT report fallback rows as paper results)."
