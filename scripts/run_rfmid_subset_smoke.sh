#!/usr/bin/env bash
# Real RFMiD subset smoke: export HF subset, run IID/Dirichlet × FedAvg/FedProx × noise.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/bin:${PATH}"

MAX_SAMPLES="${MAX_SAMPLES:-96}"
OUT_DATA="data/raw/rfmid_hf_subset"
OUT_RUN="runs/rfmid_smoke_matrix/latest"

echo "== export RFMiD HF subset (${MAX_SAMPLES} samples) =="
python3 -m fed_agent.tools.export_hf_rfmid_subset \
  --out_dir "${OUT_DATA}" \
  --max_samples "${MAX_SAMPLES}"

echo "== run RFMiD subset smoke matrix =="
python3 -m fed_agent.tools.run_rfmid_smoke_matrix \
  --labels_csv "${OUT_DATA}/labels.csv" \
  --images_dir "${OUT_DATA}/images" \
  --out_dir "${OUT_RUN}" \
  --n_clients 4 \
  --seed 0 \
  --rounds 2 \
  --image_size 32 32 \
  --publish_docs

echo "OK: see ${OUT_RUN}/ and docs/results/rfmid_subset_smoke_latest.md"
