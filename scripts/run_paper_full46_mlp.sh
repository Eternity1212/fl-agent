#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -e ".[data,torch,paper]"

python3 -m fed_agent.tools.export_hf_rfmid_subset \
  --split all \
  --out_dir data/raw/rfmid_full \
  --max_samples 0 \
  --validate

for seed in 0 1 2; do
  python3 -m fed_agent.tools.build_splits \
    --labels_csv data/raw/rfmid_full/train/labels.csv \
    --out_dir configs/splits/generated \
    --seed "${seed}" \
    --n_clients 4 \
    --alphas 0.1 0.5 1.0
done

python3 -m fed_agent.tools.run_paper_matrix \
  --matrix_yaml configs/paper_matrix_full46_mlp.yaml \
  --out_dir runs/paper_matrix/full46_mlp

python3 -m fed_agent.tools.summarize_paper_matrix \
  runs/paper_matrix/full46_mlp/summary.json \
  --out_md docs/results/paper_matrix_full46_mlp.md \
  --out_csv runs/paper_matrix/full46_mlp/summary.csv
