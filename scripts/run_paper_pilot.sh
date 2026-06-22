#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -e ".[data,torch,paper]"

python3 -m fed_agent.tools.run_paper_matrix \
  --matrix_yaml configs/paper_matrix_pilot.yaml \
  --out_dir runs/paper_matrix/pilot_head12

python3 -m fed_agent.tools.summarize_paper_matrix \
  runs/paper_matrix/pilot_head12/summary.json \
  --out_md docs/results/paper_matrix_pilot_head12.md \
  --out_csv runs/paper_matrix/pilot_head12/summary.csv
