#!/usr/bin/env bash
# Full synthetic ablation matrix: FedAvg vs FedProx × label noise → REPORT.md + CSV + optional docs snapshot.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/bin:${PATH}"

if ! python3 -c "import torch" 2>/dev/null; then
  echo "== installing torch (CPU wheels; increase timeout for slow networks) =="
  python3 -m pip install -U pip setuptools wheel
  export PIP_DEFAULT_TIMEOUT=120
  python3 -m pip install -e ".[dev,torch]"
fi

echo "== synthetic ablations + report =="
python3 -m fed_agent.tools.run_synthetic_ablations --repo_root . --publish_docs

echo "OK: see runs/synthetic_ablations/<timestamp>/ and docs/results/synthetic_ablation_latest.md"
