#!/usr/bin/env bash
# One-command synthetic baseline: 4 PNGs, 2 clients, FedProx smoke → runs/minimal/ + docs snapshot.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/bin:${PATH}"

if ! python3 -c "import torch" 2>/dev/null; then
  echo "== installing .[dev,torch] =="
  python3 -m pip install -U pip setuptools wheel
  python3 -m pip install -e ".[dev,torch]"
fi

ROOT_RUN="runs/minimal"
mkdir -p "${ROOT_RUN}/fixture"

echo "== minimal_experiment =="
python3 -m fed_agent.tools.minimal_experiment \
  --workdir "${ROOT_RUN}/fixture" \
  --out_json "${ROOT_RUN}/metrics.json" \
  --write_docs_snapshot "docs/results/minimal_synthetic_baseline.md" \
  --rounds 2 \
  --fedprox_mu 0.01

echo "== summarize =="
python3 -m fed_agent.tools.summarize_fed_smoke "${ROOT_RUN}/metrics.json"

echo "OK: wrote ${ROOT_RUN}/metrics.json and docs/results/minimal_synthetic_baseline.md"
