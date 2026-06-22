#!/usr/bin/env bash
# Dependency-light synthetic ablation matrix: runs now without PyTorch.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/bin:${PATH}"

echo "== numpy synthetic ablations + report =="
python3 -m fed_agent.tools.run_numpy_ablations --repo_root . --publish_docs

echo "OK: see runs/numpy_ablations/<timestamp>/ and docs/results/numpy_synthetic_ablation_latest.md"
