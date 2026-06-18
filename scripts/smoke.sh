#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/bin:${PATH}"

echo "== ruff =="
python3 -m ruff check src tests

echo "== pytest =="
python3 -m pytest -q

echo "OK: smoke passed"
