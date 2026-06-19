#!/usr/bin/env bash
# Install torch extras and run the same checks as smoke.sh (includes torch-only tests).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/bin:${PATH}"

echo "== pip install .[dev,torch] =="
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -e ".[dev,torch]"

echo "== ruff =="
python3 -m ruff check src tests

echo "== pytest =="
python3 -m pytest -q

echo "OK: smoke-torch passed"
