#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/bin:${PATH}"

branch="$(git rev-parse --abbrev-ref HEAD)"
echo "== push ${branch} -> origin =="
git push -u origin "${branch}"
