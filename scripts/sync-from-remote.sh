#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/bin:${PATH}"

echo "== git fetch origin =="
git fetch origin

branch="$(git rev-parse --abbrev-ref HEAD)"
echo "== current branch: ${branch} =="

if git rev-parse '@{upstream}' >/dev/null 2>&1; then
  echo "== git pull --rebase (upstream set) =="
  git pull --rebase
else
  echo "No upstream for ${branch}; skipping pull."
  echo "First push: git push -u origin ${branch}"
fi

echo "== status =="
git status -sb
