#!/usr/bin/env bash
# After adding the SSH public key to GitHub (see docs/GITHUB_SETUP.md), run:
set -euo pipefail
export PATH="${HOME}/bin:${PATH}"
cd "$(dirname "$0")/.."
echo "== SSH probe =="
ssh -T git@github.com || { echo "SSH auth failed: add ~/.ssh/id_ed25519_flagent.pub to GitHub first."; exit 1; }
echo "== Push main + dev =="
git push -u origin main
git push -u origin dev
