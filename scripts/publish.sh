#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/sync-from-remote.sh
./scripts/push-current-branch.sh
