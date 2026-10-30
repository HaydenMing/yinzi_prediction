#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
git add -- outputs
if git diff --cached --quiet; then
  echo "没有新的可回传结果"
  exit 0
fi
git diff --cached --stat
git commit -m "results: GPU4 dual-material five-framework experiments"
git pull --rebase origin main
git push origin main
