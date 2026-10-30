#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
OUTPUT="$ROOT/outputs/gpu4_formal"
if [ "$#" -gt 0 ]; then OUTPUT="$1"; fi
printf 'GPU物理编号=%s，程序内使用cuda:0\n输出目录=%s\n' "$CUDA_VISIBLE_DEVICES" "$OUTPUT"
python -X utf8 -u src/run.py --root "$ROOT" --output "$OUTPUT" --device cuda --stage all
