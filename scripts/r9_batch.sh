#!/usr/bin/env bash
# Round 9: concurrent requests (BatchGenerator) on Gemma-4-26B-A4B and Qwen3.6-35B-A3B, kernels + fused router.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-latest/bin/python
for M in mlx-community/gemma-4-26b-a4b-it-4bit mlx-community/Qwen3.6-35B-A3B-4bit; do
  N=$(basename "$M")
  for L in 1 2; do
    echo "== $N launch $L"
    $PY -m moelab.batch_bench --model "$M" --batch 1 8 32 --tokens 128 --rounds 3 --fused-router \
      --out "runs/r9/batch_${N}_fused_L$L.json" 2>&1 | grep -E "^B=|guard"
  done
done
