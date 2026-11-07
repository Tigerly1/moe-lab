#!/usr/bin/env bash
# Round 8: is MLX MoE decode partly CPU (graph build / encode) bound? Same e2e A/B with 0, 4, 8 busy CPU threads.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; O=runs/r8; mkdir -p $O
f() { grep -E "^(stock|variant|speedup|guard)" || true; }
for N in 0 4 8; do
  pids=""
  for i in $(seq 1 $N); do yes > /dev/null & pids="$pids $!"; done
  echo "== busy threads: $N"
  $PY -m moelab.e2e --variant moelab/variants/mlx/metal_q4_moe.py --tokens 256 --rounds 3 --out $O/cpu${N}_kernels.json | f
  $PY -m moelab.e2e --variant moelab/variants/mlx/metal_q4_moe.py --tokens 256 --rounds 3 --fused-router --out $O/cpu${N}_fused.json | f
  [ -n "$pids" ] && kill $pids
done
