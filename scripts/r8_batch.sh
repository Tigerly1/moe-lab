#!/usr/bin/env bash
# Round 8: concurrent requests (mlx-lm BatchGenerator), 3 models x B=1..32 x {kernels, kernels+fused router} x 2 launches.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; O=runs/r8; mkdir -p $O
args() { case $1 in
  Q) echo "--variant moelab/variants/mlx/metal_q4_moe.py" ;;
  G) echo "--model mlx-community/gpt-oss-20b-MXFP4-Q8 --variant moelab/variants/mlx/metal_mxfp4_gptoss.py" ;;
  D) echo "--model mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx --variant moelab/variants/mlx/metal_q4_moe.py" ;;
esac; }
for L in 1 2; do for M in Q G D; do
  A=$(args $M)
  echo "== $M launch $L kernels"; $PY -m moelab.batch_bench $A --tokens 128 --rounds 3 --out $O/batch_${M}_kernels_L$L.json 2>&1 | grep -E "^B=|guard"
  echo "== $M launch $L fused";   $PY -m moelab.batch_bench $A --tokens 128 --rounds 3 --fused-router --out $O/batch_${M}_fused_L$L.json 2>&1 | grep -E "^B=|guard"
done; done
echo done
