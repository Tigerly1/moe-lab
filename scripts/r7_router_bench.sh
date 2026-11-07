#!/usr/bin/env bash
# Round 7: fused router tail, end to end on three models (same day, same machine state, nothing else running).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
O=runs/r7
mkdir -p $O
args() { case $1 in
  Q) echo "--variant moelab/variants/mlx/metal_q4_moe.py" ;;
  G) echo "--model mlx-community/gpt-oss-20b-MXFP4-Q8 --variant moelab/variants/mlx/metal_mxfp4_gptoss.py" ;;
  D) echo "--model mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx --variant moelab/variants/mlx/metal_q4_moe.py" ;;
esac; }
f() { grep -E "^(stock|variant|speedup|guard|verify)|PPL|agreement" || true; }
for M in Q G D; do
  A=$(args $M)   # bash splits $A on spaces (no spaces inside the values)
  echo "== $M plain";        $PY -m moelab.e2e $A --tokens 512 --rounds 3 --out $O/${M}_plain.json | f
  echo "== $M fused";        $PY -m moelab.e2e $A --tokens 512 --rounds 3 --fused-router --out $O/${M}_fused.json | f
  echo "== $M verify plain"; $PY -m moelab.e2e $A --verify-chunks 4 16 32 --out $O/${M}_verify_plain.json | f
  echo "== $M verify fused"; $PY -m moelab.e2e $A --verify-chunks 4 16 32 --fused-router --out $O/${M}_verify_fused.json | f
  echo "== $M quality fused";$PY -m moelab.e2e $A --quality --fused-router --out $O/${M}_quality_fused.json | f
done
echo done
