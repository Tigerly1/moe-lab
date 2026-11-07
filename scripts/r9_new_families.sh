#!/usr/bin/env bash
# Round 9: families with no dedicated patch, through the universal SwitchGLU patch (moelab/universal.py):
# model-native text (fp32 arm), teacher-forced quality, single-stream speed (2 launches), concurrent requests.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-latest/bin/python
V=moelab/variants/mlx/metal_q4_moe.py
for M in mlx-community/LFM2.5-8B-A1B-MLX-4bit mlx-community/GLM-4.7-Flash-4bit; do
  N=$(basename "$M")
  echo "== $N"
  MOELAB_UNIVERSAL=1 PYTHONPATH=. $PY scripts/r9_native_text.py "$M" "runs/r9/native_$N.txt" 400 > /dev/null 2>&1
  $PY -m moelab.e2e --model "$M" --variant $V --universal --quality --quality-text "runs/r9/native_$N.txt" \
    --out "runs/r9/quality_${N}_native.json" 2>&1 | grep -E "PPL|agreement"
  for L in 1 2; do
    $PY -m moelab.e2e --model "$M" --variant $V --universal --chat --tokens 512 --rounds 3 \
      --out "runs/r9/e2e_${N}_universal_L$L.json" 2>&1 | grep -E "^(patched|speedup|guard)"
  done
  $PY -m moelab.batch_bench --model "$M" --batch 1 8 32 --tokens 128 --rounds 3 --universal \
    --out "runs/r9/batch_${N}_universal.json" 2>&1 | grep -E "^B=|guard"
done
