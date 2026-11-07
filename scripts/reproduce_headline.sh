#!/usr/bin/env bash
# Reproduce every headline number in README.md from scratch (Apple M4 Pro, AC power, ~45 min).
# Needs the two models cached: mlx-community/Qwen3-30B-A3B-4bit, mlx-community/gpt-oss-20b-MXFP4-Q8.
# Uses seeds 5000+ (never used by tuning agents or earlier verification runs).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=runs/repro
mkdir -p "$OUT"
pmset -g batt | head -1 | tee "$OUT/power.txt"

echo "== 1. correctness (unit, large/tiny scales, real checkpoints, full-model multi-token)"
MOE_SLOW=1 $PY -m pytest -q tests | tail -1 | tee "$OUT/tests.txt"

echo "== 2. layer level, real traces, fresh seeds, CI by launch"
$PY -m moelab.compare moelab/variants/mlx/mlxlm_switchglu_q4.py moelab/variants/mlx/metal_q4_moe.py \
  --shape qwen3-30b-a3b --routing trace --tokens 1 8 16 32 --launches 6 --rounds 4 --windows 8 --seed-base 5000 \
  --out "$OUT/q4_qwen3_trace.json" | tee "$OUT/q4_qwen3_trace.txt"
$PY -m moelab.compare moelab/variants/mlx/mlxlm_switchglu_mxfp4.py moelab/variants/mlx/metal_mxfp4_moe.py \
  --shape qwen3-30b-a3b --routing trace --tokens 1 8 16 32 --launches 6 --rounds 4 --windows 8 --seed-base 5000 \
  --out "$OUT/mxfp4_qwen3_trace.json" | tee "$OUT/mxfp4_qwen3_trace.txt"
$PY -m moelab.compare moelab/variants/mlx/mlxlm_switchglu_q4.py moelab/variants/mlx/metal_q4_moe.py \
  --shape qwen3-235b-a22b --routing uniform --tokens 1 16 --launches 4 --rounds 4 --windows 8 --seed-base 5000 \
  --out "$OUT/q4_qwen3-235b.json" | tee "$OUT/q4_qwen3-235b.txt"

echo "== 3. T=1 slope timing, non-repeating routing"
$PY -m moelab.slope --shape qwen3-30b-a3b --variants mlxlm_switchglu_q4 metal_q4_decode_v4 | tee "$OUT/slope_qwen3.txt"

echo "== 4. end to end"
$PY -m moelab.e2e --tokens 512 --rounds 3 --variant moelab/variants/mlx/metal_q4_moe.py --out "$OUT/e2e_qwen3_plain.json" \
  | grep -E "^(stock|variant|speedup)" | tee "$OUT/e2e_qwen3_plain.txt"
$PY -m moelab.e2e --variant moelab/variants/mlx/metal_q4_moe.py --verify-chunks 4 16 32 --out "$OUT/e2e_qwen3_verify.json" \
  | grep "^verify" | tee "$OUT/e2e_qwen3_verify.txt"
$PY -m moelab.e2e --model mlx-community/gpt-oss-20b-MXFP4-Q8 --variant moelab/variants/mlx/metal_mxfp4_gptoss.py \
  --tokens 512 --rounds 3 --out "$OUT/e2e_gptoss_plain.json" | grep -E "^(stock|variant|speedup)" | tee "$OUT/e2e_gptoss_plain.txt"
$PY -m moelab.e2e --model mlx-community/gpt-oss-20b-MXFP4-Q8 --variant moelab/variants/mlx/metal_mxfp4_gptoss.py \
  --verify-chunks 4 16 --out "$OUT/e2e_gptoss_verify.json" | grep "^verify" | tee "$OUT/e2e_gptoss_verify.txt"
echo "done: $OUT"
