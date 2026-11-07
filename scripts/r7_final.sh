#!/usr/bin/env bash
# Round 7 final pass on the final code: quality with the stock-router fp32 truth, router tie stats, clean sweeps.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; O=runs/r7; mkdir -p $O
f() { grep -E "PPL|agreement|guard|cases|token-layers" || true; }
echo "== Q quality";  $PY -m moelab.e2e --variant moelab/variants/mlx/metal_q4_moe.py --quality --fused-router --out $O/Q_quality_fused.json | f
echo "== G quality";  $PY -m moelab.e2e --model mlx-community/gpt-oss-20b-MXFP4-Q8 --variant moelab/variants/mlx/metal_mxfp4_gptoss.py --quality --fused-router --out $O/G_quality_fused.json | f
echo "== D quality";  $PY -m moelab.e2e --model mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx --variant moelab/variants/mlx/metal_q4_moe.py --quality --fused-router --out $O/D_quality_fused.json | f
echo "== router ties"; PYTHONPATH=. $PY runs/router_ties.py | f
echo "== sweeps"
$PY -m moelab.sweep --pair q4 --n 60 --seed 11 --out runs/sweep_q4_final.json > runs/sweep_q4_final.txt 2>&1; tail -2 runs/sweep_q4_final.txt
$PY -m moelab.sweep --pair mxfp4 --n 60 --seed 12 --out runs/sweep_mxfp4_final.json > runs/sweep_mxfp4_final.txt 2>&1; tail -2 runs/sweep_mxfp4_final.txt
$PY -m moelab.sweep --pair q4 --n 60 --seed 31 --out runs/sweep_q4_final_holdout.json > runs/sweep_q4_final_holdout.txt 2>&1; tail -2 runs/sweep_q4_final_holdout.txt
echo done
