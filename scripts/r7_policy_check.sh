#!/usr/bin/env bash
# Round 7: rigorous A/B (compare, 3 launches, fresh seeds 9100) at the final dispatchers' threshold boundaries.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; O=runs/r7_policy; mkdir -p $O
Q4="moelab/variants/mlx/mlxlm_switchglu_q4.py moelab/variants/mlx/metal_q4_moe.py"
FP="moelab/variants/mlx/mlxlm_switchglu_mxfp4.py moelab/variants/mlx/metal_mxfp4_moe.py"
C="--routing uniform --launches 3 --rounds 3 --windows 4 --seed-base 9100"
$PY -m moelab.compare $Q4 --shape qwen3-30b-a3b --tokens 192 256 384 512 $C --out $O/q4_qwen3.json | grep -E "^T=|guard"
$PY -m moelab.compare $Q4 --shape qwen3-235b-a22b --tokens 256 384 $C --out $O/q4_235b.json | grep -E "^T=|guard"
$PY -m moelab.compare $Q4 --shape small --dims 1024,512,32,4 --tokens 64 96 $C --out $O/q4_small.json | grep -E "^T=|guard"
$PY -m moelab.compare $Q4 --shape gpt-oss-20b --tokens 32 48 $C --out $O/q4_gptoss.json | grep -E "^T=|guard"
$PY -m moelab.compare $FP --shape gpt-oss-20b --tokens 96 128 192 $C --out $O/fp4_gptoss.json | grep -E "^T=|guard"
$PY -m moelab.compare $FP --shape small --dims 768,512,64,8 --tokens 128 192 $C --out $O/fp4_small.json | grep -E "^T=|guard"
$PY -m moelab.compare $Q4 --shape oh --dims 2560,256,64,8 --routing one_hot --tokens 8 32 --launches 3 --rounds 3 --windows 4 --seed-base 9100 --out $O/q4_onehot.json | grep -E "^T=|guard"
