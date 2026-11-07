#!/usr/bin/env bash
# Round 8, backlog #5: why does our MoE time grow with context? Mechanisms from MLX source (agent D):
#  (1) command-buffer commit points (MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER, 50/50 on applegpu_g16s) move as KV grows;
#  (3) hidden copy_gpu of non-contiguous expert ids (argpartition slice) before our custom kernels; the fused router outputs contiguous ids.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; O=runs/r8; mkdir -p $O
B="-m moelab.breakdown --ctx 1024 16384 --steps 48 --rounds 3"
echo "== default";                       $PY $B --out $O/longctx_default.json | grep -E "^ctx|guard"
echo "== default, fused router";         $PY $B --fused-router --out $O/longctx_fused.json | grep -E "^ctx|guard"
echo "== one command buffer per step";   MLX_MAX_OPS_PER_BUFFER=100000 MLX_MAX_MB_PER_BUFFER=1000000 $PY $B --out $O/longctx_bigbuf.json | grep -E "^ctx|guard"
echo "== commit after every op";         MLX_MAX_OPS_PER_BUFFER=1 $PY $B --out $O/longctx_op1.json | grep -E "^ctx|guard"
echo done
