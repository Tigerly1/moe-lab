#!/usr/bin/env bash
# Round 8: re-check the headline against the LATEST mlx-lm (0.32.2, MLX 0.32.4, Python 3.14 venv).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv-latest/bin/python; O=runs/r8/latest; mkdir -p $O
$PY -c "import mlx.core as mx, mlx_lm; print('mlx', mx.__version__, 'mlx-lm', mlx_lm.__version__)"
echo "== patch correctness on latest mlx-lm"
MOE_SLOW=1 $PY -m pytest -q tests/test_e2e_patch.py tests/test_e2e_multitoken.py tests/test_gptoss_patch.py tests/test_deepseek_patch.py tests/test_real_model_multitoken.py tests/test_router.py tests/test_fallback.py 2>&1 | tail -2
args() { case $1 in
  Q) echo "--variant moelab/variants/mlx/metal_q4_moe.py" ;;
  G) echo "--model mlx-community/gpt-oss-20b-MXFP4-Q8 --variant moelab/variants/mlx/metal_mxfp4_gptoss.py" ;;
  D) echo "--model mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx --variant moelab/variants/mlx/metal_q4_moe.py" ;;
esac; }
for M in Q G D; do
  A=$(args $M)
  echo "== $M plain decode, kernels";  $PY -m moelab.e2e $A --tokens 512 --rounds 3 --out $O/${M}_plain.json 2>&1 | grep -E "^(speedup|guard|stock|variant)"
  echo "== $M plain decode, fused";    $PY -m moelab.e2e $A --tokens 512 --rounds 3 --fused-router --out $O/${M}_fused.json 2>&1 | grep -E "^(speedup|guard|stock|variant)"
  echo "== $M batch, fused";           $PY -m moelab.batch_bench $A --batch 1 4 8 16 32 --tokens 128 --rounds 3 --fused-router --out $O/${M}_batch_fused.json 2>&1 | grep -E "^B=|guard"
done
echo done
