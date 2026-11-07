#!/usr/bin/env bash
# Portability check on a second Apple chip (written for a 16 GB M2 MacBook): is the M4 Pro speedup chip-specific?
# Same layer benchmarks and seeds as reproduce_headline.sh. End to end only on models that fit in 16 GB:
#   DeepSeek-V2-Lite 4bit (~9 GB) always; gpt-oss-20b MXFP4 (~12 GB) only with GPTOSS=1 (may exceed the GPU wired limit).
# Qwen3-30B-A3B (~17 GB) does not fit; its layer is tested with synthetic weights + the recorded real routing trace.
# Run on AC power, other apps closed. ~30 min + model download (~9 GB). Output: runs/portability_<chip>/ + .tar.gz
# Fresh clone is enough: the venv is created on first run (needs python3 >= 3.10, e.g. `brew install python@3.12`).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "== setup: creating .venv (same pins as the M4 Pro runs)"
  PYBIN=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
  "$PYBIN" -m venv .venv
  $PY -m pip install -q --upgrade pip
  $PY -m pip install -q mlx==0.32.4 mlx-lm==0.31.3 numpy==2.2.6 pytest==9.2.0 matplotlib==3.10.9
fi
CHIP=$(sysctl -n machdep.cpu.brand_string | tr ' ' '_')
OUT=runs/portability_$CHIP
mkdir -p "$OUT"
{ sysctl -n machdep.cpu.brand_string; echo "RAM $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"; pmset -g batt | head -1;
  $PY -c "import mlx.core as mx, mlx_lm; print('mlx', mx.__version__, 'mlx-lm', mlx_lm.__version__)"; } | tee "$OUT/system.txt"

echo "== 0. read bandwidth (same probe gives 213.6 GB/s on M4 Pro, whose true read ceiling is ~250)"
BW=$($PY - <<'EOF'
import mlx.core as mx, numpy as np
from moelab import mlx_backend as be
n = 1 << 30
a = mx.random.uniform(shape=(n // 2,)).astype(mx.float16); mx.eval(a)
r = n / np.median(be.time_calls(lambda a: a.sum(), [(a,)], 3, 20)) / 1e9
print(f"{r * 250 / 213.6:.1f}")   # calibrated to the M4 Pro ceiling used in RESULTS.md
EOF
)
echo "calibrated read ceiling: $BW GB/s" | tee "$OUT/bandwidth.txt"

echo "== 1. correctness (fast suite; full-model tests need models that do not fit)"
$PY -m pytest -q tests | tail -1 | tee "$OUT/tests.txt"

echo "== 2. layer level, real Qwen3 routing trace, fresh seeds"
$PY -m moelab.compare moelab/variants/mlx/mlxlm_switchglu_q4.py moelab/variants/mlx/metal_q4_moe.py \
  --shape qwen3-30b-a3b --routing trace --tokens 1 8 16 32 --launches 6 --rounds 4 --windows 8 --seed-base 5000 \
  --out "$OUT/q4_qwen3_trace.json" | tee "$OUT/q4_qwen3_trace.txt"
$PY -m moelab.compare moelab/variants/mlx/mlxlm_switchglu_mxfp4.py moelab/variants/mlx/metal_mxfp4_moe.py \
  --shape qwen3-30b-a3b --routing trace --tokens 1 8 16 32 --launches 6 --rounds 4 --windows 8 --seed-base 5000 \
  --out "$OUT/mxfp4_qwen3_trace.json" | tee "$OUT/mxfp4_qwen3_trace.txt"

echo "== 3. T=1 slope timing"
$PY -m moelab.slope --shape qwen3-30b-a3b --bw "$BW" --variants mlxlm_switchglu_q4 metal_q4_decode_v4 | tee "$OUT/slope_qwen3.txt"

echo "== 4. end to end, DeepSeek-V2-Lite"
$PY -m moelab.e2e --model mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx --variant moelab/variants/mlx/metal_q4_moe.py \
  --tokens 512 --rounds 3 --out "$OUT/e2e_dsv2_plain.json" | grep -E "^(stock|variant|speedup)" | tee "$OUT/e2e_dsv2_plain.txt"
$PY -m moelab.e2e --model mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx --variant moelab/variants/mlx/metal_q4_moe.py \
  --verify-chunks 4 16 32 --out "$OUT/e2e_dsv2_verify.json" | grep "^verify" | tee "$OUT/e2e_dsv2_verify.txt"

echo "== 4b. concurrent requests (round-8 headline: M4 Pro B=32 1.54x on DeepSeek-V2-Lite), 2 launches"
for L in 1 2; do
  $PY -m moelab.batch_bench --model mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx --batch 1 8 32 --tokens 128 --rounds 3 \
    --fused-router --out "$OUT/batch_dsv2_fused_L$L.json" 2>&1 | grep -E "^B=|guard" | tee "$OUT/batch_dsv2_fused_L$L.txt"
done

if [ "${GPTOSS:-0}" = 1 ]; then
  echo "== 5. end to end, gpt-oss-20b (opt-in)"
  $PY -m moelab.e2e --model mlx-community/gpt-oss-20b-MXFP4-Q8 --variant moelab/variants/mlx/metal_mxfp4_gptoss.py \
    --tokens 512 --rounds 3 --out "$OUT/e2e_gptoss_plain.json" | grep -E "^(stock|variant|speedup)" | tee "$OUT/e2e_gptoss_plain.txt"
fi
tar czf "$OUT.tar.gz" "$OUT"
echo "done: $OUT  -> send back $OUT.tar.gz (AirDrop is fine). M4 Pro reference: README.md, docs/RESULTS.md"
