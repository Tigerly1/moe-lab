#!/usr/bin/env bash
# Round 8: real mlx_lm.server A/B (stock vs moelab.serve), concurrent OpenAI clients, 3 models.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-latest/bin/python
for m in ${MODELS:-mlx-community/Qwen3-30B-A3B-4bit mlx-community/gpt-oss-20b-MXFP4-Q8 mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx}; do
  n=$(basename $m)
  echo "== $n"
  $PY -m moelab.serve_bench --model $m --clients 1 8 32 --max-tokens 256 --rounds 2 --port 8091 --out runs/r8/server_$n.json > runs/r8/server_$n.log 2>&1 || tail -20 runs/r8/server_$n.log; grep -E "^clients" runs/r8/server_$n.log
done
echo done
