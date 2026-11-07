#!/usr/bin/env bash
# Round 9: real mlx_lm.server A/B (stock vs moelab.serve), concurrent OpenAI clients, 3 models.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-latest/bin/python
for m in ${MODELS:-mlx-community/gemma-4-26b-a4b-it-4bit mlx-community/Qwen3.6-35B-A3B-4bit}; do
  n=$(basename $m)
  echo "== $n"
  $PY -m moelab.serve_bench --model $m --clients 1 8 32 --max-tokens 256 --rounds 2 --port 8091 --out runs/r9/server_$n.json > runs/r9/server_$n.log 2>&1 || tail -20 runs/r9/server_$n.log; grep -E "^clients" runs/r9/server_$n.log
done
echo done
