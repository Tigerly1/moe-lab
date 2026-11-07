#!/usr/bin/env bash
# Round 8: does ONE request with an empty prompt kill mlx_lm.server's generation thread for everyone? (stock mlx-lm 0.32.2)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-latest/bin/python; LOG=runs/r8/empty_prompt_bug.log
$PY -m mlx_lm.server --model mlx-community/Qwen3-0.6B-4bit --port 8093 > $LOG 2>&1 &
SRV=$!
for i in $(seq 1 120); do curl -s localhost:8093/v1/models > /dev/null && break; sleep 1; done
req() { curl -s -m 60 localhost:8093/v1/completions -H 'Content-Type: application/json' -d "{\"prompt\":\"$1\",\"max_tokens\":8,\"temperature\":0}" | head -c 160; echo; }
echo "1. normal request:";            req "Hello there"
echo "2. empty prompt:";              req ""
echo "3. normal request afterwards:"; req "Hello again"
echo "--- server errors:"; grep -E "ERROR|ValueError" $LOG | head -5
kill $SRV
