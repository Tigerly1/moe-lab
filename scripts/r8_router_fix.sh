#!/usr/bin/env bash
# Round 8: tie rate and one-line fp32-softmax fix for mlx-lm MoE routers (Qwen3, DeepSeek-V2), plus e2e quality/speed of the fix.
set -euo pipefail
cd "$(dirname "$0")/.."
for m in mlx-community/Qwen3-30B-A3B-4bit mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx; do
  PYTHONPATH=. .venv/bin/python runs/router_fix_check.py $m 2>&1 | grep "^{"
done
