#!/usr/bin/env bash
# Round 8: llama.cpp (Metal, Q4_0 = 4.5 bits/weight, same as MLX 4-bit g64) vs moelab.serve on Qwen3-30B-A3B.
set -uo pipefail
cd "$(dirname "$0")/.."
G=$HOME/Desktop/moe-thesis/gguf/Qwen3-30B-A3B-Q4_0.gguf
echo "== llama-bench single stream (tg256, pp512)"
llama-bench -m $G -p 512 -n 256 -ngl 999 -r 3 2>&1 | grep -E "^\|" | tee runs/r8/llamacpp_bench.txt
echo "== server: llama.cpp (baseline arm) vs moelab.serve"
.venv-latest/bin/python -m moelab.serve_bench --model mlx-community/Qwen3-30B-A3B-4bit --gguf $G --clients 1 8 32 \
  --max-tokens 256 --rounds 2 --port 8094 --out runs/r8/server_llamacpp_vs_ours_qwen3.json > runs/r8/server_llamacpp.log 2>&1 || tail -20 runs/r8/server_llamacpp.log
grep -E "^clients" runs/r8/server_llamacpp.log
echo done
