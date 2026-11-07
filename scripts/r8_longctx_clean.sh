#!/usr/bin/env bash
# Round 8: clean (guarded) re-runs of the default long-context breakdown, 2 launches, kernels and fused router.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for i in 1 2; do
  for fr in "" "--fused-router"; do
    tag=$([ -z "$fr" ] && echo kernels || echo fused)
    echo "== launch $i $tag"
    $PY -m moelab.breakdown --ctx 1024 16384 --steps 48 --rounds 3 $fr --out runs/r8/longctx_${tag}_clean$i.json 2>&1 | grep -E "^ctx|guard|Error|Traceback"
  done
done
echo done
