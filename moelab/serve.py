"""mlx-lm's OpenAI-compatible server with the moe-lab MoE kernels (round 8).

Same CLI as `mlx_lm.server`; every model the server loads is patched (kernel auto-selected by checkpoint
format, fused router tail on). Concurrent requests are batched by mlx-lm (--decode-concurrency), which is
where the kernels gain most (docs/RESULTS.md, Round 8).

    .venv-latest/bin/python -m moelab.serve --model mlx-community/Qwen3-30B-A3B-4bit --port 8080
    MOELAB_STOCK=1 ... same command, unpatched (for A/B)
"""
import os
import sys

import mlx_lm.server as srv

from . import e2e
from .generate import pick_variant
from .mlx_backend import load_variant

_load = srv.load


def _patched_load(*args, **kwargs):
    out = _load(*args, **kwargs)
    model = out[0]
    if not os.environ.get("MOELAB_STOCK"):
        e2e.FUSED_ROUTER = True
        variant = pick_variant(model)
        cls, n = e2e.patch(model, load_variant(variant))
        cls._use_variant = True
        print(f"[moe-lab] patched {n} MoE blocks with {os.path.basename(variant)} + fused router", file=sys.stderr, flush=True)
    else:
        print("[moe-lab] MOELAB_STOCK=1: serving unpatched mlx-lm", file=sys.stderr, flush=True)
    return out


srv.load = _patched_load

if __name__ == "__main__":
    srv.main()
