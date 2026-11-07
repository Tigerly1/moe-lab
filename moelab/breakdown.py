"""Where does decode time go at long context? (round 7)

Per decode step (one token, real model, real KV cache of `ctx` tokens), three modes on the same cache:
  stock    - mlx-lm as shipped
  variant  - MoE experts through our kernel
  no-moe   - MoE block returns zeros (removes router + experts; attention, norms, embeddings, lm_head stay)
MoE time = mode - no-moe. Attention growth = no-moe(ctx) - no-moe(small ctx).
The cache is trimmed back after each mode, so all modes see the same context.

    .venv/bin/python -m moelab.breakdown --ctx 128 1024 4096 8192 --out runs/breakdown_qwen3.json
"""
import argparse
import json
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from . import e2e
from .guard import probe, verdict, report
from .mlx_backend import load_variant


def steps(model, cache, tok_id, n):
    y = mx.array([[tok_id]])
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        logits = model(y, cache=cache)
        y = mx.argmax(logits[:, -1:], axis=-1)
        mx.eval(y)
        ts.append(time.perf_counter() - t0)
    trim_prompt_cache(cache, n)
    return float(np.median(ts)) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--variant", default="moelab/variants/mlx/metal_q4_moe.py")
    ap.add_argument("--ctx", type=int, nargs="+", default=[128, 1024, 4096, 8192])
    ap.add_argument("--steps", type=int, default=48); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fused-router", action="store_true")
    a = ap.parse_args()
    e2e.FUSED_ROUTER = a.fused_router
    model, tok = load(a.model)
    cls, n = e2e.patch(model, load_variant(a.variant))
    patched = cls.__call__

    def nomoe(self, x):
        return mx.zeros_like(x)

    ids = tok.encode(open(e2e.__file__.replace("e2e.py", "data/quality_corpus.txt")).read())
    probes, out = [probe()], []
    for ctx in a.ctx:
        prompt = mx.array([(ids * (ctx // len(ids) + 1))[:ctx]])
        cls.__call__, cls._use_variant = patched, False
        cache = make_prompt_cache(model)
        for i in range(0, ctx, 512):                       # chunked prefill (stock path)
            model(prompt[:, i:i + 512], cache=cache)
            mx.eval([c.state for c in cache])
        res = {"stock": [], "variant": [], "no-moe": []}
        for r in range(a.rounds + 1):                      # round 0 = warmup
            for mode in (["stock", "variant", "no-moe"] if r % 2 else ["no-moe", "variant", "stock"]):
                cls.__call__ = nomoe if mode == "no-moe" else patched
                cls._use_variant = mode == "variant"
                ms = steps(model, cache, ids[0], a.steps)
                if r:
                    res[mode].append(ms)
        cls.__call__ = patched
        m = {k: float(np.median(v)) for k, v in res.items()}
        row = {"ctx": ctx, **{f"{k}_ms": v for k, v in m.items()},
               "moe_stock_ms": m["stock"] - m["no-moe"], "moe_variant_ms": m["variant"] - m["no-moe"],
               "e2e_speedup": m["stock"] / m["variant"], "moe_speedup": (m["stock"] - m["no-moe"]) / (m["variant"] - m["no-moe"]),
               "moe_share_stock": (m["stock"] - m["no-moe"]) / m["stock"], "raw": res}
        out.append(row)
        print(f"ctx {ctx:5d}: stock {m['stock']:.2f} ms  variant {m['variant']:.2f}  no-moe {m['no-moe']:.2f}  | "
              f"MoE stock {row['moe_stock_ms']:.2f} -> {row['moe_variant_ms']:.2f} ms ({row['moe_speedup']:.3f}x), "
              f"MoE share {100 * row['moe_share_stock']:.0f}%, e2e {row['e2e_speedup']:.3f}x", flush=True)
        del cache
        probes.append(probe())
    g = verdict(probes)
    print(report(g))
    if a.out:
        json.dump({"args": vars(a), "rows": out, "guard": g}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
