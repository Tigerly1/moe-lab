"""Concurrent requests (round 8): mlx-lm BatchGenerator (what mlx_lm.server uses for parallel requests)
with B different prompts, greedy, no stop tokens, fixed generation length. Every decode step sends B tokens
from B different sequences through each MoE layer.

Reports generation throughput (all B streams together) for stock vs our kernels (and optionally the fused
router), interleaved rounds inside one process; run several times (--launch) for between-process spread.

    .venv/bin/python -m moelab.batch_bench --batch 1 2 4 8 16 32 --tokens 128 --out runs/r8/batch_qwen3.json
"""
import argparse
import json
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

from . import e2e
from .guard import probe, verdict, report
from .mlx_backend import load_variant


def run(model, prompts, n_tok):
    gen = BatchGenerator(model, max_tokens=n_tok, stop_tokens=[], sampler=make_sampler(temp=0.0),
                         completion_batch_size=max(32, len(prompts)), prefill_batch_size=len(prompts))
    uids = gen.insert(prompts, [n_tok] * len(prompts))
    toks = {u: [] for u in uids}
    first, count, t0 = None, 0, None
    while responses := gen.next_generated():
        for r in responses:
            toks[r.uid].append(r.token)
        if first is None:                      # first generated step ends prefill; time the rest
            first, t0 = True, time.perf_counter()
            continue
        count += len(responses)
    mx.synchronize()
    dt = time.perf_counter() - t0
    gen.close()
    run.last_tokens = [toks[u] for u in uids]
    return count / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--variant", default="moelab/variants/mlx/metal_q4_moe.py")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--fused-router", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    e2e.FUSED_ROUTER = a.fused_router
    model, tok = load(a.model)
    cls, n = e2e.patch(model, load_variant(a.variant))
    ids = tok.encode(open(e2e.__file__.replace("e2e.py", "data/quality_corpus.txt")).read())
    probes, rows = [probe()], []
    for B in a.batch:
        step = max(1, (len(ids) - a.prompt_len) // max(B, 1))
        prompts = [ids[i * step:i * step + a.prompt_len] for i in range(B)]   # B different text windows
        res = {"stock": [], "variant": []}
        outs = {}
        for r in range(a.rounds + 1):                                           # round 0 = warmup
            for name in (("stock", "variant") if r % 2 == 0 else ("variant", "stock")):
                cls._use_variant = name == "variant"
                tps = run(model, prompts, a.tokens)
                if r:
                    res[name].append(tps)
                    outs[name] = run.last_tokens
        s, v = float(np.median(res["stock"])), float(np.median(res["variant"]))
        # greedy agreement per stream: length of the identical prefix (bf16 noise makes streams drift eventually)
        pref = [next((i for i, (x, y) in enumerate(zip(a_, b_)) if x != y), len(a_)) for a_, b_ in zip(outs["stock"], outs["variant"])]
        rows.append({"B": B, "stock_tps": s, "variant_tps": v, "speedup": v / s, "raw": res,
                     "identical_prefix": pref, "lengths_equal": all(len(x) == len(y) for x, y in zip(outs["stock"], outs["variant"]))})
        print(f"B={B:3d}: stock {s:7.1f} tok/s  variant {v:7.1f} tok/s  speedup {v / s:.3f}x  "
              f"(stock {min(res['stock']):.1f}-{max(res['stock']):.1f}, variant {min(res['variant']):.1f}-{max(res['variant']):.1f}) "
              f"identical-prefix median {int(np.median(pref))}/{a.tokens}",
              flush=True)
        probes.append(probe())
    g = verdict(probes)
    print(report(g))
    if a.out:
        json.dump({"args": vars(a), "rows": rows, "guard": g}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
