"""Round 13: does a weight-reading kernel pay a fixed cost per call, on top of bytes / bandwidth?

Chain of 12 dependent 4-bit GEMVs (K -> K, bf16), for K = 512 ... 8192 (0.15 - 37.7 MB per call).
time per call = fixed + bytes / bandwidth; a least-squares line over the sizes gives both.
Light: the largest size holds ~450 MB of weights, each size runs for about a second.
"""

import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn

LAYERS, REPS, ROUNDS = 12, 40, 5


def us_per_call(K):
    layers = []
    for _ in range(LAYERS):
        lin = nn.QuantizedLinear(K, K, bias=False, group_size=64, bits=4)
        lin.set_dtype(mx.bfloat16)
        mx.eval(lin.parameters())
        layers.append(lin)
    x = mx.random.normal((1, 1, K)).astype(mx.bfloat16)
    mx.eval(x)

    def step():
        h = x
        for lin in layers:
            h = lin(h)
        return h

    mx.eval(step())
    best = float("inf")
    for _ in range(ROUNDS):
        t = time.perf_counter()
        for _ in range(REPS):
            mx.eval(step())
        best = min(best, (time.perf_counter() - t) / REPS / LAYERS * 1e6)
    nbytes = sum(v.nbytes for _, v in nn.utils.tree_flatten(layers[0].parameters()))
    return nbytes / 1e6, best


def main():
    rows = [(K, *us_per_call(K)) for K in (512, 1024, 2048, 2880, 4096, 5760, 8192)]
    mb, us = [r[1] for r in rows], [r[2] for r in rows]
    mm, mu = sum(mb) / len(mb), sum(us) / len(us)
    slope = sum((a - mm) * (b - mu) for a, b in zip(mb, us)) / sum((a - mm) ** 2 for a in mb)
    fixed = mu - slope * mm
    for K, m, u in rows:
        print(f"K={K:5d}  {m:6.2f} MB  {u:7.1f} us/call  effective {m / u * 1e3:6.1f} GB/s")
    print(f"fit: fixed {fixed:.1f} us per call, marginal {1e3 / slope:.0f} GB/s")
    if len(sys.argv) > 1:
        json.dump({"arch": mx.device_info().get("architecture"), "rows": rows, "fixed_us": round(fixed, 2),
                   "marginal_gb_s": round(1e3 / slope, 1), "peak_gb": round(mx.get_peak_memory() / 2**30, 2)},
                  open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
