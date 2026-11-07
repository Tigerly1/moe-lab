"""Slope timing: per-call device time = slope of total eval time vs chain length.
Removes the fixed per-eval cost (command-buffer commit/wait) that plain chained timing
spreads over the chain. Also measures the residual add (x + y) cost alone.

    .venv/bin/python -m moelab.slope --shape qwen3-30b-a3b --variants mlxlm_switchglu_q4 metal_q4_gemv_v31 ...
"""
import argparse
import time

import mlx.core as mx
import numpy as np

from . import mlx_backend as be, roofline
from .shapes import SHAPES

CHAINS = [10, 20, 40, 80]


def total_time(fn, inputs, chain, reps=7):
    def build():
        x = inputs[0][0]
        for i in range(chain):
            _, idx, w = inputs[i % len(inputs)]
            x = x + fn(x, idx, w).astype(x.dtype)
        return x
    mx.eval(build()); mx.synchronize()
    ts = []
    for _ in range(reps):
        x = build()
        t0 = time.perf_counter(); mx.eval(x); mx.synchronize(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def fit(fn, inputs):
    ys = [total_time(fn, inputs, c) for c in CHAINS]
    slope, icpt = np.polyfit(CHAINS, ys, 1)
    return slope * 1e6, icpt * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="qwen3-30b-a3b"); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--T", type=int, default=1); ap.add_argument("--routing", default="uniform")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--windows", type=int, default=80, help="distinct inputs; >= max chain so no expert-set repeats (audit 2)")
    ap.add_argument("--bw", type=float, default=250.0, help="measured read GB/s of this chip (M4 Pro: 250)")
    a = ap.parse_args()
    sh = SHAPES[a.shape]
    w = be.make_weights(sh, a.dtype)
    inp = [be.make_inputs(sh, a.T, a.dtype, a.routing, 1000 + s) for s in range(a.windows)]
    res_us, res_icpt = fit(lambda x, i, ww: mx.zeros_like(x), inp)
    U = float(np.mean([len(np.unique(np.array(i[1]))) for i in inp]))
    w_bits = 4.5  # default int4 g64; per-variant W_BITS below
    print(f"{a.shape} T={a.T} {a.routing} windows={a.windows}: residual-add-only slope {res_us:.1f}us (fixed {res_icpt:.0f}us)")
    base = None
    for v in a.variants:
        m = be.load_variant(f"moelab/variants/mlx/{v}.py"); st = m.prepare(*w); mx.eval(st)
        floor = roofline.bytes_moved(a.T, sh.d, sh.f, sh.E, sh.k, getattr(m, "W_BITS", w_bits), unique=U) / (a.bw * 1e9) * 1e6
        s, c = fit(lambda *z: m.run(st, *z), inp)
        net = s - res_us
        base = base or net
        print(f"  {v:22s} per-call {s:7.1f}us  minus residual {net:7.1f}us  fixed/eval {c:6.0f}us  "
              f"floor {floor:6.1f}us  {100 * floor / net:5.1f}% of floor@{a.bw:.0f}  speedup {base / net:.3f}x")
        del st


if __name__ == "__main__":
    main()
