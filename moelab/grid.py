"""Targeted grid (round 7): real-model shapes x T chosen by load = T*k/E (avg tokens per expert).
Sets the dispatcher's fallback threshold. Same quick interleaved timing as moelab.sweep.

    .venv/bin/python -m moelab.grid --pair q4 --out runs/grid_q4.json
"""
import argparse
import json

import mlx.core as mx
import numpy as np

from . import mlx_backend as be
from .guard import probe, verdict, report
from .shapes import SHAPES
from .sweep import PAIRS

GRID_SHAPES = ["qwen3-30b-a3b", "deepseek-v2-lite", "olmoe-1b-7b", "gpt-oss-20b", "qwen3-235b-a22b", "abl-gptoss-3072"]
LOADS = [1, 2, 4, 6, 8, 12, 16, 24, 32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="q4", choices=list(PAIRS))
    ap.add_argument("--shapes", nargs="+", default=GRID_SHAPES)
    ap.add_argument("--max-t", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A, B = (be.load_variant(p) for p in PAIRS[a.pair])
    probes, rows = [probe()], []
    for name in a.shapes:
        sh = SHAPES[name]
        w = be.make_weights(sh, "bfloat16")
        sa, sb = A.prepare(*w), B.prepare(*w)
        mx.eval(sa, sb)
        del w
        Ts = sorted({max(1, round(L * sh.E / sh.k)) for L in LOADS} - {t for t in range(a.max_t + 1, 10**6)})
        for T in Ts:
            inp = [be.make_inputs(sh, T, "bfloat16", "uniform", seed=900 + s) for s in range(4)]
            ta, tb = [], []
            for r in range(3):
                for which in ((0, 1) if r % 2 == 0 else (1, 0)):
                    m, s, acc = (A, sa, ta) if which == 0 else (B, sb, tb)
                    acc.append(float(np.median(be.time_chained(lambda *z: m.run(s, *z), inp, reps=5))) * 1e6)
            row = {"shape": name, "T": T, "load": T * sh.k / sh.E, "A_us": min(ta), "B_us": min(tb),
                   "speedup": min(ta) / min(tb)}
            rows.append(row)
            print(f"{name:18s} T={T:4d} load {row['load']:5.1f}  A {row['A_us']:8.1f}us B {row['B_us']:8.1f}us {row['speedup']:.3f}x", flush=True)
        del sa, sb
        probes.append(probe())
    g = verdict(probes)
    print(report(g))
    if a.out:
        json.dump({"args": vars(a), "rows": rows, "guard": g}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
