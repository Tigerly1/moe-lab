"""Wide random sweep (round 7): random MoE shapes / T / dtype / routing, correctness + quick A/B speed.

Purpose: find where a kernel is wrong or slower than the baseline outside the shapes it was tuned on.
Timing here is a screen (one process, interleaved A/B, median of 5 chained reps x 2 rounds); any case
below --flag is re-measured with moelab.compare before it counts as a finding.

    .venv/bin/python -m moelab.sweep --n 60 --seed 11 --out runs/sweep_q4.json
"""
import argparse
import json

import mlx.core as mx
import numpy as np

from . import mlx_backend as be
from .guard import probe, verdict, report
from .shapes import MoEShape

PAIRS = {"q4": ("moelab/variants/mlx/mlxlm_switchglu_q4.py", "moelab/variants/mlx/metal_q4_moe.py"),
         "mxfp4": ("moelab/variants/mlx/mlxlm_switchglu_mxfp4.py", "moelab/variants/mlx/metal_mxfp4_moe.py")}


def rand_case(rng, max_bytes=1.2e9):
    while True:
        d = int(rng.choice([512, 768, 1024, 1536, 2048, 2560, 2880, 3072, 4096, 5120]))
        f = int(rng.choice([256, 384, 512, 640, 768, 1024, 1408, 1536, 2048, 2880, 4096]))
        E = int(rng.choice([8, 16, 32, 64, 128, 256]))
        k = int(rng.choice([1, 2, 4, 6, 8]))
        if k > E or 3 * E * d * f * 2 > max_bytes:
            continue
        T = int(rng.choice([1, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]))
        return MoEShape(f"d{d}f{f}E{E}k{k}", d=d, f=f, E=E, k=k), T, str(rng.choice(["bfloat16", "float16"])), \
            str(rng.choice(["uniform", "uniform", "zipf", "one_hot"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="q4", choices=list(PAIRS))
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--flag", type=float, default=1.0, help="report cases with speedup below this")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A, B = (be.load_variant(p) for p in PAIRS[a.pair])
    rng = np.random.default_rng(a.seed)
    probes, rows = [probe()], []
    for i in range(a.n):
        sh, T, dt, routing = rand_case(rng)
        w = be.make_weights(sh, dt, seed=i)
        sa, sb = A.prepare(*w), B.prepare(*w)
        mx.eval(sa, sb)
        inp = [be.make_inputs(sh, T, dt, routing, seed=1000 * i + s) for s in range(4)]
        x, idx, wts = inp[0]
        ref = be.reference(*B.dequant(sb), x, idx, wts)
        eb = be.errors(B.run(sb, x, idx, wts), ref)
        ea = be.errors(A.run(sa, x, idx, wts), be.reference(*A.dequant(sa), x, idx, wts))
        ta, tb = [], []
        for r in range(2):
            for which in ((0, 1) if r == 0 else (1, 0)):
                m, s, acc = (A, sa, ta) if which == 0 else (B, sb, tb)
                acc.append(float(np.median(be.time_chained(lambda *z: m.run(s, *z), inp, reps=5))) * 1e6)
        row = {"shape": sh.name, "d": sh.d, "f": sh.f, "E": sh.E, "k": sh.k, "T": T, "dtype": dt, "routing": routing,
               "A_us": min(ta), "B_us": min(tb), "speedup": min(ta) / min(tb), "errA": ea["rel_fro"],
               "errB": eb["rel_fro"], "okB": eb["finite"] and eb["rel_fro"] < be.TOL[dt]}
        rows.append(row)
        mark = "" if row["okB"] and row["speedup"] >= a.flag else "  <-- FLAG"
        print(f"{i:3d} {sh.name:20s} T={T:3d} {dt:8s} {routing:8s} A {row['A_us']:8.1f}us B {row['B_us']:8.1f}us "
              f"{row['speedup']:.3f}x errB {row['errB']:.1e}{mark}", flush=True)
        del w, sa, sb
        if i % 10 == 9:
            probes.append(probe())
    probes.append(probe())
    g = verdict(probes)
    sp = np.array([r["speedup"] for r in rows])
    print(f"\n{a.pair}: {len(rows)} cases, all correct: {all(r['okB'] for r in rows)}, speedup median {np.median(sp):.3f}x "
          f"min {sp.min():.3f}x, below {a.flag}: {int((sp < a.flag).sum())}")
    print(report(g))
    if a.out:
        json.dump({"args": vars(a), "rows": rows, "guard": g}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
