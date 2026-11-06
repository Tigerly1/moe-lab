"""Rigorous A/B: interleaved rounds inside each process, several independent process
launches, paired bootstrap 95% CI on the speedup (median_A / median_B).

    .venv/bin/python -m moelab.compare A.py B.py --shape qwen3-30b-a3b --dtype bfloat16 \
        --tokens 1 2 4 8 --launches 5 --rounds 6 --out runs/cmp.json
"""
import argparse
import json
import subprocess
import sys

import numpy as np


def child(a):
    import mlx.core as mx
    from . import mlx_backend as be
    from .shapes import SHAPES
    sh = SHAPES[a.shape]
    w = be.make_weights(sh, a.dtype, seed=a.seed)
    mods = [be.load_variant(p) for p in (a.A, a.B)]
    states = [m.prepare(*w) for m in mods]
    mx.eval(states)
    for T in a.tokens:
        inp = [be.make_inputs(sh, T, a.dtype, a.routing, seed=a.seed * 100 + s) for s in range(3)]
        x, idx, wts = inp[0]
        ref = be.reference(*(mods[1].dequant(states[1]) if hasattr(mods[1], "dequant") else w), x, idx, wts)
        ok = [be.errors(m.run(s, x, idx, wts), ref)["rel_fro"] for m, s in zip(mods, states)]
        for r in range(a.rounds):
            order = (0, 1) if r % 2 == 0 else (1, 0)
            res = {}
            for i in order:
                ts = be.time_chained(lambda *z: mods[i].run(states[i], *z), inp, reps=5)
                res[i] = float(np.median(ts)) * 1e6
            print(json.dumps({"T": T, "round": r, "A_us": res[0], "B_us": res[1], "errA": ok[0], "errB": ok[1]}), flush=True)


def boot_ci(a_us, b_us, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a_us, b_us = np.asarray(a_us), np.asarray(b_us)
    idx = rng.integers(0, len(a_us), (n, len(a_us)))
    r = np.median(a_us[idx], 1) / np.median(b_us[idx], 1)
    return float(np.median(a_us) / np.median(b_us)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A"); ap.add_argument("B")
    ap.add_argument("--shape", default="qwen3-30b-a3b"); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--routing", default="uniform")
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--launches", type=int, default=5); ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--child", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.child:
        return child(a)
    rows = []
    for L in range(a.launches):
        cmd = [sys.executable, "-m", "moelab.compare", a.A, a.B, "--child", "--shape", a.shape, "--dtype", a.dtype,
               "--routing", a.routing, "--rounds", str(a.rounds), "--seed", str(L), "--tokens", *map(str, a.tokens)]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        rows += [dict(json.loads(l), launch=L) for l in out.splitlines() if l.startswith("{")]
    summary = []
    print(f"A={a.A.split('/')[-1]}  B={a.B.split('/')[-1]}  shape={a.shape} {a.dtype} {a.routing}  "
          f"{a.launches} launches x {a.rounds} rounds")
    for T in a.tokens:
        rs = [r for r in rows if r["T"] == T]
        sp, lo, hi = boot_ci([r["A_us"] for r in rs], [r["B_us"] for r in rs])
        s = {"T": T, "A_us": float(np.median([r["A_us"] for r in rs])), "B_us": float(np.median([r["B_us"] for r in rs])),
             "speedup_B_over_A": sp, "ci95": [lo, hi], "errA": rs[0]["errA"], "errB": rs[0]["errB"], "n": len(rs)}
        summary.append(s)
        print(f"T={T:3d}  A {s['A_us']:8.1f}us  B {s['B_us']:8.1f}us  B is {sp:.3f}x  [95% CI {lo:.3f}-{hi:.3f}]  "
              f"err A {s['errA']:.1e} B {s['errB']:.1e}")
    if a.out:
        json.dump({"args": vars(a), "rows": rows, "summary": summary}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
