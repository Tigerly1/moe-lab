"""Thermal / clock-boost robustness: interleaved A/B with NO pauses for --minutes.
If the GPU throttles, absolute times rise; the question is whether the A/B ratio moves.

    .venv/bin/python -m moelab.sustained A.py B.py --minutes 10 --out runs/sustained.json
Optionally, in another terminal (needs sudo, run it yourself):
    sudo powermetrics --samplers gpu_power -i 1000 > runs/powermetrics.txt
"""
import argparse
import json
import time

import mlx.core as mx
import numpy as np

from . import mlx_backend as be
from .shapes import SHAPES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A"); ap.add_argument("B")
    ap.add_argument("--shape", default="qwen3-30b-a3b"); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--T", type=int, default=1); ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--out", default="runs/sustained.json")
    a = ap.parse_args()
    sh = SHAPES[a.shape]
    w = be.make_weights(sh, a.dtype)
    mods = [be.load_variant(p) for p in (a.A, a.B)]
    st = [m.prepare(*w) for m in mods]
    mx.eval(st)
    inp = [be.make_inputs(sh, a.T, a.dtype, "uniform", s) for s in range(3)]
    rows, t0, r = [], time.time(), 0
    while time.time() - t0 < a.minutes * 60:
        order = (0, 1) if r % 2 == 0 else (1, 0)
        us = {}
        for i in order:
            us[i] = float(np.median(be.time_chained(lambda *z: mods[i].run(st[i], *z), inp, reps=5))) * 1e6
        rows.append({"t_s": time.time() - t0, "A_us": us[0], "B_us": us[1], "ratio": us[0] / us[1]})
        if r % 50 == 0:
            print(f"{rows[-1]['t_s']:6.0f}s  A {us[0]:7.1f}us  B {us[1]:7.1f}us  ratio {us[0] / us[1]:.3f}", flush=True)
        r += 1
    q = len(rows) // 4
    first, last = rows[:q], rows[-q:]
    summ = {k: {"A_us": float(np.median([x["A_us"] for x in g])), "B_us": float(np.median([x["B_us"] for x in g])),
                "ratio": float(np.median([x["ratio"] for x in g]))} for k, g in (("first_quarter", first), ("last_quarter", last))}
    summ["corr_ratio_vs_A_us"] = float(np.corrcoef([x["A_us"] for x in rows], [x["ratio"] for x in rows])[0, 1])
    print(json.dumps(summ, indent=1))
    json.dump({"args": vars(a), "rows": rows, "summary": summ}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
