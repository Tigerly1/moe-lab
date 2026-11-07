"""GPU contention guard (round 7, 2026-11-07).

Why: on 2026-11-07 a T=1 slope run with Chrome + display active gave 80% of floor instead of the 93%
measured at night. The A/B speedup barely moved (1.196x vs 1.222x), but absolute numbers did.
So every timing run now brackets itself with a read-bandwidth probe and compares it with the best probe
ever seen on this chip (runs/hw_probe.json, self-calibrating: the record only goes up).

probe()  -> GB/s of a 512 MB fp16 sum (read-bound), median of 9 calls.
verdict(probes) -> dict with best, min, ratio and contended = min < THRESH * best.
"""
import json
import os
import subprocess

THRESH = 0.92
PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "hw_probe.json")


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def probe(n_bytes=1 << 29):
    import mlx.core as mx
    import numpy as np
    from .mlx_backend import time_calls
    a = mx.random.uniform(shape=(n_bytes // 2,)).astype(mx.float16)
    mx.eval(a)
    ts = time_calls(lambda a: a.sum(), [(a,)], 3, 9)
    return float(n_bytes / np.median(ts) / 1e9)


def verdict(probes, update=True):
    rec = {}
    if os.path.exists(PATH):
        rec = json.load(open(PATH))
    c = chip()
    best = max([rec.get(c, 0.0), *probes])
    if update and best > rec.get(c, 0.0):
        rec[c] = best
        json.dump(rec, open(PATH, "w"), indent=1)
    lo = min(probes)
    return {"chip": c, "probe_best_gbs": best, "probe_min_gbs": lo, "probe_ratio": lo / best,
            "contended": lo < THRESH * best}


def report(v):
    tag = "CONTENDED - absolute numbers unreliable, close other GPU users and re-run" if v["contended"] else "ok"
    return (f"guard: probe min {v['probe_min_gbs']:.0f} GB/s vs best {v['probe_best_gbs']:.0f} on {v['chip']} "
            f"({100 * v['probe_ratio']:.0f}%) -> {tag}")
