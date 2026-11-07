"""GPU contention guard (round 7, 2026-11-07; redesigned after the round-7 audit).

Why: on 2026-11-07 a T=1 slope run with Chrome + display active gave 80% of floor instead of the 93%
measured at night. The A/B speedup barely moved (1.196x vs 1.222x), but absolute numbers did.
Every timing run brackets itself with a read-bandwidth probe and compares it with this machine's history.

Reference = 90th percentile of the last N probes recorded for the same (chip, power source) key in
runs/hw_probe.json. (v1 used the best probe ever seen: a running maximum that drifts up with noise and
re-labels old runs; audit finding.) Fewer than MIN_HISTORY probes -> verdict "calibrating", not "ok".

probe()  -> GB/s of a 512 MB fp16 sum (read-bound), median of 9 calls.
verdict(probes) -> dict with reference, min, ratio, status in {"ok", "contended", "calibrating"}.
"""
import json
import os
import subprocess

import numpy as np

THRESH, N_HISTORY, MIN_HISTORY = 0.92, 40, 5
PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "hw_probe.json")


def _sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout
    except OSError:
        return ""


def key():
    chip = _sh("sysctl", "-n", "machdep.cpu.brand_string").strip() or "unknown"
    power = "battery" if "Battery Power" in _sh("pmset", "-g", "batt") else "ac"
    return f"{chip}|{power}"


def probe(n_bytes=1 << 29):
    import mlx.core as mx
    from .mlx_backend import time_calls
    a = mx.random.uniform(shape=(n_bytes // 2,)).astype(mx.float16)
    mx.eval(a)
    ts = time_calls(lambda a: a.sum(), [(a,)], 3, 9)
    return float(n_bytes / np.median(ts) / 1e9)


def verdict(probes, update=True):
    rec = json.load(open(PATH)) if os.path.exists(PATH) else {}
    k = key()
    hist = rec.get(k, [])
    hist = hist if isinstance(hist, list) else []          # v1 stored a single float per chip
    lo = float(min(probes))
    if len(hist) >= MIN_HISTORY:
        ref = float(np.percentile(hist, 90))
        status = "contended" if lo < THRESH * ref else "ok"
    else:
        ref, status = float(max(probes)), "calibrating"
    if update:
        rec[k] = (hist + [float(p) for p in probes])[-N_HISTORY:]
        json.dump(rec, open(PATH, "w"), indent=1)
    return {"key": k, "probe_ref_gbs": ref, "probe_min_gbs": lo, "probe_ratio": lo / ref, "status": status,
            "contended": status == "contended", "history": len(hist)}


def report(v):
    tag = {"ok": "ok", "calibrating": f"calibrating ({v['history']} probes in history) - verdict not yet meaningful",
           "contended": "CONTENDED - absolute numbers unreliable, close other GPU users and re-run"}[v["status"]]
    return (f"guard: probe min {v['probe_min_gbs']:.0f} GB/s vs p90 {v['probe_ref_gbs']:.0f} for {v['key']} "
            f"({100 * v['probe_ratio']:.0f}%) -> {tag}")
