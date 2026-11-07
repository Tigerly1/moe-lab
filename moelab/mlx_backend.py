"""Apple Silicon backend: problem setup, fp32 reference, timing. All on GPU via MLX.

Variant contract (file in moelab/variants/mlx/*.py):
    META = {"name": str, "description": str}
    def prepare(w_gate, w_up, w_down) -> state            # not timed (quantize, re-layout...)
    def run(state, x, idx, wts) -> mx.array [T, d]        # timed
    def dequant(state) -> (w_gate, w_up, w_down)          # optional, for lossy formats
"""
import importlib.util
import time

import mlx.core as mx
import numpy as np

from . import roofline
from .problem import make_routing

DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}


def load_variant(path):
    spec = importlib.util.spec_from_file_location(f"variant_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_weights(shape, dtype, seed=0):
    mx.random.seed(seed)
    d, f, E = shape.d, shape.f, shape.E
    dt = DTYPES[dtype]
    wg = (mx.random.normal((E, f, d)) * d ** -0.5).astype(dt)
    wu = (mx.random.normal((E, f, d)) * d ** -0.5).astype(dt)
    wd = (mx.random.normal((E, d, f)) * f ** -0.5).astype(dt)
    mx.eval(wg, wu, wd)
    return wg, wu, wd


def make_inputs(shape, T, dtype, routing, seed):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((T, shape.d)).astype(np.float32)).astype(DTYPES[dtype])
    idx, wts = make_routing(T, shape.E, shape.k, routing, rng)
    out = (x, mx.array(idx), mx.array(wts))
    mx.eval(*out)
    return out


def reference(wg, wu, wd, x, idx, wts, act=0):
    """fp32 per-expert loop. Ground truth for the GPU (validated vs numpy fp64 in tests).
    act 0: silu(gate) * up (SwiGLU); act 1: tanh-GELU(gate) * up (Gemma-4 GeGLU)."""
    x32 = x.astype(mx.float32)
    T, d = x.shape
    y = mx.zeros((T, d), dtype=mx.float32)
    idx_np = np.array(idx)
    for e in np.unique(idx_np):
        t, j = np.nonzero(idx_np == e)
        t_mx = mx.array(t)
        xe = x32[t_mx]
        g = xe @ wg[int(e)].astype(mx.float32).T
        u = xe @ wu[int(e)].astype(mx.float32).T
        a = g * mx.sigmoid(g) if act == 0 else 0.5 * g * (1 + mx.tanh(0.7978845608028654 * (g + 0.044715 * g ** 3)))
        o = (a * u) @ wd[int(e)].astype(mx.float32).T
        y = y.at[t_mx].add(o * wts[t_mx, mx.array(j)][:, None])
    mx.eval(y)
    return y


def errors(y, y_ref):
    y = y.astype(mx.float32)
    diff = y - y_ref
    nr = mx.linalg.norm(y_ref.flatten())
    rel = (mx.linalg.norm(diff.flatten()) / (nr + 1e-30)).item()
    return {"rel_fro": rel, "max_abs": mx.abs(diff).max().item(), "finite": bool(mx.isfinite(y).all().item())}


def time_calls(fn, inputs, warmup, iters):
    """Per-call latency incl. dispatch (decode-realistic). Rotates inputs to defeat caching."""
    for i in range(warmup):
        mx.eval(fn(*inputs[i % len(inputs)]))
    mx.synchronize()
    ts = []
    for i in range(iters):
        args = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        mx.eval(fn(*args))
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return np.array(ts)


def time_chained(fn, inputs, reps, chain=10):
    """In-graph timing: chain `chain` dependent calls with a residual stream
    (x_{i+1} = x_i + y_i, like stacked layers), one eval per chain. Removes the ~130us
    host eval/sync round-trip that otherwise dominates decode-sized kernels.
    The residual keeps activations O(1); without it SwiGLU drives x to exact zeros
    after ~3 calls (found by audit 2026-11-07). The residual add is paid equally by
    every variant. Returns per-call seconds per rep."""
    def build():
        x = inputs[0][0]
        for i in range(chain):
            _, idx, wts = inputs[i % len(inputs)]
            x = x + fn(x, idx, wts).astype(x.dtype)
        return x
    mx.eval(build())  # warmup + compile
    mx.synchronize()
    ts, hs = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        x = build()                      # host: Python + lazy graph construction
        t1 = time.perf_counter()
        mx.eval(x)                       # device: encode + GPU execution
        mx.synchronize()
        t2 = time.perf_counter()
        ts.append((t2 - t1) / chain)
        hs.append((t1 - t0) / chain)
    time_chained.last_host = float(np.median(hs))
    return np.array(ts)


def measure_peak_bw(n_bytes=1 << 30):
    """STREAM-like copy: read + write n_bytes each. Returns GB/s."""
    a = mx.zeros((n_bytes // 2,), dtype=mx.float16)
    mx.eval(a)
    ts = time_calls(lambda a: a + 1, [(a,)], 3, 10)
    return 2 * n_bytes / np.median(ts) / 1e9


def bench_variant(mod, shape, tokens, dtype="float16", routing="uniform", hw="m4-pro",
                  warmup=5, iters=30, n_rot=3, check=True, weights=None, emit=print):
    wg, wu, wd = weights if weights is not None else make_weights(shape, dtype)
    state = mod.prepare(wg, wu, wd)
    mx.eval(state)
    rwg, rwu, rwd = mod.dequant(state) if hasattr(mod, "dequant") else (wg, wu, wd)
    w_bits = getattr(mod, "W_BITS", roofline.BITS[dtype])
    for T in tokens:
        inputs = [make_inputs(shape, T, dtype, routing, seed=s) for s in range(n_rot)]
        res = {"type": "result", "variant": mod.META["name"], "T": T}
        if check:
            x, idx, wts = inputs[0]
            y = mod.run(state, x, idx, wts)
            mx.eval(y)
            res.update(errors(y, reference(rwg, rwu, rwd, x, idx, wts)))
            res["ok"] = res["finite"] and res["rel_fro"] < TOL[dtype]
        fn = lambda *a: mod.run(state, *a)  # noqa: E731
        ts = time_chained(fn, inputs, reps=max(5, iters // 3))
        res["host_us"] = time_chained.last_host * 1e6  # graph-build cost; overlaps GPU under async eval
        sync = time_calls(fn, inputs, warmup=2, iters=5)
        med = float(np.median(ts))
        res["sync_us"] = float(np.median(sync)) * 1e6  # per-call incl. host eval+sync
        U = float(np.mean([len(np.unique(np.array(i[1]))) for i in inputs]))
        nbytes = roofline.bytes_moved(T, shape.d, shape.f, shape.E, shape.k, w_bits, unique=U)
        fl = roofline.flops(T, shape.d, shape.f, shape.k)
        floor = roofline.floor_time_s(T, shape.d, shape.f, shape.E, shape.k, hw, w_bits, unique=U)
        res.update({
            "median_us": med * 1e6, "p20_us": float(np.percentile(ts, 20)) * 1e6,
            "p80_us": float(np.percentile(ts, 80)) * 1e6, "unique_experts": U,
            "gbs": nbytes / med / 1e9, "tflops": fl / med / 1e12,
            "floor_us": floor * 1e6, "frac_of_floor": floor / med,
        })
        emit(res)


TOL = {"float32": 1e-4, "float16": 5e-3, "bfloat16": 1e-2}  # bf16 tightened 3e-2 -> 1e-2 (audit 2)
