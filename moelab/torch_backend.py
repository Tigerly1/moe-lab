"""NVIDIA backend (PyTorch + Triton/CUDA). Same contract and outputs as mlx_backend.

Variant contract (moelab/variants/cuda/*.py), tensors are torch on DEVICE:
    META, prepare(w_gate, w_up, w_down) -> state, run(state, x, idx, wts) -> y, dequant(state) optional
Runs on cuda when available, else cpu (for local testing of torch-only variants).
"""
import importlib.util
import platform
import time

import numpy as np
import torch

from . import roofline
from .problem import make_routing

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
TOL = {"float32": 1e-4, "float16": 5e-3, "bfloat16": 3e-2}


def env_info():
    info = {"device": DEVICE, "torch": torch.__version__, "python": platform.python_version()}
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, cc=f"{p.major}.{p.minor}", cuda=torch.version.cuda)
        try:
            import triton
            info["triton"] = triton.__version__
        except ImportError:
            pass
    return info


def load_variant(path):
    # Triton's @jit reads source via inspect, so variants must be real files (they are).
    spec = importlib.util.spec_from_file_location(f"variant_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_weights(shape, dtype, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    d, f, E = shape.d, shape.f, shape.E
    mk = lambda *s, sc: (torch.randn(*s, generator=g, device=DEVICE) * sc).to(DTYPES[dtype])  # noqa: E731
    return mk(E, f, d, sc=d ** -0.5), mk(E, f, d, sc=d ** -0.5), mk(E, d, f, sc=f ** -0.5)


def make_inputs(shape, T, dtype, routing, seed):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.standard_normal((T, shape.d)).astype(np.float32)).to(DEVICE, DTYPES[dtype])
    idx, wts = make_routing(T, shape.E, shape.k, routing, rng)
    return x, torch.from_numpy(idx).to(DEVICE), torch.from_numpy(wts).to(DEVICE)


@torch.no_grad()
def reference(wg, wu, wd, x, idx, wts):
    x32 = x.float()
    y = torch.zeros_like(x32)
    for e in idx.unique().tolist():
        t, j = (idx == e).nonzero(as_tuple=True)
        xe = x32[t]
        g, u = xe @ wg[e].float().T, xe @ wu[e].float().T
        o = (torch.nn.functional.silu(g) * u) @ wd[e].float().T
        y.index_add_(0, t, o * wts[t, j, None])
    return y


def errors(y, y_ref):
    diff = y.float() - y_ref
    return {"rel_fro": (diff.norm() / (y_ref.norm() + 1e-30)).item(), "max_abs": diff.abs().max().item(),
            "finite": bool(torch.isfinite(y).all().item())}


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def time_chained(fn, inputs, reps, chain=10):
    """Chain of dependent calls, CUDA events around the whole chain (device-wide sync
    before the end event defeats side-stream timing exploits)."""
    def once():
        x = inputs[0][0]
        for i in range(chain):
            _, idx, wts = inputs[i % len(inputs)]
            x = fn(x, idx, wts).to(x.dtype)
        return x
    once()
    _sync()
    ts = []
    for _ in range(reps):
        if DEVICE == "cuda":
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); once(); torch.cuda.synchronize(); b.record(); b.synchronize()  # noqa: E702
            ts.append(a.elapsed_time(b) / 1e3 / chain)
        else:
            t0 = time.perf_counter(); once(); ts.append((time.perf_counter() - t0) / chain)  # noqa: E702
    return np.array(ts)


@torch.no_grad()
def bench_variant(mod, shape, tokens, dtype="float16", routing="uniform", hw="h100",
                  iters=30, n_rot=3, check=True, weights=None, emit=print):
    wg, wu, wd = weights if weights is not None else make_weights(shape, dtype)
    state = mod.prepare(wg, wu, wd)
    rwg, rwu, rwd = mod.dequant(state) if hasattr(mod, "dequant") else (wg, wu, wd)
    w_bits = getattr(mod, "W_BITS", roofline.BITS[dtype])
    for T in tokens:
        inputs = [make_inputs(shape, T, dtype, routing, seed=s) for s in range(n_rot)]
        res = {"type": "result", "variant": mod.META["name"], "T": T}
        if check:
            x, idx, wts = inputs[0]
            res.update(errors(mod.run(state, x, idx, wts), reference(rwg, rwu, rwd, x, idx, wts)))
            res["ok"] = res["finite"] and res["rel_fro"] < TOL[dtype]
        ts = time_chained(lambda *a: mod.run(state, *a), inputs, reps=max(5, iters // 3))
        med = float(np.median(ts))
        U = float(np.mean([i[1].unique().numel() for i in inputs]))
        nbytes = roofline.bytes_moved(T, shape.d, shape.f, shape.E, shape.k, w_bits, unique=U)
        floor = roofline.floor_time_s(T, shape.d, shape.f, shape.E, shape.k, hw, w_bits, unique=U)
        res.update({"median_us": med * 1e6, "p20_us": float(np.percentile(ts, 20)) * 1e6,
                    "p80_us": float(np.percentile(ts, 80)) * 1e6, "unique_experts": U,
                    "gbs": nbytes / med / 1e9, "tflops": roofline.flops(T, shape.d, shape.f, shape.k) / med / 1e12,
                    "floor_us": floor * 1e6, "frac_of_floor": floor / med})
        emit(res)
