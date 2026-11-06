"""Problem generator + fp32 reference. Framework-agnostic (numpy).

The kernel contract (every variant, every backend):
    y = moe(x, w_gate, w_up, w_down, idx, wts)
    x      [T, d]
    w_gate [E, f, d], w_up [E, f, d], w_down [E, d, f]    (out, in) layout like mlx SwitchLinear
    idx    [T, k] int32  expert ids per token (from router)
    wts    [T, k] float  combine weights
    y      [T, d] = sum_j wts[t,j] * down_e( silu(gate_e x) * up_e x ),  e = idx[t,j]
Routing (router GEMM + top-k) is benchmarked separately.
"""
import numpy as np


def make_routing(T, E, k, mode="uniform", rng=None, zipf_a=1.2):
    rng = rng or np.random.default_rng(0)
    if mode == "uniform":
        logits = rng.standard_normal((T, E))
    elif mode == "zipf":  # skewed popularity, like real routers
        pop = 1.0 / np.arange(1, E + 1) ** zipf_a
        logits = rng.standard_normal((T, E)) + np.log(pop)[rng.permutation(E)] * 2.0
    elif mode == "trace":  # real Qwen3-30B-A3B routing: T consecutive tokens, random layer/domain
        return _trace_routing(T, E, k, rng)
    elif mode == "one_hot":  # all tokens to same k experts (worst imbalance)
        logits = np.tile(np.arange(E, 0, -1, dtype=np.float64), (T, 1))
    else:
        raise ValueError(mode)
    idx = np.argsort(-logits, axis=1)[:, :k].astype(np.int32)
    top = np.take_along_axis(logits, idx, axis=1)
    w = np.exp(top - top.max(1, keepdims=True))
    w = (w / w.sum(1, keepdims=True)).astype(np.float32)
    return idx, w


_TRACE = None


def _trace_routing(T, E, k, rng):
    """Windows of T consecutive tokens from recorded traces (moelab/routing_trace.py)."""
    global _TRACE
    if _TRACE is None:
        import os
        z = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs/routing_qwen3.npz"))
        _TRACE = [z[d] for d in z.files if d != "E"]
        assert int(z["E"]) == E and _TRACE[0].shape[2] == k, "trace recorded for E=128, k=8 (Qwen3) only"
    tr = _TRACE[rng.integers(len(_TRACE))]
    layer, start = rng.integers(tr.shape[0]), rng.integers(tr.shape[1] - T)
    idx = tr[layer, start:start + T].astype(np.int32)
    w = rng.random((T, k)).astype(np.float32)
    return idx, w / w.sum(1, keepdims=True)


def make_problem(shape, T, routing="uniform", seed=0):
    rng = np.random.default_rng(seed)
    d, f, E, k = shape.d, shape.f, shape.E, shape.k
    s_in, s_hid = 1.0 / np.sqrt(d), 1.0 / np.sqrt(f)
    p = {
        "x": rng.standard_normal((T, d)).astype(np.float32),
        "w_gate": (rng.standard_normal((E, f, d)) * s_in).astype(np.float32),
        "w_up": (rng.standard_normal((E, f, d)) * s_in).astype(np.float32),
        "w_down": (rng.standard_normal((E, d, f)) * s_hid).astype(np.float32),
    }
    p["idx"], p["wts"] = make_routing(T, E, k, routing, rng)
    return p


def reference(p):
    """fp64 loop over experts. Slow but obviously correct."""
    with np.errstate(all="ignore"):  # macOS Accelerate raises spurious FP flags
        return _reference(p)


def _reference(p):
    x = p["x"].astype(np.float64)
    T, d = x.shape
    y = np.zeros((T, d))
    for e in np.unique(p["idx"]):
        t, j = np.nonzero(p["idx"] == e)
        xe = x[t]
        g = xe @ p["w_gate"][e].T.astype(np.float64)
        u = xe @ p["w_up"][e].T.astype(np.float64)
        h = g / (1.0 + np.exp(-g)) * u
        o = h @ p["w_down"][e].T.astype(np.float64)
        np.add.at(y, t, o * p["wts"][t, j][:, None])
    return y.astype(np.float32)


def error_metrics(y, y_ref):
    y = np.asarray(y, dtype=np.float64)
    r = np.asarray(y_ref, dtype=np.float64)
    diff = y - r
    nr = np.linalg.norm(r) + 1e-30
    return {
        "rel_fro": float(np.linalg.norm(diff) / nr),
        "max_abs": float(np.abs(diff).max()),
        "cos": float((y * r).sum() / (np.linalg.norm(y) * nr + 1e-30)),
        "finite": bool(np.isfinite(y).all()),
    }


TOLERANCE = {"float32": 1e-4, "float16": 3e-3, "bfloat16": 2e-2, "q8": 2e-2, "q4": 2e-2}
