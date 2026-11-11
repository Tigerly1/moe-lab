"""Round 13 probe: what does one small Metal dispatch cost inside a decode-like chain?

Synthetic, no model: 48 "layers", each one 4-bit GEMV (2048 -> 4096, ~4.7 MB of weights, memory-bound)
followed by `n` tiny ops on a 2048-vector (rms_norm / add). The slope of time over n is the cost per
small dispatch. A Qwen3 decode layer issues roughly 23-29 of them (count below is from the model code).

Light by design: ~250 MB of weights, a few seconds of GPU. Run each buffer setting in a fresh process:
    MLX_MAX_OPS_PER_BUFFER=50  python scripts/r13_dispatch_probe.py
    MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=200 python scripts/r13_dispatch_probe.py
"""

import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn

LAYERS, D, O, REPS, ROUNDS = 48, 2048, 4096, 30, 5


def make_layers():
    layers = []
    for _ in range(LAYERS):
        lin = nn.QuantizedLinear(D, O, bias=False, group_size=64, bits=4)
        lin.set_dtype(mx.bfloat16)
        mx.eval(lin.parameters())  # per layer, so the fp32 init weights are freed at once
        layers.append(lin)
    return layers


# A user kernel that does as little as an add: is a mx.fast.metal_kernel call dearer than a built-in op?
_custom = mx.fast.metal_kernel(
    name="r13_probe_add",
    input_names=["a", "b"],
    output_names=["out"],
    source="uint i = thread_position_in_grid.x; out[i] = static_cast<T>(a[i] + b[i]);",
)


def step(layers, x, w, n, kind):
    for lin in layers:
        h = lin(x)[..., :D]
        for i in range(n):
            if kind == "norm":
                h = mx.fast.rms_norm(h, w, 1e-6)
            elif kind == "custom":
                h = _custom(inputs=[h, x], template=[("T", x.dtype)], grid=(D, 1, 1), threadgroup=(256, 1, 1),
                            output_shapes=[x.shape], output_dtypes=[x.dtype])[0]
            else:
                h = h + x
        x = h
    return x


def ms_per_step(layers, x, w, n, kind):
    best = float("inf")
    for _ in range(ROUNDS):
        t = time.perf_counter()
        for _ in range(REPS):
            mx.eval(step(layers, x, w, n, kind))
        best = min(best, (time.perf_counter() - t) / REPS * 1e3)
    return best


def main():
    layers = make_layers()
    x = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
    w = mx.ones((D,), mx.bfloat16)
    mx.eval(x, w)
    ms_per_step(layers, x, w, 4, "norm")  # warm up, compile kernels
    out = {
        "ops_per_buffer": os.environ.get("MLX_MAX_OPS_PER_BUFFER", "default"),
        "mb_per_buffer": os.environ.get("MLX_MAX_MB_PER_BUFFER", "default"),
        "arch": mx.device_info().get("architecture"),
    }
    for kind in ("norm", "add", "custom"):
        ns = (0, 4, 8, 16, 32)
        ts = [ms_per_step(layers, x, w, n, kind) for n in ns]
        # least-squares slope: ms per (48 * n) dispatches -> us per dispatch
        mn, mt = sum(ns) / len(ns), sum(ts) / len(ts)
        slope = sum((a - mn) * (b - mt) for a, b in zip(ns, ts)) / sum((a - mn) ** 2 for a in ns)
        out[kind] = {"n": ns, "ms_per_step": [round(t, 3) for t in ts], "us_per_dispatch": round(slope / LAYERS * 1e3, 2)}
        print(kind, out[kind])
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
