"""Cost of the Qwen3 router path (gate Linear -> softmax -> argpartition top-k -> take -> normalize) per layer, T=1,
in a 48-layer chain, vs the expert kernels. Uses the real model's gate weights/format."""
import mlx.core as mx, numpy as np, time
from mlx_lm import load
model, _ = load("mlx-community/Qwen3-30B-A3B-4bit")
blocks = [l.mlp for l in model.layers]
g = blocks[0].gate
print("gate type", type(g).__name__, {k: getattr(g, k, None) for k in ("bits", "group_size")})
x0 = (mx.random.normal((1, 1, 2048)) * 0.1).astype(mx.bfloat16)
def router(b, x):
    gates = mx.softmax(b.gate(x), axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-8, axis=-1)[..., -8:]
    sc = mx.take_along_axis(gates, inds, axis=-1)
    sc = sc / mx.sum(sc, axis=-1, keepdims=True)
    return inds, sc
def chain(mode):
    x = x0
    for b in blocks:
        if mode == "gate":
            x = x + b.gate(x)[..., :1].astype(x.dtype) * 0
        elif mode == "router":
            i, s = router(b, x)
            x = x + (s.sum() * 0 + i.sum() * 0).astype(x.dtype)
        else:
            x = x + 0 * x.sum()
    return x
def t(mode, reps=30):
    mx.eval(chain(mode)); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(chain(mode)); mx.synchronize(); ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1e3
res = {m: np.median([t(m) for _ in range(3)]) for m in ("base", "gate", "router")}
print({k: round(v, 3) for k, v in res.items()})
print(f"per token (48 layers): gate GEMV {res['gate']-res['base']:.2f} ms, full router {res['router']-res['base']:.2f} ms")
