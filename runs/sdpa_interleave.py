"""Layer chain: real SDPA over a ctx-long KV cache (Qwen3 shapes: 32 q heads, 4 kv heads, dim 128) between MoE calls."""
import mlx.core as mx, numpy as np, time
from moelab import mlx_backend as be
from moelab.shapes import SHAPES
sh = SHAPES["qwen3-30b-a3b"]
w = be.make_weights(sh, "bfloat16")
A = be.load_variant("moelab/variants/mlx/mlxlm_switchglu_q4.py"); B = be.load_variant("moelab/variants/mlx/metal_q4_moe.py")
sa, sb = A.prepare(*w), B.prepare(*w); mx.eval(sa, sb)
inp = [be.make_inputs(sh, 1, "bfloat16", "uniform", 3000 + s) for s in range(48)]
wq = (mx.random.normal((2048, 32 * 128)) * 0.02).astype(mx.bfloat16); wo = (mx.random.normal((32 * 128, 2048)) * 0.02).astype(mx.bfloat16)
def chain(fn, kvs, L=48):
    x = inp[0][0]
    for i in range(L):
        if kvs is not None:
            k, v = kvs[i % len(kvs)]
            q = (x @ wq).reshape(1, 1, 32, 128).transpose(0, 2, 1, 3)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5)
            x = x + o.transpose(0, 2, 1, 3).reshape(1, -1) @ wo
        if fn is not None:
            _, idx, wts = inp[i]
            x = x + fn(x, idx, wts).astype(x.dtype)
    return x
def t(fn, kvs, reps=15):
    mx.eval(chain(fn, kvs)); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(chain(fn, kvs)); mx.synchronize(); ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1e3
for ctx in [0, 1024, 4096, 8192, 16384]:
    kvs = None if ctx == 0 else [(mx.random.normal((1, 4, ctx, 128)).astype(mx.bfloat16), mx.random.normal((1, 4, ctx, 128)).astype(mx.bfloat16)) for _ in range(6)]
    if kvs: mx.eval(kvs)
    r = {}
    for rnd in range(3):
        order = [("stock", lambda *z: A.run(sa, *z)), ("ours", lambda *z: B.run(sb, *z)), ("attn-only", None)]
        for name, fn in (order if rnd % 2 == 0 else order[::-1]):
            if name == "attn-only" and kvs is None: r.setdefault(name, []).append(0.0); continue
            r.setdefault(name, []).append(t(fn, kvs))
    m = {k: np.median(v) for k, v in r.items()}
    print(f"ctx {ctx:5d}: stock {m['stock']:.2f} ms ours {m['ours']:.2f} attn-only {m['attn-only']:.2f} | MoE stock {m['stock']-m['attn-only']:.2f} ours {m['ours']-m['attn-only']:.2f} -> {(m['stock']-m['attn-only'])/(m['ours']-m['attn-only']):.3f}x", flush=True)
