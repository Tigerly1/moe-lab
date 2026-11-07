"""Does a large streaming read between MoE layers (like attention over a long KV cache) erase the kernel gain?"""
import mlx.core as mx, numpy as np, time
from moelab import mlx_backend as be
from moelab.shapes import SHAPES
sh = SHAPES["qwen3-30b-a3b"]
w = be.make_weights(sh, "bfloat16")
A = be.load_variant("moelab/variants/mlx/mlxlm_switchglu_q4.py"); B = be.load_variant("moelab/variants/mlx/metal_q4_moe.py")
sa, sb = A.prepare(*w), B.prepare(*w); mx.eval(sa, sb)
inp = [be.make_inputs(sh, 1, "bfloat16", "uniform", 3000 + s) for s in range(48)]
def chain(fn, kv, L=48):
    x = inp[0][0]
    for i in range(L):
        if kv is not None:
            x = x + (kv[i % len(kv)].sum() * 0).astype(x.dtype)
        if fn is not None:
            _, idx, wts = inp[i]
            x = x + fn(x, idx, wts).astype(x.dtype)
    return x
def t(fn, kv, reps=15):
    mx.eval(chain(fn, kv)); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(chain(fn, kv)); mx.synchronize(); ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1e3
for mb in [0, 4, 8, 17, 34]:
    kv = None if mb == 0 else [mx.random.uniform(shape=(mb * 2**20 // 2,)).astype(mx.float16) for _ in range(8)]
    if kv: mx.eval(kv)
    r = {}
    for rnd in range(3):
        for name, fn in ([("stock", lambda *z: A.run(sa, *z)), ("ours", lambda *z: B.run(sb, *z)), ("kv-only", None)]
                         if rnd % 2 == 0 else [("kv-only", None), ("ours", lambda *z: B.run(sb, *z)), ("stock", lambda *z: A.run(sa, *z))]):
            if name == "kv-only" and kv is None: r.setdefault(name, []).append(0.0); continue
            r.setdefault(name, []).append(t(fn, kv))
    m = {k: np.median(v) for k, v in r.items()}
    print(f"KV read per layer {mb:2d} MB: stock {m['stock']:.2f} ms ours {m['ours']:.2f} kv-only {m['kv-only']:.2f} | "
          f"MoE part stock {m['stock']-m['kv-only']:.2f} ours {m['ours']-m['kv-only']:.2f} -> {(m['stock']-m['kv-only'])/(m['ours']-m['kv-only']):.3f}x", flush=True)
