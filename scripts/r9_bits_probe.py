# Round 9. Stock mlx-lm expert path (fallback == SwitchGLU, bit-identical) at T=1: achieved GB/s per bit width.
import sys, time, mlx.core as mx, numpy as np
sys.path.insert(0, ".")
from moelab.fallback import gather_moe
from moelab.mlx_backend import load_variant
q4 = load_variant("moelab/variants/mlx/metal_q4_moe.py")
def tm(f, k=200):   # f(i): call i uses its own routing (no SLC reuse across calls)
    for i in range(20): mx.eval(f(i))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        ys = [f(i) for i in range(k)]   # every call must be evaluated (MLX is lazy)
        mx.eval(ys); ts.append((time.perf_counter() - t0) / k)
    return np.median(ts)
for name, D, F, E, K in (("qwen36", 2048, 512, 256, 8), ("gemma4", 2816, 704, 128, 8), ("qwen3", 2048, 768, 128, 8)):
    mx.random.seed(0)
    w = [mx.random.normal((E, F, D)).astype(mx.bfloat16) * 0.02, mx.random.normal((E, F, D)).astype(mx.bfloat16) * 0.02,
         mx.random.normal((E, D, F)).astype(mx.bfloat16) * 0.02]
    xs_ = [mx.random.normal((1, D)).astype(mx.bfloat16) for _ in range(200)]; x = xs_[0]
    rng = np.random.default_rng(0)
    idxs = [mx.array(rng.choice(E, K, replace=False)[None].astype(np.int32)) for _ in range(200)]
    wts = mx.full((1, K), 1.0 / K, mx.float32)
    row = []
    for bits in (4, 6, 8):
        s = {n: mx.quantize(v, group_size=64, bits=bits) for n, v in zip("gud", w)}
        mx.eval(s)
        t = tm(lambda i: gather_moe(s, xs_[i], idxs[i], wts, 64, bits))
        by = K * 3 * D * F * (bits + 32 / 64) / 8
        row.append(f"{bits}b stock {t*1e6:6.1f}us {by/t/1e9:5.0f} GB/s")
        if bits == 4:
            tv = tm(lambda i: q4.run(s, xs_[i], idxs[i], wts))
            row.append(f"4b ours {tv*1e6:6.1f}us {by/tv/1e9:5.0f} GB/s")
    print(name, " | ".join(row), flush=True)
