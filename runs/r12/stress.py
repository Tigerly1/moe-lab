"""Stress the PR head beyond its own tests: kernel path vs stock SwitchGLU (kernels disabled) and vs fp32 truth."""
import sys, numpy as np, mlx.core as mx, mlx.nn as nn, mlx_lm
from mlx_lm.models import switch_kernels as sk
from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU
print("mlx_lm from", mlx_lm.__file__.split("/")[-3], "mlx", mx.__version__)

class GeGLU(nn.Module):
    metal_act = 1
    def __call__(self, x, gate): return nn.gelu_approx(gate) * x

def truth(glu, x, idx):
    w = [mx.dequantize(p.weight, p.scales, p.biases, group_size=64, bits=4).astype(mx.float32) for p in (glu.gate_proj, glu.up_proj, glu.down_proj)]
    wg, wu, wd = (a[idx] for a in w)
    x = x.astype(mx.float32)[..., None, :, None]
    g, u = (wg @ x).squeeze(-1), (wu @ x).squeeze(-1)
    a = nn.silu(g) if glu.activation.metal_act == 0 else nn.gelu_approx(g)
    return (wd @ (a * u)[..., None]).squeeze(-1)

def rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.linalg.norm((a - b).flatten()) / mx.maximum(mx.linalg.norm(b.flatten()), 1e-30)).item()

def stock(glu, x, idx):
    k, sk._kernels = sk._kernels, None
    try:
        y = glu(x, idx); mx.eval(y); return y
    finally: sk._kernels = k

def routing(kind, shape, E, k, rng):
    T = int(np.prod(shape))
    if kind == "uniform": idx = np.stack([rng.permutation(E)[:k] for _ in range(T)])
    elif kind == "hot": idx = np.stack([np.sort(rng.choice(E, k, replace=False, p=(lambda p: p / p.sum())(1.0 / np.arange(1, E + 1) ** 1.5))) for _ in range(T)])
    elif kind == "same": idx = np.tile(rng.permutation(E)[:k], (T, 1))
    elif kind == "dup": idx = rng.integers(0, E, (T, k))  # duplicates inside a token allowed
    return mx.array(idx.reshape(*shape, k).astype(np.uint32))

rng = np.random.default_rng(0); mx.random.seed(0)
shapes = [(2048, 768, 128, 8, "qwen3"), (2816, 704, 128, 8, "gemma4"), (2048, 1408, 64, 6, "deepseek"), (2048, 512, 256, 8, "qwen3.6"),
          (1024, 256, 32, 4, "small"), (320, 192, 7, 1, "k1 odd E"), (128, 64, 4, 4, "k=E"), (64, 64, 2, 1, "tiny")]
worst, bad, n = 0.0, [], 0
for D, F, E, k, name in shapes:
    for act in (SwiGLU(), GeGLU()):
        glu = SwitchGLU(D, F, E, activation=act); nn.quantize(glu, group_size=64, bits=4); mx.eval(glu.parameters())
        for dtype in (mx.bfloat16, mx.float16):
            glu.set_dtype(dtype)  # as in real converted models: scales/biases share the activation dtype
            for shape in [(1, 1), (1, 2), (1, 5), (1, 9), (4, 1), (32, 1), (3, 7), (1, 64), (1, 128), (2, 100), (64, 1), (1, 300)]:
                T = int(np.prod(shape))
                if not sk.supported_load(T * k / E, D, F): continue
                for kind in ("uniform", "hot", "same", "dup"):
                    idx = routing(kind, shape, E, k, rng)
                    x = (mx.random.normal((*shape, D)) * (1.0 if kind != "hot" else 3.0)).astype(dtype)
                    assert sk.switch_glu(glu, x, idx) is not None
                    y = glu(x, idx); mx.eval(y)
                    ys, yt = stock(glu, x, idx), truth(glu, x, idx)
                    ok_shape = y.shape == ys.shape and y.dtype == ys.dtype
                    worst_stock = max(globals().get('worst_stock', 0.0), e_stock := rel(ys, yt)); ratio = max(globals().get('ratio', 0.0), rel(y, yt) / max(e_stock, 1e-9))
                    e_ours, e_stock = rel(y, yt), rel(ys, yt)
                    fin = bool(mx.all(mx.isfinite(y.astype(mx.float32))).item())
                    n += 1; worst = max(worst, e_ours)
                    if not ok_shape or not fin or e_ours > max(1e-2, 2 * e_stock):
                        bad.append((name, type(act).__name__, str(dtype), shape, kind, ok_shape, fin, round(e_ours, 5), round(e_stock, 5)))
print(f"cases {n}  worst rel err vs fp32: ours {worst:.2e} stock {worst_stock:.2e}  worst ours/stock ratio {ratio:.2f}  bad {len(bad)}")
for b in bad[:25]: print("  BAD", b)

# edge: zero tokens
glu = SwitchGLU(256, 128, 16); nn.quantize(glu, group_size=64, bits=4); mx.eval(glu.parameters())
for shape in [(1, 0), (0, 1)]:
    x, idx = mx.zeros((*shape, 256), mx.bfloat16), mx.zeros((*shape, 4), mx.uint32)
    for label, f in (("stock", lambda: stock(glu, x, idx)), ("branch", lambda: (lambda y: (mx.eval(y), y)[1])(glu(x, idx)))):
        try: print("T=0", shape, label, "->", f().shape)
        except Exception as e: print("T=0", shape, label, "RAISES", type(e).__name__, str(e)[:120])
# edge: int32 indices, fp32 input falls back, compile
x = mx.random.normal((1, 3, 256)).astype(mx.bfloat16); idx = mx.array(np.stack([rng.permutation(16)[:4] for _ in range(3)])[None].astype(np.int32))
print("int32 idx err vs stock", rel(glu(x, idx), stock(glu, x, idx)))
f = mx.compile(lambda x, idx: glu(x, idx))
print("mx.compile err vs stock", rel(f(x, idx), stock(glu, x, idx)), "second call", rel(f(x * 2, idx), stock(glu, x * 2, idx)))
