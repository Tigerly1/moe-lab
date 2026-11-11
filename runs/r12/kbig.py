import numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_lm.models import switch_kernels as sk
from mlx_lm.models.switch_layers import SwitchGLU
print(mx.device_info())
rng = np.random.default_rng(0)
for E, k, T in [(64, 16, 1), (64, 32, 1), (64, 40, 1), (64, 40, 3), (512, 40, 2)]:
    glu = SwitchGLU(128, 64, E); nn.quantize(glu, group_size=64, bits=4); glu.set_dtype(mx.bfloat16); mx.eval(glu.parameters())
    x = mx.random.normal((1, T, 128)).astype(mx.bfloat16)
    idx = mx.array(np.stack([rng.permutation(E)[:k] for _ in range(T)])[None].astype(np.uint32))
    used = sk.switch_glu(glu, x, idx) is not None
    try:
        y = glu(x, idx); mx.eval(y)
        k_, sk._kernels = sk._kernels, None; ys = glu(x, idx); mx.eval(ys); sk._kernels = k_
        e = (mx.linalg.norm((y.astype(mx.float32) - ys.astype(mx.float32)).flatten()) / mx.linalg.norm(ys.astype(mx.float32).flatten())).item()
        print(f"E={E} k={k} T={T} kernel_used={used} rel vs stock {e:.2e}")
    except Exception as ex:
        print(f"E={E} k={k} T={T} kernel_used={used} RAISES {type(ex).__name__}: {str(ex)[:160]}")
