"""Reference path for e2e quality checks (not a lab variant): dequantize only the selected
experts and compute the MoE in fp32. Slow; decode-sized T only."""
import mlx.core as mx

BITS, GS = 4, 64


def run(s, x, idx, wts, act=0):   # act 0 SwiGLU, 1 Gemma-4 GeGLU (tanh-GELU)
    x32 = x.astype(mx.float32)
    T, k = idx.shape
    deq = lambda q, i: mx.dequantize(q[0][i], q[1][i], q[2][i], group_size=GS, bits=BITS).astype(mx.float32)  # noqa: E731
    ys = []
    for t in range(T):
        e = idx[t]
        g = mx.einsum("kfd,d->kf", deq(s["g"], e), x32[t])
        u = mx.einsum("kfd,d->kf", deq(s["u"], e), x32[t])
        a = g * mx.sigmoid(g) if act == 0 else 0.5 * g * (1 + mx.tanh(0.7978845608028654 * (g + 0.044715 * g ** 3)))
        o = mx.einsum("kdf,kf->kd", deq(s["d"], e), a * u)
        ys.append((o * wts[t][:, None]).sum(0))
    return mx.stack(ys).astype(x.dtype)
