"""Reference path for e2e quality checks (not a lab variant): dequantize only the selected
experts and compute the MoE in fp32. Slow; decode-sized T only."""
import mlx.core as mx

BITS, GS = 4, 64


def run(s, x, idx, wts):
    x32 = x.astype(mx.float32)
    T, k = idx.shape
    deq = lambda q, i: mx.dequantize(q[0][i], q[1][i], q[2][i], group_size=GS, bits=BITS).astype(mx.float32)  # noqa: E731
    ys = []
    for t in range(T):
        e = idx[t]
        g = mx.einsum("kfd,d->kf", deq(s["g"], e), x32[t])
        u = mx.einsum("kfd,d->kf", deq(s["u"], e), x32[t])
        o = mx.einsum("kdf,kf->kd", deq(s["d"], e), g * mx.sigmoid(g) * u)
        ys.append((o * wts[t][:, None]).sum(0))
    return mx.stack(ys).astype(x.dtype)
