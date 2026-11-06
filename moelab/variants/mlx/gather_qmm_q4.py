"""MLX built-in quantized path: mx.gather_qmm with 4-bit affine weights (group 64).
This is what mlx-lm runs for mlx-community/*-4bit MoE models."""
import mlx.core as mx

META = {"name": "gather_qmm_q4", "description": "mlx-lm QuantizedSwitchLinear, 4-bit g64"}
BITS, GROUP = 4, 64
W_BITS = BITS + 32 / GROUP  # fp16 scale + bias per group


def prepare(w_gate, w_up, w_down):
    s = {}
    for n, w in (("g", w_gate), ("u", w_up), ("d", w_down)):
        s[n] = mx.quantize(w, group_size=GROUP, bits=BITS)   # (wq, scales, biases), w is [E, out, in]
    return s


def dequant(s):
    return tuple(mx.dequantize(*s[n], group_size=GROUP, bits=BITS) for n in ("g", "u", "d"))


def _qmm(x, q, ids, srt):
    wq, sc, bi = q
    return mx.gather_qmm(x, wq, sc, bi, rhs_indices=ids, transpose=True,
                         group_size=GROUP, bits=BITS, sorted_indices=srt)


def run(s, x, idx, wts):
    T, k = idx.shape
    xe = mx.expand_dims(x, (-2, -3))
    srt = idx.size >= 64
    if srt:
        flat = idx.flatten()
        order = mx.argsort(flat)
        inv = mx.argsort(order)
        xe = xe.flatten(0, -3)[order // k]
        ids = flat[order]
    else:
        ids = idx
    g = _qmm(xe, s["g"], ids, srt)
    u = _qmm(xe, s["u"], ids, srt)
    o = _qmm(g * mx.sigmoid(g) * u, s["d"], ids, srt)
    o = o[inv].reshape(T, k, -1) if srt else o.squeeze(-2)
    return (o * wts[..., None].astype(o.dtype)).sum(axis=1)
