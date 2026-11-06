"""MLX built-in: mx.gather_mm, same strategy as mlx-lm SwitchGLU
(sort tokens by expert when >= 64 slots, implicit gather inside GEMM)."""
import mlx.core as mx

META = {"name": "gather_mm", "description": "mlx-lm SwitchGLU strategy (gather_mm + sort)"}
SORT_THRESHOLD = 64


def prepare(w_gate, w_up, w_down):
    # gather_mm computes a @ b, so store weights as [E, in, out]
    return {"wg": w_gate.swapaxes(-1, -2), "wu": w_up.swapaxes(-1, -2), "wd": w_down.swapaxes(-1, -2)}


def run(s, x, idx, wts):
    T, k = idx.shape
    xe = mx.expand_dims(x, (-2, -3))                    # [T, 1, 1, d]
    do_sort = idx.size >= SORT_THRESHOLD
    if do_sort:
        flat = idx.flatten()
        order = mx.argsort(flat)
        inv = mx.argsort(order)
        xe = xe.flatten(0, -3)[order // k]               # [T*k, 1, d]
        ids = flat[order]
    else:
        ids = idx
    g = mx.gather_mm(xe, s["wg"], rhs_indices=ids, sorted_indices=do_sort)
    u = mx.gather_mm(xe, s["wu"], rhs_indices=ids, sorted_indices=do_sort)
    h = g * mx.sigmoid(g) * u
    o = mx.gather_mm(h, s["wd"], rhs_indices=ids, sorted_indices=do_sort)
    if do_sort:
        o = o[inv].reshape(T, k, -1)
    else:
        o = o.squeeze(-2)                                # [T, k, d]
    return (o * wts[..., None].astype(o.dtype)).sum(axis=1)
