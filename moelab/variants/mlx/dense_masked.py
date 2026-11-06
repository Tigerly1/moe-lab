"""Baseline: compute ALL experts for ALL tokens, weight by a dense [T, E] mask.
Wasteful in FLOPs, but one big GEMM. TRT-LLM found this wins mid-batch on B200."""
import mlx.core as mx

META = {"name": "dense_masked", "description": "all experts x all tokens, masked combine"}


def prepare(w_gate, w_up, w_down):
    return {"wg": w_gate, "wu": w_up, "wd": w_down}


def run(s, x, idx, wts):
    T = x.shape[0]
    E = s["wg"].shape[0]
    g = mx.einsum("td,efd->etf", x, s["wg"])
    u = mx.einsum("td,efd->etf", x, s["wu"])
    h = g * mx.sigmoid(g) * u
    o = mx.einsum("etf,edf->etd", h, s["wd"])          # [E, T, d]
    mask = mx.zeros((T, E), dtype=x.dtype)
    mask = mask.at[mx.arange(T)[:, None], idx].add(wts.astype(x.dtype))
    return mx.einsum("te,etd->td", mask, o)
