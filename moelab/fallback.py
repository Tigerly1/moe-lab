"""Regime policy + MLX gather_qmm fallback for the production dispatchers (round 7, 2026-11-07).

Finding (runs/sweep_*.json, runs/grid_*.json): the custom expert-tile kernels lose to MLX's sorted
gather_qmm once each active expert sees many tokens. The deciding quantity is load = T*k/E (average
tokens per expert), not T. On Qwen3 shapes load stays small up to T=256 (E=128, k=8), which is why the
earlier T-only threshold looked fine there; on small-E models it does not.

gather_moe() reproduces mlx-lm's SwitchGLU path (sort when >= 64 slots) on our state layout, so the
fallback is the baseline itself, never something slower.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort


def gather_moe(s, x, idx, wts, group_size, bits, mode="affine", act=0):
    xx = mx.expand_dims(x, (-2, -3))
    do_sort = idx.size >= 64
    ii, inv = idx, None
    if do_sort:
        xx, ii, inv = _gather_sort(xx, idx)

    def mm(z, n):
        w = s[n]
        return mx.gather_qmm(z, w[0], w[1], w[2] if len(w) > 2 else None, rhs_indices=ii, transpose=True,
                             group_size=group_size, bits=bits, mode=mode, sorted_indices=do_sort)

    g, u = mm(xx, "g"), mm(xx, "u")
    # act 0: swiglu(gate, x) = silu(gate) * x; act 1: Gemma-4 geglu = gelu_approx(gate) * x
    y = mm(swiglu(g, u) if act == 0 else nn.gelu_approx(g) * u, "d")
    if do_sort:
        y = _scatter_unsort(y, inv, idx.shape)
    return (y.squeeze(-2) * wts.astype(x.dtype)[..., None]).sum(axis=-2)
