"""One patch for every mlx-lm MoE that runs its experts through SwitchGLU (round 9).

Routing, routing weights, the weighted sum over the k experts and shared experts stay exactly as each model
implements them; only SwitchGLU (gate/up/act/down per (token, expert) slot) is replaced. Each slot is passed to
the expert kernels as its own row with k=1 and weight 1, so the kernel returns per-slot outputs (shape [..., k, D],
like SwitchGLU) and the model's own code does the weighting. At T=1 the k-slot GEMV runs without its k-reduction
(`mod.run_slots`). Cost vs a family-specific patch: no fused router and no fused weighted combine.

Supported: quantized SwitchGLU without bias whose activation is SwiGLU (mlx-lm default) or Gemma-4's GeGLU,
in the format of the kernel module (`mod.GS`, `mod.BITS`, affine). Other SwitchGLUs are left stock.
"""
import mlx.core as mx
from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU

MAX_T = 256   # real tokens per call above which the stock path runs (prefill)


def _act(a):
    if isinstance(a, SwiGLU):
        return 0
    if type(a).__name__ == "GeGLU":   # Gemma-4: gelu_approx(gate) * up
        return 1
    return None


def patch(model, mod):
    stock = SwitchGLU.__dict__.get("_stock_call", SwitchGLU.__call__)
    SwitchGLU._stock_call = stock
    n = 0
    for _, m in model.named_modules():
        if not isinstance(m, SwitchGLU):
            continue
        ps = (m.gate_proj, m.up_proj, m.down_proj)
        act = _act(m.activation)
        ok = act is not None and all(
            getattr(p, "bits", None) == mod.BITS and getattr(p, "group_size", None) == mod.GS
            and getattr(p, "mode", "affine") == "affine" and "bias" not in p for p in ps)
        m._moelab = ({n_: (p.weight, p.scales, p.biases) for n_, p in zip("gud", ps)}, act) if ok else None
        n += ok

    def call(self, x, indices):
        st = getattr(self, "_moelab", None)
        k = indices.shape[-1]
        T = indices.size // k
        if st is None or T > MAX_T or not getattr(SwitchGLU, "_use_variant", False):
            return stock(self, x, indices)
        D = x.shape[-1]
        if T == 1 and hasattr(mod, "run_slots"):   # k-slot GEMV without the k-reduction (v4 NORED)
            return mod.run_slots(st[0], x.reshape(1, D), indices.reshape(1, k), act=st[1]).reshape(*indices.shape, D)
        rows = T * k
        xr = mx.broadcast_to(x.reshape(T, 1, D), (T, k, D)).reshape(rows, D)
        y = mod.run(st[0], xr, indices.reshape(rows, 1), mx.ones((rows, 1), mx.float32), act=st[1])
        return y.reshape(*indices.shape, D)

    SwitchGLU.__call__ = call
    return SwitchGLU, n
