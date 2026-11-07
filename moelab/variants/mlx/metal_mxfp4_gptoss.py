"""Custom Metal MXFP4 MoE kernels for gpt-oss (mlx-lm SwitchGLU with bias=True).

Same kernels as metal_mxfp4_moe.py (T=1 GEMV, 1<T<8 paired GEMV, T>=8 expert groups + simdgroup MMA),
derived from its Metal sources by a fixed set of text substitutions (each asserted to hit exactly the
expected sites), so the weight decode / tiling / reduction code is byte-identical to the verified kernels.
Differences:
  * per-expert biases: gate/up bias added before the activation, down bias added before the routing weight
    (y = sum_k w_k * (W_down h + b_down), as in mlx-lm's gpt_oss.MLPBlock);
  * activation selected by template constant ACT:
      ACT=0  standard SwiGLU  silu(g) * u                         (lab contract: prepare/run/dequant)
      ACT=1  gpt-oss clamped  g=min(g,7), u=clip(u,-7,7), g*sigmoid(1.702 g) * (u + 1)
  * MMA path (T >= 8) scale fix: the base kernel folds 2^14 into a *half* group scale, which overflows
    to inf for E8M0 bytes >= 129 (scale >= 4). gpt-oss down_proj has bytes up to 136 (~11% of groups >= 129),
    the base kernel now splits the half scale into two half factors (hs2, fixed 2026-11-07) and applied as two exact power-of-two
    multiplies -- bit-identical to the base kernel wherever the base did not overflow;
  * new kernel-name prefix `mgo_` (MLX caches kernels by name).
`state(sw)` builds the run() state directly from a (quantized, mode="mxfp4") SwitchGLU module.
"""
import importlib.util
import os

import mlx.core as mx

_spec = importlib.util.spec_from_file_location(
    "_metal_mxfp4_moe_base", os.path.join(os.path.dirname(os.path.abspath(__file__)), "metal_mxfp4_moe.py"))
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)

META = {"name": "metal_mxfp4_gptoss",
        "description": "metal_mxfp4_moe kernels + per-expert biases + gpt-oss clamped SwiGLU (ACT template)"}
MODE, GS, BITS = _b.MODE, _b.GS, _b.BITS
W_BITS = _b.W_BITS
R_UP, R_DOWN, SG_PER_TG, T_GROUP, G = _b.R_UP, _b.R_DOWN, _b.SG_PER_TG, _b.T_GROUP, _b.G
ALPHA, LIMIT = 1.702, 7.0

HEADER = _b.HEADER + f"""
template <int A>
inline float act(float g, float u) {{
    if (A == 0) return g / (1.f + metal::exp(-g)) * u;
    g = metal::min(g, {LIMIT}f);
    u = metal::clamp(u, -{LIMIT}f, {LIMIT}f);
    return g / (1.f + metal::exp(-{ALPHA}f * g)) * (u + 1.f);
}}
"""


def _sub(src, *pairs):
    for old, new, n in pairs:
        assert src.count(old) == n, (old, src.count(old), n)
        src = src.replace(old, new)
    return src


def _a(g, u, i):
    """silu-gate expression of the base kernel -> act<ACT>(g + bg[i], u + bu[i])"""
    return f"{g} / (1.f + metal::exp(-{g})) * {u}", f"act<ACT>({g} + bg[{i}], {u} + bu[{i}])"


GATE_UP_SRC = _sub(_b.GATE_UP_SRC, (*_a("g", "u", "r0 + r"), 1))
GATE_UP_P_SRC = _sub(_b.GATE_UP_P_SRC, (*_a("g", "u", "r0 + r"), 2), (*_a("g2", "u2", "r0 + r"), 1))
GATE_UP_G_SRC = _sub(_b.GATE_UP_G_SRC, (*_a("ga", "ua", "rr"), 1), (*_a("gb", "ub", "rr"), 1),
                     (*_a("g", "u", "rf"), 1))
DOWN_SRC = _sub(_b.DOWN_SRC, ("= w * a;", "= w * (a + bd[r0 + r]);", 1))
DOWN_P_SRC = _sub(_b.DOWN_P_SRC, ("= wa * a;", "= wa * (a + bd[r0 + r]);", 1),
                  ("= wb * b;", "= wb * (b + bd[r0 + r]);", 1))
DOWN_G_SRC = _sub(_b.DOWN_G_SRC, ("= wts[slot] * a;", "= wts[slot] * (a + bd[rr]);", 1),
                  ("= wts[m[1]] * b2;", "= wts[m[1]] * (b2 + bd[rr]);", 1),
                  ("= wts[m[fn + i]] * a;", "= wts[m[fn + i]] * (a + bd[rf]);", 1))

_k1 = mx.fast.metal_kernel(name="mgo_gate_up_v1", input_names=["x", "gq", "gs", "uq", "us", "bg", "bu", "idx"],
                           output_names=["h"], source=GATE_UP_SRC, header=HEADER)
_k2 = mx.fast.metal_kernel(name="mgo_down_v1", input_names=["h", "dq", "ds", "bd", "idx", "wts"],
                           output_names=["y"], source=DOWN_SRC, header=HEADER)
_k1p = mx.fast.metal_kernel(name="mgo_gate_up_p_v1", input_names=["x", "gq", "gs", "uq", "us", "bg", "bu", "idx"],
                            output_names=["h"], source=GATE_UP_P_SRC, header=HEADER)
_k2p = mx.fast.metal_kernel(name="mgo_down_p_v1", input_names=["h", "dq", "ds", "bd", "idx", "wts"],
                            output_names=["yp"], source=DOWN_P_SRC, header=HEADER)
_k0 = mx.fast.metal_kernel(name="mgo_map_v1", input_names=["idx"], output_names=["tl"], source=_b.MAP_SRC)
_k1g = mx.fast.metal_kernel(name="mgo_gate_up_g_v1", input_names=["x", "gq", "gs", "uq", "us", "bg", "bu", "idx", "tl"],
                            output_names=["h"], source=GATE_UP_G_SRC, header=HEADER)
_k2g = mx.fast.metal_kernel(name="mgo_down_g_v1", input_names=["h", "dq", "ds", "bd", "idx", "wts", "tl"],
                            output_names=["yp"], source=DOWN_G_SRC, header=HEADER)
_k3 = mx.fast.metal_kernel(name="mgo_combine_v1", input_names=["yp"], output_names=["y"], source=_b.COMBINE_SRC)


def prepare(w_gate, w_up, w_down):
    """Lab contract (standard SwiGLU, zero biases) -- exercises the same kernels with ACT=0."""
    s = _b.prepare(w_gate, w_up, w_down)
    E, f, d = w_gate.shape
    s.update(bg=mx.zeros((E, f)), bu=mx.zeros((E, f)), bd=mx.zeros((E, d)), act=0)
    return s


def dequant(s):
    return _b.dequant(s)


def state(sw):
    """run() state from an mlx-lm SwitchGLU(bias=True) whose projections are QuantizedSwitchLinear(mode=mxfp4)."""
    for p in (sw.gate_proj, sw.up_proj, sw.down_proj):
        assert (p.mode, p.group_size, p.bits) == (MODE, GS, BITS), "format mismatch"
    s = {n: (p.weight, p.scales) for n, p in (("g", sw.gate_proj), ("u", sw.up_proj), ("d", sw.down_proj))}
    s.update({b: p.bias.astype(mx.float32) for b, p in (("bg", sw.gate_proj), ("bu", sw.up_proj), ("bd", sw.down_proj))})
    s["act"] = 1
    return s


def reference(s, x, idx, wts):
    """fp32 truth for quality checks: dequantize only the selected experts, compute in fp32. Slow."""
    x32 = x.astype(mx.float32)
    deq = lambda n, e: mx.dequantize(*s[n], mode=MODE)[e].astype(mx.float32)  # noqa: E731
    ys = []
    for t in range(idx.shape[0]):
        e = idx[t]
        g = mx.einsum("kfd,d->kf", deq("g", e), x32[t]) + s["bg"][e]
        u = mx.einsum("kfd,d->kf", deq("u", e), x32[t]) + s["bu"][e]
        if s["act"]:
            g, u = mx.minimum(g, LIMIT), mx.clip(u, -LIMIT, LIMIT)
            hh = g * mx.sigmoid(ALPHA * g) * (u + 1)
        else:
            hh = g * mx.sigmoid(g) * u
        o = mx.einsum("kdf,kf->kd", deq("d", e), hh) + s["bd"][e]
        ys.append((o * wts[t][:, None]).sum(0))
    return mx.stack(ys).astype(x.dtype)


def run(s, x, idx, wts):
    T, k = idx.shape
    (gq, gs), (uq, us), (dq, ds) = s["g"], s["u"], s["d"]
    bg, bu, bd = s["bg"], s["bu"], s["bd"]
    F, D = gq.shape[1], dq.shape[1]
    base = [("T", x.dtype), ("D", D), ("F", F), ("K", k), ("ACT", s["act"])]
    if 1 < T < T_GROUP:   # few repeated experts: pairs share a decode, no map prepass
        TK = T * k
        h = _k1p(inputs=[x, gq, gs, uq, us, bg, bu, idx], template=base + [("R", R_UP), ("TK", TK)],
                 grid=(32 * F // R_UP, TK, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
                 output_shapes=[(TK, F)], output_dtypes=[x.dtype])[0]
        lpr = _b._lpr(F // 32)
        rb = R_DOWN * 32 // lpr
        yp = _k2p(inputs=[h, dq, ds, bd, idx, wts.flatten()], template=base + [("R", R_DOWN), ("LPR", lpr), ("TK", TK)],
                  grid=(32 * (-(-D // rb)), TK, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
                  output_shapes=[(TK, D)], output_dtypes=[mx.float32])[0]
        return _k3(inputs=[yp], template=[("T", x.dtype), ("D", D), ("K", k), ("NT", T)],
                   grid=(T * D, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]
    if T > 1:
        TK = T * k
        rt_up, rt_down = _b._rt(F, _b.RT_UP), _b._rt(D, _b.RT_DOWN)
        tl = _k0(inputs=[idx], template=[("TK", TK), ("G", G)], grid=(TK, 1, 1), threadgroup=(min(TK, 256), 1, 1),
                 output_shapes=[(TK * (G + 1),)], output_dtypes=[mx.uint32])[0]
        h = _k1g(inputs=[x, gq, gs, uq, us, bg, bu, idx, tl], template=base + [("TK", TK), ("G", G), ("NR", F), ("NK", D), ("RT", rt_up)],
                 grid=(256 * F // rt_up, TK, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(TK, F)], output_dtypes=[x.dtype])[0]
        yp = _k2g(inputs=[h, dq, ds, bd, idx, wts.flatten(), tl], template=base + [("TK", TK), ("G", G), ("NR", D), ("NK", F), ("RT", rt_down)],
                  grid=(256 * D // rt_down, TK, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(TK, D)], output_dtypes=[mx.float32])[0]
        return _k3(inputs=[yp], template=[("T", x.dtype), ("D", D), ("K", k), ("NT", T)],
                   grid=(T * D, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]
    h = _k1(inputs=[x, gq, gs, uq, us, bg, bu, idx], template=base + [("R", R_UP)],
            grid=(32 * F // R_UP, k, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
            output_shapes=[(k, F)], output_dtypes=[x.dtype])[0]
    lpr = _b._lpr(F // 32)
    rb = R_DOWN * 32 // lpr
    return _k2(inputs=[h, dq, ds, bd, idx, wts.flatten()], template=base + [("R", R_DOWN), ("LPR", lpr)],
               grid=(32 * (-(-D // rb)), k, 1), threadgroup=(32, k, 1),
               output_shapes=[(1, D)], output_dtypes=[x.dtype])[0]
