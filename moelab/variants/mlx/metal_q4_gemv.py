"""Custom Metal: fused int4 MoE GEMV (MLX affine 4-bit, group 64). EDIT ME.

Same structure as metal_simd_gemv, but weights are read packed:
  uint32 = 8 nibbles (low bits first), w = scale * q + bias per group of GS.
Trick: dot(x, s*q + b) = s * dot(x, q) + b * sum(x)  -> 1 FMA per weight.
Each lane loads one uint4 (32 weights, 16 B) per step; a simdgroup reads 512 B
contiguous per step (coalesced). One simdgroup per output row (gate+up fused).
Knobs: ROWS_PER_TG.
"""
import mlx.core as mx

META = {"name": "metal_q4_gemv", "description": "custom Metal: fused int4 GEMV, SiLU + combine fused"}
BITS, GS = 4, 64
W_BITS = BITS + 32 / GS
ROWS_PER_TG = 4

HEADER = """
// 32 weights from one uint4 vs 32 activations -> raw dot(x, q)
template <typename T>
inline float dot_q4x32(const device T* xp, uint4 w, thread float& xsum) {
    float acc = 0.f;
    for (uint j = 0; j < 4; ++j) {
        uint word = w[j];
        for (uint i = 0; i < 8; ++i) {
            float xv = float(xp[j * 8 + i]);
            xsum += xv;
            acc += xv * float((word >> (4 * i)) & 0xFu);
        }
    }
    return acc;
}
"""

GATE_UP_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row  = thread_position_in_grid.x / 32;
    uint slot = thread_position_in_grid.y;
    if (row >= F) return;
    constexpr uint NW = D / 8;       // uint32 words per row
    constexpr uint NG = D / GS;      // groups per row
    uint e = idx[slot];
    const device T* xr = x + (slot / K) * D;
    size_t r = (size_t)e * F + row;
    const device uint4* g4 = (const device uint4*)(gq + r * NW);
    const device uint4* u4 = (const device uint4*)(uq + r * NW);
    const device S* gsc = gs + r * NG; const device S* gbi = gb + r * NG;
    const device S* usc = us + r * NG; const device S* ubi = ub + r * NG;
    float ag = 0.f, au = 0.f;
    for (uint c = lane; c < NW / 4; c += 32) {
        uint grp = (c * 32) / GS;
        const device T* xp = xr + c * 32;
        float xs = 0.f, xs2 = 0.f;
        float dg = dot_q4x32(xp, g4[c], xs);
        float du = dot_q4x32(xp, u4[c], xs2);
        ag += float(gsc[grp]) * dg + float(gbi[grp]) * xs;
        au += float(usc[grp]) * du + float(ubi[grp]) * xs;
    }
    ag = simd_sum(ag);
    au = simd_sum(au);
    if (lane == 0) h[slot * F + row] = ag / (1.f + metal::exp(-ag)) * au;
"""

DOWN_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x / 32;
    uint t   = thread_position_in_grid.y;
    if (row >= D) return;
    constexpr uint NW = F / 8;
    constexpr uint NG = F / GS;
    float acc = 0.f;
    for (uint j = 0; j < K; ++j) {
        uint slot = t * K + j;
        size_t r = (size_t)idx[slot] * D + row;
        const device uint4* w4 = (const device uint4*)(dq + r * NW);
        const device S* sc = ds + r * NG; const device S* bi = db + r * NG;
        const device float* hr = h + slot * F;
        float a = 0.f;
        for (uint c = lane; c < NW / 4; c += 32) {
            float hs = 0.f;
            float d = dot_q4x32(hr + c * 32, w4[c], hs);
            uint grp = (c * 32) / GS;
            a += float(sc[grp]) * d + float(bi[grp]) * hs;
        }
        acc += wts[slot] * a;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[t * D + row] = T(acc);
"""

_k1 = mx.fast.metal_kernel(name="moe_q4_gate_up", input_names=["x", "gq", "gs", "gb", "uq", "us", "ub", "idx"],
                           output_names=["h"], source=GATE_UP_SRC, header=HEADER)
_k2 = mx.fast.metal_kernel(name="moe_q4_down", input_names=["h", "dq", "ds", "db", "idx", "wts"],
                           output_names=["y"], source=DOWN_SRC, header=HEADER)


def prepare(w_gate, w_up, w_down):
    d, f = w_gate.shape[2], w_gate.shape[1]
    assert d % 32 == 0 and f % 32 == 0 and GS % 32 == 0, "d, f, GS must be multiples of 32"
    return {n: mx.quantize(w, group_size=GS, bits=BITS) for n, w in (("g", w_gate), ("u", w_up), ("d", w_down))}


def dequant(s):
    return tuple(mx.dequantize(*s[n], group_size=GS, bits=BITS) for n in ("g", "u", "d"))


def run(s, x, idx, wts):
    T, k = idx.shape
    (gq, gs, gb), (uq, us, ub), (dq, ds, db) = s["g"], s["u"], s["d"]
    E, F = gq.shape[0], gq.shape[1]
    D = dq.shape[1]
    tmpl = [("T", x.dtype), ("S", gs.dtype), ("D", D), ("F", F), ("K", k), ("GS", GS)]
    idx = idx.flatten().astype(mx.uint32)
    h = _k1(inputs=[x, gq, gs, gb, uq, us, ub, idx], template=tmpl,
            grid=(32 * F, T * k, 1), threadgroup=(32 * ROWS_PER_TG, 1, 1),
            output_shapes=[(T * k, F)], output_dtypes=[mx.float32])[0]
    return _k2(inputs=[h, dq, ds, db, idx, wts.flatten()], template=tmpl,
               grid=(32 * D, T, 1), threadgroup=(32 * ROWS_PER_TG, 1, 1),
               output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]
