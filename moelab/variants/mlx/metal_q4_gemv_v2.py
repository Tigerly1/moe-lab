"""Custom Metal: fused int4 MoE GEMV v2. EDIT ME.

vs metal_q4_gemv (v1), two changes:
 1. R rows per simdgroup: one x chunk in registers serves R gate + R up rows
    (x loads and sum(x) amortized R times).
 2. Pre-scaled x: x_i / 16^i, so nibble i is used as (word & (0xF << 4i)) without
    a shift: 3 ops/weight (and, cvt, fma) instead of ~6. Exact in fp32 (q*2^4i, q<16).
Knobs: R (rows per simdgroup), SG_PER_TG.
"""
import mlx.core as mx

META = {"name": "metal_q4_gemv_v2", "description": "int4 GEMV: R rows/simdgroup + prescaled-x nibble trick"}
BITS, GS = 4, 64
W_BITS = BITS + 32 / GS
R = 4
SG_PER_TG = 2

HEADER = """
// load 32 activations, prescale by 1/16^(i%8), return sum(x)
template <typename T>
inline float load_x32(const device T* xp, thread float* xs) {
    float s = 0.f;
    for (uint j = 0; j < 4; ++j) {
        float sc = 1.f;
        for (uint i = 0; i < 8; ++i) {
            float v = float(xp[j * 8 + i]);
            s += v;
            xs[j * 8 + i] = v * sc;
            sc *= 0.0625f;
        }
    }
    return s;
}
inline float dot_q4x32(thread const float* xs, uint4 w) {
    float acc = 0.f;
    for (uint j = 0; j < 4; ++j) {
        uint word = w[j];
        for (uint i = 0; i < 8; ++i) acc += xs[j * 8 + i] * float(word & (0xFu << (4 * i)));
    }
    return acc;
}
"""

GATE_UP_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row0 = (thread_position_in_grid.x / 32) * R;
    uint slot = thread_position_in_grid.y;
    if (row0 >= F) return;
    constexpr uint NW = D / 8, NG = D / GS;
    uint e = idx[slot];
    const device T* xr = x + (slot / K) * D;
    size_t r0 = (size_t)e * F + row0;
    float ag[R] = {0.f}, au[R] = {0.f};
    float xs[32];
    for (uint c = lane; c < NW / 4; c += 32) {
        float xsum = load_x32(xr + c * 32, xs);
        uint grp = (c * 32) / GS;
        for (uint r = 0; r < R; ++r) {
            size_t rr = r0 + r;
            float dg = dot_q4x32(xs, ((const device uint4*)(gq + rr * NW))[c]);
            float du = dot_q4x32(xs, ((const device uint4*)(uq + rr * NW))[c]);
            ag[r] += float(gs[rr * NG + grp]) * dg + float(gb[rr * NG + grp]) * xsum;
            au[r] += float(us[rr * NG + grp]) * du + float(ub[rr * NG + grp]) * xsum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float g = simd_sum(ag[r]), u = simd_sum(au[r]);
        if (lane == 0) h[slot * F + row0 + r] = g / (1.f + metal::exp(-g)) * u;
    }
"""

DOWN_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row0 = (thread_position_in_grid.x / 32) * R;
    uint t = thread_position_in_grid.y;
    if (row0 >= D) return;
    constexpr uint NW = F / 8, NG = F / GS;
    float acc[R] = {0.f};
    float hs[32];
    for (uint j = 0; j < K; ++j) {
        uint slot = t * K + j;
        size_t r0 = (size_t)idx[slot] * D + row0;
        float w = wts[slot];
        for (uint c = lane; c < NW / 4; c += 32) {
            float hsum = load_x32(h + slot * F + c * 32, hs);
            uint grp = (c * 32) / GS;
            for (uint r = 0; r < R; ++r) {
                size_t rr = r0 + r;
                float d = dot_q4x32(hs, ((const device uint4*)(dq + rr * NW))[c]);
                acc[r] += w * (float(ds[rr * NG + grp]) * d + float(db[rr * NG + grp]) * hsum);
            }
        }
    }
    for (uint r = 0; r < R; ++r) {
        float a = simd_sum(acc[r]);
        if (lane == 0) y[t * D + row0 + r] = T(a);
    }
"""

_k1 = mx.fast.metal_kernel(name="moe_q4v2_gate_up", input_names=["x", "gq", "gs", "gb", "uq", "us", "ub", "idx"],
                           output_names=["h"], source=GATE_UP_SRC, header=HEADER)
_k2 = mx.fast.metal_kernel(name="moe_q4v2_down", input_names=["h", "dq", "ds", "db", "idx", "wts"],
                           output_names=["y"], source=DOWN_SRC, header=HEADER)


def prepare(w_gate, w_up, w_down):
    E, f, d = w_gate.shape
    assert d % 32 == 0 and f % 32 == 0 and f % R == 0 and d % R == 0, "shape constraints"
    return {n: mx.quantize(w, group_size=GS, bits=BITS) for n, w in (("g", w_gate), ("u", w_up), ("d", w_down))}


def dequant(s):
    return tuple(mx.dequantize(*s[n], group_size=GS, bits=BITS) for n in ("g", "u", "d"))


def run(s, x, idx, wts):
    T, k = idx.shape
    (gq, gs, gb), (uq, us, ub), (dq, ds, db) = s["g"], s["u"], s["d"]
    F, D = gq.shape[1], dq.shape[1]
    tmpl = [("T", x.dtype), ("S", gs.dtype), ("D", D), ("F", F), ("K", k), ("GS", GS), ("R", R)]
    idx = idx.flatten().astype(mx.uint32)
    h = _k1(inputs=[x, gq, gs, gb, uq, us, ub, idx], template=tmpl,
            grid=(32 * F // R, T * k, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
            output_shapes=[(T * k, F)], output_dtypes=[mx.float32])[0]
    return _k2(inputs=[h, dq, ds, db, idx, wts.flatten()], template=tmpl,
               grid=(32 * D // R, T, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
               output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]
