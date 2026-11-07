"""Custom Metal MXFP4 MoE kernels (autoresearch round 4).

Format: mx.quantize(mode="mxfp4"): uint32 words of 8 E2M1 codes (low nibble first), one E8M0
scale byte per 32 weights = exactly one uint4 load per scale group.
E2M1 -> half is exact by bit placement: sign -> bit 15, the 3 magnitude bits -> bits 9..11
(exponent field e, mantissa m). That half equals value * 2^-14 (also for e=0: subnormal),
so the 2^14 is folded into the group scale: float with exponent field s + 14.

T == 1: v4-style GEMV (gate+up+SiLU fused, one simdgroup per row; down with LPR lanes per row and
the k slots reduced in threadgroup memory -> 2 dispatches).
T > 1: map prepass groups slots by expert into tiles of <= VT; each simdgroup decodes a weight
once and applies it to all slots of its tile (scalar), then a combine kernel sums over k.
"""
import mlx.core as mx

META = {"name": "ar_mxfp4", "description": "custom Metal MXFP4: v4 GEMV at T=1, expert-tiled scalar GEMV at T>1"}
MODE, GS, BITS = "mxfp4", 32, 4
W_BITS = 4 + 8 / 32
R_UP, R_DOWN = 1, 2
SG_PER_TG = 4
ITERS = 3
VT = 8

HEADER = """
// two E2M1 codes (bits 0..3 and 16..19 of p) -> half2, value * 2^-14 (exact)
inline half2 fp4x2(uint p) {
    return as_type<half2>(((p & 0x00080008u) << 12) | ((p & 0x00070007u) << 9));
}
// E8M0 scale byte -> float 2^(s-127) * 2^14
inline float e8s(uint s) { return as_type<float>((s + 14u) << 23); }
template <typename T>
inline void load_x32(const device T* xp, thread float* xs) {
    for (uint i = 0; i < 32; ++i) xs[i] = float(xp[i]);
}
// dot of 32 fp4 weights (one uint4) with 32 floats. nibble i of word j <-> weight 8j + i.
inline float dot_fp4x32(thread const float* xs, uint4 w) {
    float acc = 0.f;
    for (uint j = 0; j < 4; ++j) {
        uint word = w[j];
        for (uint i = 0; i < 4; ++i) {
            float2 v = float2(fp4x2(word >> (4 * i)));
            acc = fma(xs[j * 8 + i], v.x, acc);
            acc = fma(xs[j * 8 + i + 4], v.y, acc);
        }
    }
    return acc;
}
// decode 8 weights of one word, times scale
inline void dec8(uint word, float sc, thread float* w) {
    for (uint i = 0; i < 4; ++i) {
        float2 v = float2(fp4x2(word >> (4 * i))) * sc;
        w[i] = v.x; w[i + 4] = v.y;
    }
}
template <typename T>
inline float dot8(const device T* xp, thread const float* w) {
    float a = 0.f;
    for (uint i = 0; i < 8; ++i) a = fma(float(xp[i]), w[i], a);
    return a;
}
"""

GATE_UP_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row0 = (thread_position_in_grid.x / 32) * R;
    uint slot = thread_position_in_grid.y;
    if (row0 >= F) return;
    constexpr uint NC = D / 32;
    uint e = idx[slot];
    const device T* xr = x + (slot / K) * D;
    size_t r0 = (size_t)e * F + row0;
    float ag[R] = {0.f}, au[R] = {0.f};
    float xs[32];
    for (uint c = lane; c < NC; c += 32) {
        load_x32(xr + c * 32, xs);
        for (uint r = 0; r < R; ++r) {
            size_t q = (r0 + r) * NC + c;
            ag[r] += e8s(gs[q]) * dot_fp4x32(xs, ((const device uint4*)gq)[q]);
            au[r] += e8s(us[q]) * dot_fp4x32(xs, ((const device uint4*)uq)[q]);
        }
    }
    for (uint r = 0; r < R; ++r) {
        float g = simd_sum(ag[r]), u = simd_sum(au[r]);
        if (lane == 0) h[slot * F + row0 + r] = T(g / (1.f + metal::exp(-g)) * u);
    }
"""

DOWN_SRC = """
    constexpr uint RB = R * 32 / LPR;
    uint lane = thread_index_in_simdgroup;
    uint sub = lane % LPR;
    uint j = thread_position_in_threadgroup.y;
    uint rb0 = (thread_position_in_grid.x / 32) * RB;
    uint row0 = rb0 + (lane / LPR) * R;
    uint t = thread_position_in_grid.y / K;
    uint slot = t * K + j;
    bool ok = row0 < D;
    constexpr uint NC = F / 32;
    threadgroup float buf[K][RB];
    size_t r0 = (size_t)idx[slot] * D + (ok ? row0 : 0);
    float acc[R] = {0.f};
    float hs[32];
    for (uint c = sub; ok && c < NC; c += LPR) {
        load_x32(h + slot * F + c * 32, hs);
        for (uint r = 0; r < R; ++r) {
            size_t q = (r0 + r) * NC + c;
            acc[r] += e8s(ds[q]) * dot_fp4x32(hs, ((const device uint4*)dq)[q]);
        }
    }
    float w = wts[slot];
    for (uint r = 0; r < R; ++r) {
        float a = acc[r];
        for (uint o = LPR / 2; o > 0; o >>= 1) a += simd_shuffle_xor(a, o);
        if (sub == 0) buf[j][(lane / LPR) * R + r] = w * a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tid = j * 32 + lane;
    if (tid < RB && rb0 + tid < D) {
        float a = 0.f;
        for (uint q = 0; q < K; ++q) a += buf[q][tid];
        y[t * D + rb0 + tid] = T(a);
    }
"""

MAP_SRC = """
    uint p = thread_position_in_grid.x;
    if (p >= TK) return;
    int e = idx[p];
    uint rank = 0;
    for (uint q = 0; q < p; ++q) rank += (idx[q] == e) ? 1u : 0u;
    device uint* t = tl + p * (VT + 1);
    if (rank % VT != 0) { t[0] = 0; return; }
    uint n = 0;
    for (uint q = p; q < TK && n < VT; ++q) if (idx[q] == e) t[1 + n++] = q;
    t[0] = n;
"""

# tiled gate/up: one simdgroup per row and tile; each weight word decoded once for all n slots
GATE_UP_E_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x / 32;
    uint p = thread_position_in_grid.y;
    if (row >= F) return;
    const device uint* tt = tl + p * (VT + 1);
    uint n = tt[0];
    if (n == 0) return;
    constexpr uint NC = D / 32;
    uint sl[VT];
    for (uint i = 0; i < VT; ++i) sl[i] = tt[1 + min(i, n - 1)];
    size_t rr = (size_t)idx[sl[0]] * F + row;
    float ag[VT] = {0.f}, au[VT] = {0.f};
    for (uint c = lane; c < NC; c += 32) {
        size_t q = rr * NC + c;
        uint4 wg = ((const device uint4*)gq)[q], wu = ((const device uint4*)uq)[q];
        float sg = e8s(gs[q]), su = e8s(us[q]);
        for (uint j = 0; j < 4; ++j) {
            float fg[8], fu[8];
            dec8(wg[j], sg, fg);
            dec8(wu[j], su, fu);
            for (uint t = 0; t < VT; ++t) {
                if (t < n) {
                    const device T* xp = x + (sl[t] / K) * D + c * 32 + j * 8;
                    ag[t] += dot8(xp, fg);
                    au[t] += dot8(xp, fu);
                }
            }
        }
    }
    for (uint t = 0; t < VT; ++t) {
        if (t < n) {
            float g = simd_sum(ag[t]), u = simd_sum(au[t]);
            if (lane == 0) h[sl[t] * F + row] = T(g / (1.f + metal::exp(-g)) * u);
        }
    }
"""

# tiled down: LPR lanes per row, RB rows per simdgroup; writes w * y per slot (fp32), combine sums k
DOWN_E_SRC = """
    constexpr uint RB = 32 / LPR;
    uint lane = thread_index_in_simdgroup;
    uint sub = lane % LPR;
    uint row = (thread_position_in_grid.x / 32) * RB + lane / LPR;
    uint p = thread_position_in_grid.y;
    const device uint* tt = tl + p * (VT + 1);
    uint n = tt[0];
    if (n == 0 || row >= D) return;
    constexpr uint NC = F / 32;
    uint sl[VT];
    for (uint i = 0; i < VT; ++i) sl[i] = tt[1 + min(i, n - 1)];
    size_t rr = (size_t)idx[sl[0]] * D + row;
    float acc[VT] = {0.f};
    for (uint c = sub; c < NC; c += LPR) {
        size_t q = rr * NC + c;
        uint4 w = ((const device uint4*)dq)[q];
        float s = e8s(ds[q]);
        for (uint j = 0; j < 4; ++j) {
            float f[8];
            dec8(w[j], s, f);
            for (uint t = 0; t < VT; ++t)
                if (t < n) acc[t] += dot8(h + sl[t] * F + c * 32 + j * 8, f);
        }
    }
    for (uint t = 0; t < VT; ++t) {
        if (t < n) {
            float a = acc[t];
            for (uint o = LPR / 2; o > 0; o >>= 1) a += simd_shuffle_xor(a, o);
            if (sub == 0) yp[sl[t] * D + row] = wts[sl[t]] * a;
        }
    }
"""

COMBINE_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= NT * D) return;
    uint t = i / D, r = i % D;
    float a = 0.f;
    for (uint j = 0; j < K; ++j) a += yp[(t * K + j) * D + r];
    y[i] = T(a);
"""

_k1 = mx.fast.metal_kernel(name="ar_mxfp4_gate_up_v1", input_names=["x", "gq", "gs", "uq", "us", "idx"],
                           output_names=["h"], source=GATE_UP_SRC, header=HEADER)
_k2 = mx.fast.metal_kernel(name="ar_mxfp4_down_v1", input_names=["h", "dq", "ds", "idx", "wts"],
                           output_names=["y"], source=DOWN_SRC, header=HEADER)
_k0 = mx.fast.metal_kernel(name="ar_mxfp4_map_v1", input_names=["idx"], output_names=["tl"], source=MAP_SRC)
_k1e = mx.fast.metal_kernel(name="ar_mxfp4_gate_up_e_v1", input_names=["x", "gq", "gs", "uq", "us", "idx", "tl"],
                            output_names=["h"], source=GATE_UP_E_SRC, header=HEADER)
_k2e = mx.fast.metal_kernel(name="ar_mxfp4_down_e_v1", input_names=["h", "dq", "ds", "idx", "wts", "tl"],
                            output_names=["yp"], source=DOWN_E_SRC, header=HEADER)
_k3 = mx.fast.metal_kernel(name="ar_mxfp4_combine_v1", input_names=["yp"], output_names=["y"], source=COMBINE_SRC)


def _lpr(chunks, iters=ITERS):
    """lanes per row: smallest power of 2 >= chunks / iters, capped at 32."""
    l = 1
    while l < 32 and l * iters < chunks:
        l *= 2
    return l


def prepare(w_gate, w_up, w_down):
    E, f, d = w_gate.shape
    assert d % 32 == 0 and f % 32 == 0 and d % R_DOWN == 0, "shape constraints"
    return {n: mx.quantize(w, mode=MODE) for n, w in (("g", w_gate), ("u", w_up), ("d", w_down))}


def dequant(s):
    return tuple(mx.dequantize(*s[n], mode=MODE) for n in ("g", "u", "d"))


def run(s, x, idx, wts):
    T, k = idx.shape
    (gq, gs), (uq, us), (dq, ds) = s["g"], s["u"], s["d"]
    F, D = gq.shape[1], dq.shape[1]
    base = [("T", x.dtype), ("D", D), ("F", F), ("K", k)]
    if T == 1:
        h = _k1(inputs=[x, gq, gs, uq, us, idx], template=base + [("R", R_UP)],
                grid=(32 * F // R_UP, k, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
                output_shapes=[(k, F)], output_dtypes=[x.dtype])[0]
        lpr = _lpr(F // 32)
        rb = R_DOWN * 32 // lpr
        return _k2(inputs=[h, dq, ds, idx, wts.flatten()], template=base + [("R", R_DOWN), ("LPR", lpr)],
                   grid=(32 * (-(-D // rb)), k, 1), threadgroup=(32, k, 1),
                   output_shapes=[(1, D)], output_dtypes=[x.dtype])[0]
    TK = T * k
    tl = _k0(inputs=[idx], template=[("VT", VT), ("TK", TK)], grid=(TK, 1, 1), threadgroup=(min(TK, 256), 1, 1),
             output_shapes=[(TK * (VT + 1),)], output_dtypes=[mx.uint32])[0]
    h = _k1e(inputs=[x, gq, gs, uq, us, idx, tl], template=base + [("VT", VT)],
             grid=(32 * F, TK, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
             output_shapes=[(TK, F)], output_dtypes=[x.dtype])[0]
    lpr = _lpr(F // 32)
    rb = 32 // lpr
    yp = _k2e(inputs=[h, dq, ds, idx, wts.flatten(), tl], template=base + [("VT", VT), ("LPR", lpr)],
              grid=(32 * (-(-D // rb)), TK, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
              output_shapes=[(TK, D)], output_dtypes=[mx.float32])[0]
    return _k3(inputs=[yp], template=[("T", x.dtype), ("D", D), ("K", k), ("NT", T)],
               grid=(T * D, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]
