"""Custom Metal: int4 MoE decode with expert reuse across the T tokens of a step. EDIT ME.

T == 1: v3.1 kernels unchanged. T > 1: a map kernel groups slots by expert into tiles of <= 8.
Tiles of 1 slot run the v3.1 per-slot GEMV; larger tiles run simdgroup_matrix 8x8 (half A/B,
fp32 acc) so each weight is loaded and dequantized (0x6400 trick) once per tile, not per slot.

v3.1 vs v3: fewer dispatches (5 -> 3). idx passed as int32 (no astype op);
sum over k + cast fused into one tiny combine kernel. On M4 Pro each dispatch
costs several us of GPU-side time, which matters for small experts (Qwen3).
---- v3 notes:

v3 vs v2: down kernel parallel over (token, slot) instead of looping the k experts
serially inside one simdgroup (8x more simdgroups in flight -> more memory-level
parallelism); partial outputs [T*k, D] summed over k afterwards; h kept in fp16
(halves activation load traffic). Separate R for each kernel.
---- v2 notes:

vs metal_q4_gemv (v1), two changes:
 1. R rows per simdgroup: one x chunk in registers serves R gate + R up rows
    (x loads and sum(x) amortized R times).
 2. Pre-scaled x: x_i / 16^i, so nibble i is used as (word & (0xF << 4i)) without
    a shift: 3 ops/weight (and, cvt, fma) instead of ~6. Exact in fp32 (q*2^4i, q<16).
Knobs: R (rows per simdgroup), SG_PER_TG.
"""
import mlx.core as mx

META = {"name": "ar_tiles", "description": "v3.1 at T=1; expert-tiled MMA reuse kernels at T>1"}
BITS, GS = 4, 64
W_BITS = BITS + 32 / GS
R_UP, R_DOWN = 1, 2  # tuned on M4 Pro, T=1 (scan: R_UP=1 best on all shapes)
SG_PER_TG = 4

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
// A-fragment pair for MMA step s: P interleaves two words (low 16 bits: word a nibbles, high: word b).
// (q | 0x6400) as half is exactly 1024 + q.
inline half2 ar_dq2(uint P, uint s, half2 sc, half2 bi) {
    half2 q = as_type<half2>(((P >> (4 * s)) & 0x000F000Fu) | 0x64006400u) - half2(1024.h);
    return fma(q, sc, bi);
}
// read the tile built by the map kernel: n slots (0 = no tile here), ids into sl[0..n)
inline uint ar_tile(const device uint* tl, uint p, thread uint* sl, uint VT) {
    const device uint* t = tl + p * (VT + 1);
    uint n = t[0];
    for (uint i = 0; i < VT; ++i) sl[i] = (i < n) ? t[1 + i] : 0u;
    return n;
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
        if (lane == 0) h[slot * F + row0 + r] = T(g / (1.f + metal::exp(-g)) * u);
    }
"""

DOWN_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row0 = (thread_position_in_grid.x / 32) * R;
    uint slot = thread_position_in_grid.y;
    if (row0 >= D) return;
    constexpr uint NW = F / 8, NG = F / GS;
    size_t r0 = (size_t)idx[slot] * D + row0;
    float acc[R] = {0.f};
    float hs[32];
    for (uint c = lane; c < NW / 4; c += 32) {
        float hsum = load_x32(h + slot * F + c * 32, hs);
        uint grp = (c * 32) / GS;
        for (uint r = 0; r < R; ++r) {
            size_t rr = r0 + r;
            float d = dot_q4x32(hs, ((const device uint4*)(dq + rr * NW))[c]);
            acc[r] += float(ds[rr * NG + grp]) * d + float(db[rr * NG + grp]) * hsum;
        }
    }
    float w = wts[slot];
    for (uint r = 0; r < R; ++r) {
        float a = simd_sum(acc[r]);
        if (lane == 0) y[slot * D + row0 + r] = w * a;
    }
"""

COMBINE_SRC = """
    uint i = thread_position_in_grid.x;            // over T*D
    if (i >= NT * D) return;
    uint t = i / D, r = i % D;
    float a = 0.f;
    for (uint j = 0; j < K; ++j) a += yp[(t * K + j) * D + r];
    y[i] = T(a);
"""
# Expert-reuse kernels (T > 1), simdgroup_matrix 8x8: one simdgroup per (8 rows, tile of <= 8 slots
# of one expert). C[row, tok] += A[row, c] * B[c, tok] where, inside a 64-wide quant block,
# logical (c, step s) <-> k = 8c + s, so each lane's A elements are 8 contiguous nibbles (one word)
# and its B elements 8 contiguous activations. Weights dequantized once per tile.
MMA_HEAD = """
    uint lane = thread_index_in_simdgroup;
    uint ks = simdgroup_index_in_threadgroup;   // K split: SGK simdgroups share 8 rows
    uint row0 = threadgroup_position_in_grid.x * 8;
    uint p = thread_position_in_grid.y;
    threadgroup float red[SGK * 128];
    const device uint* tt = tl + p * (VT + 1);
    uint n = tt[0];
    if (n == 0 || row0 >= NR) return;
    size_t ex = idx[tt[1]];
    uint rw0 = row0 + ks * RS;  // small tiles: v31 per-slot GEMV, RS = 8 / SGK rows per simdgroup
""" + "%SMALL%" + """
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);   // fragment row of this lane
    uint fn = (qid & 2) * 2 + (lane % 2) * 2; // fragment cols fn, fn+1
    uint s0 = tt[1 + min(fn, n - 1)], s1 = tt[1 + min(fn + 1, n - 1)];
    size_t rr = (size_t)idx[tt[1]] * NR + row0 + fm;
    constexpr uint NW = NK / 8, NB = NK / 64;
"""
GATE_UP_E_SRC = MMA_HEAD.replace("%SMALL%", """    if (n <= NSMALL) {
        constexpr uint NWs = D / 8, NGs = D / GS;
        for (uint t = 0; t < n; ++t) {
            uint slot = tt[1 + t];
            const device T* xr = x + (slot / K) * D;
            float ag[RS] = {0.f}, au[RS] = {0.f}, xs[32];
            for (uint c = lane; c < NWs / 4; c += 32) {
                float xsum = load_x32(xr + c * 32, xs);
                uint grp = (c * 32) / GS;
                for (uint r = 0; r < RS; ++r) {
                    size_t q = ex * F + rw0 + r;
                    float dg = dot_q4x32(xs, ((const device uint4*)(gq + q * NWs))[c]);
                    float du = dot_q4x32(xs, ((const device uint4*)(uq + q * NWs))[c]);
                    ag[r] += float(gs[q * NGs + grp]) * dg + float(gb[q * NGs + grp]) * xsum;
                    au[r] += float(us[q * NGs + grp]) * du + float(ub[q * NGs + grp]) * xsum;
                }
            }
            for (uint r = 0; r < RS; ++r) {
                float g = simd_sum(ag[r]), u = simd_sum(au[r]);
                if (lane == 0) h[slot * F + rw0 + r] = T(g / (1.f + metal::exp(-g)) * u);
            }
        }
        return;
    }
""") + """
    const device T* xa = x + (s0 / K) * D + fm * 16;
    const device T* xb = x + (s1 / K) * D + fm * 16;
    const device uint4* g4 = (const device uint4*)(gq + rr * NW) + fn / 2;
    const device uint4* u4 = (const device uint4*)(uq + rr * NW) + fn / 2;
    simdgroup_matrix<float, 8, 8> Cg = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> Cu = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<half, 8, 8> Ag, Au, B;
    for (uint b = ks; b < NB / 2; b += SGK) {   // 128-wide superblocks: col c, step s <-> k = 16c + s
        uint4 wg = g4[b * 4], wu = u4[b * 4];
        uint pg[4] = {(wg.x & 0xFFFFu) | (wg.z << 16), (wg.x >> 16) | (wg.z & 0xFFFF0000u),
                      (wg.y & 0xFFFFu) | (wg.w << 16), (wg.y >> 16) | (wg.w & 0xFFFF0000u)};
        uint pu[4] = {(wu.x & 0xFFFFu) | (wu.z << 16), (wu.x >> 16) | (wu.z & 0xFFFF0000u),
                      (wu.y & 0xFFFFu) | (wu.w << 16), (wu.y >> 16) | (wu.w & 0xFFFF0000u)};
        uint grp = rr * NB + b * 2 + fn / 4;
        half2 gsc = half2(half(gs[grp])), gbi = half2(half(gb[grp]));
        half2 usc = half2(half(us[grp])), ubi = half2(half(ub[grp]));
        const device vec<T, 4>* pa = (const device vec<T, 4>*)(xa + b * 128);
        const device vec<T, 4>* pb = (const device vec<T, 4>*)(xb + b * 128);
        half2 xv[16];
        for (uint v = 0; v < 4; ++v) {
            half4 av = half4(pa[v]), bv = half4(pb[v]);
            for (uint i = 0; i < 4; ++i) xv[v * 4 + i] = half2(av[i], bv[i]);
        }
        for (uint s = 0; s < 16; ++s) {
            { half2 v_ = ar_dq2(pg[s / 4], s % 4, gsc, gbi); Ag.thread_elements()[0] = v_.x; Ag.thread_elements()[1] = v_.y; }
            { half2 v_ = ar_dq2(pu[s / 4], s % 4, usc, ubi); Au.thread_elements()[0] = v_.x; Au.thread_elements()[1] = v_.y; }
            { half2 v_ = xv[s]; B.thread_elements()[0] = v_.x; B.thread_elements()[1] = v_.y; }
            simdgroup_multiply_accumulate(Cg, Ag, B, Cg);
            simdgroup_multiply_accumulate(Cu, Au, B, Cu);
        }
    }
    for (uint i = 0; i < 2; ++i) {
        red[ks * 128 + lane * 2 + i] = Cg.thread_elements()[i];
        red[ks * 128 + 64 + lane * 2 + i] = Cu.thread_elements()[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    for (uint i = 0; i < 2; ++i) {
        float g = 0.f, u = 0.f;
        for (uint j = 0; j < SGK; ++j) { g += red[j * 128 + lane * 2 + i]; u += red[j * 128 + 64 + lane * 2 + i]; }
        if (fn + i < n) h[(i ? s1 : s0) * F + row0 + fm] = T(g / (1.f + metal::exp(-g)) * u);
    }
"""

DOWN_E_SRC = MMA_HEAD.replace("%SMALL%", """    if (n <= NSMALL) {
        constexpr uint NWs = F / 8, NGs = F / GS;
        for (uint t = 0; t < n; ++t) {
            uint slot = tt[1 + t];
            float acc[RS] = {0.f}, hs[32];
            for (uint c = lane; c < NWs / 4; c += 32) {
                float hsum = load_x32(h + slot * F + c * 32, hs);
                uint grp = (c * 32) / GS;
                for (uint r = 0; r < RS; ++r) {
                    size_t q = ex * D + rw0 + r;
                    float d = dot_q4x32(hs, ((const device uint4*)(dq + q * NWs))[c]);
                    acc[r] += float(ds[q * NGs + grp]) * d + float(db[q * NGs + grp]) * hsum;
                }
            }
            float w = wts[slot];
            for (uint r = 0; r < RS; ++r) {
                float a = simd_sum(acc[r]);
                if (lane == 0) y[slot * D + rw0 + r] = w * a;
            }
        }
        return;
    }
""") + """
    const device T* ha = h + s0 * F + fm * 16;
    const device T* hb = h + s1 * F + fm * 16;
    const device uint4* d4 = (const device uint4*)(dq + rr * NW) + fn / 2;
    simdgroup_matrix<float, 8, 8> C = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<half, 8, 8> A, B;
    for (uint b = ks; b < NB / 2; b += SGK) {
        uint4 w = d4[b * 4];
        uint pw[4] = {(w.x & 0xFFFFu) | (w.z << 16), (w.x >> 16) | (w.z & 0xFFFF0000u),
                      (w.y & 0xFFFFu) | (w.w << 16), (w.y >> 16) | (w.w & 0xFFFF0000u)};
        uint grp = rr * NB + b * 2 + fn / 4;
        half2 sc = half2(half(ds[grp])), bi = half2(half(db[grp]));
        const device vec<T, 4>* pa = (const device vec<T, 4>*)(ha + b * 128);
        const device vec<T, 4>* pb = (const device vec<T, 4>*)(hb + b * 128);
        half2 xv[16];
        for (uint v = 0; v < 4; ++v) {
            half4 av = half4(pa[v]), bv = half4(pb[v]);
            for (uint i = 0; i < 4; ++i) xv[v * 4 + i] = half2(av[i], bv[i]);
        }
        for (uint s = 0; s < 16; ++s) {
            { half2 v_ = ar_dq2(pw[s / 4], s % 4, sc, bi); A.thread_elements()[0] = v_.x; A.thread_elements()[1] = v_.y; }
            { half2 v_ = xv[s]; B.thread_elements()[0] = v_.x; B.thread_elements()[1] = v_.y; }
            simdgroup_multiply_accumulate(C, A, B, C);
        }
    }
    for (uint i = 0; i < 2; ++i) red[ks * 128 + lane * 2 + i] = C.thread_elements()[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    for (uint i = 0; i < 2; ++i) {
        float a = 0.f;
        for (uint j = 0; j < SGK; ++j) a += red[j * 128 + lane * 2 + i];
        uint slot = i ? s1 : s0;
        if (fn + i < n) y[slot * D + row0 + fm] = wts[slot] * a;
    }
"""
# map kernel: slot p leads a tile iff its rank among slots of the same expert is a multiple of VT.
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
_k0 = mx.fast.metal_kernel(name="ar_tiles_map", input_names=["idx"], output_names=["tl"], source=MAP_SRC)
_k1e = mx.fast.metal_kernel(name="ar_tiles_gate_up_e", input_names=["x", "gq", "gs", "gb", "uq", "us", "ub", "idx", "tl"],
                            output_names=["h"], source=GATE_UP_E_SRC, header=HEADER)
_k2e = mx.fast.metal_kernel(name="ar_tiles_down_e", input_names=["h", "dq", "ds", "db", "idx", "wts", "tl"],
                            output_names=["y"], source=DOWN_E_SRC, header=HEADER)
VT = 8
SGK_UP, SGK_DOWN = 8, 2  # simdgroups per 8-row block: K split on the MMA path, 8 // SGK rows each on the small path
NSMALL = 1  # tiles with <= NSMALL slots use the per-slot GEMV path

_k3 = mx.fast.metal_kernel(name="ar_tiles_combine", input_names=["yp"], output_names=["y"], source=COMBINE_SRC)

_k1 = mx.fast.metal_kernel(name="ar_tiles_gate_up", input_names=["x", "gq", "gs", "gb", "uq", "us", "ub", "idx"],
                           output_names=["h"], source=GATE_UP_SRC, header=HEADER)
_k2 = mx.fast.metal_kernel(name="ar_tiles_down", input_names=["h", "dq", "ds", "db", "idx", "wts"],
                           output_names=["y"], source=DOWN_SRC, header=HEADER)


def prepare(w_gate, w_up, w_down):
    E, f, d = w_gate.shape
    assert d % 128 == 0 and f % 128 == 0 and GS == 64 and f % R_UP == 0 and d % R_DOWN == 0, "shape constraints"
    return {n: mx.quantize(w, group_size=GS, bits=BITS) for n, w in (("g", w_gate), ("u", w_up), ("d", w_down))}


def dequant(s):
    return tuple(mx.dequantize(*s[n], group_size=GS, bits=BITS) for n in ("g", "u", "d"))


def run(s, x, idx, wts):
    T, k = idx.shape
    (gq, gs, gb), (uq, us, ub), (dq, ds, db) = s["g"], s["u"], s["d"]
    F, D = gq.shape[1], dq.shape[1]
    base = [("T", x.dtype), ("S", gs.dtype), ("D", D), ("F", F), ("K", k), ("GS", GS)]
    if T > 1:
        return _run_e(base, x, gq, gs, gb, uq, us, ub, dq, ds, db, idx, wts, T, k, F, D)
    h = _k1(inputs=[x, gq, gs, gb, uq, us, ub, idx], template=base + [("R", R_UP)],
            grid=(32 * F // R_UP, T * k, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
            output_shapes=[(T * k, F)], output_dtypes=[x.dtype])[0]
    yp = _k2(inputs=[h, dq, ds, db, idx, wts.flatten()], template=base + [("R", R_DOWN)],
             grid=(32 * D // R_DOWN, T * k, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
             output_shapes=[(T * k, D)], output_dtypes=[mx.float32])[0]
    return _k3(inputs=[yp], template=[("T", x.dtype), ("D", D), ("K", k), ("NT", T)],
               grid=(T * D, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]


def _run_e(base, x, gq, gs, gb, uq, us, ub, dq, ds, db, idx, wts, T, k, F, D):
    tm = base + [("VT", VT), ("TK", T * k)]
    tl = _k0(inputs=[idx], template=[("VT", VT), ("TK", T * k)], grid=(T * k, 1, 1), threadgroup=(min(T * k, 256), 1, 1),
             output_shapes=[(T * k * (VT + 1),)], output_dtypes=[mx.uint32])[0]
    h = _k1e(inputs=[x, gq, gs, gb, uq, us, ub, idx, tl], template=tm + [("NR", F), ("NK", D), ("SGK", SGK_UP), ("NSMALL", NSMALL), ("RS", 8 // SGK_UP)],
             grid=(32 * SGK_UP * F // 8, T * k, 1), threadgroup=(32 * SGK_UP, 1, 1),
             output_shapes=[(T * k, F)], output_dtypes=[x.dtype])[0]
    yp = _k2e(inputs=[h, dq, ds, db, idx, wts.flatten(), tl], template=tm + [("NR", D), ("NK", F), ("SGK", SGK_DOWN), ("NSMALL", NSMALL), ("RS", 8 // SGK_DOWN)],
              grid=(32 * SGK_DOWN * D // 8, T * k, 1), threadgroup=(32 * SGK_DOWN, 1, 1),
              output_shapes=[(T * k, D)], output_dtypes=[mx.float32])[0]
    return _k3(inputs=[yp], template=[("T", x.dtype), ("D", D), ("K", k), ("NT", T)],
               grid=(T * D, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]
