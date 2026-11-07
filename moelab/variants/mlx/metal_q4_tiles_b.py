"""Custom Metal: fused int4 MoE GEMV v3.1. EDIT ME.

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

META = {"name": "metal_q4_tiles_b", "description": "v3 + 3 dispatches (combine kernel, int32 idx)"}
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
# Expert-reuse kernels (T > 1): one simdgroup per (row, tile of <= VT slots of one expert).
# Each weight uint4 is loaded and dequantized once, then applied to all slots of the tile.
GATE_UP_E_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x / 32;
    if (row >= F) return;
    uint sl[VT];
    uint n = ar_tile(tl, thread_position_in_grid.y, sl, VT);
    if (n == 0) return;
    constexpr uint NW = D / 8, NG = D / GS;
    size_t rr = (size_t)idx[sl[0]] * F + row;
    if (n == 1) {  // v31 per-slot path
        const device T* xr = x + (sl[0] / K) * D;
        float ag = 0.f, au = 0.f, xs[32];
        for (uint c = lane; c < NW / 4; c += 32) {
            float xsum = load_x32(xr + c * 32, xs);
            uint grp = (c * 32) / GS;
            float dg = dot_q4x32(xs, ((const device uint4*)(gq + rr * NW))[c]);
            float du = dot_q4x32(xs, ((const device uint4*)(uq + rr * NW))[c]);
            ag += float(gs[rr * NG + grp]) * dg + float(gb[rr * NG + grp]) * xsum;
            au += float(us[rr * NG + grp]) * du + float(ub[rr * NG + grp]) * xsum;
        }
        float g = simd_sum(ag), u = simd_sum(au);
        if (lane == 0) h[sl[0] * F + row] = T(g / (1.f + metal::exp(-g)) * u);
        return;
    }
    float ag[VT], au[VT];
    for (uint i = 0; i < VT; ++i) { ag[i] = 0.f; au[i] = 0.f; }
    for (uint c = lane; c < NW / 4; c += 32) {
        uint grp = (c * 32) / GS;
        uint4 gw = ((const device uint4*)(gq + rr * NW))[c];
        uint4 uw = ((const device uint4*)(uq + rr * NW))[c];
        float gsc = float(gs[rr * NG + grp]), gbi = float(gb[rr * NG + grp]);
        float usc = float(us[rr * NG + grp]), ubi = float(ub[rr * NG + grp]);
        for (uint j = 0; j < 4; ++j) {
            float wg[8], wu[8], sc = 1.f;
            for (uint i = 0; i < 8; ++i) {
                wg[i] = float(gw[j] & (0xFu << (4 * i))) * (gsc * sc) + gbi;
                wu[i] = float(uw[j] & (0xFu << (4 * i))) * (usc * sc) + ubi;
                sc *= 0.0625f;
            }
            for (uint t = 0; t < VT; ++t) {
                if (t < n) {
                    const device T* xp = x + (sl[t] / K) * D + c * 32 + j * 8;
                    for (uint i = 0; i < 8; ++i) { float v = float(xp[i]); ag[t] += wg[i] * v; au[t] += wu[i] * v; }
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

DOWN_E_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row0 = (thread_position_in_grid.x / 32) * R;
    if (row0 >= D) return;
    uint sl[VT];
    uint n = ar_tile(tl, thread_position_in_grid.y, sl, VT);
    if (n == 0) return;
    constexpr uint NW = F / 8, NG = F / GS;
    size_t r0 = (size_t)idx[sl[0]] * D + row0;
    if (n == 1) {  // v31 per-slot path
        uint slot = sl[0];
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
        return;
    }
    float acc[R][VT];
    for (uint r = 0; r < R; ++r) for (uint t = 0; t < VT; ++t) acc[r][t] = 0.f;
    for (uint c = lane; c < NW / 4; c += 32) {
        uint grp = (c * 32) / GS;
        uint4 dw[R]; float dsc[R], dbi[R];
        for (uint r = 0; r < R; ++r) {
            size_t rr = r0 + r;
            dw[r] = ((const device uint4*)(dq + rr * NW))[c];
            dsc[r] = float(ds[rr * NG + grp]); dbi[r] = float(db[rr * NG + grp]);
        }
        for (uint j = 0; j < 4; ++j) {
            float wd[R][8];
            for (uint r = 0; r < R; ++r) {
                float sc = 1.f;
                for (uint i = 0; i < 8; ++i) { wd[r][i] = float(dw[r][j] & (0xFu << (4 * i))) * (dsc[r] * sc) + dbi[r]; sc *= 0.0625f; }
            }
            for (uint t = 0; t < VT; ++t) {
                if (t < n) {
                    const device T* hp = h + sl[t] * F + c * 32 + j * 8;
                    for (uint i = 0; i < 8; ++i) {
                        float v = float(hp[i]);
                        for (uint r = 0; r < R; ++r) acc[r][t] += wd[r][i] * v;
                    }
                }
            }
        }
    }
    for (uint t = 0; t < VT; ++t) {
        if (t < n) {
            float w = wts[sl[t]];
            for (uint r = 0; r < R; ++r) {
                float a = simd_sum(acc[r][t]);
                if (lane == 0) y[sl[t] * D + row0 + r] = w * a;
            }
        }
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
_k0 = mx.fast.metal_kernel(name="mq4tb_map", input_names=["idx"], output_names=["tl"], source=MAP_SRC)
_k1e = mx.fast.metal_kernel(name="mq4tb_gate_up_e", input_names=["x", "gq", "gs", "gb", "uq", "us", "ub", "idx", "tl"],
                            output_names=["h"], source=GATE_UP_E_SRC, header=HEADER)
_k2e = mx.fast.metal_kernel(name="mq4tb_down_e", input_names=["h", "dq", "ds", "db", "idx", "wts", "tl"],
                            output_names=["y"], source=DOWN_E_SRC, header=HEADER)
VT = 4

_k3 = mx.fast.metal_kernel(name="mq4tb_combine", input_names=["yp"], output_names=["y"], source=COMBINE_SRC)

_k1 = mx.fast.metal_kernel(name="mq4tb_gate_up", input_names=["x", "gq", "gs", "gb", "uq", "us", "ub", "idx"],
                           output_names=["h"], source=GATE_UP_SRC, header=HEADER)
_k2 = mx.fast.metal_kernel(name="mq4tb_down", input_names=["h", "dq", "ds", "db", "idx", "wts"],
                           output_names=["y"], source=DOWN_SRC, header=HEADER)


def prepare(w_gate, w_up, w_down):
    E, f, d = w_gate.shape
    assert d % 32 == 0 and f % 32 == 0 and f % R_UP == 0 and d % R_DOWN == 0, "shape constraints"
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
    h = _k1e(inputs=[x, gq, gs, gb, uq, us, ub, idx, tl], template=tm,
             grid=(32 * F, T * k, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
             output_shapes=[(T * k, F)], output_dtypes=[x.dtype])[0]
    yp = _k2e(inputs=[h, dq, ds, db, idx, wts.flatten(), tl], template=tm + [("R", R_DOWN)],
              grid=(32 * D // R_DOWN, T * k, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
              output_shapes=[(T * k, D)], output_dtypes=[mx.float32])[0]
    return _k3(inputs=[yp], template=[("T", x.dtype), ("D", D), ("K", k), ("NT", T)],
               grid=(T * D, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]
