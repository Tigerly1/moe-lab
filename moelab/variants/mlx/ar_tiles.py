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
// two tokens share one weight decode: and + cvt once, two fma
inline void dot2_q4x32(thread const float* xa, thread const float* xb, uint4 w, thread float& da, thread float& db) {
    float a = 0.f, b = 0.f;
    for (uint j = 0; j < 4; ++j) {
        uint word = w[j];
        for (uint i = 0; i < 8; ++i) { float q = float(word & (0xFu << (4 * i))); a += xa[j * 8 + i] * q; b += xb[j * 8 + i] * q; }
    }
    da += a; db += b;
}
// A-fragment pair for MMA step s: P interleaves two words (low 16 bits: word a nibbles, high: word b).
// Raw codes plus an exact offset: (q | 0x6400) as half = 1024 + q, (q | 0x4300) as bfloat = 128 + q.
// Scale and bias are applied in fp32 per scale group (bias corrected by -OFF * scale).
template <typename M> inline vec<M, 2> ar_qo2(uint P, uint s);
template <> inline half2 ar_qo2<half>(uint P, uint s) { return as_type<half2>(((P >> (4 * s)) & 0x000F000Fu) | 0x64006400u); }
template <> inline vec<bfloat, 2> ar_qo2<bfloat>(uint P, uint s) { return as_type<vec<bfloat, 2>>(((P >> (4 * s)) & 0x000F000Fu) | 0x43004300u); }
// per-token sums of one 64-wide activation group for this lane's fragment cols (fn, fn+1):
// lane-local sum over its 8 values, then over the 8 fragment rows (lane bits 1, 2, 4).
template <typename T>
inline float2 ar_colsum(vec<T, 4> a0, vec<T, 4> a1, vec<T, 4> b0, vec<T, 4> b1) {
    float2 v = float2(0.f);
    for (uint i = 0; i < 4; ++i) { v.x += float(a0[i]) + float(a1[i]); v.y += float(b0[i]) + float(b1[i]); }
    v += simd_shuffle_xor(v, 2); v += simd_shuffle_xor(v, 4); v += simd_shuffle_xor(v, 16);
    return v;
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
    threadgroup float red[SGK * 128 * MR];
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
GATE_UP_E_SRC = MMA_HEAD.replace("%SMALL%", """    if (n == 2 && NSMALL >= 2) {   // 2-slot tile: scalar GEMV, weights decoded once for both tokens
        constexpr uint NWs = D / 8, NGs = D / GS;
        uint sa = tt[1], sb = tt[2];
        const device T* xra = x + (sa / K) * D;
        const device T* xrb = x + (sb / K) * D;
        float ga[RS] = {0.f}, gb_[RS] = {0.f}, ua[RS] = {0.f}, ub_[RS] = {0.f}, xs[32], xt[32];
        for (uint c = lane; c < NWs / 4; c += 32) {
            float xsa = load_x32(xra + c * 32, xs), xsb = load_x32(xrb + c * 32, xt);
            uint grp = (c * 32) / GS;
            for (uint r = 0; r < RS; ++r) {
                size_t q = ex * F + rw0 + r;
                float dga = 0.f, dgb = 0.f, dua = 0.f, dub = 0.f;
                dot2_q4x32(xs, xt, ((const device uint4*)(gq + q * NWs))[c], dga, dgb);
                dot2_q4x32(xs, xt, ((const device uint4*)(uq + q * NWs))[c], dua, dub);
                float sg = float(gs[q * NGs + grp]), bg = float(gb[q * NGs + grp]);
                float su = float(us[q * NGs + grp]), bu = float(ub[q * NGs + grp]);
                ga[r] += sg * dga + bg * xsa; gb_[r] += sg * dgb + bg * xsb;
                ua[r] += su * dua + bu * xsa; ub_[r] += su * dub + bu * xsb;
            }
        }
        for (uint r = 0; r < RS; ++r) {
            float g = simd_sum(ga[r]), u = simd_sum(ua[r]);
            float g2 = simd_sum(gb_[r]), u2 = simd_sum(ub_[r]);
            if (lane == 0) {
                h[sa * F + rw0 + r] = T(g / (1.f + metal::exp(-g)) * u);
                h[sb * F + rw0 + r] = T(g2 / (1.f + metal::exp(-g2)) * u2);
            }
        }
        return;
    }
    if (n == 1) {
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
    if (threadgroup_position_in_grid.x % MR) return;   // MMA tiles: one threadgroup per MR x 8 rows (B reused MR times)
    const device T* xa = x + (s0 / K) * D + fm * 8;
    const device T* xb = x + (s1 / K) * D + fm * 8;
    const device uint2* g2 = (const device uint2*)(gq + rr * NW) + fn / 2;
    const device uint2* u2 = (const device uint2*)(uq + rr * NW) + fn / 2;
    // two explicit 8-row blocks (arrays of simdgroup_matrix spill)
    simdgroup_matrix<float, 8, 8> Cg0 = make_filled_simdgroup_matrix<float, 8, 8>(0.f), Cu0 = Cg0, Cg1 = Cg0, Cu1 = Cg0;
    simdgroup_matrix<float, 8, 8> Pg0, Pu0, Pg1, Pu1;
    simdgroup_matrix<MT, 8, 8> Ag, Au, B;
    for (uint g = ks; g < NB; g += SGK) {   // one 64-wide scale group: col c, step s <-> k = 8c + s
        uint2 wg = g2[g * 4], wu = u2[g * 4], wg1 = g2[4 * NW + g * 4], wu1 = u2[4 * NW + g * 4];
        uint pg[2] = {(wg.x & 0xFFFFu) | (wg.y << 16), (wg.x >> 16) | (wg.y & 0xFFFF0000u)};
        uint pu[2] = {(wu.x & 0xFFFFu) | (wu.y << 16), (wu.x >> 16) | (wu.y & 0xFFFF0000u)};
        uint qg[2] = {(wg1.x & 0xFFFFu) | (wg1.y << 16), (wg1.x >> 16) | (wg1.y & 0xFFFF0000u)};
        uint qu[2] = {(wu1.x & 0xFFFFu) | (wu1.y << 16), (wu1.x >> 16) | (wu1.y & 0xFFFF0000u)};
        vec<T, 4> a0 = ((const device vec<T, 4>*)(xa + g * 64))[0], a1 = ((const device vec<T, 4>*)(xa + g * 64))[1];
        vec<T, 4> b0 = ((const device vec<T, 4>*)(xb + g * 64))[0], b1 = ((const device vec<T, 4>*)(xb + g * 64))[1];
        for (uint s = 0; s < 8; ++s) {
            B.thread_elements()[0] = MT(s < 4 ? a0[s] : a1[s - 4]); B.thread_elements()[1] = MT(s < 4 ? b0[s] : b1[s - 4]);
            { vec<MT, 2> v_ = ar_qo2<MT>(pg[s / 4], s % 4); Ag.thread_elements()[0] = v_.x; Ag.thread_elements()[1] = v_.y; }
            { vec<MT, 2> v_ = ar_qo2<MT>(pu[s / 4], s % 4); Au.thread_elements()[0] = v_.x; Au.thread_elements()[1] = v_.y; }
            if (s == 0) { simdgroup_multiply(Pg0, Ag, B); simdgroup_multiply(Pu0, Au, B); }
            else { simdgroup_multiply_accumulate(Pg0, Ag, B, Pg0); simdgroup_multiply_accumulate(Pu0, Au, B, Pu0); }
            { vec<MT, 2> v_ = ar_qo2<MT>(qg[s / 4], s % 4); Ag.thread_elements()[0] = v_.x; Ag.thread_elements()[1] = v_.y; }
            { vec<MT, 2> v_ = ar_qo2<MT>(qu[s / 4], s % 4); Au.thread_elements()[0] = v_.x; Au.thread_elements()[1] = v_.y; }
            if (s == 0) { simdgroup_multiply(Pg1, Ag, B); simdgroup_multiply(Pu1, Au, B); }
            else { simdgroup_multiply_accumulate(Pg1, Ag, B, Pg1); simdgroup_multiply_accumulate(Pu1, Au, B, Pu1); }
        }
        float2 sx = ar_colsum(a0, a1, b0, b1);
        uint grp = rr * NB + g, grp1 = grp + 8 * NB;
        float sg_ = float(gs[grp]), su = float(us[grp]), sg1 = float(gs[grp1]), su1 = float(us[grp1]);
        float bg = float(gb[grp]) - OFF * sg_, bu = float(ub[grp]) - OFF * su;   // A holds q + OFF
        float bg1 = float(gb[grp1]) - OFF * sg1, bu1 = float(ub[grp1]) - OFF * su1;
        for (uint i = 0; i < 2; ++i) {
            Cg0.thread_elements()[i] = fma(sg_, Pg0.thread_elements()[i], fma(bg, sx[i], Cg0.thread_elements()[i]));
            Cu0.thread_elements()[i] = fma(su, Pu0.thread_elements()[i], fma(bu, sx[i], Cu0.thread_elements()[i]));
            Cg1.thread_elements()[i] = fma(sg1, Pg1.thread_elements()[i], fma(bg1, sx[i], Cg1.thread_elements()[i]));
            Cu1.thread_elements()[i] = fma(su1, Pu1.thread_elements()[i], fma(bu1, sx[i], Cu1.thread_elements()[i]));
        }
    }
    for (uint i = 0; i < 2; ++i) {
        red[ks * 256 + lane * 2 + i] = Cg0.thread_elements()[i];
        red[ks * 256 + 64 + lane * 2 + i] = Cu0.thread_elements()[i];
        red[ks * 256 + 128 + lane * 2 + i] = Cg1.thread_elements()[i];
        red[ks * 256 + 192 + lane * 2 + i] = Cu1.thread_elements()[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    for (uint r = 0; r < 2; ++r)
    for (uint i = 0; i < 2; ++i) {
        float g = 0.f, u = 0.f;
        for (uint j = 0; j < SGK; ++j) { g += red[(j * 2 + r) * 128 + lane * 2 + i]; u += red[(j * 2 + r) * 128 + 64 + lane * 2 + i]; }
        if (fn + i < n) h[(i ? s1 : s0) * F + row0 + 8 * r + fm] = T(g / (1.f + metal::exp(-g)) * u);
    }
"""

DOWN_E_SRC = MMA_HEAD.replace("%SMALL%", """    if (n == 2 && NSMALL >= 2) {   // 2-slot tile: LPR GEMV, weights decoded once for both slots
        constexpr uint NWs = F / 8, NGs = F / GS;
        constexpr uint RR = RS * LPR / 32;
        uint sa = tt[1], sb = tt[2];
        uint sub = lane % LPR, rg = rw0 + (lane / LPR) * RR;
        float aa[RR] = {0.f}, ab[RR] = {0.f}, hs[32], ht[32];
        for (uint c = sub; c < NWs / 4; c += LPR) {
            float hsa = load_x32(h + sa * F + c * 32, hs), hsb = load_x32(h + sb * F + c * 32, ht);
            uint grp = (c * 32) / GS;
            for (uint r = 0; r < RR; ++r) {
                size_t q = ex * D + rg + r;
                float da = 0.f, db_ = 0.f;
                dot2_q4x32(hs, ht, ((const device uint4*)(dq + q * NWs))[c], da, db_);
                float sc = float(ds[q * NGs + grp]), bi = float(db[q * NGs + grp]);
                aa[r] += sc * da + bi * hsa; ab[r] += sc * db_ + bi * hsb;
            }
        }
        float wa = wts[sa], wb = wts[sb];
        for (uint r = 0; r < RR; ++r) {
            float a = aa[r], b = ab[r];
            for (uint o = LPR / 2; o > 0; o >>= 1) { a += simd_shuffle_xor(a, o); b += simd_shuffle_xor(b, o); }
            if (sub == 0) { y[sa * D + rg + r] = wa * a; y[sb * D + rg + r] = wb * b; }
        }
        return;
    }
    if (n == 1) {
        constexpr uint NWs = F / 8, NGs = F / GS;
        for (uint t = 0; t < n; ++t) {
            uint slot = tt[1 + t];
            // LPR lanes per row (v4 trick): 32 / LPR row groups of RR rows, all lanes busy when F/32 < 32
            constexpr uint RR = RS * LPR / 32;
            uint sub = lane % LPR, rg = rw0 + (lane / LPR) * RR;
            float acc[RR] = {0.f}, hs[32];
            for (uint c = sub; c < NWs / 4; c += LPR) {
                float hsum = load_x32(h + slot * F + c * 32, hs);
                uint grp = (c * 32) / GS;
                for (uint r = 0; r < RR; ++r) {
                    size_t q = ex * D + rg + r;
                    float d = dot_q4x32(hs, ((const device uint4*)(dq + q * NWs))[c]);
                    acc[r] += float(ds[q * NGs + grp]) * d + float(db[q * NGs + grp]) * hsum;
                }
            }
            float w = wts[slot];
            for (uint r = 0; r < RR; ++r) {
                float a = acc[r];
                for (uint o = LPR / 2; o > 0; o >>= 1) a += simd_shuffle_xor(a, o);
                if (sub == 0) y[slot * D + rg + r] = w * a;
            }
        }
        return;
    }
""") + """
    if (threadgroup_position_in_grid.x % MR) return;   // MMA tiles: one threadgroup per 16 rows (B reused twice)
    const device T* ha = h + s0 * F + fm * 8;
    const device T* hb = h + s1 * F + fm * 8;
    const device uint2* d2 = (const device uint2*)(dq + rr * NW) + fn / 2;
    simdgroup_matrix<float, 8, 8> C0 = make_filled_simdgroup_matrix<float, 8, 8>(0.f), C1 = C0;
    simdgroup_matrix<float, 8, 8> P0, P1;
    simdgroup_matrix<MT, 8, 8> A, B;
    for (uint g = ks; g < NB; g += SGK) {
        uint2 w = d2[g * 4], w1 = d2[4 * NW + g * 4];
        uint pw[2] = {(w.x & 0xFFFFu) | (w.y << 16), (w.x >> 16) | (w.y & 0xFFFF0000u)};
        uint qw[2] = {(w1.x & 0xFFFFu) | (w1.y << 16), (w1.x >> 16) | (w1.y & 0xFFFF0000u)};
        vec<T, 4> a0 = ((const device vec<T, 4>*)(ha + g * 64))[0], a1 = ((const device vec<T, 4>*)(ha + g * 64))[1];
        vec<T, 4> b0 = ((const device vec<T, 4>*)(hb + g * 64))[0], b1 = ((const device vec<T, 4>*)(hb + g * 64))[1];
        for (uint s = 0; s < 8; ++s) {
            B.thread_elements()[0] = MT(s < 4 ? a0[s] : a1[s - 4]); B.thread_elements()[1] = MT(s < 4 ? b0[s] : b1[s - 4]);
            { vec<MT, 2> v_ = ar_qo2<MT>(pw[s / 4], s % 4); A.thread_elements()[0] = v_.x; A.thread_elements()[1] = v_.y; }
            if (s == 0) simdgroup_multiply(P0, A, B); else simdgroup_multiply_accumulate(P0, A, B, P0);
            { vec<MT, 2> v_ = ar_qo2<MT>(qw[s / 4], s % 4); A.thread_elements()[0] = v_.x; A.thread_elements()[1] = v_.y; }
            if (s == 0) simdgroup_multiply(P1, A, B); else simdgroup_multiply_accumulate(P1, A, B, P1);
        }
        float2 sx = ar_colsum(a0, a1, b0, b1);
        uint grp = rr * NB + g, grp1 = grp + 8 * NB;
        float sc = float(ds[grp]), bi = float(db[grp]) - OFF * sc;
        float sc1 = float(ds[grp1]), bi1 = float(db[grp1]) - OFF * sc1;
        for (uint i = 0; i < 2; ++i) {
            C0.thread_elements()[i] = fma(sc, P0.thread_elements()[i], fma(bi, sx[i], C0.thread_elements()[i]));
            C1.thread_elements()[i] = fma(sc1, P1.thread_elements()[i], fma(bi1, sx[i], C1.thread_elements()[i]));
        }
    }
    for (uint i = 0; i < 2; ++i) { red[ks * 128 + lane * 2 + i] = C0.thread_elements()[i]; red[ks * 128 + 64 + lane * 2 + i] = C1.thread_elements()[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    for (uint r = 0; r < 2; ++r)
    for (uint i = 0; i < 2; ++i) {
        float a = 0.f;
        for (uint j = 0; j < SGK; ++j) a += red[j * 128 + r * 64 + lane * 2 + i];
        uint slot = i ? s1 : s0;
        if (fn + i < n) y[slot * D + row0 + 8 * r + fm] = wts[slot] * a;
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
_k1e = mx.fast.metal_kernel(name="ar_tiles_gate_up_e6", input_names=["x", "gq", "gs", "gb", "uq", "us", "ub", "idx", "tl"],
                            output_names=["h"], source=GATE_UP_E_SRC, header=HEADER)
_k2e = mx.fast.metal_kernel(name="ar_tiles_down_e6_lpr", input_names=["h", "dq", "ds", "db", "idx", "wts", "tl"],
                            output_names=["y"], source=DOWN_E_SRC, header=HEADER)
VT = 8
SGK_UP = 8  # simdgroups per 8-row block: K split on the MMA path, 8 // SGK rows each on the small path
# down: SGK chosen per shape so each small-path lane owns R_DOWN rows x ~3 chunks (v4 rule): SGK = LPR / (4 R_DOWN)
NSMALL = 2  # tiles with <= NSMALL slots use the scalar GEMV path (n=2: one weight decode for both slots)

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


def _lpr(chunks, iters=3):
    """lanes per row for the small down path: smallest power of 2 >= chunks / iters, <= 32."""
    l = 1
    while l < 32 and l * iters < chunks:
        l *= 2
    return l


def _run_e(base, x, gq, gs, gb, uq, us, ub, dq, ds, db, idx, wts, T, k, F, D):
    bf = x.dtype == mx.bfloat16  # MMA operand type: bfloat keeps bf16 activations exact and in range
    tm = base + [("VT", VT), ("TK", T * k), ("MT", mx.bfloat16 if bf else mx.float16), ("OFF", 128 if bf else 1024)]
    tl = _k0(inputs=[idx], template=[("VT", VT), ("TK", T * k)], grid=(T * k, 1, 1), threadgroup=(min(T * k, 256), 1, 1),
             output_shapes=[(T * k * (VT + 1),)], output_dtypes=[mx.uint32])[0]
    h = _k1e(inputs=[x, gq, gs, gb, uq, us, ub, idx, tl], template=tm + [("NR", F), ("NK", D), ("SGK", SGK_UP), ("MR", 2), ("NSMALL", NSMALL), ("RS", 8 // SGK_UP)],
             grid=(32 * SGK_UP * F // 8, T * k, 1), threadgroup=(32 * SGK_UP, 1, 1),
             output_shapes=[(T * k, F)], output_dtypes=[x.dtype])[0]
    lpr = _lpr(F // 32)
    sgk = max(1, lpr // (4 * R_DOWN))
    lpr = max(lpr, 4 * sgk)  # RS = 8 / sgk rows must fill 32 / lpr row groups
    yp = _k2e(inputs=[h, dq, ds, db, idx, wts.flatten(), tl], template=tm + [("NR", D), ("NK", F), ("SGK", sgk), ("MR", 2), ("NSMALL", NSMALL), ("RS", 8 // sgk), ("LPR", lpr)],
              grid=(32 * sgk * D // 8, T * k, 1), threadgroup=(32 * sgk, 1, 1),
              output_shapes=[(T * k, D)], output_dtypes=[mx.float32])[0]
    return _k3(inputs=[yp], template=[("T", x.dtype), ("D", D), ("K", k), ("NT", T)],
               grid=(T * D, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]
