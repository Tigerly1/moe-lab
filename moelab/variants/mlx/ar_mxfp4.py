"""Custom Metal MXFP4 MoE kernels (autoresearch round 4).

Format: mx.quantize(mode="mxfp4"): uint32 words of 8 E2M1 codes (low nibble first), one E8M0
scale byte per 32 weights = exactly one uint4 load per scale group.
E2M1 -> half is exact by bit placement: sign -> bit 15, the 3 magnitude bits -> bits 9..11
(exponent field e, mantissa m). That half equals value * 2^-14 (also for e=0: subnormal),
so the 2^14 is folded into the group scale: float with exponent field s + 14.

T == 1: v4-style GEMV (gate+up+SiLU fused, one simdgroup per row; down with LPR lanes per row and
the k slots reduced in threadgroup memory -> 2 dispatches). At the bandwidth floor.
1 < T < T_GROUP: same GEMVs per slot, but slots of one expert are paired in-kernel (rank parity scan)
so a pair shares every weight decode (the scalar path is ALU-bound once weights hit in cache);
down writes w*y per slot (fp32) and a combine kernel sums k.
T >= T_GROUP: map prepass groups slots by expert (<= G); groups of 1-2 run the scalar GEMV,
groups of >= 3 run simdgroup_matrix 8x8 MMA (half A/B, fp32 acc) with one decode per group.
"""
import mlx.core as mx

META = {"name": "ar_mxfp4", "description": "custom Metal MXFP4: v4 GEMV (T=1), paired-slot GEMV (T<8), expert groups + simdgroup MMA (T>=8)"}
MODE, GS, BITS = "mxfp4", 32, 4
W_BITS = 4 + 8 / 32
R_UP, R_DOWN = 1, 2
SG_PER_TG = 4
ITERS = 3
RT_UP, RT_DOWN = 16, 32  # T > 1: rows per threadgroup (8 simdgroups)
T_GROUP = 8  # T >= T_GROUP: map prepass + groups of <= G (scalar / MMA); 1 < T < T_GROUP: pairs
G = 8  # T > 1: max group size (slots of one expert sharing weight decode)

HEADER = """
// two E2M1 codes (bits 0..3 and 16..19 of p) -> half2, value * 2^-14 (exact)
inline half2 fp4x2(uint p) {
    return as_type<half2>(((p & 0x00080008u) << 12) | ((p & 0x00070007u) << 9));
}
// E8M0 scale byte -> float 2^(s-127), exact for s = 1..254 (s = 0 -> 0). The decode's 2^-14 is undone once
// per output (swiglu / routing weight * 2^14), not folded in here: (s + 14) << 23 overflows for s >= 241.
inline float e8s(uint s) { return as_type<float>(s << 23); }
inline float swiglu(float g, float u) { g *= 0x1p14f; return g / (1.f + metal::exp(-g)) * (u * 0x1p14f); }
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
// two activation vectors share one weight decode
inline void dot2_fp4x32(thread const float* xa, thread const float* xb, uint4 w, thread float& da, thread float& db) {
    float a = 0.f, b = 0.f;
    for (uint j = 0; j < 4; ++j) {
        uint word = w[j];
        for (uint i = 0; i < 4; ++i) {
            float2 v = float2(fp4x2(word >> (4 * i)));
            a = fma(xa[j * 8 + i], v.x, a); a = fma(xa[j * 8 + i + 4], v.y, a);
            b = fma(xb[j * 8 + i], v.x, b); b = fma(xb[j * 8 + i + 4], v.y, b);
        }
    }
    da = a; db = b;
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
        if (lane == 0) h[slot * F + row0 + r] = T(swiglu(g, u));
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
    float w = wts[slot] * 0x1p14f;
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

# T > 1: threadgroup = 8 simdgroups x 8 rows. Slots of one expert form groups of <= 8 in-kernel
# (rank among same-expert slots; rank % 8 == 0 leads, the rest return). Group size n:
#   1 -> per-slot GEMV (simdgroup j: row row0 + j), 2 -> GEMV with one decode for both slots,
#   >= 3 -> simdgroup_matrix 8x8 (half A/B, fp32 acc), the 8 simdgroups split K, reduced in threadgroup
#   memory. MMA k-remap inside a 64-block: k = 8c + s (c = fragment col, s = step), so each lane's A
#   pair comes from words c, c+1 (same scale group) and its B values are 8 contiguous activations.
MAP_SRC = """
    uint p = thread_position_in_grid.x;
    if (p >= TK) return;
    int e = idx[p];
    uint rank = 0;
    for (uint q = 0; q < p; ++q) rank += (idx[q] == e) ? 1u : 0u;
    device uint* t = tl + p * (G + 1);
    if (rank % G != 0) { t[0] = 0; return; }
    uint n = 0;
    for (uint q = p; q < TK && n < G; ++q) if (idx[q] == e) t[1 + n++] = q;
    t[0] = n;
"""
GROUP_SCAN = """
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row0 = threadgroup_position_in_grid.x * RT;
    constexpr uint RS = RT / 8, LPR = 32 / RS, SK = 64 / RT;  // scalar: RS rows/simdgroup; MMA: SK-way K split
    uint slot = thread_position_in_grid.y;
    threadgroup float red[8 * 128];
    if (row0 >= NR) return;
    const device uint* tt = tl + slot * (G + 1);
    uint n = tt[0];
    if (n == 0) return;
    int e = idx[slot];
    uint m[G];
    for (uint i = 0; i < G; ++i) m[i] = tt[1 + min(i, n - 1)];
    uint sub = lane % LPR;
    uint row = row0 + sg * RS + lane / LPR;
    uint rbk = sg / SK, ks = sg % SK;
    constexpr uint NC = NK / 32;
    // MMA fragment coordinates
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    uint s0 = m[min(fn, n - 1)], s1 = m[min(fn + 1, n - 1)];
    uint rowm = row0 + rbk * 8 + fm;
    size_t rf = (size_t)e * NR + rowm;
"""
GATE_UP_G_SRC = GROUP_SCAN + """
    size_t rr = (size_t)e * F + row;
    if (n <= 2) {
        const device T* xr = x + (slot / K) * D;
        float xs[32], ga = 0.f, ua = 0.f, gb = 0.f, ub = 0.f;
        if (n == 1) {
            for (uint c = sub; c < NC; c += LPR) {
                load_x32(xr + c * 32, xs);
                size_t q = rr * NC + c;
                ga += e8s(gs[q]) * dot_fp4x32(xs, ((const device uint4*)gq)[q]);
                ua += e8s(us[q]) * dot_fp4x32(xs, ((const device uint4*)uq)[q]);
            }
        } else {
            const device T* xr2 = x + (m[1] / K) * D;
            float xt[32];
            for (uint c = sub; c < NC; c += LPR) {
                load_x32(xr + c * 32, xs);
                load_x32(xr2 + c * 32, xt);
                size_t q = rr * NC + c;
                float a, b, sg_ = e8s(gs[q]), su = e8s(us[q]);
                dot2_fp4x32(xs, xt, ((const device uint4*)gq)[q], a, b); ga += sg_ * a; gb += sg_ * b;
                dot2_fp4x32(xs, xt, ((const device uint4*)uq)[q], a, b); ua += su * a; ub += su * b;
            }
        }
        for (uint o = LPR / 2; o > 0; o >>= 1) {
            ga += simd_shuffle_xor(ga, o); ua += simd_shuffle_xor(ua, o);
            gb += simd_shuffle_xor(gb, o); ub += simd_shuffle_xor(ub, o);
        }
        if (sub == 0) {
            h[slot * F + row] = T(swiglu(ga, ua));
            if (n == 2) h[m[1] * F + row] = T(swiglu(gb, ub));
        }
        return;
    }
    const device T* xa = x + (s0 / K) * D + fm * 4;
    const device T* xb = x + (s1 / K) * D + fm * 4;
    const device uint* gw = gq + rf * (NK / 8) + fn / 2;
    const device uint* uw = uq + rf * (NK / 8) + fn / 2;
    simdgroup_matrix<float, 8, 8> Cg = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> Cu = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> Pg, Pu;
    simdgroup_matrix<half, 8, 8> Ag, Au, B;
    for (uint g = ks; g < NC; g += SK) {
        uint wg = gw[g * 4], wu = uw[g * 4];
        float sg_ = e8s(gs[rf * NC + g]), su = e8s(us[rf * NC + g]);
        half4 a0 = half4(*(const device vec<T, 4>*)(xa + g * 32)), b0 = half4(*(const device vec<T, 4>*)(xb + g * 32));
        for (uint s = 0; s < 4; ++s) {
            half2 vg = fp4x2(wg >> (4 * s)), vu = fp4x2(wu >> (4 * s));
            Ag.thread_elements()[0] = vg.x; Ag.thread_elements()[1] = vg.y;
            Au.thread_elements()[0] = vu.x; Au.thread_elements()[1] = vu.y;
            B.thread_elements()[0] = a0[s]; B.thread_elements()[1] = b0[s];
            if (s == 0) { simdgroup_multiply(Pg, Ag, B); simdgroup_multiply(Pu, Au, B); }
            else { simdgroup_multiply_accumulate(Pg, Ag, B, Pg); simdgroup_multiply_accumulate(Pu, Au, B, Pu); }
        }
        for (uint i = 0; i < 2; ++i) {
            Cg.thread_elements()[i] = fma(sg_, Pg.thread_elements()[i], Cg.thread_elements()[i]);
            Cu.thread_elements()[i] = fma(su, Pu.thread_elements()[i], Cu.thread_elements()[i]);
        }
    }
    for (uint i = 0; i < 2; ++i) {
        red[sg * 128 + lane * 2 + i] = Cg.thread_elements()[i];
        red[sg * 128 + 64 + lane * 2 + i] = Cu.thread_elements()[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    for (uint i = 0; i < 2; ++i) {
        float g = 0.f, u = 0.f;
        for (uint j = 0; j < SK; ++j) { g += red[(sg + j) * 128 + lane * 2 + i]; u += red[(sg + j) * 128 + 64 + lane * 2 + i]; }
        if (fn + i < n) h[m[fn + i] * F + rowm] = T(swiglu(g, u));
    }
"""
# grouped down: writes w * y per slot (fp32); a combine kernel sums the k slots of each token
DOWN_G_SRC = GROUP_SCAN + """
    size_t rr = (size_t)e * D + row;
    if (n <= 2) {
        float hs[32], a = 0.f, b2 = 0.f;
        if (n == 1) {
            for (uint c = sub; c < NC; c += LPR) {
                load_x32(h + slot * F + c * 32, hs);
                size_t q = rr * NC + c;
                a += e8s(ds[q]) * dot_fp4x32(hs, ((const device uint4*)dq)[q]);
            }
        } else {
            float ht[32];
            for (uint c = sub; c < NC; c += LPR) {
                load_x32(h + slot * F + c * 32, hs);
                load_x32(h + m[1] * F + c * 32, ht);
                size_t q = rr * NC + c;
                float da, db, sd = e8s(ds[q]);
                dot2_fp4x32(hs, ht, ((const device uint4*)dq)[q], da, db);
                a += sd * da; b2 += sd * db;
            }
        }
        for (uint o = LPR / 2; o > 0; o >>= 1) { a += simd_shuffle_xor(a, o); b2 += simd_shuffle_xor(b2, o); }
        if (sub == 0) {
            yp[slot * D + row] = wts[slot] * 0x1p14f * a;
            if (n == 2) yp[m[1] * D + row] = wts[m[1]] * 0x1p14f * b2;
        }
        return;
    }
    const device T* ha = h + s0 * F + fm * 4;
    const device T* hb = h + s1 * F + fm * 4;
    const device uint* dw = dq + rf * (NK / 8) + fn / 2;
    simdgroup_matrix<float, 8, 8> C = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> P;
    simdgroup_matrix<half, 8, 8> A, B;
    for (uint g = ks; g < NC; g += SK) {
        uint w = dw[g * 4];
        float sd = e8s(ds[rf * NC + g]);
        half4 a0 = half4(*(const device vec<T, 4>*)(ha + g * 32)), b0 = half4(*(const device vec<T, 4>*)(hb + g * 32));
        for (uint s = 0; s < 4; ++s) {
            half2 v = fp4x2(w >> (4 * s));
            A.thread_elements()[0] = v.x; A.thread_elements()[1] = v.y;
            B.thread_elements()[0] = a0[s]; B.thread_elements()[1] = b0[s];
            if (s == 0) simdgroup_multiply(P, A, B);
            else simdgroup_multiply_accumulate(P, A, B, P);
        }
        for (uint i = 0; i < 2; ++i) C.thread_elements()[i] = fma(sd, P.thread_elements()[i], C.thread_elements()[i]);
    }
    for (uint i = 0; i < 2; ++i) red[sg * 128 + lane * 2 + i] = C.thread_elements()[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    for (uint i = 0; i < 2; ++i) {
        float a = 0.f;
        for (uint j = 0; j < SK; ++j) a += red[(sg + j) * 128 + lane * 2 + i];
        if (fn + i < n) yp[m[fn + i] * D + rowm] = wts[m[fn + i]] * 0x1p14f * a;
    }
"""

PAIR_SCAN = """
    uint rank = 0, nxt = TK;
    for (uint q = lane; q < TK; q += 32) {
        bool same = idx[q] == e;
        rank += (same && q < slot) ? 1u : 0u;
        if (same && q > slot) nxt = min(nxt, q);
    }
    rank = simd_sum(rank); nxt = simd_min(nxt);
    if (rank & 1u) return;
"""
# paired down: writes w * y per slot (fp32); a combine kernel sums the k slots of each token
DOWN_P_SRC = """
    constexpr uint RB = R * 32 / LPR;
    uint lane = thread_index_in_simdgroup;
    uint sub = lane % LPR;
    uint row0 = (thread_position_in_grid.x / 32) * RB + (lane / LPR) * R;
    uint slot = thread_position_in_grid.y;
    int e = idx[slot];
""" + PAIR_SCAN + """
    bool ok = row0 < D;
    constexpr uint NC = F / 32;
    size_t r0 = (size_t)e * D + (ok ? row0 : 0);
    float hs[32], ht[32];
    float aa[R] = {0.f}, ab[R] = {0.f};
    if (nxt == TK) {
        for (uint c = sub; ok && c < NC; c += LPR) {
            load_x32(h + slot * F + c * 32, hs);
            for (uint r = 0; r < R; ++r) {
                size_t q = (r0 + r) * NC + c;
                aa[r] += e8s(ds[q]) * dot_fp4x32(hs, ((const device uint4*)dq)[q]);
            }
        }
    } else {
        for (uint c = sub; ok && c < NC; c += LPR) {
            load_x32(h + slot * F + c * 32, hs);
            load_x32(h + nxt * F + c * 32, ht);
            for (uint r = 0; r < R; ++r) {
                size_t q = (r0 + r) * NC + c;
                float a, b, sd = e8s(ds[q]);
                dot2_fp4x32(hs, ht, ((const device uint4*)dq)[q], a, b);
                aa[r] += sd * a; ab[r] += sd * b;
            }
        }
    }
    float wa = wts[slot] * 0x1p14f, wb = wts[min(nxt, uint(TK - 1))] * 0x1p14f;
    for (uint r = 0; r < R; ++r) {
        float a = aa[r], b = ab[r];
        for (uint o = LPR / 2; o > 0; o >>= 1) { a += simd_shuffle_xor(a, o); b += simd_shuffle_xor(b, o); }
        if (sub == 0 && ok) {
            yp[slot * D + row0 + r] = wa * a;
            if (nxt < TK) yp[nxt * D + row0 + r] = wb * b;
        }
    }
"""
# 1 < T < T_GROUP: slots of the same expert are paired in-kernel (rank among same-expert slots: even rank leads,
# odd rank returns); a pair shares every weight decode. Unpaired slots run the plain GEMV.
GATE_UP_P_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row0 = (thread_position_in_grid.x / 32) * R;
    uint slot = thread_position_in_grid.y;
    if (row0 >= F) return;
    constexpr uint NC = D / 32;
    int e = idx[slot];
""" + PAIR_SCAN + """
    size_t r0 = (size_t)e * F + row0;
    const device T* xr = x + (slot / K) * D;
    float xs[32];
    if (nxt == TK) {
        float ag[R] = {0.f}, au[R] = {0.f};
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
            if (lane == 0) h[slot * F + row0 + r] = T(swiglu(g, u));
        }
        return;
    }
    const device T* xr2 = x + (nxt / K) * D;
    float xt[32];
    float ga[R] = {0.f}, gb[R] = {0.f}, ua[R] = {0.f}, ub[R] = {0.f};
    for (uint c = lane; c < NC; c += 32) {
        load_x32(xr + c * 32, xs);
        load_x32(xr2 + c * 32, xt);
        for (uint r = 0; r < R; ++r) {
            size_t q = (r0 + r) * NC + c;
            float a, b, sg = e8s(gs[q]), su = e8s(us[q]);
            dot2_fp4x32(xs, xt, ((const device uint4*)gq)[q], a, b); ga[r] += sg * a; gb[r] += sg * b;
            dot2_fp4x32(xs, xt, ((const device uint4*)uq)[q], a, b); ua[r] += su * a; ub[r] += su * b;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float g = simd_sum(ga[r]), u = simd_sum(ua[r]), g2 = simd_sum(gb[r]), u2 = simd_sum(ub[r]);
        if (lane == 0) {
            h[slot * F + row0 + r] = T(swiglu(g, u));
            h[nxt * F + row0 + r] = T(swiglu(g2, u2));
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

_k1 = mx.fast.metal_kernel(name="ar_mxfp4_gate_up_v2", input_names=["x", "gq", "gs", "uq", "us", "idx"],
                           output_names=["h"], source=GATE_UP_SRC, header=HEADER)
_k2 = mx.fast.metal_kernel(name="ar_mxfp4_down_v2", input_names=["h", "dq", "ds", "idx", "wts"],
                           output_names=["y"], source=DOWN_SRC, header=HEADER)
_k1p = mx.fast.metal_kernel(name="ar_mxfp4_gate_up_p_v2", input_names=["x", "gq", "gs", "uq", "us", "idx"],
                            output_names=["h"], source=GATE_UP_P_SRC, header=HEADER)
_k2p = mx.fast.metal_kernel(name="ar_mxfp4_down_p_v2", input_names=["h", "dq", "ds", "idx", "wts"],
                            output_names=["yp"], source=DOWN_P_SRC, header=HEADER)
_k0 = mx.fast.metal_kernel(name="ar_mxfp4_map_v1", input_names=["idx"], output_names=["tl"], source=MAP_SRC)
_k1g = mx.fast.metal_kernel(name="ar_mxfp4_gate_up_g_v10", input_names=["x", "gq", "gs", "uq", "us", "idx", "tl"],
                            output_names=["h"], source=GATE_UP_G_SRC, header=HEADER)
_k2g = mx.fast.metal_kernel(name="ar_mxfp4_down_g_v10", input_names=["h", "dq", "ds", "idx", "wts", "tl"],
                            output_names=["yp"], source=DOWN_G_SRC, header=HEADER)
_k3 = mx.fast.metal_kernel(name="ar_mxfp4_combine_v1", input_names=["yp"], output_names=["y"], source=COMBINE_SRC)


def _lpr(chunks, iters=ITERS):
    """lanes per row: smallest power of 2 >= chunks / iters, capped at 32."""
    l = 1
    while l < 32 and l * iters < chunks:
        l *= 2
    return l


def prepare(w_gate, w_up, w_down):
    E, f, d = w_gate.shape
    assert d % 64 == 0 and f % 64 == 0 and d % R_DOWN == 0, "shape constraints"
    return {n: mx.quantize(w, mode=MODE) for n, w in (("g", w_gate), ("u", w_up), ("d", w_down))}


def dequant(s):
    return tuple(mx.dequantize(*s[n], mode=MODE) for n in ("g", "u", "d"))


def _rt(nr, rt):
    """rows per threadgroup on the T > 1 path: rt (8, 16 or 32), halved until it divides nr."""
    while nr % rt:
        rt //= 2
    return rt


def run(s, x, idx, wts):
    T, k = idx.shape
    (gq, gs), (uq, us), (dq, ds) = s["g"], s["u"], s["d"]
    F, D = gq.shape[1], dq.shape[1]
    base = [("T", x.dtype), ("D", D), ("F", F), ("K", k)]
    if 1 < T < T_GROUP:   # few repeated experts: pairs share a decode, no map prepass
        TK = T * k
        h = _k1p(inputs=[x, gq, gs, uq, us, idx], template=base + [("R", R_UP), ("TK", TK)],
                 grid=(32 * F // R_UP, TK, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
                 output_shapes=[(TK, F)], output_dtypes=[x.dtype])[0]
        lpr = _lpr(F // 32)
        rb = R_DOWN * 32 // lpr
        yp = _k2p(inputs=[h, dq, ds, idx, wts.flatten()], template=base + [("R", R_DOWN), ("LPR", lpr), ("TK", TK)],
                  grid=(32 * (-(-D // rb)), TK, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
                  output_shapes=[(TK, D)], output_dtypes=[mx.float32])[0]
        return _k3(inputs=[yp], template=[("T", x.dtype), ("D", D), ("K", k), ("NT", T)],
                   grid=(T * D, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]
    if T > 1:
        TK = T * k
        tl = _k0(inputs=[idx], template=[("TK", TK), ("G", G)], grid=(TK, 1, 1), threadgroup=(min(TK, 256), 1, 1),
                 output_shapes=[(TK * (G + 1),)], output_dtypes=[mx.uint32])[0]
        h = _k1g(inputs=[x, gq, gs, uq, us, idx, tl], template=base + [("TK", TK), ("G", G), ("NR", F), ("NK", D), ("RT", _rt(F, RT_UP))],
                 grid=(256 * F // _rt(F, RT_UP), TK, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(TK, F)], output_dtypes=[x.dtype])[0]
        yp = _k2g(inputs=[h, dq, ds, idx, wts.flatten(), tl], template=base + [("TK", TK), ("G", G), ("NR", D), ("NK", F), ("RT", _rt(D, RT_DOWN))],
                  grid=(256 * D // _rt(D, RT_DOWN), TK, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(TK, D)], output_dtypes=[mx.float32])[0]
        return _k3(inputs=[yp], template=[("T", x.dtype), ("D", D), ("K", k), ("NT", T)],
                   grid=(T * D, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]
    h = _k1(inputs=[x, gq, gs, uq, us, idx], template=base + [("R", R_UP)],
            grid=(32 * F // R_UP, k, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
            output_shapes=[(k, F)], output_dtypes=[x.dtype])[0]
    lpr = _lpr(F // 32)
    rb = R_DOWN * 32 // lpr
    return _k2(inputs=[h, dq, ds, idx, wts.flatten()], template=base + [("R", R_DOWN), ("LPR", lpr)],
               grid=(32 * (-(-D // rb)), k, 1), threadgroup=(32, k, 1),
               output_shapes=[(1, D)], output_dtypes=[x.dtype])[0]
