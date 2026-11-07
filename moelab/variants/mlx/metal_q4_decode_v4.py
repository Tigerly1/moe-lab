"""Custom Metal: fused int4 MoE GEMV, autoresearch round 2 (from v3.1). EDIT ME.

ar round 2 vs v3.1: 2 dispatches. The down kernel runs threadgroups of (32, K): simdgroup j
computes slot t*K+j and the K partials are reduced in threadgroup memory straight into
y[t] (no fp32 partials, no combine kernel). Down rows are split over LPR lanes per row
(power of 2 giving ~ITERS uint4 chunks per lane) so all 32 lanes work when F/32 < 32.
---- v3.1 notes:

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

META = {"name": "metal_q4_decode_v4", "description": "v3.1 + k-reduction in down threadgroup (2 dispatches) + LPR lanes per down row"}
BITS, GS = 4, 64
W_BITS = BITS + 32 / GS
R_UP, R_DOWN = 1, 2  # tuned on M4 Pro, T=1 (scan: R_UP=1 best on all shapes)
SG_PER_TG = 4
ITERS = 3  # target uint4 chunks per lane in the down kernel

HEADER = """
// GLU activation on the gate (ACT 0 = SiLU as mlx-lm swiglu, 1 = tanh-GELU as nn.gelu_approx, Gemma-4 geglu)
template <int A> inline float glu_act(float g) {
    if (A == 1) return 0.5f * g * (1.f + metal::precise::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g)));
    return g / (1.f + metal::exp(-g));
}
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
        if (lane == 0) h[slot * F + row0 + r] = T(glu_act<ACT>(g) * u);
    }
"""

DOWN_SRC = """
    // threadgroup = (32, K): simdgroup j computes the rows of slot t*K+j; the K partials
    // are reduced in threadgroup memory and written to y[t] (no combine dispatch).
    // LPR lanes per row group (32/LPR groups of R rows per simdgroup) keeps all 32 lanes
    // busy when F/32 chunks < 32 (e.g. F=768: 24 chunks).
    constexpr uint RB = R * 32 / LPR;  // rows per simdgroup
    uint lane = thread_index_in_simdgroup;
    uint sub = lane % LPR;
    uint j = thread_position_in_threadgroup.y;
    uint rb0 = (thread_position_in_grid.x / 32) * RB;
    uint row0 = rb0 + (lane / LPR) * R;
    uint t = thread_position_in_grid.y / K;
    uint slot = t * K + j;
    bool ok = row0 < D;
    constexpr uint NW = F / 8, NG = F / GS;
    threadgroup float buf[K][RB];
    size_t r0 = (size_t)idx[slot] * D + (ok ? row0 : 0);
    float acc[R] = {0.f};
    float hs[32];
    for (uint c = sub; ok && c < NW / 4; c += LPR) {
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


_k1 = mx.fast.metal_kernel(name="mq4d4_gate_up", input_names=["x", "gq", "gs", "gb", "uq", "us", "ub", "idx"],
                           output_names=["h"], source=GATE_UP_SRC, header=HEADER)
_k2 = mx.fast.metal_kernel(name="mq4d4_down_lpr", input_names=["h", "dq", "ds", "db", "idx", "wts"],
                           output_names=["y"], source=DOWN_SRC, header=HEADER)


def _lpr(chunks):
    """lanes per row: smallest power of 2 >= chunks / ITERS, capped at 32."""
    l = 1
    while l < 32 and l * ITERS < chunks:
        l *= 2
    return l


def prepare(w_gate, w_up, w_down):
    E, f, d = w_gate.shape
    assert d % 32 == 0 and f % 32 == 0 and f % R_UP == 0, "shape constraints"
    return {n: mx.quantize(w, group_size=GS, bits=BITS) for n, w in (("g", w_gate), ("u", w_up), ("d", w_down))}


def dequant(s):
    return tuple(mx.dequantize(*s[n], group_size=GS, bits=BITS) for n in ("g", "u", "d"))


def run(s, x, idx, wts, act=0):
    T, k = idx.shape
    (gq, gs, gb), (uq, us, ub), (dq, ds, db) = s["g"], s["u"], s["d"]
    F, D = gq.shape[1], dq.shape[1]
    base = [("T", x.dtype), ("S", gs.dtype), ("D", D), ("F", F), ("K", k), ("GS", GS), ("ACT", act)]
    h = _k1(inputs=[x, gq, gs, gb, uq, us, ub, idx], template=base + [("R", R_UP)],
            grid=(32 * F // R_UP, T * k, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
            output_shapes=[(T * k, F)], output_dtypes=[x.dtype])[0]
    lpr = _lpr(F // 32)
    rb = R_DOWN * 32 // lpr
    return _k2(inputs=[h, dq, ds, db, idx, wts.flatten()], template=base + [("R", R_DOWN), ("LPR", lpr)],
               grid=(32 * (-(-D // rb)), T * k, 1), threadgroup=(32, k, 1),
               output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]
