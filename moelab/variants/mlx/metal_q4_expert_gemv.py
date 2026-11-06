"""Custom Metal: expert-centric int4 MoE GEMV (v4). EDIT ME.

Problem with v1-v3 and MLX gather_qmm at 1 < T < 64: every (token, slot) re-reads its
expert's weights. With skewed routing many slots share an expert -> wasted bandwidth.

Fix: argsort slots by expert. Slot position p (in sorted order) is a *leader* if its
rank inside the expert's segment is a multiple of MAXN. A leader loads each weight
chunk once (uint4 = 32 int4 weights) and applies it to up to MAXN tokens.
Non-leaders exit immediately. Weight bytes ~ U experts instead of T*k slots.
Same math tricks as v3: prescaled x (no shifts), s*dot(x,q) + b*sum(x).
Knobs: MAXN, SG_PER_TG.
"""
import mlx.core as mx

META = {"name": "metal_q4_expert_gemv", "description": "int4, each expert row read once per <=MAXN tokens"}
BITS, GS = 4, 64
W_BITS = BITS + 32 / GS
MAXN = 8
SG_PER_TG = 4

HEADER = """
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
// segment of equal experts in sorted order: returns n tokens to handle (0 = not a leader)
template <typename P>
inline uint leader_count(P se, uint p, uint total, uint maxn) {
    uint e = se[p];
    uint rank = 0;
    while (p - rank > 0 && se[p - rank - 1] == e) ++rank;
    if (rank % maxn != 0) return 0;
    uint n = 1;
    while (n < maxn && p + n < total && se[p + n] == e) ++n;
    return n;
}
"""

GATE_UP_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row  = thread_position_in_grid.x / 32;
    uint p    = thread_position_in_grid.y;           // position in expert-sorted order
    if (row >= F) return;
    uint n = leader_count(se, p, TK, MAXN);
    if (n == 0) return;
    constexpr uint NW = D / 8, NG = D / GS;
    size_t r = (size_t)se[p] * F + row;
    const device uint4* g4 = (const device uint4*)(gq + r * NW);
    const device uint4* u4 = (const device uint4*)(uq + r * NW);
    float ag[MAXN], au[MAXN];
    for (uint i = 0; i < MAXN; ++i) { ag[i] = 0.f; au[i] = 0.f; }
    float xs[32];
    for (uint c = lane; c < NW / 4; c += 32) {
        uint grp = (c * 32) / GS;
        uint4 gw = g4[c], uw = u4[c];
        float gsc = float(gs[r * NG + grp]), gbi = float(gb[r * NG + grp]);
        float usc = float(us[r * NG + grp]), ubi = float(ub[r * NG + grp]);
        for (uint i = 0; i < MAXN; ++i) {
            if (i < n) {
                uint t = order[p + i] / K;
                float xsum = load_x32(x + t * D + c * 32, xs);
                ag[i] += gsc * dot_q4x32(xs, gw) + gbi * xsum;
                au[i] += usc * dot_q4x32(xs, uw) + ubi * xsum;
            }
        }
    }
    for (uint i = 0; i < MAXN; ++i) {
        if (i < n) {
            float g = simd_sum(ag[i]), u = simd_sum(au[i]);
            if (lane == 0) h[order[p + i] * F + row] = T(g / (1.f + metal::exp(-g)) * u);
        }
    }
"""

DOWN_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row  = thread_position_in_grid.x / 32;
    uint p    = thread_position_in_grid.y;
    if (row >= D) return;
    uint n = leader_count(se, p, TK, MAXN);
    if (n == 0) return;
    constexpr uint NW = F / 8, NG = F / GS;
    size_t r = (size_t)se[p] * D + row;
    const device uint4* w4 = (const device uint4*)(dq + r * NW);
    float acc[MAXN];
    for (uint i = 0; i < MAXN; ++i) acc[i] = 0.f;
    float hs[32];
    for (uint c = lane; c < NW / 4; c += 32) {
        uint grp = (c * 32) / GS;
        uint4 w = w4[c];
        float sc = float(ds[r * NG + grp]), bi = float(db[r * NG + grp]);
        for (uint i = 0; i < MAXN; ++i) {
            if (i < n) {
                float hsum = load_x32(h + order[p + i] * F + c * 32, hs);
                acc[i] += sc * dot_q4x32(hs, w) + bi * hsum;
            }
        }
    }
    for (uint i = 0; i < MAXN; ++i) {
        if (i < n) {
            float a = simd_sum(acc[i]);
            uint slot = order[p + i];
            if (lane == 0) y[slot * D + row] = wts[slot] * a;
        }
    }
"""

_k1 = mx.fast.metal_kernel(name="moe_q4e_gate_up", input_names=["x", "gq", "gs", "gb", "uq", "us", "ub", "se", "order"],
                           output_names=["h"], source=GATE_UP_SRC, header=HEADER)
_k2 = mx.fast.metal_kernel(name="moe_q4e_down", input_names=["h", "dq", "ds", "db", "se", "order", "wts"],
                           output_names=["y"], source=DOWN_SRC, header=HEADER)


def prepare(w_gate, w_up, w_down):
    E, f, d = w_gate.shape
    assert d % 32 == 0 and f % 32 == 0, "d, f must be multiples of 32"
    return {n: mx.quantize(w, group_size=GS, bits=BITS) for n, w in (("g", w_gate), ("u", w_up), ("d", w_down))}


def dequant(s):
    return tuple(mx.dequantize(*s[n], group_size=GS, bits=BITS) for n in ("g", "u", "d"))


def run(s, x, idx, wts):
    T, k = idx.shape
    (gq, gs, gb), (uq, us, ub), (dq, ds, db) = s["g"], s["u"], s["d"]
    F, D = gq.shape[1], dq.shape[1]
    flat = idx.flatten().astype(mx.uint32)
    order = mx.argsort(flat).astype(mx.uint32)
    se = flat[order]
    tmpl = [("T", x.dtype), ("S", gs.dtype), ("D", D), ("F", F), ("K", k), ("GS", GS),
            ("MAXN", MAXN), ("TK", T * k)]
    h = _k1(inputs=[x, gq, gs, gb, uq, us, ub, se, order], template=tmpl,
            grid=(32 * F, T * k, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
            output_shapes=[(T * k, F)], output_dtypes=[x.dtype])[0]
    yp = _k2(inputs=[h, dq, ds, db, se, order, wts.flatten()], template=tmpl,
             grid=(32 * D, T * k, 1), threadgroup=(32 * SG_PER_TG, 1, 1),
             output_shapes=[(T * k, D)], output_dtypes=[mx.float32])[0]
    return yp.reshape(T, k, D).sum(axis=1).astype(x.dtype)
