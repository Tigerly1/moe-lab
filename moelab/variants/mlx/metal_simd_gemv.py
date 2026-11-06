"""Custom Metal: fused decode-oriented MoE in 2 kernels. EDIT ME.

K1 gate_up: one simdgroup per output row h[slot, r]; lanes stride over d,
            simd_sum reduce, fused SiLU(g)*u.
K2 down+combine: one simdgroup per y[t, r]; loops over the k slots of token t,
            so the weighted combine is fused (no atomics, no scatter).
Weights read once per (token, slot) -> great for small T, bad for big T
(no reuse across tokens of same expert). That's the point: compare vs gather_mm.
"""
import mlx.core as mx

META = {"name": "metal_simd_gemv", "description": "custom Metal: simdgroup GEMV, fused SiLU + combine"}
ROWS_PER_TG = 4  # simdgroups per threadgroup

GATE_UP_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row  = thread_position_in_grid.x / 32;      // output row in [0, F)
    uint slot = thread_position_in_grid.y;           // t*K + j
    if (row >= F) return;
    uint e = idx[slot];
    uint t = slot / K;
    const device T* xr = x + t * D;
    const device T* g  = w_gate + ((size_t)e * F + row) * D;
    const device T* u  = w_up   + ((size_t)e * F + row) * D;
    float ag = 0.f, au = 0.f;
    for (uint i = lane * 4; i < D; i += 32 * 4) {
        float4 xv = float4(*(const device vec<T, 4>*)(xr + i));
        ag += dot(xv, float4(*(const device vec<T, 4>*)(g + i)));
        au += dot(xv, float4(*(const device vec<T, 4>*)(u + i)));
    }
    ag = simd_sum(ag);
    au = simd_sum(au);
    if (lane == 0) {
        h[slot * F + row] = ag / (1.f + metal::exp(-ag)) * au;
    }
"""

DOWN_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x / 32;       // output row in [0, D)
    uint t   = thread_position_in_grid.y;
    if (row >= D) return;
    float acc = 0.f;
    for (uint j = 0; j < K; ++j) {
        uint slot = t * K + j;
        uint e = idx[slot];
        const device float* hr = h + slot * F;
        const device T* w = w_down + ((size_t)e * D + row) * F;
        float a = 0.f;
        for (uint i = lane * 4; i < F; i += 32 * 4) {
            a += dot(*(const device float4*)(hr + i), float4(*(const device vec<T, 4>*)(w + i)));
        }
        acc += wts[slot] * a;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[t * D + row] = T(acc);
"""

_k1 = mx.fast.metal_kernel(name="moe_gate_up", input_names=["x", "w_gate", "w_up", "idx"],
                           output_names=["h"], source=GATE_UP_SRC)
_k2 = mx.fast.metal_kernel(name="moe_down", input_names=["h", "w_down", "idx", "wts"],
                           output_names=["y"], source=DOWN_SRC)


def prepare(w_gate, w_up, w_down):
    assert w_gate.shape[2] % 4 == 0 and w_gate.shape[1] % 4 == 0, "d and f must be multiples of 4"
    return {"wg": w_gate, "wu": w_up, "wd": w_down}


def run(s, x, idx, wts):
    T, k = idx.shape
    E, F, D = s["wg"].shape
    tmpl = [("T", x.dtype), ("D", D), ("F", F), ("K", k)]
    idx = idx.flatten().astype(mx.uint32)
    h = _k1(inputs=[x, s["wg"], s["wu"], idx], template=tmpl,
            grid=(32 * F, T * k, 1), threadgroup=(32 * ROWS_PER_TG, 1, 1),
            output_shapes=[(T * k, F)], output_dtypes=[mx.float32])[0]
    return _k2(inputs=[h, s["wd"], idx, wts.flatten()], template=tmpl,
               grid=(32 * D, T, 1), threadgroup=(32 * ROWS_PER_TG, 1, 1),
               output_shapes=[(T, D)], output_dtypes=[x.dtype])[0]
