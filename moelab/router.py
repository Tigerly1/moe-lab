"""Fused router tail (round 7): logits [T, E] -> top-k expert ids [T, k] int32 + normalized weights [T, k] fp32.

Replaces softmax(precise) -> argpartition -> take_along_axis -> renormalize (4-5 small dispatches per layer,
~26 us/layer at T=1 on M4 Pro, 1.2 ms per Qwen3 token) with one simdgroup per token.
With norm_topk_prob, softmax over all E followed by renormalizing the top-k equals softmax over the
top-k logits, so the kernel only needs the k largest logits. Ties: lowest expert id wins (argpartition's
order is unspecified; the kernels are slot-order invariant, see tests/test_edge_cases.py).
Without renormalization (DeepSeek-V2 "greedy", norm_topk_prob=False) the weights are the full-softmax
probabilities of the chosen experts: NORM=false computes the denominator over all E first.
Any E <= 1024 (padding lanes hold -inf), k <= min(E, 32). NaN logits count as -inf; with fewer than k
finite logits the remaining slots get the lowest unused ids and weight 0 (stock mlx-lm would propagate NaN).
"""
import mlx.core as mx

SRC = """
    uint t = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    constexpr int J = (E + 31) / 32;
    float v[J];
    for (int j = 0; j < J; ++j) {
        uint e = uint(j * 32) + lane;
        float l = e < uint(E) ? float(logits[t * E + e]) : -INFINITY;
        v[j] = metal::isnan(l) ? -INFINITY : l;          // NaN logits are never selected
    }
    float m0 = -INFINITY;
    for (int j = 0; j < J; ++j) m0 = metal::max(m0, v[j]);
    m0 = simd_max(m0);
    float den = 0.f;
    if (!NORM) {
        for (int j = 0; j < J; ++j) den += v[j] == -INFINITY ? 0.f : metal::precise::exp(v[j] - m0);
        // (all -inf/NaN: den = 0 -> NaN weights, like stock softmax)
        den = simd_sum(den);
    }
    float topv[K];
    uint topi[K];
    for (int r = 0; r < K; ++r) {
        float b = -INFINITY; uint bi = 0xFFFFFFFFu;
        for (int j = 0; j < J; ++j) {
            if (v[j] > b) { b = v[j]; bi = uint(j * 32) + lane; }
        }
        float m = simd_max(b);
        uint sel = simd_min(b == m ? bi : 0xFFFFFFFFu);
        if (sel == 0xFFFFFFFFu) {                         // fewer than K finite logits: lowest unused id, weight 0
            sel = 0;
            for (bool used = true; used; ) {
                used = false;
                for (int q = 0; q < r; ++q) used = used || topi[q] == sel;
                if (used) ++sel;
            }
            m = -INFINITY;
        } else if (sel % 32 == lane) {
            v[sel / 32] = -INFINITY;
        }
        topv[r] = m; topi[r] = sel;
    }
    float s = 0.f;
    for (int r = 0; r < K; ++r) s += metal::precise::exp(topv[r] - topv[0]);
    float off = NORM ? topv[0] : m0;
    if (!NORM) s = den;
    if (lane < uint(K)) {
        float vr = topv[0]; uint ir = 0;
        for (int r = 0; r < K; ++r) if (uint(r) == lane) { vr = topv[r]; ir = topi[r]; }
        idx[t * K + lane] = int(ir);
        wts[t * K + lane] = metal::precise::exp(vr - off) / s;
    }
"""

_k = mx.fast.metal_kernel(name="router_topk_norm_v4", input_names=["logits"], output_names=["idx", "wts"], source=SRC)


def topk_norm(logits, k, norm=True):
    """logits [..., E] -> (idx [T, k] int32, wts [T, k] float32), T = prod of leading dims.
    norm=True: softmax over the top-k logits. norm=False: full-softmax probabilities of the top-k."""
    E = logits.shape[-1]
    assert E <= 1024 and k <= min(E, 32), (E, k)
    lg = logits.reshape(-1, E)
    T = lg.shape[0]
    idx, wts = _k(inputs=[lg], template=[("E", E), ("K", k), ("NORM", bool(norm))], grid=(32, T, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(T, k), (T, k)], output_dtypes=[mx.int32, mx.float32])
    return idx, wts
