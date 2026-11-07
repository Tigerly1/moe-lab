"""Fused router tail vs the stock mlx-lm sequence (softmax precise -> argpartition -> take -> renormalize)."""
import mlx.core as mx
import numpy as np
import pytest

from moelab.router import topk_norm


def stock(logits, k):
    g = mx.softmax(logits.astype(mx.float32), axis=-1, precise=True)
    i = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
    s = mx.take_along_axis(g, i, axis=-1)
    return i, s / s.sum(-1, keepdims=True)


@pytest.mark.parametrize("E,k", [(8, 2), (16, 6), (32, 4), (60, 4), (64, 6), (128, 8), (256, 8), (128, 1)])
@pytest.mark.parametrize("T", [1, 7, 64])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_matches_stock(E, k, T, dtype):
    mx.random.seed(E + T + k)
    lg = (mx.random.normal((T, E)) * 3).astype(dtype)
    i1, w1 = topk_norm(lg, k)
    i0, w0 = stock(lg, k)
    L = np.array(lg.astype(mx.float32))
    for t in range(T):
        a, b = np.array(i1[t]), np.array(i0[t])
        assert len(set(a.tolist())) == k
        # same multiset of selected logits; ids may differ only among exact ties at the k-th value (bf16 ties)
        np.testing.assert_array_equal(np.sort(L[t, a]), np.sort(L[t, b]))
        kth = np.sort(L[t])[-k]
        assert set(a[L[t, a] > kth].tolist()) == set(b[L[t, b] > kth].tolist())
        np.testing.assert_allclose(np.sort(np.array(w1[t])), np.sort(np.array(w0[t])), atol=1e-5)
    np.testing.assert_allclose(np.array(w1.sum(-1)), 1.0, atol=1e-5)


def test_ties_pick_distinct_experts():
    lg = mx.zeros((3, 64))                      # all tied: must still return k distinct ids, equal weights
    i, w = topk_norm(lg, 6)
    for t in range(3):
        assert len(set(np.array(i[t]).tolist())) == 6
    np.testing.assert_allclose(np.array(w), 1 / 6, atol=1e-6)


@pytest.mark.parametrize("E,k", [(16, 6), (64, 6), (128, 8)])
@pytest.mark.parametrize("T", [1, 33])
def test_full_softmax_weights(E, k, T):
    """norm=False (DeepSeek-V2 greedy): weights are the full-softmax probabilities of the chosen experts."""
    mx.random.seed(T + E)
    lg = mx.random.normal((T, E)) * 2
    i, w = topk_norm(lg, k, norm=False)
    P = np.array(mx.softmax(lg, axis=-1, precise=True))
    for t in range(T):
        ids = np.array(i[t])
        np.testing.assert_array_equal(np.sort(ids), np.sort(np.argsort(-P[t])[:k]))
        np.testing.assert_allclose(np.array(w[t]), P[t, ids], rtol=1e-5)
