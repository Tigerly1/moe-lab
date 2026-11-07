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


@pytest.mark.parametrize("E,k", [(8, 2), (16, 6), (32, 4), (60, 4), (64, 6), (128, 8), (256, 8), (128, 1), (512, 32), (1000, 8)])
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


@pytest.mark.parametrize("E,k", [(16, 6), (60, 4), (64, 6), (128, 8)])
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


def test_bf16_stock_path_differs_only_at_ties():
    """Against the real mlx-lm sequence (softmax output in bf16, then argpartition): any id difference must be
    an exact tie of the bf16 probabilities."""
    mx.random.seed(7)
    lg = (mx.random.normal((256, 128)) * 2).astype(mx.bfloat16)
    g = mx.softmax(lg, axis=-1, precise=True)
    i0 = np.array(mx.argpartition(g, kth=-8, axis=-1)[..., -8:])
    i1 = np.array(topk_norm(lg, 8)[0])
    G = np.array(g.astype(mx.float32))
    for t in range(256):
        a, b = set(i0[t].tolist()), set(i1[t].tolist())
        if a != b:
            np.testing.assert_array_equal(np.sort(G[t, list(a - b)]), np.sort(G[t, list(b - a)]))


@pytest.mark.parametrize("bad", ["nan", "neginf"])
def test_nonfinite_logits_stay_in_range(bad):
    """Audit: with fewer than k finite logits the kernel must still return k distinct in-range ids."""
    lg = np.random.default_rng(0).standard_normal((4, 64)).astype(np.float32)
    lg[0, :] = np.nan if bad == "nan" else -np.inf          # no finite logit at all
    lg[1, 3:] = np.nan if bad == "nan" else -np.inf         # only 3 finite logits, k = 6
    i, w = topk_norm(mx.array(lg), 6)
    i = np.array(i)
    assert ((i >= 0) & (i < 64)).all(), i
    for t in range(4):
        assert len(set(i[t].tolist())) == 6
    assert set(i[1, :3].tolist()) == {0, 1, 2}
    np.testing.assert_allclose(np.array(w[1]).sum(), 1.0, atol=1e-5)    # weight only on the finite ones
    assert np.allclose(np.array(w[1, 3:]), 0)
