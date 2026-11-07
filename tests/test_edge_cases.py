"""Narrow edge cases for the production kernels (2026-11-07 round 7): dispatcher boundaries, shapes that
are not multiples of 128, degenerate routing. Each case vs the fp32 reference on dequantized weights
(itself checked against fp64 numpy in test_mlx_variants)."""
import mlx.core as mx
import numpy as np
import pytest

from moelab import mlx_backend as be
from moelab.problem import make_problem
from moelab.shapes import MoEShape

PROD = ["moelab/variants/mlx/metal_q4_moe.py", "moelab/variants/mlx/metal_mxfp4_moe.py"]
_MODS = {}


def mod(path):
    if path not in _MODS:
        _MODS[path] = be.load_variant(path)
    return _MODS[path]


def check(path, shape, T, dtype="bfloat16", routing="uniform", seed=0, idx=None, wts=None, tol=None):
    m = mod(path)
    p = make_problem(shape, T, routing, seed)
    dt = be.DTYPES[dtype]
    w = [mx.array(p[n]).astype(dt) for n in ("w_gate", "w_up", "w_down")]
    st = m.prepare(*w)
    x = mx.array(p["x"]).astype(dt)
    idx = mx.array(p["idx"] if idx is None else idx, dtype=mx.int32)
    wts = mx.array(p["wts"] if wts is None else wts, dtype=mx.float32)
    y = m.run(st, x, idx, wts)
    err = be.errors(y, be.reference(*m.dequant(st), x, idx, wts))
    assert y.shape == x.shape and err["finite"] and err["rel_fro"] < (tol or be.TOL[dtype]), err
    return y


# T values around every dispatch boundary: 1 (GEMV), 2 (2-slot scalar), 3 (MMA starts), <8 / >=8 (MXFP4 paired
# vs grouped), and odd sizes that leave partial tiles.
@pytest.mark.parametrize("path", PROD, ids=lambda p: p.split("/")[-1])
@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 7, 8, 9, 15, 17, 31, 63, 65, 127, 129, 257])
def test_dispatch_boundaries(path, T):
    check(path, MoEShape("q", d=256, f=384, E=16, k=4), T, routing="zipf", seed=T)


# Shapes: d or f a multiple of 64 but not 128 (int4 falls to scalar tiles), gpt-oss-like odd multiples,
# k = 1, k = E (every expert), E = 1.
SHAPES = [MoEShape("d192", d=192, f=256, E=8, k=2), MoEShape("f320", d=256, f=320, E=8, k=2),
          MoEShape("odd", d=320, f=448, E=8, k=3), MoEShape("k1", d=128, f=256, E=8, k=1),
          MoEShape("kE", d=128, f=256, E=4, k=4), MoEShape("E1", d=128, f=256, E=1, k=1)]


@pytest.mark.parametrize("path", PROD, ids=lambda p: p.split("/")[-1])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
@pytest.mark.parametrize("T", [1, 2, 6, 33])
def test_shapes(path, shape, T):
    check(path, shape, T)


@pytest.mark.parametrize("path", PROD, ids=lambda p: p.split("/")[-1])
@pytest.mark.parametrize("T", [1, 4, 16])
def test_zero_and_unnormalized_weights(path, T):
    """Some combine weights exactly 0 (and a row of all zeros), others > 1: kernel must not normalize or skip wrongly."""
    sh = MoEShape("w", d=256, f=256, E=8, k=4)
    rng = np.random.default_rng(T)
    wts = rng.random((T, 4)).astype(np.float32) * 3
    wts[:, 1] = 0
    wts[0] = 0
    y = check(path, sh, T, wts=wts)
    assert float(mx.abs(y[0].astype(mx.float32)).max()) == 0.0   # all-zero weights -> exactly zero output


@pytest.mark.parametrize("path", PROD, ids=lambda p: p.split("/")[-1])
@pytest.mark.parametrize("T", [2, 8, 32])
def test_all_tokens_one_expert_set(path, T):
    """Worst imbalance: every token picks the same k experts (one expert tile holds all T tokens)."""
    sh = MoEShape("o", d=256, f=256, E=16, k=4)
    check(path, sh, T, idx=np.tile([[3, 7, 0, 15]], (T, 1)))


@pytest.mark.parametrize("path", PROD, ids=lambda p: p.split("/")[-1])
@pytest.mark.parametrize("T", [1, 8])
def test_permutation_invariance(path, T):
    """Order of the k slots must not matter (kernels group slots by expert)."""
    sh = MoEShape("p", d=256, f=256, E=16, k=4)
    p = make_problem(sh, T, "uniform", 1)
    perm = np.array([2, 0, 3, 1])
    y1 = check(path, sh, T, seed=1)
    y2 = check(path, sh, T, seed=1, idx=p["idx"][:, perm], wts=p["wts"][:, perm])
    rel = float(mx.linalg.norm((y1 - y2).astype(mx.float32).flatten()) / mx.linalg.norm(y1.astype(mx.float32).flatten()))
    assert rel < 1e-2, rel


@pytest.mark.parametrize("path", PROD, ids=lambda p: p.split("/")[-1])
def test_batch_consistency(path):
    """Token t's output must not depend on which other tokens share the call (T=1 vs inside T=16)."""
    sh = MoEShape("b", d=256, f=256, E=16, k=4)
    m = mod(path)
    p = make_problem(sh, 16, "uniform", 2)
    w = [mx.array(p[n]).astype(mx.bfloat16) for n in ("w_gate", "w_up", "w_down")]
    st = m.prepare(*w)
    x = mx.array(p["x"]).astype(mx.bfloat16)
    idx, wts = mx.array(p["idx"]), mx.array(p["wts"])
    full = m.run(st, x, idx, wts).astype(mx.float32)
    for t in (0, 5, 15):
        one = m.run(st, x[t:t + 1], idx[t:t + 1], wts[t:t + 1]).astype(mx.float32)
        rel = float(mx.linalg.norm(one - full[t:t + 1]) / mx.linalg.norm(full[t:t + 1]))
        assert rel < 1e-2, (t, rel)


@pytest.mark.parametrize("path", PROD, ids=lambda p: p.split("/")[-1])
def test_large_activations(path):
    """Activations ~100x larger (outlier tokens, as in real residual streams) stay finite and accurate in bf16."""
    sh = MoEShape("big", d=256, f=256, E=8, k=2)
    m = mod(path)
    p = make_problem(sh, 8, "uniform", 3)
    w = [mx.array(p[n]).astype(mx.bfloat16) for n in ("w_gate", "w_up", "w_down")]
    st = m.prepare(*w)
    x = (mx.array(p["x"]) * 100).astype(mx.bfloat16)
    idx, wts = mx.array(p["idx"]), mx.array(p["wts"])
    y = m.run(st, x, idx, wts)
    err = be.errors(y, be.reference(*m.dequant(st), x, idx, wts))
    assert err["finite"] and err["rel_fro"] < 1e-2, err
