"""Round 9: GeGLU (Gemma-4) through the int4 kernels. ACT=1 must match the fp32 tanh-GELU reference on every
dispatch path (T=1 GEMV, scalar tiles, MMA tiles, fallback), on the real Gemma-4 and Qwen3.6 expert shapes
and on small shapes that hit partial tiles. The policy fixture forces the custom kernels in one pass."""
import mlx.core as mx
import pytest

from moelab import mlx_backend as be
from moelab.problem import make_problem
from moelab.shapes import MoEShape

Q4 = "moelab/variants/mlx/metal_q4_moe.py"
_M = {}


def q4():
    if Q4 not in _M:
        _M[Q4] = be.load_variant(Q4)
    return _M[Q4]


def _no_fallback(*a, **k):
    raise AssertionError("fallback ran in forced-custom mode")


@pytest.fixture(autouse=True, params=["default", "forced-custom"])
def policy(request, monkeypatch):
    if request.param == "forced-custom":
        monkeypatch.setattr(q4(), "custom_wins", lambda *a: True)
        monkeypatch.setattr(q4(), "gather_moe", _no_fallback)
    return request.param


SHAPES = [MoEShape("gemma4", d=2816, f=704, E=128, k=8), MoEShape("qwen36", d=2048, f=512, E=256, k=8),
          MoEShape("small", d=256, f=320, E=8, k=2), MoEShape("mma", d=256, f=384, E=16, k=4)]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
@pytest.mark.parametrize("T", [1, 2, 3, 8, 17, 64])
@pytest.mark.parametrize("act", [0, 1])
def test_act(shape, T, act):
    m = q4()
    p = make_problem(shape, T, "zipf", T)
    w = [mx.array(p[n]).astype(mx.bfloat16) for n in ("w_gate", "w_up", "w_down")]
    st = m.prepare(*w)
    x, idx = mx.array(p["x"]).astype(mx.bfloat16), mx.array(p["idx"], dtype=mx.int32)
    wts = mx.array(p["wts"], dtype=mx.float32)
    y = m.run(st, x, idx, wts, act)
    ref = be.reference(*m.dequant(st), x, idx, wts, act=act)
    err = be.errors(y, ref)
    assert err["finite"] and err["rel_fro"] < be.TOL["bfloat16"], err
    other = be.errors(y, be.reference(*m.dequant(st), x, idx, wts, act=1 - act))
    assert other["rel_fro"] > 3 * err["rel_fro"], "activation did not reach the kernel"
