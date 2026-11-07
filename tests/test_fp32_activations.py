"""Audit 3: fp32 activations at T>1 silently lost ~10% accuracy in the round-6 MMA path.
Production dispatchers must be exact for fp32 inputs in every regime."""
import pytest

from moelab import mlx_backend as be
from moelab.problem import make_problem
from moelab.shapes import MoEShape
import mlx.core as mx

SHAPES = [MoEShape("a", d=256, f=384, E=8, k=2), MoEShape("b", d=2048, f=1408, E=16, k=6)]


@pytest.mark.parametrize("variant", ["metal_q4_moe", "metal_mxfp4_moe"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
@pytest.mark.parametrize("T,routing", [(1, "uniform"), (3, "uniform"), (16, "one_hot"), (32, "uniform")])
def test_fp32_activations(variant, shape, T, routing):
    mod = be.load_variant(f"moelab/variants/mlx/{variant}.py")
    p = make_problem(shape, T, routing)
    st = mod.prepare(*[mx.array(p[n]).astype(mx.bfloat16) for n in ("w_gate", "w_up", "w_down")])
    x, idx, wts = mx.array(p["x"]), mx.array(p["idx"]), mx.array(p["wts"])     # x stays fp32
    e = be.errors(mod.run(st, x, idx, wts), be.reference(*mod.dequant(st), x, idx, wts))
    assert e["finite"] and e["rel_fro"] < 1e-2, e
