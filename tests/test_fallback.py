"""gather_moe (the dispatchers' high-load fallback) must equal the fp32 reference and mlx-lm's SwitchGLU path."""
import mlx.core as mx
import pytest

from moelab import mlx_backend as be
from moelab.fallback import gather_moe
from moelab.problem import make_problem
from moelab.shapes import MoEShape


@pytest.mark.parametrize("mode,gs,bits", [("affine", 64, 4), ("mxfp4", 32, 4)])
@pytest.mark.parametrize("T", [1, 5, 40])           # 40*4 = 160 slots -> sorted path
def test_gather_moe(mode, gs, bits, T):
    sh = MoEShape("g", d=256, f=384, E=16, k=4)
    p = make_problem(sh, T, "zipf", T)
    w = [mx.array(p[n]).astype(mx.bfloat16) for n in ("w_gate", "w_up", "w_down")]
    kw = {"mode": mode} if mode == "mxfp4" else {"group_size": gs, "bits": bits}
    s = {n: mx.quantize(v, **kw) for n, v in zip("gud", w)}
    deq = [mx.dequantize(*s[n], **kw) for n in "gud"]
    x, idx, wts = mx.array(p["x"]).astype(mx.bfloat16), mx.array(p["idx"]), mx.array(p["wts"])
    err = be.errors(gather_moe(s, x, idx, wts, gs, bits, mode), be.reference(*deq, x, idx, wts))
    assert err["finite"] and err["rel_fro"] < 1e-2, err


def test_gather_moe_equals_mlxlm_switchglu():
    """Audit: the fallback must be the baseline itself, not just close to the fp32 reference."""
    base = be.load_variant("moelab/variants/mlx/mlxlm_switchglu_q4.py")
    sh = MoEShape("g", d=256, f=384, E=16, k=4)
    for T in (3, 40):
        p = make_problem(sh, T, "zipf", T)
        w = [mx.array(p[n]).astype(mx.bfloat16) for n in ("w_gate", "w_up", "w_down")]
        sb = base.prepare(*w)
        m = sb["m"]
        s = {n: (pr.weight, pr.scales, pr.biases) for n, pr in zip("gud", (m.gate_proj, m.up_proj, m.down_proj))}
        x, idx, wts = mx.array(p["x"]).astype(mx.bfloat16), mx.array(p["idx"]), mx.array(p["wts"])
        a, b = base.run(sb, x, idx, wts).astype(mx.float32), gather_moe(s, x, idx, wts, 64, 4).astype(mx.float32)
        assert mx.array_equal(a, b).item(), float(mx.abs(a - b).max())
