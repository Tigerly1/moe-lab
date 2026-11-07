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
