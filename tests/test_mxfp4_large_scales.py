"""Regression (found on the real gpt-oss checkpoint): MXFP4 groups with E8M0 scale bytes >= 129
overflowed a half scale in the T>=8 MMA path. Random-normal lab weights never reach such scales,
so scale weights up until the bytes do, and check every regime of every MXFP4 variant."""
import glob
import os

import mlx.core as mx
import numpy as np
import pytest

from moelab import mlx_backend as be
from moelab.shapes import MoEShape

ROOT = os.path.dirname(os.path.dirname(__file__))
VARIANTS = [p for p in sorted(glob.glob(os.path.join(ROOT, "moelab/variants/mlx/*.py")))
            if getattr(be.load_variant(p), "MODE", None) == "mxfp4"]
SHAPE = MoEShape("big-scales", d=256, f=256, E=8, k=2)


@pytest.mark.parametrize("path", VARIANTS, ids=os.path.basename)
@pytest.mark.parametrize("T,routing", [(1, "uniform"), (3, "uniform"), (16, "one_hot"), (32, "uniform")])
def test_large_e8m0_scales(path, T, routing):
    mod = be.load_variant(path)
    w = [x * 256.0 for x in be.make_weights(SHAPE, "bfloat16", seed=1)]
    state = mod.prepare(*w)
    if "m" in state:   # mlx-lm baseline wrapper
        scales = [np.array(p.scales) for p in (state["m"].gate_proj, state["m"].up_proj, state["m"].down_proj)]
    else:              # mx.quantize(...) -> [wq, scales]
        scales = [np.array(q[1]) for q in state.values() if isinstance(q, (list, tuple))]
    assert max(int(s.max()) for s in scales) >= 129, "test must exercise E8M0 bytes >= 129"
    x, idx, wts = be.make_inputs(SHAPE, T, "bfloat16", routing, seed=T)
    x = x / 256.0   # large weight scales, normal-sized activations (like real gpt-oss); h stays in half range
    err = be.errors(mod.run(state, x, idx, wts), be.reference(*mod.dequant(state), x, idx, wts))
    assert err["finite"] and err["rel_fro"] < 1e-2, err
