"""Hazard from audit 2: int4 MMA paths form dequantized weights / scales in half, so affine scales
below the half subnormal range (~6e-8) would flush to zero. Real Qwen3 min scale is 8.6e-6 (safe),
but other checkpoints may be smaller. Tiny weights + large activations keep outputs O(1)."""
import glob
import os

import mlx.core as mx
import numpy as np
import pytest

from moelab import mlx_backend as be
from moelab.shapes import MoEShape

ROOT = os.path.dirname(os.path.dirname(__file__))
VARIANTS = [p for p in sorted(glob.glob(os.path.join(ROOT, "moelab/variants/mlx/*.py")))
            if getattr(be.load_variant(p), "MODE", "affine") != "mxfp4" and "BITS" in vars(be.load_variant(p))]
SHAPE = MoEShape("tiny-scales", d=256, f=256, E=8, k=2)
HISTORICAL = {"metal_q4_tiles_mma.py"}  # kept for the ablation story, not used in production


@pytest.mark.parametrize("path", VARIANTS, ids=os.path.basename)
@pytest.mark.parametrize("T,routing", [(1, "uniform"), (3, "uniform"), (16, "one_hot"), (32, "uniform")])
def test_tiny_affine_scales(path, T, routing):
    if os.path.basename(path) in HISTORICAL and T > 1:
        pytest.xfail("historical round-1 kernel: half dequant in MMA path (fixed in tiles_v2 round 6)")
    mod = be.load_variant(path)
    w = [x * 1e-6 for x in be.make_weights(SHAPE, "bfloat16", seed=2)]
    state = mod.prepare(*w)
    x, idx, wts = be.make_inputs(SHAPE, T, "bfloat16", routing, seed=T)
    x = x * 1e6
    err = be.errors(mod.run(state, x, idx, wts), be.reference(*mod.dequant(state), x, idx, wts))
    assert err["finite"] and err["rel_fro"] < 1e-2, err
