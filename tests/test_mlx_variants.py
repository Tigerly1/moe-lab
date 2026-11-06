"""Correctness gate: every MLX variant vs fp64 numpy reference, incl. edge cases."""
import glob
import os

import mlx.core as mx
import numpy as np
import pytest

from moelab import mlx_backend as be
from moelab.problem import make_problem, reference, error_metrics
from moelab.shapes import MoEShape

ROOT = os.path.dirname(os.path.dirname(__file__))
VARIANTS = sorted(glob.glob(os.path.join(ROOT, "moelab/variants/mlx/*.py")))
SHAPE = MoEShape("t", d=128, f=256, E=8, k=2)
DTYPES = ["float16", "bfloat16"]
CASES = [(1, "uniform"), (7, "uniform"), (33, "zipf"), (100, "uniform"), (64, "one_hot")]


def test_mlx_reference_matches_numpy():
    p = make_problem(SHAPE, 17)
    w = [mx.array(p[n]) for n in ("w_gate", "w_up", "w_down")]
    y = be.reference(*w, mx.array(p["x"]), mx.array(p["idx"]), mx.array(p["wts"]))
    assert error_metrics(np.array(y), reference(p))["rel_fro"] < 1e-5


@pytest.mark.parametrize("path", VARIANTS, ids=os.path.basename)
@pytest.mark.parametrize("T,routing", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_variant(path, T, routing, dtype):
    mod = be.load_variant(path)
    p = make_problem(SHAPE, T, routing)
    w32 = [mx.array(p[n]) for n in ("w_gate", "w_up", "w_down")]
    dt = be.DTYPES[dtype]
    w16 = [w.astype(dt) for w in w32]
    state = mod.prepare(*w16)
    ref_w = mod.dequant(state) if hasattr(mod, "dequant") else w16
    x = mx.array(p["x"]).astype(dt)
    idx, wts = mx.array(p["idx"]), mx.array(p["wts"])
    y = mod.run(state, x, idx, wts)
    err = be.errors(y, be.reference(*ref_w, x, idx, wts))
    assert err["finite"] and err["rel_fro"] < be.TOL[dtype], err
