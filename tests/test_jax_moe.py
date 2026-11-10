"""JAX backend (XLA ragged_dot + Pallas grouped GEMM, interpret mode) vs numpy fp64 reference."""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from moelab import jax_moe
from moelab.problem import error_metrics, make_problem, reference
from moelab.shapes import MoEShape

SHAPE = MoEShape("t", d=128, f=256, E=8, k=2)
CASES = [(1, "uniform"), (7, "uniform"), (33, "zipf"), (64, "one_hot")]


def args(p):
    return [jnp.asarray(p[n]) for n in ("x", "w_gate", "w_up", "w_down", "idx", "wts")]


@pytest.mark.parametrize("T,routing", CASES)
def test_ragged_dot(T, routing):
    p = make_problem(SHAPE, T, routing)
    e = error_metrics(np.asarray(jax_moe.moe_ragged(*args(p))), reference(p))
    assert e["rel_fro"] < 1e-4, e


@pytest.mark.parametrize("T,routing", CASES)
def test_pallas_gmm(T, routing):
    p = make_problem(SHAPE, T, routing)
    e = error_metrics(np.asarray(jax_moe.moe_pallas(*args(p), BM=8)), reference(p))
    assert e["rel_fro"] < 1e-4, e
