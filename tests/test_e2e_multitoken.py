"""Multi-token decode path (speculative verify, T=2..8) through the patched mlx-lm model
must match stock logits, like the T=1 test in test_e2e_patch.py."""
import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache

import moelab.e2e as e2e
from moelab.mlx_backend import load_variant
from test_e2e_patch import tiny_model  # pytest puts tests/ on sys.path (prepend import mode)

VARIANTS = ["moelab/variants/mlx/metal_q4_tiles_mma.py", "moelab/variants/mlx/metal_q4_tiles_b.py", "moelab/variants/mlx/metal_q4_moe.py"]


def logits(m, cls, use_variant, chunks):
    cls._use_variant = use_variant
    cache = make_prompt_cache(m)
    m(mx.array([[1, 5, 9, 42, 7, 3, 2, 8, 11, 13, 17, 19, 23, 29, 31, 37, 41]]), cache=cache)  # prefill > MAX_T: stock
    outs = [m(mx.array([c]), cache=cache) for c in chunks]
    return mx.concatenate(outs, axis=1).astype(mx.float32)


@pytest.mark.parametrize("path", VARIANTS)
def test_multitoken_matches_stock(path, monkeypatch):
    monkeypatch.setattr(e2e, "MAX_T", 16)
    m = tiny_model()
    cls, _ = e2e.patch(m, load_variant(path))
    chunks = [[3, 4], [5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]   # T = 2, 3, 8
    a, b = logits(m, cls, False, chunks), logits(m, cls, True, chunks)
    rel = (mx.linalg.norm((a - b).flatten()) / mx.linalg.norm(a.flatten())).item()
    assert rel < 1e-2, rel
    assert (a.argmax(-1) == b.argmax(-1)).mean().item() > 0.95


@pytest.fixture(autouse=True, params=[False, True], ids=["stock-router", "fused-router"])
def _router(request, monkeypatch):
    """Round 7: every patch test also runs with the fused router tail (moelab/router.py)."""
    monkeypatch.setattr(e2e, "FUSED_ROUTER", request.param)
