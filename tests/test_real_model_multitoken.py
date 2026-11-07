"""Opt-in (MOE_SLOW=1): full real models, multi-token (verify-sized) chunks through the patched MoE
blocks must match stock logits (audit 2: the T>1 path behind the verify-bench numbers had no output
check on real models). Loads 12-16 GB per model; skipped unless the model is cached locally."""
import os

import mlx.core as mx
import numpy as np
import pytest

import moelab.e2e as e2e

SLOW = os.environ.get("MOE_SLOW") == "1"
CASES = [("mlx-community/Qwen3-30B-A3B-4bit", "moelab/variants/mlx/metal_q4_moe.py"),
         ("mlx-community/gpt-oss-20b-MXFP4-Q8", "moelab/variants/mlx/metal_mxfp4_gptoss.py"),
         ("mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx", "moelab/variants/mlx/metal_q4_moe.py")]


def _cached(repo):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not SLOW, reason="set MOE_SLOW=1 to run full-model tests")
@pytest.mark.parametrize("repo,variant", CASES)
def test_real_model_chunks_match_stock(repo, variant, monkeypatch):
    if not _cached(repo):
        pytest.skip(f"{repo} not cached")
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    import moelab.e2e as e2e
    from moelab.mlx_backend import load_variant
    monkeypatch.setattr(e2e, "MAX_T", 16)
    model, tok = load(repo)
    cls, _ = e2e.patch(model, load_variant(variant))
    ids = tok.encode(open(os.path.join(os.path.dirname(e2e.__file__), "data/quality_corpus.txt")).read())[:32 + 4 + 8 + 16]
    prompt, rest = ids[:32], ids[32:]
    chunks = [rest[:4], rest[4:12], rest[12:28]]                  # T = 4, 8, 16

    def run(use):
        cls._use_variant = use
        cache = make_prompt_cache(model)
        model(mx.array([prompt]), cache=cache)
        return [model(mx.array([c]), cache=cache)[0].astype(mx.float32) for c in chunks]

    a, b = run(False), run(True)
    agree = []
    for la, lb in zip(a, b):
        assert bool(mx.isfinite(lb).all().item())
        rel = (mx.linalg.norm((la - lb).flatten()) / mx.linalg.norm(la.flatten())).item()
        assert rel < 0.08, rel                                # bf16 model noise floor ~0.01-0.05
        agree += list(np.array(la.argmax(-1)) == np.array(lb.argmax(-1)))
    # top-1 agreement over all 28 positions (a 4-position chunk alone is too few: one near-tie flip = 75%)
    assert np.mean(agree) >= 0.85, np.mean(agree)


@pytest.fixture(autouse=True, params=[False, True], ids=["stock-router", "fused-router"])
def _router(request, monkeypatch):
    """Round 7: every patch test also runs with the fused router tail (moelab/router.py)."""
    monkeypatch.setattr(e2e, "FUSED_ROUTER", request.param)
