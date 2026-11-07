"""Round 9 model families: Gemma-4 (GeGLU experts, own Router, fused router variant) and Qwen3.5/3.6-MoE (gated
shared expert), on tiny random quantized models.

Glue check: with the expert compute swapped for gather_moe (bit-identical to mlx-lm's SwitchGLU), the patched
forward must reproduce stock logits EXACTLY at T=1 and multi-token chunks; any routing, weighting, activation or
shared-expert mistake in the patch shows up as a non-zero difference. A logits tolerance vs stock is useless on
these tiny random models: even the fp32-truth experts differ from stock by 4% (Qwen3.5) and 14% (Gemma-4),
so the kernels' own accuracy is tested in test_geglu.py and on real models (scripts/r9_layer_error.py)."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

gemma4_text = pytest.importorskip("mlx_lm.models.gemma4_text")
qwen3_5_moe = pytest.importorskip("mlx_lm.models.qwen3_5_moe")
from mlx_lm.models.cache import make_prompt_cache

import moelab.e2e as e2e
from moelab.fallback import gather_moe
from moelab.generate import pick_variant
from moelab.mlx_backend import load_variant

Q4 = "moelab/variants/mlx/metal_q4_moe.py"


def _finish(m):
    m.set_dtype(mx.bfloat16)
    pred = getattr(m, "quant_predicate", None)
    nn.quantize(m, group_size=64, bits=4,
                class_predicate=(lambda p, mod: hasattr(mod, "to_quantized") and pred(p, mod) is not False) if pred else None)
    mx.eval(m.parameters())
    return m


def tiny_gemma4():
    mx.random.seed(0)
    a = gemma4_text.ModelArgs(hidden_size=256, num_hidden_layers=3, intermediate_size=512, num_attention_heads=4,
                              head_dim=64, global_head_dim=64, vocab_size=1000, vocab_size_per_layer_input=1000,
                              num_key_value_heads=1, num_kv_shared_layers=0, hidden_size_per_layer_input=0,
                              sliding_window=64, enable_moe_block=True, num_experts=16, top_k_experts=4,
                              moe_intermediate_size=192)
    m = gemma4_text.Model(a)
    for l in m.layers:   # non-trivial per-expert scales so a dropped scale would show
        l.router.per_expert_scale = mx.random.uniform(0.5, 1.5, (16,))
    return _finish(m)


def tiny_qwen35():
    mx.random.seed(0)
    a = qwen3_5_moe.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": dict(
        model_type="qwen3_5_moe_text", hidden_size=256, intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=1000, linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32, num_experts=16,
        num_experts_per_tok=4, moe_intermediate_size=128, shared_expert_intermediate_size=128)})
    return _finish(qwen3_5_moe.Model(a))


CHUNKS = ([3], [4, 5], [6, 7, 8], list(range(9, 17)))


def logits(m, cls, use):
    cls._use_variant = use
    cache = make_prompt_cache(m)
    m(mx.array([[1, 5, 9, 42, 7, 3, 2, 8, 11, 13, 17, 19, 23, 29, 31, 37, 41]]), cache=cache)   # 17 > MAX_T: stock
    return mx.concatenate([m(mx.array([c]), cache=cache) for c in CHUNKS], axis=1).astype(mx.float32)


def exact_fallback(*a, act=0):
    s, x, i, w = a
    return gather_moe(s, x, i, w, 64, 4, act=act)


@pytest.mark.parametrize("make", [tiny_gemma4, tiny_qwen35], ids=["gemma4", "qwen3_5_moe"])
def test_patch_glue_is_exact(make, monkeypatch):
    monkeypatch.setattr(e2e, "MAX_T", 16)
    m = make()
    assert pick_variant(m).endswith("metal_q4_moe.py")
    mod = load_variant(Q4)
    runs = []
    monkeypatch.setattr(mod, "run", lambda *a, **k: (runs.append(k.get("act", 0)), exact_fallback(*a, **k))[1])
    cls, n = e2e.patch(m, mod)
    assert n == len(m.layers)
    a = logits(m, cls, False)
    assert not runs
    b = logits(m, cls, True)
    assert mx.array_equal(a, b).item()
    assert len(runs) == n * len(CHUNKS)
    assert set(runs) == {1 if make is tiny_gemma4 else 0}   # Gemma-4 experts run with GeGLU


@pytest.mark.parametrize("make", [tiny_gemma4, tiny_qwen35], ids=["gemma4", "qwen3_5_moe"])
@pytest.mark.parametrize("fused", [False, True])
def test_kernels_and_fused_router_run(make, fused, monkeypatch):
    monkeypatch.setattr(e2e, "MAX_T", 16)
    monkeypatch.setattr(e2e, "FUSED_ROUTER", fused)
    m = make()
    routed, real = [], e2e.topk_norm
    monkeypatch.setattr(e2e, "topk_norm", lambda *a, **k: (routed.append(1), real(*a, **k))[1])
    cls, n = e2e.patch(m, load_variant(Q4))
    a, b = logits(m, cls, False), logits(m, cls, True)
    assert mx.all(mx.isfinite(b)).item() and not mx.array_equal(a, b).item()
    assert len(routed) == (n * len(CHUNKS) if fused else 0)

