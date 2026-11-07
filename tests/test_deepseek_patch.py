"""DeepSeek-V2 path of e2e.patch (stock gate incl. routed scaling, shared experts stock, routed experts
via our int4 kernels) must match stock logits on a tiny random model, at T=1 and multi-token chunks."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models import deepseek_v2
from mlx_lm.models.cache import make_prompt_cache

import moelab.e2e as e2e
from moelab.mlx_backend import load_variant


def tiny(topk_method="gready"):
    args = deepseek_v2.ModelArgs(
        vocab_size=1000, hidden_size=256, intermediate_size=512, moe_intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4, n_shared_experts=2,
        n_routed_experts=16, routed_scaling_factor=1.0, kv_lora_rank=64, q_lora_rank=None,
        qk_rope_head_dim=16, v_head_dim=32, qk_nope_head_dim=32, topk_method=topk_method,
        n_group=4 if topk_method != "gready" else None, topk_group=2 if topk_method != "gready" else None,
        num_experts_per_tok=6, first_k_dense_replace=1,
        rope_scaling={"type": "yarn", "factor": 1.0, "original_max_position_embeddings": 2048, "beta_fast": 32,
                      "beta_slow": 1, "mscale": 0.707, "mscale_all_dim": 0.707})
    mx.random.seed(0)
    m = deepseek_v2.Model(args)
    m.set_dtype(mx.bfloat16)
    nn.quantize(m, group_size=64, bits=4)
    mx.eval(m.parameters())
    return m


@pytest.mark.parametrize("topk_method", ["gready", "group_limited_greedy"])
def test_deepseek_patch_matches_stock(topk_method, monkeypatch):
    monkeypatch.setattr(e2e, "MAX_T", 16)
    m = tiny(topk_method)
    cls, n = e2e.patch(m, load_variant("moelab/variants/mlx/metal_q4_moe.py"))
    assert n == 2   # 3 layers, first one dense

    def run(use):
        cls._use_variant = use
        cache = make_prompt_cache(m)
        m(mx.array([[1, 5, 9, 42, 7, 3, 2, 8, 11, 13, 17, 19, 23, 29, 31, 37, 41]]), cache=cache)  # 17 > MAX_T: stock
        outs = [m(mx.array([c]), cache=cache) for c in ([3], [4, 5], [6, 7, 8], list(range(9, 17)))]
        return mx.concatenate(outs, axis=1).astype(mx.float32)

    a, b = run(False), run(True)
    rel = (mx.linalg.norm((a - b).flatten()) / mx.linalg.norm(a.flatten())).item()
    assert 0 < rel < 2e-2, rel
    assert float(np.mean(np.array(a.argmax(-1)) == np.array(b.argmax(-1)))) >= 0.85
