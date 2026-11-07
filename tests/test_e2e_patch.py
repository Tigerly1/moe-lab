"""The mlx-lm patch must give the same decode logits as the stock Qwen3-MoE block."""
import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import qwen3_moe
from mlx_lm.models.cache import make_prompt_cache

from moelab.e2e import patch
from moelab.mlx_backend import load_variant

VARIANTS = ["moelab/variants/mlx/metal_q4_gemv_v31.py", "moelab/variants/mlx/metal_q4_gemv_v3.py", "moelab/variants/mlx/metal_q4_moe.py"]


def tiny_model():
    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=256, num_hidden_layers=2, intermediate_size=512,
        num_attention_heads=4, num_experts=16, num_experts_per_tok=4, decoder_sparse_step=1,
        mlp_only_layers=[], moe_intermediate_size=128, rms_norm_eps=1e-6, vocab_size=1000,
        num_key_value_heads=2, head_dim=64, rope_theta=1e6, tie_word_embeddings=False,
        max_position_embeddings=512, norm_topk_prob=True)
    mx.random.seed(0)
    m = qwen3_moe.Model(args)
    m.set_dtype(mx.float16)
    nn.quantize(m, group_size=64, bits=4)
    mx.eval(m.parameters())
    return m


def decode_logits(m, use_variant, cls):
    cls._use_variant = use_variant
    cache = make_prompt_cache(m)
    prompt = mx.array([[1, 5, 9, 42, 7, 3]])
    m(prompt, cache=cache)                       # prefill: stock path (T > MAX_T)
    out = [m(mx.array([[t]]), cache=cache) for t in (11, 12, 13)]   # decode: variant path
    return mx.concatenate(out, axis=1).astype(mx.float32)


@pytest.mark.parametrize("path", VARIANTS)
def test_patch_matches_stock(path):
    m = tiny_model()
    cls, n = patch(m, load_variant(path))
    assert n == 2
    a, b = decode_logits(m, False, cls), decode_logits(m, True, cls)
    rel = (mx.linalg.norm((a - b).flatten()) / mx.linalg.norm(a.flatten())).item()
    assert rel < 1e-2, rel
    assert (a.argmax(-1) == b.argmax(-1)).all().item()
