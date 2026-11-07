"""Round 9: the universal SwitchGLU patch (moelab/universal.py) on tiny random quantized models of six families
with different routers (softmax top-k, sigmoid + expert bias, sigmoid + correction bias + scaling, GeGLU experts,
gated shared expert, shared experts).

1. glue: with gather_moe (bit-identical to SwitchGLU) as the expert kernel, logits must equal stock exactly;
2. kernels: every SwitchGLU call through our kernels matches the stock SwitchGLU output (per-slot, same inputs)
   within the bf16 kernel tolerance; the patch must reach every MoE layer."""
import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.switch_layers import SwitchGLU

import moelab.e2e as e2e
from moelab import mlx_backend as be
from moelab import universal
from moelab.fallback import gather_moe

Q4 = "moelab/variants/mlx/metal_q4_moe.py"
COMMON = dict(vocab_size=1000, hidden_size=256, intermediate_size=512, num_attention_heads=4, num_key_value_heads=2)


def _finish(m):
    mx.random.seed(1)
    m.set_dtype(mx.bfloat16)
    pred = getattr(m, "quant_predicate", None)
    nn.quantize(m, group_size=64, bits=4,
                class_predicate=(lambda p, mod: hasattr(mod, "to_quantized") and pred(p, mod) is not False) if pred else None)
    mx.eval(m.parameters())
    return m


def qwen3_moe():
    from mlx_lm.models import qwen3_moe as q
    return q.Model(q.ModelArgs(model_type="qwen3_moe", num_hidden_layers=2, num_experts=16, num_experts_per_tok=4,
                               decoder_sparse_step=1, mlp_only_layers=[], moe_intermediate_size=128, rms_norm_eps=1e-6,
                               head_dim=64, rope_theta=1e6, tie_word_embeddings=False, max_position_embeddings=4096,
                               norm_topk_prob=True, **COMMON))


def qwen3_5_moe():
    q = pytest.importorskip("mlx_lm.models.qwen3_5_moe")
    return q.Model(q.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": dict(
        model_type="qwen3_5_moe_text", num_hidden_layers=4, head_dim=64, linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32, num_experts=16,
        num_experts_per_tok=4, moe_intermediate_size=128, shared_expert_intermediate_size=128, **COMMON)}))


def gemma4():
    g = pytest.importorskip("mlx_lm.models.gemma4_text")
    return g.Model(g.ModelArgs(hidden_size=256, num_hidden_layers=3, intermediate_size=512, num_attention_heads=4,
                               head_dim=64, global_head_dim=64, vocab_size=1000, vocab_size_per_layer_input=1000,
                               num_key_value_heads=1, num_kv_shared_layers=0, hidden_size_per_layer_input=0,
                               sliding_window=64, enable_moe_block=True, num_experts=16, top_k_experts=4,
                               moe_intermediate_size=192))


def lfm2_moe():
    l = pytest.importorskip("mlx_lm.models.lfm2_moe")
    m = l.Model(l.ModelArgs(model_type="lfm2_moe", moe_intermediate_size=128, num_hidden_layers=3, num_experts=16,
                            num_experts_per_tok=4, norm_topk_prob=True, max_position_embeddings=4096,
                            use_expert_bias=True, num_dense_layers=1, norm_eps=1e-5, conv_bias=False, conv_L_cache=3,
                            routed_scaling_factor=2.0, full_attn_idxs=[1], **COMMON))
    for layer in m.model.layers:
        b = getattr(layer.feed_forward, "expert_bias", None) if hasattr(layer, "feed_forward") else None
        if b is not None:
            layer.feed_forward.expert_bias = mx.random.normal(b.shape) * 0.1
    return m


def glm4_moe_lite():
    g = pytest.importorskip("mlx_lm.models.glm4_moe_lite")
    return g.Model(g.ModelArgs(vocab_size=1000, hidden_size=256, intermediate_size=512, moe_intermediate_size=128,
                               num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4, n_routed_experts=16,
                               kv_lora_rank=64, q_lora_rank=64, qk_rope_head_dim=16, qk_nope_head_dim=64,
                               v_head_dim=64, num_experts_per_tok=4, max_position_embeddings=4096))


def deepseek_v2():
    from mlx_lm.models import deepseek_v2 as d
    return d.Model(d.ModelArgs(
        vocab_size=1000, hidden_size=256, intermediate_size=512, moe_intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=4, n_shared_experts=2, n_routed_experts=16,
        routed_scaling_factor=1.0, kv_lora_rank=64, q_lora_rank=None, qk_rope_head_dim=16, v_head_dim=32,
        qk_nope_head_dim=32, topk_method="gready", num_experts_per_tok=6, first_k_dense_replace=1,
        rope_scaling={"type": "yarn", "factor": 1.0, "original_max_position_embeddings": 2048, "beta_fast": 32,
                      "beta_slow": 1, "mscale": 0.707, "mscale_all_dim": 0.707}))


FAMILIES = [qwen3_moe, qwen3_5_moe, gemma4, lfm2_moe, glm4_moe_lite, deepseek_v2]
CHUNKS = ([3], [4, 5], [6, 7, 8], list(range(9, 17)))


def build(make):
    mx.random.seed(0)
    return _finish(make())


def n_switch(m):
    return sum(isinstance(x, SwitchGLU) for _, x in m.named_modules())


def logits(m, use):
    SwitchGLU._use_variant = use
    cache = make_prompt_cache(m)
    m(mx.array([[1, 5, 9, 42, 7, 3, 2, 8, 11, 13, 17, 19, 23, 29, 31, 37, 41]]), cache=cache)   # 17 > MAX_T: stock
    out = mx.concatenate([m(mx.array([c]), cache=cache) for c in CHUNKS], axis=1).astype(mx.float32)
    SwitchGLU._use_variant = False
    return out


@pytest.fixture(autouse=True)
def restore(monkeypatch):
    monkeypatch.setattr(universal, "MAX_T", 16)
    yield
    if "_stock_call" in SwitchGLU.__dict__:
        SwitchGLU.__call__ = SwitchGLU._stock_call
    SwitchGLU._use_variant = False


@pytest.mark.parametrize("make", FAMILIES, ids=lambda f: f.__name__)
def test_glue_exact(make, monkeypatch):
    m = build(make)
    mod = be.load_variant(Q4)
    calls = []

    def exact(s, x, idx, wts, act=0):
        calls.append(act)
        return gather_moe(s, x, idx, wts, 64, 4, act=act)

    def exact_slots(s, x, idx, act=0):   # T=1 path: per-slot outputs [k, D]
        rows = idx.size
        xr = mx.broadcast_to(x, (rows, x.shape[-1]))
        return exact(s, xr, idx.reshape(rows, 1), mx.ones((rows, 1), mx.float32), act=act)

    monkeypatch.setattr(mod, "run", exact)
    monkeypatch.setattr(mod, "run_slots", exact_slots)
    cls, n = universal.patch(m, mod)
    assert n == n_switch(m) > 0
    a = logits(m, False)
    assert not calls
    b = logits(m, True)
    assert len(calls) == n * len(CHUNKS)
    assert set(calls) == {1 if make is gemma4 else 0}
    assert mx.array_equal(a, b).item(), (mx.abs(a - b).max().item())


@pytest.mark.parametrize("make", FAMILIES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("T", [1, 3, 8])
def test_kernel_matches_switchglu(make, T):
    m = build(make)
    mod = be.load_variant(Q4)
    universal.patch(m, mod)
    sw = next(x for _, x in m.named_modules() if isinstance(x, SwitchGLU))
    E = sw.gate_proj.weight.shape[0]
    k = 4
    x = (mx.random.normal((1, T, sw.gate_proj.weight.shape[-1] * 8)) * 0.5).astype(mx.bfloat16)
    idx = mx.argsort(mx.random.uniform(shape=(1, T, E)), axis=-1)[..., :k].astype(mx.uint32)
    ref = SwitchGLU._stock_call(sw, x, idx).astype(mx.float32)
    SwitchGLU._use_variant = True
    y = sw(x, idx).astype(mx.float32)
    assert y.shape == ref.shape
    rel = (mx.linalg.norm((y - ref).flatten()) / mx.linalg.norm(ref.flatten())).item()
    assert rel < 2e-2, rel


@pytest.mark.parametrize("make", [lfm2_moe, glm4_moe_lite], ids=lambda f: f.__name__)
def test_pick_variant_falls_back_to_universal(make, monkeypatch):
    from moelab.generate import pick_variant
    monkeypatch.setattr(e2e, "UNIVERSAL", False)
    m = build(make)
    assert pick_variant(m).endswith("metal_q4_moe.py") and e2e.UNIVERSAL
    cls, n = e2e.patch(m, be.load_variant(Q4))
    assert cls is SwitchGLU and n == n_switch(m)
