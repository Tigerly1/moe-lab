"""The gpt-oss patch (metal_mxfp4_gptoss: biases + clamped SwiGLU) must give the same logits as the
stock mlx-lm gpt_oss MLPBlock, for T=1 decode and multi-token chunks (T = 2, 3, 8)."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import gpt_oss
from mlx_lm.models.cache import make_prompt_cache

import moelab.e2e as e2e
from moelab.mlx_backend import load_variant

VARIANT = "moelab/variants/mlx/metal_mxfp4_gptoss.py"


def tiny_model(dtype):
    args = gpt_oss.ModelArgs(
        model_type="gpt_oss", num_hidden_layers=2, num_local_experts=16, num_experts_per_tok=4,
        vocab_size=1000, hidden_size=256, intermediate_size=192, head_dim=64, num_attention_heads=4,
        num_key_value_heads=2, sliding_window=8, layer_types=["sliding_attention", "full_attention"])
    mx.random.seed(0)
    m = gpt_oss.Model(args)
    for l in m.layers:  # non-zero biases; gate/up large enough that the +-7 clamps are hit
        e = l.mlp.experts
        e.gate_proj.bias = 4 * mx.random.normal(e.gate_proj.bias.shape)
        e.up_proj.bias = 4 * mx.random.normal(e.up_proj.bias.shape)
        e.down_proj.bias = 0.1 * mx.random.normal(e.down_proj.bias.shape)
        l.mlp.router.bias = mx.random.normal(l.mlp.router.bias.shape)
    m.set_dtype(dtype)
    # as in the mlx-community MXFP4-Q8 checkpoint: experts mxfp4 (gs 32), everything else affine 8-bit gs 64
    nn.quantize(m, group_size=32, bits=4, mode="mxfp4", class_predicate=lambda p, _: p.endswith("experts.gate_proj")
                or p.endswith("experts.up_proj") or p.endswith("experts.down_proj"))
    nn.quantize(m, group_size=64, bits=8, class_predicate=lambda p, mod: hasattr(mod, "to_quantized") and "experts" not in p)
    mx.eval(m.parameters())
    return m


def logits(m, cls, use_variant, prompt, chunks):
    cls._use_variant = use_variant
    cache = make_prompt_cache(m)
    m(mx.array([prompt]), cache=cache)                    # prefill > MAX_T: stock path
    return mx.concatenate([m(mx.array([c]), cache=cache) for c in chunks], axis=1).astype(mx.float32)


def check(a, b):
    rel = (mx.linalg.norm((a - b).flatten()) / mx.linalg.norm(a.flatten())).item()
    assert 0 < rel < 1e-2, rel   # rel == 0 would mean the variant path never ran
    assert (a.argmax(-1) == b.argmax(-1)).mean().item() > 0.95


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gptoss_quant_format(dtype):
    sw = tiny_model(dtype).layers[0].mlp.experts
    assert (sw.gate_proj.mode, sw.gate_proj.group_size, sw.gate_proj.bits) == ("mxfp4", 32, 4)
    assert (sw.gate_proj.bias > 7).any().item() and (mx.abs(sw.up_proj.bias) > 7).any().item()  # clamps reachable


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gptoss_decode_matches_stock(dtype):
    m = tiny_model(dtype)
    cls, n = e2e.patch(m, load_variant(VARIANT))
    assert n == 2
    prompt, chunks = [1, 5, 9, 42, 7, 3], [[11], [12], [13], [14]]   # T = 1 decode (MAX_T = 4)
    check(logits(m, cls, False, prompt, chunks), logits(m, cls, True, prompt, chunks))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gptoss_multitoken_matches_stock(dtype, monkeypatch):
    monkeypatch.setattr(e2e, "MAX_T", 16)
    m = tiny_model(dtype)
    cls, _ = e2e.patch(m, load_variant(VARIANT))
    prompt = [1, 5, 9, 42, 7, 3, 2, 8, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    chunks = [[3, 4], [5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], list(range(100, 116)),   # T = 2, 3, 8, 16
              [43, 44], [45, 46, 47], list(range(50, 58))]
    a, b = logits(m, cls, False, prompt, chunks), logits(m, cls, True, prompt, chunks)
    check(a, b)
    # fp32 truth (same dequantized weights): the variant must be at least as close to it as stock
    # (random tiny model -> near-tied logits; a stock bf16 argmax flip at a tie is noise, not a bug)
    mod = load_variant(VARIANT)
    e2e.patch(m, SimpleNamespace(state=mod.state, run=mod.reference))
    c = logits(m, cls, True, prompt, chunks)
    err = lambda p: (mx.linalg.norm((p - c).flatten()) / mx.linalg.norm(c.flatten())).item()  # noqa: E731
    assert err(b) <= err(a), (err(b), err(a))


@pytest.mark.parametrize("T", [1, 3, 16])
def test_gptoss_kernel_large_scales(T):
    """E8M0 scale bytes up to ~136 occur in the real gpt-oss down_proj (scale up to 2^9); every regime
    (incl. the half-precision MMA path at T >= 8) must stay finite and match the fp32 reference."""
    mod = load_variant(VARIANT)
    mx.random.seed(1)
    E, f, d, k = 8, 128, 256, 4
    sw = gpt_oss.SwitchGLU(d, f, E, activation=gpt_oss.SwiGLU(), bias=True)
    sw.gate_proj.weight = 0.05 * mx.random.normal((E, f, d))
    sw.up_proj.weight = 0.05 * mx.random.normal((E, f, d))
    sw.down_proj.weight = 300 * mx.random.normal((E, d, f))      # group max ~1000 -> scale bytes ~135
    for p in (sw.gate_proj, sw.up_proj, sw.down_proj):
        p.bias = mx.random.normal(p.bias.shape)
    nn.quantize(sw, group_size=32, bits=4, mode="mxfp4")
    assert sw.down_proj.scales.max().item() >= 133
    s = mod.state(sw)
    x = mx.random.normal((T, d)).astype(mx.bfloat16)
    idx = mx.stack([mx.random.permutation(E)[:k] for _ in range(T)]).astype(mx.uint32)
    if T == 16:
        idx = idx % 2                                             # heavy repeats -> MMA groups of >= 3
    wts = mx.softmax(mx.random.normal((T, k)), axis=-1)
    y, r = mod.run(s, x, idx, wts).astype(mx.float32), mod.reference(s, x, idx, wts).astype(mx.float32)
    assert mx.isfinite(y).all().item()
    rel = (mx.linalg.norm((y - r).flatten()) / mx.linalg.norm(r.flatten())).item()
    assert rel < 1e-2, rel


@pytest.fixture(autouse=True, params=[False, True], ids=["stock-router", "fused-router"])
def _router(request, monkeypatch):
    """Round 7: every patch test also runs with the fused router tail (moelab/router.py)."""
    monkeypatch.setattr(e2e, "FUSED_ROUTER", request.param)
