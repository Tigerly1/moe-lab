"""Round 13: how much single-stream decode time do the small ops of a Qwen3-MoE layer cost?

Synthetic model with the real layer dimensions of Qwen3-30B-A3B (hidden 2048, 32 heads, 4 kv heads,
experts 2048x768, top-8) but 6 layers, 16 experts and random weights: ~320 MB, no download.
Baseline = mlx-lm with the expert kernels of mlx-lm#2044. Each ablation removes or fuses small ops.
Ablations marked (bound) change the output; they only show what a fused kernel could save at most.

    PYTHONPATH=<mlx-lm checkout with switch_kernels>:. python scripts/r13_layer_ablation.py out.json
"""

import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_moe
from mlx_lm.models.cache import make_prompt_cache

from moelab.router import SRC, topk_norm

_w = "wts[t * K + lane] = metal::precise::exp(vr - off) / s;"
assert _w in SRC
router_kernel = mx.fast.metal_kernel(
    name="r13_router_typed", input_names=["logits"], output_names=["idx", "wts"],
    source=SRC.replace(_w, "wts[t * K + lane] = static_cast<WT>(metal::precise::exp(vr - off) / s);"),
)

L, E, K, STEPS, ROUNDS = 6, 16, 8, 200, 7
Block, Layer = qwen3_moe.Qwen3MoeSparseMoeBlock, qwen3_moe.Qwen3MoeDecoderLayer
stock_block, stock_layer = Block.__call__, Layer.__call__


def build():
    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=2048, num_hidden_layers=L, intermediate_size=6144,
        num_attention_heads=32, num_experts=E, num_experts_per_tok=K, decoder_sparse_step=1,
        mlp_only_layers=[], moe_intermediate_size=768, rms_norm_eps=1e-6, vocab_size=1024,
        num_key_value_heads=4, head_dim=128, rope_theta=1e6, tie_word_embeddings=False,
        max_position_embeddings=4096, norm_topk_prob=True,
    )
    model = qwen3_moe.Model(args)
    for layer in model.layers:  # one layer at a time, so the fp32 init weights are freed at once
        nn.quantize(layer, group_size=64, bits=4)
        layer.set_dtype(mx.bfloat16)
        mx.eval(layer.parameters())
    nn.quantize(model, group_size=64, bits=4)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def block_router(tail):
    def call(self, x):
        gates = self.gate(x)
        if tail == "fused":  # exact up to tie order: softmax over the top-k logits
            inds, scores = topk_norm(gates, K)
            inds, scores = inds.reshape(*x.shape[:-1], K), scores.reshape(*x.shape[:-1], K).astype(x.dtype)
        elif tail == "fused_typed":  # same kernel, writing uint32 ids and weights in the model dtype: no casts after it
            T = gates.size // E
            inds, scores = router_kernel(
                inputs=[gates.reshape(T, E)], template=[("E", E), ("K", K), ("NORM", True), ("WT", x.dtype)], grid=(32, T, 1),
                threadgroup=(32, 1, 1), output_shapes=[(*x.shape[:-1], K)] * 2, output_dtypes=[mx.uint32, x.dtype])
        else:  # (bound) no softmax / argpartition / take / sum / divide
            inds = mx.broadcast_to(mx.arange(K, dtype=mx.uint32), (*x.shape[:-1], K))
            scores = gates[..., :K]
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

    return call


def layer_no_adds(self, x, mask=None, cache=None):  # (bound) drops both residual adds
    r = self.self_attn(self.input_layernorm(x), mask, cache)
    return self.mlp(self.post_attention_layernorm(r))


class Identity(nn.Module):
    def __call__(self, x):
        return x


def configure(model, name):
    Block.__call__, Layer.__call__ = stock_block, stock_layer
    for layer in model.layers:
        a = layer.self_attn
        if not hasattr(a, "_qk"):
            a._qk = (a.q_norm, a.k_norm)
        a.q_norm, a.k_norm = a._qk
    if name in ("router_fused", "all"):
        Block.__call__ = block_router("fused")
    if name == "router_fused_typed":
        Block.__call__ = block_router("fused_typed")
    if name == "router_tail_removed":
        Block.__call__ = block_router("none")
    if name in ("adds_removed", "all"):
        Layer.__call__ = layer_no_adds
    if name in ("qk_norm_removed", "all"):
        for layer in model.layers:
            layer.self_attn.q_norm = layer.self_attn.k_norm = Identity()


def ms_per_token(model):
    cache = make_prompt_cache(model)
    tok = mx.array([[1]])
    mx.eval(model(mx.ones((1, 16), mx.int32), cache=cache), [c.state for c in cache])
    t = time.perf_counter()
    for _ in range(STEPS):
        mx.eval(model(tok, cache=cache), [c.state for c in cache])
    return (time.perf_counter() - t) / STEPS * 1e3


def main():
    model = build()
    names = ["base", "router_fused", "router_fused_typed", "router_tail_removed", "adds_removed", "qk_norm_removed", "all"]
    best = {n: float("inf") for n in names}
    for n in names:  # compile kernels
        configure(model, n)
        ms_per_token(model)
    for _ in range(ROUNDS):
        for n in names:
            configure(model, n)
            best[n] = min(best[n], ms_per_token(model))
    out = {"layers": L, "arch": mx.device_info().get("architecture"), "peak_gb": round(mx.get_peak_memory() / 2**30, 2), "variants": {}}
    for n in names:
        out["variants"][n] = {"us_per_layer": round(best[n] / L * 1e3, 1), "saved_us_per_layer": round((best["base"] - best[n]) / L * 1e3, 1)}
        print(f"{n:22s} {best[n]:.3f} ms/token  {out['variants'][n]['us_per_layer']:6.1f} us/layer  saved {out['variants'][n]['saved_us_per_layer']:5.1f}")
    print("peak GB", out["peak_gb"])
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
