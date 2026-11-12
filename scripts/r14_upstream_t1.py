"""Round 14: single-token expert path of mlx-lm#2044 (ours) against the upstream PRs #2053 and #2054.

Same synthetic model as scripts/r13_layer_ablation.py: real Qwen3-30B-A3B layer dimensions, 6 layers,
16 experts, random weights (~320 MB, no download). T = 1 decode only.

  #2053: SwitchGLU.combine = gate+up+GLU+weight kernel, then down+weighted-sum kernel (one token)
  #2054: softmax_topk = softmax + top-k + renormalize in one kernel

The two upstream kernel files are loaded from a directory (not vendored here):
    PYTHONPATH=<mlx-lm checkout with switch_kernels> python scripts/r14_upstream_t1.py <dir> out.json
<dir> holds up_switch_glu.py (kernels/switch_glu.py of #2053) and up_softmax_topk.py (kernels/softmax_topk.py of #2054).
"""

import importlib.util
import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_moe, switch_layers
from mlx_lm.models.cache import make_prompt_cache


def load(name):
    spec = importlib.util.spec_from_file_location(name, f"{sys.argv[1]}/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


up_glu, up_router = load("up_switch_glu"), load("up_softmax_topk")
ours_hook = switch_layers.switch_kernels.switch_glu

L, E, K, STEPS, ROUNDS = 6, 16, 8, 200, 7
Block = qwen3_moe.Qwen3MoeSparseMoeBlock


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


def block(router, experts):
    """router: 'stock' | 'fused' (#2054).  experts: 'stock' | 'ours' (#2044) | 'combine' (#2053)."""

    def call(self, x):
        gates = self.gate(x)
        if router == "fused":
            inds, scores = up_router.softmax_topk(gates, top_k=K, normalize=True)
        else:
            gates = mx.softmax(gates, axis=-1, precise=True)
            inds = mx.stop_gradient(mx.argpartition(gates, kth=-K, axis=-1)[..., -K:])
            scores = mx.take_along_axis(gates, inds, axis=-1)
            scores /= mx.sum(scores, axis=-1, keepdims=True)
        if experts == "combine":
            assert up_glu.can_combine(self.switch_mlp, x)
            return up_glu.switch_glu_combine(self.switch_mlp, x, inds, scores).astype(x.dtype)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

    return call


VARIANTS = {  # name: (router, experts)
    "main": ("stock", "stock"),
    "pr2044": ("stock", "ours"),
    "pr2053": ("stock", "combine"),
    "pr2054": ("fused", "stock"),
    "pr2044+2054": ("fused", "ours"),
    "pr2053+2054": ("fused", "combine"),
}


def configure(name):
    router, experts = VARIANTS[name]
    switch_layers.switch_kernels.switch_glu = ours_hook if experts == "ours" else (lambda *a: None)
    Block.__call__ = block(router, experts)


def run(model, steps):
    cache = make_prompt_cache(model)
    tok = mx.array([[1]])
    mx.eval(model(mx.ones((1, 16), mx.int32), cache=cache), [c.state for c in cache])
    t = time.perf_counter()
    for _ in range(steps):
        out = model(tok, cache=cache)
        mx.eval(out, [c.state for c in cache])
    return (time.perf_counter() - t) / steps * 1e3, out.astype(mx.float32)


def main():
    model = build()
    best = {n: float("inf") for n in VARIANTS}
    logits = {}
    for n in VARIANTS:  # compile kernels; logits after the same 16 + 4 tokens in every arm
        configure(n)
        logits[n] = run(model, 4)[1]
    ref = logits["main"]
    err = {n: (mx.abs(v - ref).max() / mx.abs(ref).max()).item() for n, v in logits.items()}
    for _ in range(ROUNDS):
        for n in VARIANTS:
            configure(n)
            best[n] = min(best[n], run(model, STEPS)[0])
    out = {"layers": L, "arch": mx.device_info().get("architecture"), "peak_gb": round(mx.get_peak_memory() / 2**30, 2), "variants": {}}
    for n in VARIANTS:
        out["variants"][n] = {
            "us_per_layer": round(best[n] / L * 1e3, 1),
            "saved_us_per_layer": round((best["main"] - best[n]) / L * 1e3, 1),
            "speedup": round(best["main"] / best[n], 3),
            "logit_err_vs_main": round(err[n], 5),
        }
        v = out["variants"][n]
        print(f"{n:13s} {v['us_per_layer']:6.1f} us/layer  saved {v['saved_us_per_layer']:5.1f}  {v['speedup']:.3f}x  err {v['logit_err_vs_main']:.1e}")
    print("peak GB", out["peak_gb"])
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
