"""Round 9: per-layer expert-output error on REAL activations, stock mlx-lm vs our kernel, both against the fp32
truth on the same inputs and routing. Inputs are captured while decoding model-native text one token at a time.

    .venv-latest/bin/python scripts/r9_layer_error.py MODEL TEXT.txt [N_STEPS]
"""
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from moelab import e2e
from moelab.mlx_backend import load_variant

model_id, text_path = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 64
model, tok = load(model_id)
var, truth = load_variant("moelab/variants/mlx/metal_q4_moe.py"), load_variant("moelab/variants/fp32_truth_q4.py")
gemma = model.model_type == "gemma4"
cls, _ = e2e.patch(model, var)
cls._use_variant = False
caught = []
stock = cls._stock_call
if gemma:
    def grab(self, x, inds, w):
        caught.append((self, x, inds, w)); return stock(self, x, inds, w)
else:   # Qwen3-MoE / Qwen3.5-MoE blocks: routing recomputed exactly as the stock block does
    def grab(self, x):
        g = mx.softmax(self.gate(x), axis=-1, precise=True)
        i = mx.argpartition(g, kth=-self.top_k, axis=-1)[..., -self.top_k:]
        w = mx.take_along_axis(g, i, axis=-1)
        if self.norm_topk_prob:
            w = w / w.sum(axis=-1, keepdims=True)
        caught.append((self, x, i, w)); return stock(self, x)
cls.__call__ = grab
ids = tok.encode(open(text_path).read())[:steps + 1]
cache = make_prompt_cache(model)
for i in range(steps):
    mx.eval(model(mx.array([[ids[i]]]), cache=cache))
es, ev = [], []
for blk, x, inds, w in caught:
    T, k = 1, inds.shape[-1]
    args = (blk._variant_state, x.reshape(T, -1), inds.reshape(T, k), w.reshape(T, k).astype(mx.float32))
    ref = truth.run(*args, act=int(gemma)).astype(mx.float32)
    if gemma:
        ys = stock(blk, x, inds, w)
    else:   # stock routed-expert part only (shared expert excluded on both sides)
        ys = (blk.switch_mlp(x, inds) * w[..., None]).sum(axis=-2)
    ys = ys.reshape(T, -1).astype(mx.float32)
    yv = var.run(*args, act=int(gemma)).astype(mx.float32)
    n = mx.linalg.norm(ref)
    es.append((mx.linalg.norm(ys - ref) / n).item()); ev.append((mx.linalg.norm(yv - ref) / n).item())
es, ev = np.array(es), np.array(ev)
print(f"{len(es)} layer calls: rel error vs fp32 - stock median {np.median(es):.2e} mean {es.mean():.2e} | "
      f"ours median {np.median(ev):.2e} mean {ev.mean():.2e} | ours closer in {100 * (ev < es).mean():.0f}% of calls")
