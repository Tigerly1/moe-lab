"""Round 9: one batched decode step (B sequences, T=B tokens through every MoE layer) vs the fp32-expert truth on
the same dequantized weights. Reports per-sequence relative logit error and argmax agreement for stock and ours.
Prefill is stock for all arms, so every arm decodes from the same cache.

    .venv-latest/bin/python scripts/r9_batch_truth.py MODEL TEXT.txt OUT.json
"""
import json
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from moelab import e2e
from moelab.mlx_backend import load_variant

model_id, text_path, out = sys.argv[1:4]
model, tok = load(model_id)
ours, truth = load_variant("moelab/variants/mlx/metal_q4_moe.py"), load_variant("moelab/variants/fp32_truth_q4.py")
ids = tok.encode(open(text_path).read())
res = {}
for B in (1, 8, 32):
    step_len = max(1, (len(ids) - 65) // B)
    P = mx.array([ids[i * step_len:i * step_len + 64] for i in range(B)])
    nxt = mx.array([[ids[i * step_len + 64]] for i in range(B)])

    def step(mod, use):
        cls, _ = e2e.patch(model, mod)
        cls._use_variant = False
        c = make_prompt_cache(model)
        model(P, cache=c)
        cls._use_variant = use
        return model(nxt, cache=c)[:, -1].astype(mx.float32)

    st, ou, tr = step(ours, False), step(ours, True), step(truth, True)

    def err(x):
        r = np.array(mx.linalg.norm(x - tr, axis=-1) / mx.linalg.norm(tr, axis=-1))
        return {"median_rel": float(np.median(r)), "max_rel": float(r.max()),
                "argmax_agree": float((x.argmax(-1) == tr.argmax(-1)).astype(mx.float32).mean().item())}

    res[B] = {"stock": err(st), "ours": err(ou)}
    print(B, res[B], flush=True)
json.dump(res, open(out, "w"), indent=1)
