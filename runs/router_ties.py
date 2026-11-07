"""On real Qwen3 hidden states: how often does stock routing (softmax -> bf16 probs -> argpartition) pick a different
expert set than exact top-k on logits, and is every difference a tie after bf16 rounding of the probabilities?"""
import mlx.core as mx, numpy as np
from mlx_lm import load
from moelab.router import topk_norm
model, tok = load("mlx-community/Qwen3-30B-A3B-4bit")
ids = tok.encode(open("moelab/data/quality_corpus.txt").read())[:512]
caught = []
cls = type(model.layers[0].mlp); orig = cls.__call__
def spy(self, x):
    lg = self.gate(x)
    caught.append(lg.reshape(-1, lg.shape[-1]))
    return orig(self, x)
cls.__call__ = spy
model(mx.array([ids])); cls.__call__ = orig
diff = ties = n = 0
for lg in caught:
    g = mx.softmax(lg, axis=-1, precise=True)                  # bf16 output, as stock
    i0 = np.array(mx.argpartition(g, kth=-8, axis=-1)[..., -8:])
    i1 = np.array(topk_norm(lg, 8)[0])
    G = np.array(g.astype(mx.float32))
    for t in range(lg.shape[0]):
        n += 1
        a, b = set(i0[t].tolist()), set(i1[t].tolist())
        if a != b:
            diff += 1
            ties += np.allclose(sorted(G[t, list(a - b)]), sorted(G[t, list(b - a)]), rtol=0, atol=0)
import json; json.dump({"token_layers": n, "differ": diff, "differ_are_bf16_ties": int(ties), "tokens": len(ids)}, open("runs/router_ties.json", "w"), indent=1)
print(f"{n} token-layers: stock and exact top-8 differ in {diff} ({100*diff/n:.2f}%); of those, bf16-prob ties: {ties} ({100*ties/max(diff,1):.0f}%)")
