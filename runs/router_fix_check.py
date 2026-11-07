"""mlx-lm MoE router precision, per model, with the model's OWN stock selection code:
qwen3_moe : softmax(precise) [bf16 out] -> argpartition(g, kth=-k)[..., -k:] -> take -> renorm
deepseek_v2: softmax(precise) [bf16 out] -> argpartition(-g, kth=k-1)[..., :k] -> take (no renorm)
Fix A (one line): softmax on fp32 logits.   Fix B: rank on logits, fp32 softmax over the top-k only (SGLang-style;
exact for norm_topk_prob; for no-renorm models needs the full fp32 denominator).
Counts token-layers where stock picks a different expert set than exact (larger logit; logit ties excepted) and
checks that each difference is an exact tie of bf16 probabilities; router tail cost per token for stock / A / B."""
import sys, json, time
import mlx.core as mx, numpy as np
from mlx_lm import load
repo = sys.argv[1]
model, tok = load(repo)
ids = tok.encode(open("moelab/data/quality_corpus.txt").read())[:512]
mt = model.model_type
blocks = [l.mlp for l in model.layers if hasattr(l.mlp, "switch_mlp")]
k = blocks[0].top_k if hasattr(blocks[0], "top_k") else blocks[0].gate.top_k
logit = (lambda b, x: b.gate(x)) if mt != "deepseek_v2" else (lambda b, x: x @ b.gate.weight.T)
def sel(g):
    if mt == "deepseek_v2":
        return mx.argpartition(-g, kth=k - 1, axis=-1)[..., :k]
    return mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
caught = []
cls = type(blocks[0]); orig = cls.__call__
def spy(self, x):
    l = logit(self, x); caught.append(l.reshape(-1, l.shape[-1])); return orig(self, x)
cls.__call__ = spy; model(mx.array([ids])); cls.__call__ = orig
n = diff = ties = diffA = 0
for lg in caught:
    g = mx.softmax(lg, axis=-1, precise=True)
    i0 = np.array(sel(g)); iA = np.array(sel(mx.softmax(lg.astype(mx.float32), axis=-1)))
    G = np.array(g.astype(mx.float32)); L = np.array(lg.astype(mx.float32))
    for t in range(lg.shape[0]):
        n += 1
        kth = np.sort(L[t])[-k]
        strict = set(np.nonzero(L[t] > kth)[0].tolist())          # experts that must be chosen
        def ok(s):
            s = set(s.tolist()); return strict <= s and all(L[t, e] >= kth for e in s)
        if not ok(i0[t]):
            diff += 1
            wrong = [e for e in i0[t] if L[t, e] < kth]; missed = [e for e in strict if e not in set(i0[t].tolist())]
            ties += np.array_equal(np.sort(G[t, wrong]), np.sort(G[t, missed])) if len(wrong) == len(missed) else 0
        diffA += not ok(iA[t])
x0 = (mx.random.normal((1, 1, model.args.hidden_size), key=mx.random.key(0)) * 0.1).astype(mx.bfloat16)
def chain(kind):
    x = x0
    for b in blocks:
        lg = logit(b, x)
        if kind == "gate":
            x = x + 0 * lg.sum().astype(x.dtype); continue
        if kind == "stock":
            g = mx.softmax(lg, axis=-1, precise=True); i = sel(g); s = mx.take_along_axis(g, i, axis=-1)
        elif kind == "A":
            g = mx.softmax(lg.astype(mx.float32), axis=-1); i = sel(g); s = mx.take_along_axis(g, i, axis=-1)
        else:  # B: rank on logits
            i = mx.argpartition(lg, kth=-k, axis=-1)[..., -k:]
            s = mx.softmax(mx.take_along_axis(lg, i, axis=-1).astype(mx.float32), axis=-1)
        if mt != "deepseek_v2" or kind != "B":
            s = s / s.sum(-1, keepdims=True) if mt != "deepseek_v2" else s
        x = x + (s.sum() * 0 + i.sum() * 0).astype(x.dtype)
    return x
def t(kind, reps=30):
    mx.eval(chain(kind)); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(chain(kind)); mx.synchronize(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)
cost = {}
for r in range(3):
    for kd in (("gate", "stock", "A", "B") if r % 2 == 0 else ("B", "A", "stock", "gate")):
        cost.setdefault(kd, []).append(t(kd))
c = {kd: float(np.median(v)) for kd, v in cost.items()}
res = {"model": repo, "selection": "argpartition(-g)[:k]" if mt == "deepseek_v2" else "argpartition(g)[-k:]",
       "token_layers": n, "stock_wrong_set": diff, "of_which_exact_bf16_prob_ties": int(ties), "fixA_wrong_set": diffA,
       "tail_ms_stock": c["stock"] - c["gate"], "tail_ms_fixA_fp32softmax": c["A"] - c["gate"], "tail_ms_fixB_rank_on_logits": c["B"] - c["gate"]}
print(json.dumps(res))
json.dump(res, open(f"runs/r8/router_fix_{mt}.json", "w"), indent=1)
