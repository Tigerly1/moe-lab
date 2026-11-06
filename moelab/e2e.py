"""End-to-end decode benchmark: mlx-lm Qwen3-MoE with the stock MoE block vs a variant kernel.

    .venv/bin/python -m moelab.e2e --model mlx-community/Qwen3-30B-A3B-4bit \
        --variant moelab/variants/mlx/metal_q4_gemv_v31.py --tokens 256 --rounds 3

Router (gate + softmax + top-k) is kept identical; only the expert compute is swapped.
The variant is used for decode-sized calls (<= MAX_T tokens); prefill keeps the stock path.
Rounds alternate stock/variant to spread thermal drift. Greedy outputs are compared.
"""
import argparse
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from .mlx_backend import load_variant

MAX_T = 4
PROMPT = "Explain, step by step, how a mixture-of-experts transformer layer routes tokens to experts."


def patch(model, mod):
    """Swap expert compute of every Qwen3MoeSparseMoeBlock for `mod.run` at decode."""
    blocks = [l.mlp for l in model.layers if hasattr(l.mlp, "switch_mlp")]
    cls = type(blocks[0])
    stock = cls.__dict__.get("_stock_call", cls.__call__)  # idempotent re-patching
    cls._stock_call = stock
    for b in blocks:
        sw = b.switch_mlp
        assert sw.gate_proj.group_size == mod.GS and sw.gate_proj.bits == mod.BITS, "format mismatch"
        b._variant_state = {n: (p.weight, p.scales, p.biases) for n, p in
                            (("g", sw.gate_proj), ("u", sw.up_proj), ("d", sw.down_proj))}

    def call(self, x):
        T = x.size // x.shape[-1]
        if T > MAX_T or not getattr(cls, "_use_variant", False):
            return stock(self, x)
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)
        y = mod.run(self._variant_state, x.reshape(T, -1), inds.reshape(T, k),
                    scores.reshape(T, k).astype(mx.float32))
        return y.reshape(x.shape)

    cls.__call__ = call
    return cls, len(blocks)


def decode(model, tok, n):
    out, tps = [], None
    for r in stream_generate(model, tok, PROMPT, max_tokens=n, sampler=make_sampler(temp=0.0)):
        out.append(r.token)
        tps = r.generation_tps
    return out, tps


def teacher_forced(model, tok, cls, text, use_variant):
    """Feed `text` one token at a time (decode path), return per-token NLL and argmax."""
    from mlx_lm.models.cache import make_prompt_cache
    cls._use_variant = use_variant
    ids = tok.encode(text)
    cache = make_prompt_cache(model)
    nll, top = [], []
    for i in range(len(ids) - 1):
        logits = model(mx.array([[ids[i]]]), cache=cache)[0, -1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits)
        nll.append(-lp[ids[i + 1]].item())
        top.append(int(mx.argmax(logits).item()))
    return np.array(nll), np.array(top)


def quality_check(model, tok, cls, variant_path):
    """Stock vs variant vs fp32 truth (same dequantized weights), teacher-forced."""
    text = open(__file__.replace("moelab/e2e.py", "docs/RESULTS.md")).read()[:3000]
    truth_mod = load_variant(__file__.replace("e2e.py", "variants/fp32_truth_q4.py"))
    var_mod = load_variant(variant_path)
    runs = {"stock": (False, None)}
    runs["variant"] = (True, var_mod)
    runs["fp32"] = (True, truth_mod)
    out = {}
    for name, (use, mod) in runs.items():
        if mod is not None:
            patch(model, mod)
        out[name] = teacher_forced(model, tok, cls, text, use)
        print(f"{name:8s} PPL {np.exp(out[name][0].mean()):.4f}")
    for a, b in (("stock", "fp32"), ("variant", "fp32"), ("stock", "variant")):
        (na, ta), (nb, tb) = out[a], out[b]
        print(f"{a:7s} vs {b:7s}: top-1 agreement {100 * (ta == tb).mean():6.2f}%  mean |dNLL| {np.abs(na - nb).mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--variant", default="moelab/variants/mlx/metal_q4_gemv_v31.py")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--quality", action="store_true", help="teacher-forced PPL comparison only")
    a = ap.parse_args()
    model, tok = load(a.model)
    cls, n = patch(model, load_variant(a.variant))
    print(f"patched {n} MoE blocks with {a.variant}")
    if a.quality:
        return quality_check(model, tok, cls, a.variant)
    res = {"stock": [], "variant": []}
    outs = {}
    for r in range(a.rounds + 1):  # round 0 = warmup
        for name in (("stock", "variant") if r % 2 == 0 else ("variant", "stock")):
            cls._use_variant = name == "variant"
            o, tps = decode(model, tok, a.tokens)
            if r > 0:
                res[name].append(tps)
                outs[name] = o
                print(f"round {r} {name:8s} {tps:6.1f} tok/s")
            time.sleep(2)
    s, v = np.median(res["stock"]), np.median(res["variant"])
    same = sum(x == y for x, y in zip(outs["stock"], outs["variant"]))
    first_diff = next((i for i, (x, y) in enumerate(zip(outs["stock"], outs["variant"])) if x != y), None)
    print(f"\nstock   {s:.1f} tok/s  {res['stock']}")
    print(f"variant {v:.1f} tok/s  {res['variant']}")
    print(f"speedup {v / s:.3f}x | greedy tokens identical: {same}/{len(outs['stock'])} (first diff at {first_diff})")


if __name__ == "__main__":
    main()
