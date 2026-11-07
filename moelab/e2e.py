"""End-to-end decode benchmark: mlx-lm Qwen3-MoE with the stock MoE block vs a variant kernel.

    .venv/bin/python -m moelab.e2e --model mlx-community/Qwen3-30B-A3B-4bit \
        --variant moelab/variants/mlx/metal_q4_gemv_v31.py --tokens 256 --rounds 3

Router (gate + softmax + top-k) is kept identical; only the expert compute is swapped.
The variant is used for decode-sized calls (<= MAX_T tokens); prefill keeps the stock path.
Rounds alternate stock/variant to spread thermal drift. Greedy outputs are compared.
"""
import argparse
import json
import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from .mlx_backend import load_variant
from .router import topk_norm

FUSED_ROUTER = False  # round 7: moelab/router.py replaces softmax/argpartition/take/renorm (--fused-router)
MAX_T = 256  # measured crossover: our kernels win up to T=256, parity at T=512 (runs/crossover_*.json)
PROMPT = "Explain, step by step, how a mixture-of-experts transformer layer routes tokens to experts."


def patch(model, mod):
    """Swap expert compute of every Qwen3MoeSparseMoeBlock (or gpt-oss MLPBlock) for `mod.run` at decode."""
    if getattr(model, "model_type", None) == "gpt_oss":
        return _patch_gptoss(model, mod)
    if getattr(model, "model_type", None) in ("deepseek_v2", "deepseek_v3"):
        return _patch_deepseek(model, mod)
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
        k = self.top_k
        if FUSED_ROUTER and self.norm_topk_prob:
            inds, scores = topk_norm(self.gate(x), k)
        else:
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if self.norm_topk_prob:
                scores /= mx.sum(scores, axis=-1, keepdims=True)
        y = mod.run(self._variant_state, x.reshape(T, -1), inds.reshape(T, k),
                    scores.reshape(T, k).astype(mx.float32))
        return y.reshape(x.shape)

    cls.__call__ = call
    return cls, len(blocks)


def _patch_deepseek(model, mod):
    """DeepSeek-V2/V3 MoE block: the stock MoEGate (softmax/sigmoid, group-limited top-k, routed scaling)
    is called unchanged; only the routed SwitchGLU is replaced; shared experts run stock and are added."""
    blocks = [l.mlp for l in model.layers if hasattr(l.mlp, "switch_mlp")]   # first layer(s) are dense
    cls = type(blocks[0])
    stock = cls.__dict__.get("_stock_call", cls.__call__)
    cls._stock_call = stock
    for b in blocks:
        sw = b.switch_mlp
        for p in (sw.gate_proj, sw.up_proj, sw.down_proj):
            assert p.group_size == mod.GS and p.bits == mod.BITS, "format mismatch"
        b._variant_state = {n: (p.weight, p.scales, p.biases) for n, p in
                            (("g", sw.gate_proj), ("u", sw.up_proj), ("d", sw.down_proj))}

    def call(self, x):
        T = x.size // x.shape[-1]
        if T > MAX_T or not getattr(cls, "_use_variant", False):
            return stock(self, x)
        assert getattr(self, "sharding_group", None) is None, "tensor-parallel DeepSeek MoE not supported by the patch"
        g = self.gate
        if FUSED_ROUTER and g.topk_method != "group_limited_greedy":   # mlx-lm: anything else is greedy
            inds, scores = topk_norm(x @ g.weight.T, g.top_k, norm=False)
            if g.routed_scaling_factor != 1.0:
                scores = scores * g.routed_scaling_factor
        else:
            inds, scores = g(x)
        k = inds.shape[-1]
        y = mod.run(self._variant_state, x.reshape(T, -1), inds.reshape(T, k),
                    scores.reshape(T, k).astype(mx.float32)).reshape(x.shape)
        if getattr(self, "shared_experts", None) is not None:
            y = y + self.shared_experts(x)
        return y

    cls.__call__ = call
    return cls, len(blocks)


def _patch_gptoss(model, mod):
    """gpt-oss MLPBlock: router = Linear(bias) -> mlx_topk -> softmax over the top-k (as stock);
    experts = SwitchGLU(bias=True, mxfp4) with clamped SwiGLU, run by `mod.run(mod.state(experts), ...)`."""
    from mlx_lm.models.gpt_oss import mlx_topk
    blocks = [l.mlp for l in model.layers]
    cls = type(blocks[0])
    stock = cls.__dict__.get("_stock_call", cls.__call__)  # idempotent re-patching
    cls._stock_call = stock
    for b in blocks:
        b._variant_state = mod.state(b.experts)  # asserts mode/group_size/bits

    def call(self, x):
        T = x.size // x.shape[-1]
        k = self.num_experts_per_tok
        if T > MAX_T or not getattr(cls, "_use_variant", False) or \
                (T > 1 and T * k / self._variant_state["g"][0].shape[0] > 8):   # round 7 load policy
            return stock(self, x)
        if FUSED_ROUTER:
            inds, w = topk_norm(self.router(x), k)
        else:
            vals, inds = mlx_topk(self.router(x), k=k, axis=-1)
            w = mx.softmax(vals, axis=-1, precise=True)
        y = mod.run(self._variant_state, x.reshape(T, -1), inds.reshape(T, k), w.reshape(T, k).astype(mx.float32))
        return y.reshape(x.shape)

    cls.__call__ = call
    return cls, len(blocks)


def decode(model, tok, n, draft=None, num_draft=3, prompt=None):
    out, tps, acc = [], None, 0
    kw = {"draft_model": draft, "num_draft_tokens": num_draft} if draft is not None else {}
    for r in stream_generate(model, tok, prompt if prompt is not None else PROMPT, max_tokens=n,
                             sampler=make_sampler(temp=0.0), **kw):
        out.append(r.token)
        tps = r.generation_tps
        acc += bool(getattr(r, "from_draft", False))
    decode.last_accept = acc / max(1, len(out))
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
    text = open(__file__.replace("e2e.py", "data/quality_corpus.txt")).read()[:3000]  # frozen corpus (audit fix)
    var_mod = load_variant(variant_path)
    if getattr(model, "model_type", None) == "gpt_oss":  # fp32 truth with biases + clamped SwiGLU
        truth_mod = SimpleNamespace(state=var_mod.state, run=var_mod.reference)
    else:
        truth_mod = load_variant(__file__.replace("e2e.py", "variants/fp32_truth_q4.py"))
    runs = {"stock": (False, None)}
    runs["variant"] = (True, var_mod)
    runs["fp32"] = (True, truth_mod)
    out = {}
    for name, (use, mod) in runs.items():
        if mod is not None:
            patch(model, mod)
        out[name] = teacher_forced(model, tok, cls, text, use)
        print(f"{name:8s} PPL {np.exp(out[name][0].mean()):.4f}")
    res = {"ppl": {k: float(np.exp(v[0].mean())) for k, v in out.items()}, "pairs": {}}
    for a, b in (("stock", "fp32"), ("variant", "fp32"), ("stock", "variant")):
        (na, ta), (nb, tb) = out[a], out[b]
        res["pairs"][f"{a}_vs_{b}"] = {"top1_agreement": float((ta == tb).mean()), "mean_abs_dnll": float(np.abs(na - nb).mean())}
        print(f"{a:7s} vs {b:7s}: top-1 agreement {100 * (ta == tb).mean():6.2f}%  mean |dNLL| {np.abs(na - nb).mean():.4f}")
    return res


def verify_bench(model, tok, cls, chunk, rounds=3):
    """Acceptance-free model-level comparison: feed the SAME fixed text through the full
    model in chunks of `chunk` tokens (= a speculative verify step of chunk-1 drafts) and
    time it. Both arms do identical work, unlike tok/s of free-running speculative decoding."""
    from mlx_lm.models.cache import make_prompt_cache
    ids = tok.encode(open(__file__.replace("e2e.py", "data/quality_corpus.txt")).read())[:1024]
    prompt, rest = ids[:32], ids[32:]
    chunks = [rest[i:i + chunk] for i in range(0, len(rest) - chunk + 1, chunk)]

    def run_once():
        cache = make_prompt_cache(model)
        mx.eval(model(mx.array([prompt]), cache=cache))
        t0 = time.perf_counter()
        for c in chunks:
            mx.eval(model(mx.array([c]), cache=cache))
        return time.perf_counter() - t0

    res = {"stock": [], "variant": []}
    for r in range(rounds + 1):
        for name in (("stock", "variant") if r % 2 == 0 else ("variant", "stock")):
            cls._use_variant = name == "variant"
            t = run_once()
            if r > 0:
                res[name].append(len(chunks) * chunk / t)
    s, v = np.median(res["stock"]), np.median(res["variant"])
    print(f"verify chunk={chunk}: stock {s:.1f} tok/s  variant {v:.1f} tok/s  speedup {v / s:.3f}x  "
          f"({len(chunks)} steps, rounds {res})", flush=True)
    return {"chunk": chunk, "max_t": MAX_T, "variant_used": chunk <= MAX_T, "stock": res["stock"], "variant": res["variant"], "speedup": v / s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--variant", default="moelab/variants/mlx/metal_q4_gemv_v31.py")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--quality", action="store_true", help="teacher-forced PPL comparison only")
    ap.add_argument("--draft", default=None, help="draft model for speculative decoding (verify T = num_draft+1)")
    ap.add_argument("--num-draft", type=int, default=3)
    ap.add_argument("--max-t", type=int, default=MAX_T, help="use the variant for calls with <= this many tokens")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fused-router", action="store_true", help="fused top-k/normalize router tail (moelab/router.py)")
    ap.add_argument("--context", type=int, default=0, help="prompt of this many tokens (frozen corpus, repeated) before decoding")
    ap.add_argument("--verify-chunks", type=int, nargs="*", default=None,
                    help="acceptance-free verify benchmark with these chunk sizes, e.g. 1 2 4 7 16")
    a = ap.parse_args()
    globals()["MAX_T"] = a.max_t
    globals()["FUSED_ROUTER"] = a.fused_router
    model, tok = load(a.model)
    draft = load(a.draft)[0] if a.draft else None
    cls, n = patch(model, load_variant(a.variant))
    print(f"patched {n} MoE blocks with {a.variant}")
    if a.quality:
        q = quality_check(model, tok, cls, a.variant)
        if a.out:
            json.dump({"args": vars(a), **q}, open(a.out, "w"), indent=1)
        return
    if a.verify_chunks:
        out = [verify_bench(model, tok, cls, c) for c in a.verify_chunks]
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1)
        return
    ctx_prompt = None
    if a.context:
        ids = tok.encode(open(__file__.replace("e2e.py", "data/quality_corpus.txt")).read())
        ctx_prompt = (ids * (a.context // len(ids) + 1))[:a.context]
    res = {"stock": [], "variant": []}
    outs = {}
    from .guard import probe, verdict, report
    probes = [probe()]
    for r in range(a.rounds + 1):  # round 0 = warmup
        for name in (("stock", "variant") if r % 2 == 0 else ("variant", "stock")):
            cls._use_variant = name == "variant"
            o, tps = decode(model, tok, a.tokens, draft, a.num_draft, ctx_prompt)
            if r > 0:
                res[name].append(tps)
                outs[name] = o
                print(f"round {r} {name:8s} {tps:6.1f} tok/s  len={len(o)}"
                      + (f"  draft-accept={decode.last_accept:.2f}" if draft is not None else ""), flush=True)
            time.sleep(2)
    probes.append(probe())
    g = verdict(probes)
    print(report(g))
    s, v = np.median(res["stock"]), np.median(res["variant"])
    same = sum(x == y for x, y in zip(outs["stock"], outs["variant"]))
    first_diff = next((i for i, (x, y) in enumerate(zip(outs["stock"], outs["variant"])) if x != y), None)
    print(f"\nstock   {s:.1f} tok/s  {res['stock']}")
    print(f"variant {v:.1f} tok/s  {res['variant']}")
    print(f"speedup {v / s:.3f}x | greedy tokens identical: {same}/{len(outs['stock'])} (first diff at {first_diff})")
    if a.out:
        json.dump({"args": vars(a), "stock": res["stock"], "variant": res["variant"], "speedup": v / s,
                   "tokens_identical": int(same), "first_diff": first_diff,
                   "len_stock": len(outs["stock"]), "len_variant": len(outs["variant"]), "guard": g}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
