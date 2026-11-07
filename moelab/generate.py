"""Generate text with mlx-lm using the moe-lab MoE kernels (auto-selected by checkpoint format).

    .venv/bin/python -m moelab.generate --model mlx-community/Qwen3-30B-A3B-4bit --prompt "Hello"
    .venv/bin/python -m moelab.generate --model mlx-community/gpt-oss-20b-MXFP4-Q8 --prompt "Hello" --compare

--compare also runs stock mlx-lm with the same prompt and prints both decode speeds.
"""
import argparse
import os

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from . import e2e
from .mlx_backend import load_variant

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {  # (model_type, expert quant mode) -> kernel file
    ("qwen3_moe", "affine"): "variants/mlx/metal_q4_moe.py",
    ("gpt_oss", "mxfp4"): "variants/mlx/metal_mxfp4_gptoss.py",
    ("deepseek_v2", "affine"): "variants/mlx/metal_q4_moe.py",
    ("qwen3_5_moe", "affine"): "variants/mlx/metal_q4_moe.py",   # Qwen3.5/3.6-35B-A3B (round 9)
    ("gemma4", "affine"): "variants/mlx/metal_q4_moe.py",        # Gemma-4-26B-A4B, GeGLU (round 9)
    ("gemma4_text", "affine"): "variants/mlx/metal_q4_moe.py",
}


def _experts(layer):
    m = getattr(layer, "mlp", None)
    return (getattr(m, "switch_mlp", None) or getattr(m, "experts", None)
            or getattr(getattr(layer, "experts", None), "switch_glu", None))   # Gemma-4: layer.experts.switch_glu


def pick_variant(model):
    mtype = getattr(model, "model_type", None) or model.args.model_type
    sw = next(e for e in map(_experts, model.layers) if e is not None)
    mode = getattr(sw.gate_proj, "mode", "affine")
    key = (mtype, mode)
    if key not in VARIANTS:
        raise SystemExit(f"no moe-lab kernel for {key}; supported: {sorted(VARIANTS)}")
    if mode == "affine":
        assert (sw.gate_proj.bits, sw.gate_proj.group_size) == (4, 64), "affine kernels need 4-bit, group 64"
    return os.path.join(HERE, VARIANTS[key])


def run(model, tok, prompt, max_tokens):
    text, last = "", None
    for r in stream_generate(model, tok, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)):
        text += r.text
        last = r
    return text, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Explain mixture-of-experts layers in two sentences.")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--no-fused-router", action="store_true", help="keep mlx-lm's softmax/argpartition router tail")
    a = ap.parse_args()
    e2e.FUSED_ROUTER = not a.no_fused_router   # round 7: faster (fewer dispatches); ties broken by lowest id, PPL within 0.9% of stock (docs/RESULTS.md)
    model, tok = load(a.model)
    variant = pick_variant(model)
    cls, n = e2e.patch(model, load_variant(variant))
    prompt = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True, tokenize=False) \
        if getattr(tok, "chat_template", None) else a.prompt
    print(f"[moe-lab] {n} MoE blocks patched with {os.path.basename(variant)}\n")
    for use in (True, False):            # warm up both paths (Metal kernel JIT compile is one-time)
        cls._use_variant = use
        run(model, tok, prompt, 8)
    cls._use_variant = True
    text, r = run(model, tok, prompt, a.max_tokens)
    print(text)
    print(f"\n[moe-lab kernels] prompt {r.prompt_tps:.1f} tok/s | decode {r.generation_tps:.1f} tok/s")
    if a.compare:
        cls._use_variant = False
        _, s = run(model, tok, prompt, a.max_tokens)
        print(f"[stock mlx-lm  ] prompt {s.prompt_tps:.1f} tok/s | decode {s.generation_tps:.1f} tok/s")
        print(f"decode speedup {r.generation_tps / s.generation_tps:.3f}x (single run; use moelab.e2e for statistics)")


if __name__ == "__main__":
    main()
