"""Record real top-k routing of an mlx-lm Qwen3-MoE model on real text.

    .venv/bin/python -m moelab.routing_trace --out runs/routing_qwen3.npz

Saves inds[domain] -> int16 [layers, tokens, k]. Analysis: moelab/routing_stats.py.
"""
import argparse
import glob
import os

import mlx.core as mx
import numpy as np
from mlx_lm import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLISH = ("Mieszanina ekspertów to architektura sieci neuronowej, w której router wybiera dla każdego tokenu "
          "kilka spośród wielu podsieci. Dzięki temu model ma bardzo dużo parametrów, ale na każdy token "
          "zużywa tylko ich niewielką część. W inferencji na laptopie liczy się przede wszystkim "
          "przepustowość pamięci, ponieważ wagi aktywnych ekspertów trzeba odczytać przy każdym kroku. ") * 12


def domains():
    md = "\n".join(open(p).read() for p in sorted(glob.glob(f"{ROOT}/docs/*.md")))
    code = "\n".join(open(p).read() for p in sorted(glob.glob(f"{ROOT}/moelab/*.py")))
    return {"english_tech": md, "python_code": code, "polish": POLISH}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default="runs/routing_qwen3.npz")
    a = ap.parse_args()
    model, tok = load(a.model)
    # Qwen3-MoE: layer.mlp.switch_mlp ; gpt-oss: layer.mlp.experts (both mlx-lm SwitchGLU)
    sws = [getattr(l.mlp, "switch_mlp", None) or getattr(l.mlp, "experts", None) for l in model.layers]
    sws = [w for w in sws if w is not None]
    rec = []
    orig_switch = type(sws[0]).__call__

    def spy(self, x, indices):  # SwitchGLU.__call__(x, indices): indices [B, L, k]
        rec.append(indices)
        return orig_switch(self, x, indices)

    type(sws[0]).__call__ = spy
    out = {}
    for name, text in domains().items():
        ids = tok.encode(text)[: a.max_tokens]
        rec.clear()
        mx.eval(model(mx.array([ids])))
        inds = np.stack([np.array(r)[0] for r in rec]).astype(np.int16)   # [layers, L, k]
        out[name] = inds
        print(f"{name:13s} tokens={inds.shape[1]} layers={inds.shape[0]} k={inds.shape[2]}")
    np.savez_compressed(a.out, E=sws[0].gate_proj.weight.shape[0], **out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
