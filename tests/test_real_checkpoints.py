"""Kernels vs fp32 reference on REAL expert weights (lesson of 2026-11-07: random-normal lab
weights hid a scale overflow that real checkpoints trigger). Skipped if a model isn't cached locally."""
import json
import os

import mlx.core as mx
import numpy as np
import pytest

from moelab import mlx_backend as be
from moelab.problem import make_routing


def _local(repo):
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download(repo, allow_patterns=["*.json"], local_files_only=True)
        idx = json.load(open(os.path.join(p, "model.safetensors.index.json")))["weight_map"]
        if not all(os.path.exists(os.path.join(p, f)) for f in set(idx.values())):
            return None
        return p, idx
    except Exception:
        return None


def _get(p, idx, k):
    return mx.load(os.path.join(p, idx[k]))[k]


QWEN = _local("mlx-community/Qwen3-30B-A3B-4bit")
GPTOSS = _local("mlx-community/gpt-oss-20b-MXFP4-Q8")


@pytest.mark.skipif(QWEN is None, reason="Qwen3-30B-A3B-4bit not cached")
@pytest.mark.parametrize("variant", ["metal_q4_moe", "metal_q4_tiles_v2", "metal_q4_decode_v4"])
@pytest.mark.parametrize("layer", [0, 47])
def test_qwen3_real_weights(variant, layer):
    p, idx = QWEN
    m = be.load_variant(f"moelab/variants/mlx/{variant}.py")
    P = f"model.layers.{layer}.mlp.switch_mlp."
    st = {n: [_get(p, idx, P + f"{pr}.{s}") for s in ("weight", "scales", "biases")]
          for n, pr in (("g", "gate_proj"), ("u", "up_proj"), ("d", "down_proj"))}
    rw = tuple(mx.dequantize(*st[n], group_size=64, bits=4) for n in ("g", "u", "d"))
    E, d = rw[0].shape[0], rw[0].shape[2]
    rng = np.random.default_rng(layer)
    for T in ([1] if variant == "metal_q4_decode_v4" else [1, 3, 8, 16, 32]):
        for r in ("uniform", "one_hot", "trace"):
            x = mx.array(rng.standard_normal((T, d)).astype(np.float32)).astype(mx.bfloat16)
            i, w = make_routing(T, E, 8, r, rng)
            i, w = mx.array(i), mx.array(w)
            e = be.errors(m.run(st, x, i, w), be.reference(*rw, x, i, w))
            assert e["finite"] and e["rel_fro"] < 1e-2, (T, r, e)


@pytest.mark.skipif(GPTOSS is None, reason="gpt-oss-20b-MXFP4-Q8 not cached")
@pytest.mark.parametrize("layer", [0, 23])
def test_gptoss_real_weights(layer):
    p, idx = GPTOSS
    m = be.load_variant("moelab/variants/mlx/metal_mxfp4_gptoss.py")
    P = f"model.layers.{layer}.mlp.experts."
    s = {n: (_get(p, idx, P + f"{pr}.weight"), _get(p, idx, P + f"{pr}.scales"))
         for n, pr in (("g", "gate_proj"), ("u", "up_proj"), ("d", "down_proj"))}
    s.update(bg=_get(p, idx, P + "gate_proj.bias").astype(mx.float32),
             bu=_get(p, idx, P + "up_proj.bias").astype(mx.float32),
             bd=_get(p, idx, P + "down_proj.bias").astype(mx.float32), act=1)
    assert int(np.array(s["d"][1]).max()) >= 129, "checkpoint layer should contain E8M0 bytes >= 129"
    E, d = s["g"][0].shape[0], s["d"][0].shape[1]
    rng = np.random.default_rng(layer)
    for T in (1, 3, 8, 16):
        for r in ("uniform", "one_hot"):
            x = mx.array(rng.standard_normal((T, d)).astype(np.float32)).astype(mx.bfloat16)
            i, w = make_routing(T, E, 4, r, rng)
            i, w = mx.array(i), mx.array(w)
            e = be.errors(m.run(s, x, i, w), m.reference(s, x, i, w))
            assert e["finite"] and e["rel_fro"] < 1e-2, (T, r, e)
