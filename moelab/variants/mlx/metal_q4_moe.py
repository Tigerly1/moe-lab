"""Production dispatcher (regime-adaptive), int4 affine g64 MoE experts on Apple GPUs.

T == 1                      -> metal_q4_decode_v4 (GEMV at ~95-100% of measured bandwidth)
T > 1, d,f % 128 == 0       -> metal_q4_tiles_mma (expert tiles, simdgroup 8x8 MMA, weight reuse)
T > 1, otherwise (gpt-oss)  -> metal_q4_tiles_b   (expert tiles, scalar)
All three share the mx.quantize state layout, so prepare() is done once.
"""
import os

import mlx.core as mx

from moelab.mlx_backend import load_variant

META = {"name": "metal_q4_moe", "description": "dispatcher: v4 at T=1, expert-tile kernels at T>1"}
BITS, GS = 4, 64
W_BITS = BITS + 32 / GS
_HERE = os.path.dirname(os.path.abspath(__file__))
_v4 = load_variant(os.path.join(_HERE, "metal_q4_decode_v4.py"))
_mma = load_variant(os.path.join(_HERE, "metal_q4_tiles_mma.py"))
_tb = load_variant(os.path.join(_HERE, "metal_q4_tiles_b.py"))


def prepare(w_gate, w_up, w_down):
    E, f, d = w_gate.shape
    assert d % 32 == 0 and f % 32 == 0, "d, f must be multiples of 32"
    return {n: mx.quantize(w, group_size=GS, bits=BITS) for n, w in (("g", w_gate), ("u", w_up), ("d", w_down))}


def dequant(s):
    return tuple(mx.dequantize(*s[n], group_size=GS, bits=BITS) for n in ("g", "u", "d"))


def run(s, x, idx, wts):
    T = idx.shape[0]
    if T == 1:
        return _v4.run(s, x, idx, wts)
    F, D = s["g"][0].shape[1], s["d"][0].shape[1]
    if D % 128 == 0 and F % 128 == 0:
        return _mma.run(s, x, idx, wts)
    return _tb.run(s, x, idx, wts)
