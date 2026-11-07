"""Production dispatcher (regime-adaptive), int4 affine g64 MoE experts on Apple GPUs.

T == 1                      -> metal_q4_decode_v4 (GEMV at ~95-100% of measured bandwidth)
T > 1, d,f % 128 == 0       -> metal_q4_tiles_v2 (expert tiles, 8x8 MMA for >=3 slots, scalar 2-slot, LPR down; round 3)
T > 1, otherwise (gpt-oss)  -> metal_q4_tiles_b   (expert tiles, scalar)
All three share the mx.quantize state layout, so prepare() is done once.

Round 7 (2026-11-07): above a load threshold (load = T*k/E, avg tokens per expert) MLX's sorted
gather_qmm is as fast or faster (runs/grid_q4.json, runs/sweep_q4.json), so run() falls back to it:
  aligned d, f and d*f >= 2^20: custom up to load 8; otherwise (gpt-oss-like or tiny experts): up to load 4.
"""
import os

import mlx.core as mx

from moelab.fallback import gather_moe
from moelab.mlx_backend import load_variant

META = {"name": "metal_q4_moe", "description": "dispatcher: v4 at T=1, expert-tile kernels at T>1"}
BITS, GS = 4, 64
W_BITS = BITS + 32 / GS
_HERE = os.path.dirname(os.path.abspath(__file__))
_v4 = load_variant(os.path.join(_HERE, "metal_q4_decode_v4.py"))
_mma = load_variant(os.path.join(_HERE, "metal_q4_tiles_v2.py"))
_tb = load_variant(os.path.join(_HERE, "metal_q4_tiles_b.py"))


def prepare(w_gate, w_up, w_down):
    E, f, d = w_gate.shape
    assert d % 32 == 0 and f % 32 == 0, "d, f must be multiples of 32"
    return {n: mx.quantize(w, group_size=GS, bits=BITS) for n, w in (("g", w_gate), ("u", w_up), ("d", w_down))}


def dequant(s):
    return tuple(mx.dequantize(*s[n], group_size=GS, bits=BITS) for n in ("g", "u", "d"))


def custom_wins(T, k, E, D, F):
    load = T * k / E
    if D % 128 == 0 and F % 128 == 0 and D * F >= 1 << 20:
        return load <= 8
    return load <= 4


def run(s, x, idx, wts):
    T, k = idx.shape
    if T == 1:
        return _v4.run(s, x, idx, wts)
    E, F, D = s["g"][0].shape[0], s["g"][0].shape[1], s["d"][0].shape[1]
    if not custom_wins(T, k, E, D, F):
        return gather_moe(s, x, idx, wts, GS, BITS)
    if D % 128 == 0 and F % 128 == 0 and x.dtype in (mx.bfloat16, mx.float16):
        return _mma.run(s, x, idx, wts)   # fp32 activations -> scalar tiles (exact), see tiles_v2 note
    return _tb.run(s, x, idx, wts)
