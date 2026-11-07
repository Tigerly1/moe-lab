"""Production dispatcher (regime-adaptive), int4 affine g64 MoE experts on Apple GPUs.

T == 1                      -> metal_q4_decode_v4 (GEMV at ~95-100% of measured bandwidth)
T > 1, d,f % 128 == 0       -> metal_q4_tiles_v2 (expert tiles, 8x8 MMA for >=3 slots, scalar 2-slot, LPR down; round 3)
T > 1, otherwise (gpt-oss)  -> metal_q4_tiles_b   (expert tiles, scalar)
All three share the mx.quantize state layout, so prepare() is done once.

Round 7 (2026-11-07): above a load threshold (load = T*k/E, avg tokens per expert) MLX's sorted
gather_qmm is as fast or faster, so run() falls back to it. Thresholds fitted on clean grids with the
custom kernels forced (runs/grid_q4.json, runs/grid_small_q4.json); first version (load 8 everywhere)
was too conservative (audit):
  d, f multiples of 128 and d*f >= 1.5 * 2^20 (Qwen3-30B and larger): custom up to load 24
  d, f multiples of 128, smaller experts: up to load 8
  otherwise (scalar tile path, e.g. d=2880): up to load 4
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
    # round 9: with the MMA tiles on every D, F % 64 shape the round-7 rule for %128 shapes holds for all
    # (runs/r9/grid_q4_mma64.json: Gemma-4 F=704 crosses at load 24-32, Qwen3.6 at 8-12)
    load = T * k / E
    return load <= (24 if D * F >= 3 << 19 else 8)


def run(s, x, idx, wts, act=0):
    """act: 0 = SwiGLU (silu(gate) * up), 1 = GeGLU with tanh-GELU (Gemma-4)."""
    T, k = idx.shape
    if T == 1:
        return _v4.run(s, x, idx, wts, act)
    E, F, D = s["g"][0].shape[0], s["g"][0].shape[1], s["d"][0].shape[1]
    if not custom_wins(T, k, E, D, F):
        return gather_moe(s, x, idx, wts, GS, BITS, act=act)
    if x.dtype in (mx.bfloat16, mx.float16):   # round 9: MMA tiles need only D, F % 64 (always true at group 64)
        return _mma.run(s, x, idx, wts, act)   # fp32 activations -> scalar tiles (exact), see tiles_v2 note
    return _tb.run(s, x, idx, wts, act)
