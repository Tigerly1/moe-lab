"""Analytic cost model of a SwiGLU MoE layer (expert part only).

FLOPs  = 6 * d * f * T * k              (gate + up + down, 2 FLOP/MAC)
Bytes  = U * 3 * d * f * bw/8 + acts    (U = unique experts touched)
E[U]   = E * (1 - (1 - k/E)^T)          (uniform routing, coupon-collector)
"""
import numpy as np

# Peak numbers. Datasheet values; replace with measured peaks (see bench.measure_peak_bw).
HARDWARE = {
    "m4-pro":  {"bw_gbs": 273.0,  "tflops_fp16": 9.2,   "label": "Apple M4 Pro (20c GPU)"},
    "m4-max":  {"bw_gbs": 546.0,  "tflops_fp16": 18.4,  "label": "Apple M4 Max (40c GPU)"},
    "l4":      {"bw_gbs": 300.0,  "tflops_fp16": 121.0, "label": "NVIDIA L4"},
    "a100":    {"bw_gbs": 2039.0, "tflops_fp16": 312.0, "label": "NVIDIA A100 80GB SXM"},
    "h100":    {"bw_gbs": 3350.0, "tflops_fp16": 989.0, "label": "NVIDIA H100 SXM"},
}

BITS = {"float16": 16, "bfloat16": 16, "float32": 32, "q8": 8.5, "q4": 4.5}  # q*: MLX affine g64 incl. scale+bias


def expected_unique_experts(T, E, k):
    return E * (1.0 - (1.0 - k / E) ** T)


def flops(T, d, f, k):
    return 6.0 * d * f * T * k


def bytes_moved(T, d, f, E, k, w_bits=16, a_bytes=2, unique=None):
    U = expected_unique_experts(T, E, k) if unique is None else unique
    weights = U * 3.0 * d * f * w_bits / 8.0
    acts = 2.0 * T * d * a_bytes + 2.0 * T * k * f * a_bytes
    return weights + acts


def floor_time_s(T, d, f, E, k, hw, w_bits=16, unique=None):
    """Lower bound on runtime: max(memory time, compute time)."""
    h = HARDWARE[hw]
    t_mem = bytes_moved(T, d, f, E, k, w_bits, unique=unique) / (h["bw_gbs"] * 1e9)
    t_cmp = flops(T, d, f, k) / (h["tflops_fp16"] * 1e12)
    return max(t_mem, t_cmp)


def crossover_tokens(d, f, E, k, hw, w_bits=16):
    """Smallest T (power of 2 scan) where layer becomes compute-bound."""
    h = HARDWARE[hw]
    for T in np.unique(np.logspace(0, 16, 400, base=2).astype(int)):
        if flops(T, d, f, k) / (h["tflops_fp16"] * 1e12) >= bytes_moved(T, d, f, E, k, w_bits) / (h["bw_gbs"] * 1e9):
            return int(T)
    return None
