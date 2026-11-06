"""FAITHFUL baseline: the real mlx-lm module (SwitchGLU -> QuantizedSwitchLinear, compiled
swiglu) plus the Qwen3 combine `(y * scores[..., None]).sum(-2)` with scores in model dtype.
This is exactly what mlx-lm runs at decode. Use this, not gather_qmm_q4, for speedup claims.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

META = {"name": "mlxlm_switchglu_q4", "description": "real mlx-lm SwitchGLU (4-bit g64) + Qwen3 combine"}
BITS, GS = 4, 64
W_BITS = BITS + 32 / GS


def prepare(w_gate, w_up, w_down):
    E, f, d = w_gate.shape
    m = SwitchGLU(d, f, E)
    m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight = w_gate, w_up, w_down
    nn.quantize(m, group_size=GS, bits=BITS)
    mx.eval(m.parameters())
    return {"m": m}


def dequant(s):
    m = s["m"]
    return tuple(mx.dequantize(p.weight, p.scales, p.biases, group_size=GS, bits=BITS)
                 for p in (m.gate_proj, m.up_proj, m.down_proj))


def run(s, x, idx, wts):
    y = s["m"](x, idx)                                   # [T, k, d]
    return (y * wts.astype(x.dtype)[..., None]).sum(axis=-2)
