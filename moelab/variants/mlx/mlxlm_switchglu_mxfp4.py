"""FAITHFUL baseline, MXFP4: real mlx-lm SwitchGLU quantized with mode="mxfp4" (E2M1 values,
E8M0 scale per 32, 4.25 bits/weight - the gpt-oss expert format) + Qwen3-style combine."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

META = {"name": "mlxlm_switchglu_mxfp4", "description": "real mlx-lm SwitchGLU, mxfp4"}
MODE, GS, BITS = "mxfp4", 32, 4
W_BITS = 4 + 8 / 32


def prepare(w_gate, w_up, w_down):
    E, f, d = w_gate.shape
    m = SwitchGLU(d, f, E)
    m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight = w_gate, w_up, w_down
    nn.quantize(m, mode=MODE)
    mx.eval(m.parameters())
    return {"m": m}


def dequant(s):
    m = s["m"]
    return tuple(mx.dequantize(p.weight, p.scales, mode=MODE) for p in (m.gate_proj, m.up_proj, m.down_proj))


def run(s, x, idx, wts):
    y = s["m"](x, idx)
    return (y * wts.astype(x.dtype)[..., None]).sum(axis=-2)
