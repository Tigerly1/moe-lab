"""Baseline: PyTorch eager, loop over active experts (HF transformers style)."""
import torch

META = {"name": "torch_loop", "description": "eager per-expert loop (HF style)"}


def prepare(w_gate, w_up, w_down):
    return {"wg": w_gate, "wu": w_up, "wd": w_down}


def run(s, x, idx, wts):
    y = torch.zeros_like(x)
    for e in idx.unique().tolist():
        t, j = (idx == e).nonzero(as_tuple=True)
        xe = x[t]
        h = torch.nn.functional.silu(xe @ s["wg"][e].T) * (xe @ s["wu"][e].T)
        y.index_add_(0, t, (h @ s["wd"][e].T) * wts[t, j, None].to(x.dtype))
    return y
