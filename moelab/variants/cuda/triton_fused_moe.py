"""Triton fused MoE, simplified vLLM `fused_moe` design. EDIT ME.

1. align: sort (token, slot) pairs by expert, pad each expert segment to BLOCK_M
   -> sorted_ids (padding = T*k sentinel), expert of each M-block.
2. GEMM1 on [gate; up] stacked weights with implicit gather of x rows, -> SiLU*up (torch).
3. GEMM2 down-proj, routed weight multiplied in epilogue, scattered to [T*k, d]; sum over k.
Needs CUDA + triton. Not runnable on the Mac (run via Modal backend).
"""
import torch
import triton
import triton.language as tl

META = {"name": "triton_fused_moe", "description": "vLLM-style fused_moe (align + grouped GEMM)"}
BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 16, 64, 64, 8


@triton.jit
def moe_gemm(a_ptr, b_ptr, c_ptr, w_ptr, sorted_ids, block_expert, n_valid,
             N, K, stride_am, stride_be, stride_bn, stride_cm, top_k,
             MUL_W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(tl.load(n_valid), BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * num_n)
    first_m = group * GROUP_M
    gsize = min(num_m - first_m, GROUP_M)
    if gsize <= 0:
        return
    pid_m = first_m + (pid % gsize)
    pid_n = (pid % (GROUP_M * num_n)) // gsize

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = tl.load(sorted_ids + offs_m)                 # flat slot id = t*k + j, or sentinel
    valid = rows < tl.load(n_valid + 1)                 # n_valid[1] = T*k
    e = tl.load(block_expert + pid_m)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_rows = rows // top_k                              # GEMM1 (top_k=k) gathers x[t]; GEMM2 (top_k=1) reads h[slot]
    a = a_ptr + a_rows[:, None] * stride_am + offs_k[None, :]
    b = b_ptr + e * stride_be + offs_n[None, :] * stride_bn + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k < K - k0 * BLOCK_K
        av = tl.load(a, mask=valid[:, None] & kmask[None, :], other=0.0)
        bv = tl.load(b, mask=kmask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(av, bv)
        a += BLOCK_K
        b += BLOCK_K
    if MUL_W:
        acc *= tl.load(w_ptr + rows, mask=valid, other=0.0)[:, None]
    c = c_ptr + rows[:, None] * stride_cm + offs_n[None, :]
    tl.store(c, acc.to(c_ptr.dtype.element_ty), mask=valid[:, None] & (offs_n[None, :] < N))


def align(idx, E):
    flat = idx.flatten()
    n = flat.numel()
    counts = torch.bincount(flat, minlength=E)
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    starts = torch.cumsum(padded, 0) - padded
    total = int(padded.sum())                          # host sync; fine for a first version
    order = torch.argsort(flat, stable=True)
    rank = torch.arange(n, device=idx.device) - (torch.cumsum(counts, 0) - counts)[flat[order]]
    sorted_ids = torch.full((total + BLOCK_M,), n, dtype=torch.int32, device=idx.device)
    sorted_ids[starts[flat[order]] + rank] = order.to(torch.int32)
    block_expert = torch.repeat_interleave(torch.arange(E, device=idx.device), padded // BLOCK_M).to(torch.int32)
    n_valid = torch.tensor([total, n], dtype=torch.int32, device=idx.device)
    return sorted_ids, block_expert, n_valid, total


def _gemm(a, w, sorted_ids, block_expert, n_valid, total, out, top_k, wts=None):
    E, N, K = w.shape
    grid = (triton.cdiv(total, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    moe_gemm[grid](a, w, out, wts if wts is not None else out, sorted_ids, block_expert, n_valid,
                   N, K, a.stride(0), w.stride(0), w.stride(1), out.stride(0), top_k,
                   MUL_W=wts is not None, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)


def prepare(w_gate, w_up, w_down):
    return {"w13": torch.cat([w_gate, w_up], dim=1).contiguous(), "wd": w_down.contiguous(), "f": w_gate.shape[1]}


def run(s, x, idx, wts):
    T, k = idx.shape
    E, f = s["w13"].shape[0], s["f"]
    sid, bexp, nval, total = align(idx, E)
    gu = torch.empty((T * k, 2 * f), device=x.device, dtype=x.dtype)
    _gemm(x, s["w13"], sid, bexp, nval, total, gu, top_k=k)
    h = (torch.nn.functional.silu(gu[:, :f].float()) * gu[:, f:].float()).to(x.dtype)
    out = torch.empty((T * k, x.shape[1]), device=x.device, dtype=x.dtype)
    _gemm(h, s["wd"], sid, bexp, nval, total, out, top_k=1, wts=wts.flatten().float())
    return out.view(T, k, -1).sum(1)
