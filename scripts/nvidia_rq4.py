"""RQ4 on NVIDIA via Modal: does the decode-regime picture from Apple (bandwidth floor at small T, grouping by
expert paying off as T*k/E grows) hold on an NVIDIA GPU?

Per shape (Qwen3-30B-A3B, Gemma-4-26B-A4B, Qwen3.6-35B-A3B expert layers, bf16) and T = 1..64, with distinct
uniform routing per call (no L2 reuse across calls):
  - correctness of every arm vs an fp32 per-expert reference;
  - time of: per-expert torch loop (HF style), our Triton vLLM-style fused_moe, and vLLM's own
    `fused_experts` when it imports;
  - % of the bytes floor (unique experts actually read / measured copy bandwidth).

    .venv/bin/modal run scripts/nvidia_rq4.py --gpu L4 --out runs/nvidia/rq4_L4.json
"""
import json
import os

import modal

app = modal.App("moe-lab-rq4")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "triton", "numpy", "vllm")
         .add_local_python_source("moelab"))

SHAPES = {"qwen3-30b-a3b": (2048, 768, 128, 8), "gemma-4-26b-a4b": (2816, 704, 128, 8),
          "qwen3.6-35b-a3b": (2048, 512, 256, 8)}
TS = [1, 2, 4, 8, 16, 32, 64]


def _bench(gpu_name):
    import importlib.util

    import numpy as np
    import torch

    def load(path):
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    root = os.path.dirname(importlib.util.find_spec("moelab").origin)
    arms = {"torch_loop": load(f"{root}/variants/cuda/torch_loop.py"),
            "triton_fused_moe": load(f"{root}/variants/cuda/triton_fused_moe.py")}
    try:
        from vllm.model_executor.layers.fused_moe import fused_experts
    except Exception as e:  # noqa: BLE001
        fused_experts, vllm_err = None, repr(e)[:300]
    else:
        vllm_err = None

    dev = torch.device("cuda")
    # copy bandwidth (read + write of 1 GiB), the practical ceiling
    a = torch.empty(1 << 29, dtype=torch.bfloat16, device=dev)
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(10):
        b.copy_(a)
    ev1.record()
    torch.cuda.synchronize()
    bw = 2 * a.numel() * 2 * 10 / (ev0.elapsed_time(ev1) / 1e3)
    del a, b

    def timeit(fn, n):
        for i in range(5):
            fn(i)
        torch.cuda.synchronize()
        ts = []
        for _ in range(5):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for i in range(n):
                fn(i)
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) / 1e3 / n)
        return float(np.median(ts))

    rows = []
    for name, (D, F, E, K) in SHAPES.items():
        g = torch.Generator(device=dev).manual_seed(0)
        wg, wu = (torch.randn(E, F, D, device=dev, generator=g, dtype=torch.bfloat16) * 0.02 for _ in range(2))
        wd = torch.randn(E, D, F, device=dev, generator=g, dtype=torch.bfloat16) * 0.02
        states = {k: m.prepare(wg, wu, wd) for k, m in arms.items()}
        w13 = torch.cat([wg, wu], dim=1).contiguous()
        per_expert = 3 * D * F * 2  # bf16 bytes
        rng = np.random.default_rng(0)
        for T in TS:
            N = 40
            idxs = [torch.tensor(np.stack([rng.choice(E, K, replace=False) for _ in range(T)]), device=dev,
                                 dtype=torch.int32) for _ in range(N)]
            xs = [torch.randn(T, D, device=dev, dtype=torch.bfloat16) for _ in range(N)]
            wts = torch.full((T, K), 1.0 / K, device=dev, dtype=torch.float32)
            U = float(np.mean([len(np.unique(i.cpu().numpy())) for i in idxs]))
            floor = U * per_expert / bw
            # fp32 reference for call 0
            x0, i0 = xs[0].float(), idxs[0].long()
            ref = torch.zeros(T, D, device=dev)
            for t in range(T):
                for j in range(K):
                    e = i0[t, j]
                    h = torch.nn.functional.silu(wg[e].float() @ x0[t]) * (wu[e].float() @ x0[t])
                    ref[t] += wts[t, j] * (wd[e].float() @ h)
            row = {"shape": name, "T": T, "unique_experts": U, "floor_us": floor * 1e6, "gpu": gpu_name}
            fns = {k: (lambda i, m=m, s=states[k]: m.run(s, xs[i % N], idxs[i % N], wts)) for k, m in arms.items()}
            if fused_experts is not None:
                fns["vllm_fused_experts"] = lambda i: fused_experts(xs[i % N], w13, wd, wts, idxs[i % N])
            for k, fn in fns.items():
                try:
                    y = fn(0).float()
                    err = float((y - ref).norm() / ref.norm())
                    t = timeit(fn, N)
                    row[k] = {"us": t * 1e6, "pct_floor": 100 * floor / t, "rel_err": err}
                except Exception as e:  # noqa: BLE001
                    row[k] = {"error": repr(e)[:300]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    return {"gpu": gpu_name, "copy_bw_GBs": bw / 1e9, "vllm_error": vllm_err, "rows": rows}


@app.function(gpu="L4", image=image, timeout=3600)
def bench_l4():
    return _bench("L4")


@app.function(gpu="H100", image=image, timeout=3600)
def bench_h100():
    return _bench("H100")


@app.local_entrypoint()
def main(gpu: str = "L4", out: str = "runs/nvidia/rq4_L4.json"):
    res = (bench_h100 if gpu.upper() == "H100" else bench_l4).remote()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"copy bandwidth {res['copy_bw_GBs']:.0f} GB/s, vllm: {res['vllm_error'] or 'ok'} -> {out}")
