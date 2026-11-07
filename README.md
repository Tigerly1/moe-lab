# moe-lab

A harness and live lab for optimizing Mixture-of-Experts layer kernels on Apple Silicon (MLX/Metal) and NVIDIA (Triton/CUDA).

## Quick start (Mac)
```bash
python3 -m venv .venv && .venv/bin/pip install mlx mlx-lm numpy fastapi "uvicorn[standard]" pytest torch modal
.venv/bin/python -m pytest -q tests          # correctness gate
.venv/bin/uvicorn server.app:app --port 8765  # open http://localhost:8765
```

## Headline result (M4 Pro, 2026-11-07, audit-corrected, AC power)
Production kernel: `moelab/variants/mlx/metal_q4_moe.py`, a dispatcher over `metal_q4_decode_v4` (T=1) and `metal_q4_tiles_v2` / `metal_q4_tiles_b` (T>1).

- **Plain decode, Qwen3-30B-A3B-4bit, mlx-lm:** 91.6 → **99.8 tok/s (+9.0%)**, with perplexity unchanged.
- **Speculative-verify / small-batch step** (full model, fixed text, chunk of 4 / 7 / 16 tokens): **+23% / +34% / +45%**.
- **MoE layer at T=1 vs real mlx-lm:**
  - **1.24×** on Qwen3, 1.10–1.12× on gpt-oss, 1.04× on Mixtral (slope timing).
  - Our kernel reads memory at **95–100% of the measured ~255 GB/s** hardware limit; MLX reaches 79% on Qwen3.
- **MoE layer at T>1, real routing traces:** 1.72× at T=8, 1.98× at T=16, 2.44× at T=32. Held-out shapes: Mixtral 1.95× at T=8, Qwen3-235B 1.27× at T=16. Consecutive tokens share ~45% of experts, and the kernel reads each expert once per tile.
- **MXFP4 (gpt-oss format):** `metal_mxfp4_moe` beats mlx-lm's MXFP4 path:
  - 1.16× (gpt-oss) and 1.26× (Qwen3) at T=1;
  - up to 2.42× at T=32;
  - held-out shapes (gpt-oss-120b, Mixtral, DeepSeek-V2-Lite) at 1.05–1.92×.
- **Mechanism at T=1:**
  - About half the gain is MLX's qmv fast path requiring K % 512 == 0, which Qwen3 (K=768) and gpt-oss (2880) miss.
  - The rest is fusion and full lane use.
- **Not a clock boost:** 10-minute sustained runs and battery vs AC leave the ratio unchanged. Other Apple chips are untested.
- **Corrections:**
  - An earlier 1.25× claim was inflated by an unfair baseline; an independent audit caught it.
  - Free-running speculative tok/s is confounded by draft acceptance, so we use the verify benchmark instead.
  - Details: [docs/RESULTS.md](docs/RESULTS.md). Visual summary: `docs/atlas/index.html`.

```bash
.venv/bin/python -m moelab.e2e --tokens 256 --rounds 3   # tok/s, stock vs variant
.venv/bin/python -m moelab.e2e --quality                 # PPL: stock vs variant vs fp32
```

## Lab workflow
1. Click a variant to open its source in the editor.
2. **Fork…** copies it under a new name.
3. Edit the Metal source or the launch code.
4. Press **Cmd+Enter**. This saves the file and benchmarks every checked variant. Results stream into the charts per token count T:
   - latency
   - GB/s
   - speedup vs. the baseline
   - correctness (✓ / ✗ WRONG)
5. Re-running a variant draws its previous run as a dashed line, so you can see the effect of each edit straight away.
6. Every result is appended to `runs/history.jsonl`.

## Variant contract
A variant is one file in `moelab/variants/<backend>/`:
```python
META = {"name": ..., "description": ...}
W_BITS = 4.5                                  # optional, for the roofline (default = dtype bits)
def prepare(w_gate, w_up, w_down) -> state     # untimed: quantize, re-layout
def run(state, x, idx, wts) -> y               # timed
def dequant(state) -> (wg, wu, wd)             # optional, reference weights for lossy formats
```
Shapes:
- `x [T,d]`
- `w_gate`, `w_up` `[E,f,d]`
- `w_down [E,d,f]`
- `idx [T,k]` int32
- `wts [T,k]` fp32
- `y [T,d]`

Routing (router GEMM + top-k) is benchmarked as a separate stage.

## Measurement
- **Correctness:** every T is compared with an fp32 reference, which is itself checked against fp64 numpy in the tests. Pass criterion: relative Frobenius error below the dtype tolerance.
- **Timing:** chained in-graph timing. Ten dependent calls (y feeds the next x) run per evaluation, and the median over reps is reported.
  - On the M4 Pro each `mx.eval` + sync costs about 130 µs, which would swamp decode kernels.
  - The per-call figure including sync is reported separately as `sync_us`.
- **Roofline:** `%roof` is the analytic floor `max(bytes/BW, FLOPs/peak)` divided by the measured time, where bytes are counted for the *actual* unique experts touched. A value above 100% means a bug.

## NVIDIA (Modal)
```bash
.venv/bin/modal setup
MOE_LAB_GPU=L4 .venv/bin/modal deploy moelab/remote_modal.py
```
In the UI, choose the backend "CUDA / Triton (Modal)". The lab sends the current editor source, so edits never need a redeploy.

`triton_fused_moe.py` has **not yet been run on a GPU**. Its first run will show whether it is correct.

## Layout
```
moelab/shapes.py         MoE shapes of real models (Qwen3, DeepSeek-V3, Mixtral, gpt-oss, ...)
moelab/roofline.py       analytic FLOPs / bytes / U(T) / floor
moelab/problem.py        routing generators, fp64 reference
moelab/mlx_backend.py    Apple: weights, reference, timing
moelab/torch_backend.py  NVIDIA: same API (CUDA events)
moelab/worker.py         isolated job runner (fresh process per run, NDJSON output)
moelab/remote_modal.py   remote GPU runner
moelab/e2e.py            end-to-end mlx-lm decode benchmark + quality check
moelab/compare.py        rigorous interleaved A/B with bootstrap CI
moelab/slope.py          per-call time as slope over chain length (removes fixed eval cost)
moelab/sustained.py      thermal/clock robustness (long nonstop A/B)
moelab/routing_trace.py  record real router top-k; routing_stats.py analyses U(T), skew, locality
autoresearch/            Karpathy-style agent loop: program.md (human), bench.py (read-only), ar_expert.py (agent)
moelab/variants/mlx/     baselines (dense_masked, gather_mm, gather_qmm_q4) + custom Metal
                         (metal_simd_gemv, metal_q4_gemv v1/v2/v3/v31, metal_q4_expert_gemv)
moelab/variants/cuda/    torch_loop, triton_fused_moe
server/app.py            FastAPI: edit / run / stream
web/index.html           Monaco editor + Chart.js
docs/RESEARCH.md         literature synthesis
docs/THESIS_PLAN.md      research questions, chapters, timeline
```
