# moe-lab

A harness and live lab for optimizing Mixture-of-Experts layer kernels on Apple Silicon (MLX/Metal) and NVIDIA (Triton/CUDA).

## Quick start (Mac)
```bash
python3 -m venv .venv && .venv/bin/pip install mlx mlx-lm numpy fastapi "uvicorn[standard]" pytest torch modal
.venv/bin/python -m pytest -q tests          # correctness gate
.venv/bin/uvicorn server.app:app --port 8765  # open http://localhost:8765
```

## Headline result (M4 Pro, 2026-11-07, audit-corrected)
- **Layer level:** the fused int4 Metal kernel `metal_q4_gemv_v31` beats the **real mlx-lm MoE path** at decode (T=1, bf16):
  - **1.17×** [95% CI 1.16–1.18] on Qwen3-30B-A3B experts
  - 1.12× on gpt-oss-20b
  - 1.03× on Mixtral
- **Why:** about half the gain comes from MLX's qmv fast path requiring K % 512 == 0, which Qwen3 (K=768) and gpt-oss (K=2880) miss; the other half from fusion and fewer dispatches.
- **End to end:** Qwen3-30B-A3B-4bit decode goes from 85.3 to **91.3 tok/s (+7.1%)** in mlx-lm, with perplexity unchanged.
- **Real routing:** T consecutive tokens touch about half the experts the uniform formula predicts. This is the next target (`autoresearch/`).
- An earlier 1.25× claim was inflated by an unfair baseline; it was found by an independent audit and corrected. Details: [docs/RESULTS.md](docs/RESULTS.md).

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
