# moe-lab

A harness and live lab for optimizing Mixture-of-Experts layer kernels on Apple Silicon (MLX/Metal) and NVIDIA (Triton/CUDA).

## Quick start (Mac)
```bash
python3 -m venv .venv && .venv/bin/pip install mlx mlx-lm numpy fastapi "uvicorn[standard]" pytest torch modal
.venv/bin/python -m pytest -q tests          # correctness gate
.venv/bin/uvicorn server.app:app --port 8765  # open http://localhost:8765
```

## Headline result (M4 Pro, 2026-11-07, after two independent audits, AC power)
Production kernels:
- `moelab/variants/mlx/metal_q4_moe.py` (int4 affine g64): v4 GEMV at T=1, expert-tile MMA at T>1;
- `metal_mxfp4_moe.py` (MXFP4, the gpt-oss format), plus `metal_mxfp4_gptoss.py` (adds biases and the clamped SwiGLU).

**End to end (mlx-lm, real models):**

| model | plain decode | verify step, 4 tokens | verify step, 16 tokens |
|---|---|---|---|
| Qwen3-30B-A3B-4bit | 91.6 → **99.8 tok/s (+9.0%)** | +23% | +45% |
| gpt-oss-20b-MXFP4 | 75.5 → **81.5 tok/s (+7.9%)** | +27% | +70% |

- Quality: perplexity is unchanged; the kernels are as close to fp32 as stock or closer.
- The verify step is what speculative decoding and small-batch serving run. On this setup, free-running speculative decoding with a 0.6B draft is itself slower than plain decoding (stock 72–77 vs 91.5 tok/s).

**MoE layer vs the real mlx-lm path** (bf16; fresh seeds; CIs resampled by launch):

| | T=1 | T=8 (real traces) | T=16 (real traces) | T=32 (real traces) |
|---|---|---|---|---|
| int4 dispatcher, Qwen3 | 1.20 [1.19, 1.21] | 1.75 [1.63, 1.83] | 2.02 [1.90, 2.13] | 2.46 [2.40, 2.53] |
| MXFP4, Qwen3 | 1.26 [1.25, 1.27] | 1.76 [1.68, 1.80] | 2.01 [1.90, 2.08] | 2.37 [2.32, 2.42] |

- **Crossover (RQ1):** our kernels win up to T=256 and reach parity with mlx-lm's sorted grouped GEMM at T=512. Whole-model 32-token chunks (short prefill or batch): **1.86×**.
- **Clean held-out shape (Qwen3-235B, uniform routing):** int4 1.06–1.26×, MXFP4 1.07–1.23×. Partially seen shapes: Mixtral up to 1.82× and DeepSeek-V2-Lite up to 1.42× (MXFP4).
- **T=1 is at the memory wall.** With non-repeating routing our GEMV reads at 94% (Qwen3), 100% (gpt-oss) and ~103% (Mixtral) of 250 GB/s. mlx-lm reaches 77% on Qwen3.
- **Mechanism at T=1:**
  - about half the gain comes from MLX's qmv fast path requiring K % 512 == 0, which Qwen3 (K=768) and gpt-oss (2880) miss;
  - the rest from fusion and full lane use.
- **Mechanism at T>1:**
  - consecutive tokens share ~45% of their experts (real routing traces);
  - the kernels group slots by expert and read each expert's weights once per tile (simdgroup 8×8 MMA);
  - at T≥8 they become compute-bound (~80% of the measured 6.2 TFLOPS MMA issue rate).
- **Not a clock boost:** 10-minute sustained runs and battery vs AC leave the ratio unchanged. Other Apple chips are untested.

**Corrections made along the way** (all documented in [docs/RESULTS.md](docs/RESULTS.md)):
- an unfair first baseline;
- activations decaying to zero in the timing chain;
- a speculative-decoding acceptance confound;
- two MXFP4 scale-overflow bugs found on the real gpt-oss checkpoint;
- CIs too narrow for trace routing;
- overstated "held-out" labels and bandwidth percentages.

Visual summary: `docs/atlas/index.html`. Night log: `docs/NIGHT_LOG.md`.

```bash
.venv/bin/python -m moelab.e2e --tokens 256 --rounds 3   # tok/s, stock vs variant
.venv/bin/python -m moelab.e2e --quality                 # PPL: stock vs variant vs fp32
```

## Use the kernels
```bash
.venv/bin/python -m moelab.generate --model mlx-community/Qwen3-30B-A3B-4bit --prompt "Hello" --compare
.venv/bin/python -m moelab.generate --model mlx-community/gpt-oss-20b-MXFP4-Q8 --prompt "Hello" --compare
```
The first call JIT-compiles the Metal kernels (~0.4 s cold start).

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
