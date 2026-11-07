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

| model | plain decode | verify step, 4 tokens | verify step, 16 tokens | chunk of 32 tokens |
|---|---|---|---|---|
| Qwen3-30B-A3B-4bit | 91.5–91.7 → **99.7–99.8 tok/s (+8.6–9.0%)** | +23% | +48% | +83% |
| gpt-oss-20b-MXFP4 | 75.2–75.3 → **81.3 tok/s (+8.0–8.1%)** | +27% | +69% | +52% |
| DeepSeek-V2-Lite-4bit (shared experts, MLA) | 126.3–126.4 → **135.5 tok/s (+7.2–7.3%)** | +13.6% | +33.4% | +63.3% |

Every cell was reproduced on the final code in one clean run of `scripts/reproduce_headline.sh` (2026-11-07 06:22–06:33, seeds 5000+, output in `runs/repro/`). Ranges cover this run and the earlier 05:46 run.

- Quality: perplexity is within 0.1% of stock on all three models. Top-1 agreement with an fp32-expert reference is as high as stock or higher. On DeepSeek our perplexity is 0.1% further from fp32 than stock (41.40 vs 41.35; fp32 41.26). Files: `runs/e2e_dsv2_quality.json`; Qwen3 and gpt-oss in `docs/RESULTS.md`.
- Plain-decode greedy outputs are identical to stock only on DeepSeek (225/225 tokens). On Qwen3 and gpt-oss they diverge after ~22 tokens (bf16 noise); both arms still decode the same number of tokens.
- The decode gain shrinks with context length (Amdahl's law: attention grows): Qwen3 +8.1% at 1k, +7.2% at 4k, +4.3% at 8k context (256 tokens × 2 rounds, `runs/e2e_context_*.json`).
- The verify step is what speculative decoding and small-batch serving run. On this setup, free-running speculative decoding with a 0.6B draft is itself slower than plain decoding (stock 72–77 vs 91.5 tok/s).

**MoE layer vs the real mlx-lm path** (bf16; fresh seeds; CIs resampled by launch):

| | T=1 | T=8 (real traces) | T=16 (real traces) | T=32 (real traces) |
|---|---|---|---|---|
| int4 dispatcher, Qwen3 | 1.21 [1.20, 1.21] | 1.72 [1.67, 1.75] | 2.05 [1.89, 2.11] | 2.45 [2.27, 2.57] |
| MXFP4, Qwen3 | 1.26 [1.25, 1.27] | 1.73 [1.70, 1.76] | 2.07 [1.95, 2.10] | 2.40 [2.24, 2.51] |

The table values come from the reproduction run on the final kernels (`runs/repro/*_qwen3_trace.json`, seeds 5000+).

- **Crossover (RQ1):** our kernels win up to T=256 (1.09–1.10×). At T=512 they are 1–2% slower than mlx-lm's sorted grouped GEMM, so e2e uses stock above 256 tokens. Whole-model 32-token chunks (short prefill or batch): **1.83×** (`runs/crossover_r6_*`, `runs/e2e_prefill_chunks_qwen3_r6.json`).
- **Clean held-out shape (Qwen3-235B, uniform routing):** int4 1.06–1.26×, MXFP4 1.07–1.23×. Partially seen shapes: Mixtral up to 1.82× and DeepSeek-V2-Lite up to 1.42× (MXFP4).
- **T=1 is at the memory wall.** With non-repeating routing our GEMV reads at 93% (Qwen3), 100% (gpt-oss) and ~103% (Mixtral) of 250 GB/s. mlx-lm reaches 77% on Qwen3.
- **Mechanism at T=1:**
  - about half the gain comes from MLX's qmv fast path requiring K % 512 == 0, which Qwen3 (K=768) and gpt-oss (2880) miss;
  - the rest from fusion and full lane use.
- **Mechanism at T>1:**
  - consecutive tokens share 41–50% of their experts in Qwen3 and gpt-oss, but only 18–29% in DeepSeek-V2-Lite (real routing traces);
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
