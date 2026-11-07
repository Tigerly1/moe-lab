# moe-lab

A harness and live lab for optimizing Mixture-of-Experts layer kernels on Apple Silicon (MLX/Metal) and NVIDIA (Triton/CUDA).

## Quick start (Mac)
```bash
python3 -m venv .venv && .venv/bin/pip install mlx mlx-lm numpy fastapi "uvicorn[standard]" pytest torch modal
.venv/bin/python -m pytest -q tests          # correctness gate
.venv/bin/uvicorn server.app:app --port 8765  # open http://localhost:8765
```

## When does it help? (M4 Pro, latest mlx-lm 0.32.2 + MLX 0.32.4, 2026-11-07, every run guard-checked)
Stock = unmodified mlx-lm. "Ours" = our expert kernels + fused router tail (`python -m moelab.generate`, or `e2e.py --fused-router`).

| situation | Qwen3-30B-A3B | gpt-oss-20b | DeepSeek-V2-Lite | worth it? |
|---|---|---|---|---|
| one chat, short context (tok/s) | 90 → **105 (1.17×)** | 74 → 81 (1.10×) | 124 → 138 (1.12×) | modest |
| one chat, 16k context | 1.07× (MoE part 1.27×; attention dominates) | — | — | small |
| **4 requests at once** (mlx-lm BatchGenerator, aggregate tok/s) | 136 → 167 (1.22×) | 129 → 160 (1.24×) | 192 → 220 (1.15×) | yes |
| **8 at once** | 167 → 204 (1.23×) | 138 → 197 (**1.43×**) | 209 → 265 (1.27×) | yes |
| **16 at once** | 189 → 252 (1.33×) | 167 → 259 (**1.55×**) | 257 → 326 (1.27×) | yes |
| **32 at once** | 222 → 352 (**1.59×**) | 224 → 359 (**1.60×**) | 312 → 481 (**1.54×**) | **yes, the main win** |
| **real `mlx_lm.server`, 32 concurrent HTTP clients** (`moelab.serve`, 256 tokens each) | 156 → 215 tok/s (**1.38×**), median reply 52 s → 38 s | 157 → 217 (**1.38×**), 52 s → 37 s | 199 → 262 (**1.32×**), 41 s → 31 s | **yes** |
| **vs llama.cpp** (`llama-server`, default settings, GGUF Q4_0 = 4.5 bits/weight like MLX 4-bit g64), Qwen3 only | 1 stream: llama.cpp 82 tok/s (llama-bench tg256), ours 105. Server, 8 clients: llama.cpp 144, stock MLX 121, ours 142. **32 clients: llama.cpp 150, ours 222 (1.48×), median reply 54.6 s → 36.7 s** | — | — | stock MLX trails llama.cpp at 8 clients; with our kernels MLX matches it at 8 and leads at 32 |
| same server, 8 clients | 1.16×, 17.0 s → 14.6 s | 1.35×, 19.3 s → 14.2 s | 1.18×, 14.0 s → 11.9 s | yes |
| verify step of 16 / 32 tokens (speculative decoding, chunked prefill) | 1.47× / 1.79× | 1.71× / 1.54× | 1.33× / 1.63× | yes, if the stack uses it |
| CPU fully busy (8 threads of other work) | 1.01–1.07× | — | — | no |

**Newer popular local models (round 9, 2026-11-07).** We picked these by downloads: they are the two most-downloaded mlx-community MoE families we did not support yet. Kernels + fused router, 2 launches each:

| situation | Gemma-4-26B-A4B 4-bit | Qwen3.6-35B-A3B 4-bit |
|---|---|---|
| one chat (chat prompt, 512 tokens) | 72.3 → 79.8 (**1.10×**) | 83.8 → 92.7 (**1.11×**) |
| 8 requests at once | 1.14× | 1.09× |
| 32 requests at once | 230 → 290 (**1.25×**) | 289 → 358 (**1.24×**) |

Below 1.5× at B=32: Qwen3.6 has 256 small experts (~1 token per expert at B=32), and Gemma-4's F=704 runs on scalar tiles. Details: `docs/ACHIEVEMENTS.md`.

- **Use it:** `.venv-latest/bin/python -m moelab.serve --model <mlx-community MoE repo> --decode-concurrency 32`. It is a drop-in `mlx_lm.server` with the kernels; set `MOELAB_STOCK=1` for the unpatched A/B. Server gains are below the BatchGenerator benchmark (1.3–1.4× vs 1.5–1.6×) because the server also spends time on prefill, tokenization and HTTP, which the kernels do not touch.
- **Where it pays off:** a Mac serving several requests at once. Examples: the `mlx_lm.server` with concurrent users, an agent issuing parallel calls, batch evaluation, synthetic-data generation. Each decode step then sends B tokens from B different sequences through every MoE layer.
- **Why the gain grows with B:** stock mlx-lm re-reads each expert's weights per (token, slot) until 64 slots, while our kernels group slots by expert and read each expert once per tile. This technique is standard on NVIDIA (vLLM fused_moe, llama.cpp map0 + mul_mm_id); here it is brought to MLX/Metal with measured serving gains (`docs/related_work_r8.md`).
- **Where it does not help:**
  - very long context: attention takes over (Amdahl);
  - a CPU-saturated machine;
  - many tokens per expert (load T·k/E > 8–24), where the dispatchers fall back to the stock path, bit-identical;
  - all k experts receiving every token (synthetic only, 0.76×);
  - experts smaller than any studied model (0.93–0.95×).
- **Correctness:**
  - 742 tests;
  - every path vs an fp32 reference, plus real-model multi-token logits;
  - in batched decode at B=32, our logits are closer to the fp32-expert truth than stock's (max per-row error 0.07 vs 0.19, `runs/r8/batch_truth_check.json`).
- **Output differences:** greedy text differs from stock after a few tokens. The cause is bf16 rounding plus a different tie-breaking between experts with exactly equal bf16 gate logits (9% of Qwen3 token-layers; both choices are valid). Perplexity is within 0.9% of stock on all three models.
- **Sources:**
  - single-stream and batch: `runs/r8/latest/`, same ratios on mlx-lm 0.31.3 in `runs/r8/batch_*` (2 launches each);
  - long context: `runs/r8/longctx_*_clean*.json`;
  - verify steps: `runs/r7/`;
  - CPU stress: `runs/r8/cpu*`.

## Headline result, night run (M4 Pro, 2026-11-07, after two independent audits, AC power)
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
- The decode gain shrinks with context length: Qwen3 +8.1% at 1k, +7.2% at 4k, +4.3% at 8k context (`runs/e2e_context_*.json`); per-step breakdown +9.0% at 1k, +3.8% at 8k, +1.2% at 16k (`runs/breakdown_qwen3.json`). Attention grows (Amdahl), and the MoE-only gain also falls from 1.18× to 1.04×. One hypothesis (KV streaming interference) was tested and rejected; the cause is open.
- The verify step is what speculative decoding and small-batch serving run. On this setup, free-running speculative decoding with a 0.6B draft is itself slower than plain decoding (stock 72–77 vs 91.5 tok/s).

**MoE layer vs the real mlx-lm path** (bf16; fresh seeds; CIs resampled by launch):

| | T=1 | T=8 (real traces) | T=16 (real traces) | T=32 (real traces) |
|---|---|---|---|---|
| int4 dispatcher, Qwen3 | 1.21 [1.20, 1.21] | 1.72 [1.67, 1.75] | 2.05 [1.89, 2.11] | 2.45 [2.27, 2.57] |
| MXFP4, Qwen3 | 1.26 [1.25, 1.27] | 1.73 [1.70, 1.76] | 2.07 [1.95, 2.10] | 2.40 [2.24, 2.51] |

The table values come from the reproduction run on the final kernels (`runs/repro/*_qwen3_trace.json`, seeds 5000+).

- **Crossover (RQ1), revised in round 7:** what decides the winner at T>1 is load = T·k/E (average tokens per expert), not T. The custom kernels win up to load 24 on Qwen3-30B-sized and larger experts (Qwen3 1.07–1.09× at T=192–384, Qwen3-235B 1.10–1.12×), up to load 8 on small experts, and up to load 4 on int4 d=2880. Above that, the dispatchers fall back to mlx-lm's own sorted `gather_qmm` path, which is bit-identical to the baseline (gpt-oss int4 at T=128 went 0.51× → 0.98–1.00×). Whole-model 32-token chunks: **1.83×**. Files: `runs/r7_policy/`, `runs/grid_*`, `runs/sweep_*_final*`.
- **Fused router tail (round 7):** one Metal kernel replaces mlx-lm's softmax → argpartition → take → renormalize. With it, plain decode goes from +11.4% to **+17.3%** on Qwen3, +7.7% to **+9.3%** on gpt-oss and +7.7% to **+12.1%** on DeepSeek (same-session runs, `runs/r7/`).
  - Caveat: in 9% of Qwen3 token-layers two experts tie exactly on the bf16 gate logit. Stock and the fused kernel break such ties differently (both are valid top-k), so greedy outputs differ from stock from token 2. Perplexity stays within 0.9%: a different tie-breaking, not a better model (`runs/r8/router_fix_qwen3_moe.json`, `runs/r7/*_quality_fused.json`).
- **Tested wide and narrow (round 7):** 480 random shapes and all edge cases are correct. Every screen flag was re-measured with `compare`. Remaining losses:
  - saturated skew (all k experts get every token, 0.76×), never seen in real traces;
  - two tiny-expert shapes (0.93×, 0.95×), smaller than any studied model.
  
  A GPU-contention guard flags runs disturbed by other processes. Details: `docs/RESULTS.md`, "Round 7".
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
