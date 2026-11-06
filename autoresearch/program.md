# autoresearch: expert-reuse int4 MoE decode kernel (Metal, Apple M4 Pro)

Adapted from karpathy/autoresearch. The human owns this file. The agent owns exactly one file.

## Goal
Maximize `score` printed by `autoresearch/bench.py`. Score is the geometric-mean speedup of
`moelab/variants/mlx/ar_expert.py` over the **real mlx-lm MoE path** on Qwen3-30B-A3B expert
shapes:
- d=2048, f=768, E=128, top-8, int4 affine group 64, bf16
- real routing traces (T consecutive tokens)
- T ∈ {1, 2, 4, 8, 16}

Starting point: ar_expert = `metal_q4_gemv_v31`, score ≈ 1.10.

## Why there is headroom (measured, docs/RESULTS.md)
With real routing, T consecutive tokens touch far fewer unique experts than T·k slots:

| T | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| unique experts U | ~12.5 | ~18.5 | ~28 | ~38 |
| slots T·k | 16 | 32 | 64 | 128 |

Both MLX and v31 re-read each expert's weights once per slot. Reading each active expert once
per step would cut weight traffic by about 2.3× at T=8 and 3.4× at T=16. The T=1 kernel is
already near measured bandwidth, so do not regress T=1.

## Rules
1. **Edit only** `moelab/variants/mlx/ar_expert.py`.
   - Never edit `autoresearch/bench.py`, `moelab/compare.py`, `moelab/mlx_backend.py`, tests, or other variants.
   - No new dependencies.
   - Keep the contract `prepare/run/dequant`, `META`, `BITS`, `GS`, `W_BITS`.
   - Kernel names must stay unique (prefix `ar_expert_`).
2. **Git:**
   - Work on branch `autoresearch/expert-reuse`.
   - First run: the unmodified baseline, so you measure its score.
   - Each experiment:
     1. Edit, then `git commit -am "<short idea>"`.
     2. Run `.venv/bin/python autoresearch/bench.py > run.log 2>&1`.
     3. Run `grep "^score:\|^status:\|^per_T:\|^max_ci" run.log`.
   - **Keep** the commit only if status is ok and score improves on the best kept score by more than `max(0.01, max_ci_width/2)`. Otherwise run `git reset --hard HEAD~1`.
   - On a crash, read `tail -n 60 run.log` and try at most 2 quick fixes, then discard.
3. **Log every run** to `autoresearch/results.tsv` (tab separated, not committed), header:
   `commit	score	T1	T2	T4	T8	T16	status	description`
4. **Simplicity:**
   - Keep a small gain that adds a lot of complexity only if the gain is ≥ 3%.
   - Removing code at equal score counts as a win.
5. **Budget:** at most 25 experiments, or 2.5 hours. Kill any single bench run after 6 min.
   - Never stop to ask the human whether to continue. When out of ideas, re-read the references below and combine near-misses.
6. **Honesty:** never special-case the benchmark.
   - Do not read the trace file in the kernel.
   - Do not cache outputs.
   - Do not depend on seeds.
   - The kernel must be correct for any routing; the gate checks uniform and one_hot too.

## Ideas, from research (docs/RESEARCH.md and the scout reports)
- **Segment by expert inside one step.**
  - Sort the slot list by expert id: `mx.argsort` costs about 1 dispatch.
  - Better: a tiny map0-style prepass kernel (one threadgroup) that builds (expert, start, count) tiles. This mirrors llama.cpp `kernel_mul_mm_id_map0`. Its grid is an upper bound on tiles, and spare threadgroups exit early.
- **Design B, register-broadcast wide GEMV.**
  - Each simdgroup owns R output rows of ONE expert and a tile of ≤ VT tokens routed to it.
  - Each lane loads a weight `uint4` once and applies it to all VT tokens.
  - Token activations are prescaled ONCE per token, not per row. Do this in a prepass kernel writing `xs [T, d]` fp32 plus group sums, or stage the tile's tokens in threadgroup memory shared by all simdgroups of the threadgroup.
  - Segments of 1 token degrade to today's v31 GEMV.
- **Design A, simdgroup_matrix 8×8 (MLX `affine_qmv_mma`, llama.cpp `mul_mv_mma`).**
  - Dequantize weights straight into A fragments (`as_type<half2>(q | 0x6400) - 1024`, exact), tokens as B fragments.
  - Split K across 4 simdgroups and reduce in threadgroup memory.
  - Worth it for segments of ≥ 3 tokens.
- **Previous failure:** expert-centric v4 (`metal_q4_expert_gemv.py`) was correct but slower.
  - One "leader" simdgroup did all the work while the others exited, so parallelism collapsed.
  - Activations were reloaded and prescaled per row per token, so ALU ballooned.
  - Fix both.
- **Keep:** fp16/bf16 `h`, down-proj parallel over (slot), fused combine kernel, few dispatches.
  - Every dispatch costs a few µs of GPU time.
- **Apple GPU facts:** SIMD 32, 32 KB threadgroup memory, no FP8/FP4.
  - `mx.fast.metal_kernel` inputs smaller than ~8 elements arrive in `constant` address space, so template the helper pointer types.
  - Template constants are not visible inside `header` functions; pass them as arguments.

## Output at the end
Print a final summary:
- best commit and score
- the per-T table of kept experiments
- 3 bullet lessons: what worked, what failed, and why
