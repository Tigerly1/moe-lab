# autoresearch round 2: plain decode (T=1) int4 MoE kernel, Apple M4 Pro

The human owns this file. The agent owns exactly one file: `moelab/variants/mlx/ar_decode.py`, which starts as `metal_q4_gemv_v31`.

## Metric
`.venv/bin/python autoresearch/bench_decode.py > run.log 2>&1` prints `score`.
- **What it measures:** the geometric-mean speedup at T=1 over the real mlx-lm path (bf16, int4 g64).
- **Shapes:** Qwen3-30B-A3B, OLMoE-1B-7B, DeepSeek-V2-Lite, gpt-oss-20b, Mixtral-8x7B.
- **Baseline:** score 1.1017. Per shape: qwen3 1.164, olmoe 1.069, dsv2lite 1.143, gpt-oss 1.104, mixtral 1.034. CI half-widths are about 0.005–0.009.

## Why there is headroom (measured)
- **Real memory bandwidth:** the M4 Pro reads at about 250 GB/s (a read-only kernel measured 240–250); copy reaches about 224 GB/s.
- **Qwen3 at T=1:** 8 experts × 2.65 MB = 21 MB, so the floor is about 85 µs at 250 GB/s. v31 takes about 115 µs (74%); MLX takes 133 µs.
- **Mixtral:** v31 is already at about 88% of the real peak.
- **So the gap is fixed cost, and it is worst for small experts.**
  - v31 issues 3 dispatches: gate_up+SiLU, down per slot (fp32 partials), and combine.
  - Each dispatch costs on the order of 5–15 µs of GPU time on this machine.
  - Tail effects come from small grids: Qwen3 gate_up has 768 rows × 8 slots of simdgroups, while down K=768 only uses 24 of 32 lanes for one step.

## Ideas, untested
- **Remove the combine dispatch.** Have the down kernel accumulate into y:
  - use `atomic_outputs=True` with fp32 atomics, or
  - assign one simdgroup per output row that loops over the k slots, with the partial dots of the k slots computed in parallel by different simdgroups of the same threadgroup and reduced in threadgroup memory. This avoids atomics and the k-serial latency of v1.
  - Check whether MLX's `init_value` costs a separate fill dispatch.
- **Down kernel at K=768:** use the full simdgroup. For example, 2 rows per simdgroup with 16 lanes each, or uint2 loads so 32 lanes × 24 = 768 values per step.
- **Better occupancy and latency hiding:** issue independent loads before use (prefetch the next uint4), unroll, and tune threadgroup size per kernel.
- **Gate_up:** experiment with rows-per-simdgroup and threadgroup shape per K, trading tail effects against occupancy.
- **Do not regress T>1:** the gate also checks T=2,3,8 correctness. Speed is only scored at T=1.

## Rules (adapted from karpathy/autoresearch)
1. **Edit only** `moelab/variants/mlx/ar_decode.py`.
   - Keep the `prepare/run/dequant` contract and `META`/`BITS`/`GS`/`W_BITS`.
   - Kernel names must keep the `ar_decode_` prefix.
   - No new dependencies.
   - Don't touch bench, compare, backend, tests or other variants.
2. **Git:** you are on branch `autoresearch/decode-t1`; the baseline is at HEAD.
   - Per experiment:
     1. Commit with `git commit -am "<idea>"`.
     2. Run the bench into `run.log`.
     3. Run `grep "^score:\|^status:\|^per_shape:\|^ci:" run.log`.
   - **Keep only if** status is ok AND score > best_kept + max(0.01, 2 × mean(ci of the candidate)) AND no shape regresses by more than 0.02 vs the best kept. This rule uses the candidate's own CI; it is fixed from round 1.
   - Otherwise run `git reset --hard HEAD~1`.
   - On a crash, look at `tail -n 60 run.log` and try at most 2 fixes.
3. **Log every run** to `autoresearch/results_decode.tsv` (tab-separated), with these columns:
   `commit score qwen3 olmoe dsv2lite gptoss mixtral status description`
4. **Simplicity:**
   - A gain under 3% that adds lots of code is not worth keeping.
   - Removing code at equal score is a win.
5. **Budget:** at most 30 experiments or 2.5 hours. Kill a bench run after 6 minutes.
   - Never ask the human.
   - If stuck, re-read the ideas list and combine near-misses.
6. **Honesty:**
   - No benchmark-specific special cases (shape-keyed hacks are fine only if they are general rules based on d/f/k).
   - No caching of outputs.
   - The kernel must be correct for any routing.

## Final report
- Best commit, score and per-shape results.
- The full `results_decode.tsv`.
- 3–5 lessons, each tied to a measurement.
- Before reporting, copy the best `ar_decode.py` to `autoresearch/ar_decode_best.py`.
