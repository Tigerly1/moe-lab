# autoresearch round 3: T>1 expert-reuse int4 MoE kernel (Apple M4 Pro)

The human owns this file. The agent owns exactly one file: `moelab/variants/mlx/ar_tiles.py`, which starts as `metal_q4_tiles_mma`, the round-1 winner.

## Metric
- Run: `.venv/bin/python autoresearch/bench_tiles.py > run.log 2>&1`, about 45 s.
- **score** = geometric mean of 10 speedups over the real mlx-lm path (bf16, int4 g64), taken at these points:

  | model | routing | T |
  |---|---|---|
  | Qwen3-30B-A3B | real traces | 2, 4, 8, 16 |
  | OLMoE-1B-7B | uniform | 2, 4, 8 |
  | DeepSeek-V2-Lite | uniform | 2, 4, 8 |

- **Baseline:** score 1.2639, mean CI half-width 0.039.
  - qwen3: T2=1.201, T4=1.350, T8=1.619, T16=1.852
  - olmoe: T2=1.032, T4=1.137, T8=1.144
  - deepseek: T2=1.072, T4=1.166, T8=1.275

## What we know (measured; see docs/RESULTS.md)
- **T=1 GEMV is at the bandwidth ceiling.** `metal_q4_decode_v4` reaches ~95–100% of ~255 GB/s. Its main trick is that the down kernel spreads each row over LPR lanes with `simd_shuffle_xor`, so all 32 lanes are busy at K=768, and it reduces the k slots in threadgroup memory with no combine dispatch. Read it: `moelab/variants/mlx/metal_q4_decode_v4.py`.
- **ar_tiles at T>1:**
  - Builds expert tiles of up to 8 slots in a map prepass.
  - Runs simdgroup 8×8 MMA on tiles with ≥2 slots, using the exact `q|0x6400` half dequant and 128-wide K superblocks.
  - Single-slot tiles use the v31 GEMV inside the same threadgroup.
  - It does about 6 dispatches.
- **Uniform routing (OLMoE, DeepSeek at T=2–8)** has few tiles with ≥2 slots. Most work there is single-slot GEMV, so the v4 down-kernel tricks (LPR lanes, no combine) should transfer.
- **Dispatch cost per kernel is small** (GPU overlaps launches), so fusion for its own sake did not pay off in round 2. Use measurements, not intuition.

## Rules (karpathy/autoresearch, adapted)
1. Edit **only** `moelab/variants/mlx/ar_tiles.py`.
   - Keep the contract (`prepare/run/dequant`, `META`, `BITS`, `GS`, `W_BITS`).
   - Keep the `ar_tiles_` kernel-name prefix, with unique names per distinct kernel.
   - No new dependencies.
   - Do not touch the bench, compare, backend, tests or any other variant.
2. **Git:** this worktree is on branch `autoresearch/tiles-r3`, with HEAD = baseline.
   - For each experiment:
     1. `git commit -am "<idea>"`
     2. Run the bench.
     3. `grep "^score:\|^status:\|^per_point:\|^ci:" run.log`
   - **Keep** only if all three hold:
     - status is ok;
     - score > best_kept + max(0.01, ci), where ci is the candidate's own mean half-width;
     - no point regresses by more than 0.03 vs best kept.
   - Otherwise `git reset --hard HEAD~1`.
   - On a crash, read `tail -n 60 run.log` and make at most 2 fix attempts.
   - If a candidate looks better but falls inside the threshold, re-run the bench once more. Keep it if both runs clear the threshold on average.
3. **Log every run** to `autoresearch/results_tiles.tsv` (untracked, tab-separated) with columns: `commit score per_point status description`.
4. **Simplicity:** a <3% gain that adds lots of code is not worth it. Deleting code at equal score is a win.
5. **Budget:** ≤ 30 experiments or 2.5 h. Kill any bench run after 6 min.
   - Never ask the human.
   - Quick timing probes of single kernels are allowed when needed to learn why something failed.
6. **Honesty:**
   - No benchmark-specific special cases. Shape rules based on d/f/k/T are fine.
   - No output caching.
   - The kernel must be correct for any routing (the gate checks uniform, one_hot and trace, T up to 32).

## Final report
- Best commit, score, per-point table.
- Full `results_tiles.tsv`.
- 3–5 measured lessons.
- Copy the best `ar_tiles.py` to `autoresearch/ar_tiles_best.py`, and `results_tiles.tsv` next to it.
