# autoresearch round 4: MXFP4 MoE expert kernels (Apple M4 Pro)

The human owns this file. The agent owns exactly one file: `moelab/variants/mlx/ar_mxfp4.py`.
It starts as a thin wrapper around the real mlx-lm MXFP4 path (score = 1.000).
Your job is to replace it with custom Metal kernels.

## Format: MLX `mode="mxfp4"`, the gpt-oss expert format
- `mx.quantize(w, mode="mxfp4")` returns `(wq uint32 [..., in/8], scales uint8 [..., in/32])`.
- Each uint32 packs 8 FP4 E2M1 codes, low nibble first. Verify the order yourself with `mx.dequantize`.
- E2M1 code → value: bit 3 is the sign, magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
- The scale is E8M0: value = 2^(s − 127), one per 32 weights. There is no bias, so the `b·Σx` term disappears.
- That is 4.25 bits per weight, against 4.5 for affine int4 g64, so the bandwidth floor is about 6% lower.
- The dequantized weights from `dequant()` are the correctness reference. They must match `mx.dequantize(..., mode="mxfp4")`.

## Metric
- Command: `.venv/bin/python autoresearch/bench_mxfp4.py > run.log 2>&1`. One run takes about 40 s.
- Score is the geometric mean of 7 speedups over the real mlx-lm MXFP4 SwitchGLU path, measured on:
  - gpt-oss-20b shapes (d=f=2880, E=32, k=4), uniform routing, T = 1, 2, 4, 8;
  - Qwen3-30B-A3B shapes, real routing traces, T = 1, 4, 16.
- Baseline: 0.9997, CI 0.003.
- Gate: unit tests, plus bf16 correctness at gpt-oss, Qwen3, Mixtral and OLMoE shapes with T ≤ 16, routing uniform, one_hot and trace. Requirement: rel-err < 1e-2.

## Starting knowledge (measured on the affine int4 path: docs/RESULTS.md and the kernels named below)
- **T = 1:** `metal_q4_decode_v4.py` reaches 95–100% of about 255–260 GB/s. Its tricks:
  - one simdgroup per gate/up row, gate+up+SiLU fused;
  - in down, LPR lanes per row with `simd_shuffle_xor` and the k slots reduced in threadgroup memory, so no combine dispatch;
  - uint4 loads.
- **T > 1:** `metal_q4_tiles_v2.py`:
  - a map prepass groups slots by expert into tiles of up to 8;
  - tiles of 3 or more use simdgroup 8×8 MMA with half dequant;
  - 2-slot tiles use a scalar path, single-slot tiles use GEMV.
  - At T ≥ 8 it is bound by MMA instruction count.
- **FP4 dequant ideas:** a 16-entry constant LUT, or `half` bit tricks. E2M1 → half is exact: build the half from the sign, exponent and mantissa bits and multiply by the power-of-two scale with `ldexp`, or by building the exponent directly.
- **Pitfalls:**
  - d = 2880 is not a multiple of 128, so the MMA superblocks need a tail path or a different blocking (2880 = 90 × 32).
  - MLX passes inputs with fewer than about 8 elements in `constant` address space, so template the helper pointer types.
  - Template constants are not visible inside `header` functions.
  - Kernels are cached by name: keep the `ar_mxfp4_` prefix and give each distinct kernel source a unique name.
- **Process tip:** start with T = 1 by porting v4 to FP4 decoding, then add tiles for T > 1.

## Rules (karpathy/autoresearch, adapted)
1. Edit only `moelab/variants/mlx/ar_mxfp4.py`.
   - Keep the contract: `prepare/run/dequant`, `META`, and `MODE`/`GS`/`BITS`/`W_BITS`.
   - No new dependencies.
   - Do not touch the bench, compare, backend, tests or any other file.
2. Git: you are on branch `autoresearch/mxfp4-r4` in this worktree. For each experiment:
   1. `git commit -am "<idea>"`
   2. Run the bench.
   3. `grep "^score:\|^status:\|^per_point:\|^ci:" run.log`
   - Keep the commit only if status is ok, score > best_kept + max(0.01, ci), and no point regresses by more than 0.03.
   - Otherwise `git reset --hard HEAD~1`.
   - A crash gets at most 2 fix attempts.
   - If a result is borderline, re-run it once.
3. Log every run to `autoresearch/results_mxfp4.tsv` (untracked, tab-separated): `commit score per_point status description`.
4. Simplicity: a gain under 3% that adds lots of code is not worth keeping. Deleting code at equal score counts as a win.
5. Budget: at most 30 experiments or 2.5 hours. Kill any bench run after 6 minutes. Never ask the human. Quick probes are fine.
6. Honesty:
   - No benchmark-specific special cases. Shape rules based on d, f, k and T are fine.
   - No caching of outputs.
   - The kernel must be correct for any routing.

## Final report
- Best commit, score and per-point table.
- The full results tsv.
- 3–5 measured lessons, including what FP4 decoding costs compared with affine int4.
- Copy the best `ar_mxfp4.py` to `autoresearch/ar_mxfp4_best.py`, and the tsv next to it.
