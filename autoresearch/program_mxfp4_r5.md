# autoresearch round 5: MXFP4 after the scale-overflow fix

Everything in `autoresearch/program_mxfp4.md` still applies: format, metric, rules, honesty, and the single editable file `moelab/variants/mlx/ar_mxfp4.py`. Differences:

1. **Starting point.** `ar_mxfp4.py` is the round-4 winner plus a correctness fix.
   - **The bug:** the T≥8 MMA path stored the E8M0 group scale, with 2^14 folded in, as a single half. That overflows to inf for scale bytes ≥ 129. About 11% of real gpt-oss down_proj groups have such bytes.
   - **The fix:** `hs2()` splits the scale into two exact half factors, which costs two multiplies.
   - **What it cost:** 8–11% at T≥16 (Qwen3 T16 2.06 → 1.89, gpt T16 1.69 → 1.56).
   - **Baseline now:** score **1.3765**. Per point: gptT1 1.163, gptT2 1.158, gptT4 1.224, gptT8 1.504, qwen3T1 1.262, qwen3T4 1.587, qwen3T16 1.887. CI 0.04.
2. **Gate.** It now also runs `tests/test_mxfp4_large_scales.py`, which forces E8M0 bytes ≥ 129 in every regime. Never let real-checkpoint scales overflow again.
3. **Goal.** Recover and beat the pre-fix T≥8 speed while staying exact for every E8M0 byte 0..254. Ideas, all untested:
   - Apply the group scale in fp32 to the accumulator. MMA accumulates per K-block, and a scale group is 32 values, so the block structure can line up with groups.
   - Pick a per-row or per-tile power-of-two pre-scale that keeps half values in range, and undo it on the fp32 output.
   - Fold the 2^-14 into the activations (the B fragment) instead of the weights, if B stays within half range.
   - Use bf16 fragments if `simdgroup_matrix<bfloat>` is supported and fast on M4 (exponent range like fp32). Measure first.
4. **Logging and handoff.**
   - Log to `autoresearch/results_mxfp4_r5.tsv`, untracked, starting with the baseline row.
   - At the end, copy the best file to `autoresearch/ar_mxfp4_r5_best.py`, with the tsv next to it.
5. **Budget.** At most 25 experiments or 2 hours.
