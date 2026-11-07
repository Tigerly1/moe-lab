# autoresearch round 6: make the int4 tile kernel exact for any affine scale

Everything in `autoresearch/program_tiles.md` (round 3) still applies: rules, git keep/discard, simplicity, honesty, and the single editable file `moelab/variants/mlx/ar_tiles.py`, which is the production `metal_q4_tiles_v2` kernel. Differences:

1. **Correctness bug (latent).** The MMA paths (tiles of ≥3 slots, T ≥ 3) form dequantized weights, scales and biases in **half**. Affine scales below the half subnormal range (~6e-8) flush to zero, which gives a wrong output.
   - Real Qwen3-30B-A3B scales are ≥ 8.6e-6, so they are safe today. Other checkpoints may not be.
   - The bench gate now also runs `tests/test_int4_small_scales.py`, which uses weights × 1e-6 and x × 1e6. The current kernel **fails** it at T = 16 and T = 32. GEMV and scalar paths pass.
2. **Baseline speed**, measured with the gate bypassed: score **1.3140**, CI 0.041.
   - qwen3: T2 1.246, T4 1.433, T8 1.719, T16 1.977
   - olmoe: T2 1.031, T4 1.153, T8 1.189
   - deepseek: T2 1.110, T4 1.205, T8 1.340
   - The first kept commit must pass the gate. Its score must be ≥ 1.3140 − max(0.01, ci), meaning a correctness fix may cost at most noise.
   - After that, normal keep rules apply: score > best + max(0.01, ci), and no point regresses by more than 0.03.
3. **Proven idea from MXFP4 round 5** (`moelab/variants/mlx/metal_mxfp4_moe.py`, which is the same structure):
   - Run the MMA on the **raw integer codes** q ∈ [0, 15], which are exact in half, for one 64-wide scale group: P = Σ q·x.
   - Then apply in fp32: `C += s * P + b * Σx`, where Σx is the per-token sum of the activation group.
   - The K remap makes each scale group's MMA steps contiguous.
   - This removes half-precision scales and biases entirely, and it removes the per-weight `fma(q, s, b)` dequant.
   - Σx per (token, group) can come from an extra MMA with an all-ones A fragment, a cheap pre-pass, or reuse of the activation loads.
4. **Log and hand-off.**
   - Log every run to `autoresearch/results_r6.tsv` (untracked). Start it with a header and a baseline row marked `gate-fail`.
   - At the end, copy the best file to `autoresearch/ar_tiles_r6_best.py`, with the tsv next to it.
5. **Budget.** At most 20 experiments or 90 minutes.
