# Night log, 2026-11-07 (00:40 → 08:00, autonomous run)

Every number below is in `runs/*.json` and documented in `docs/RESULTS.md`. Times are CET.

| time | what | outcome |
|---|---|---|
| 00:40 | Moved the project to `~/Desktop/moe-thesis`; rebuilt the venv; local git | – |
| 00:45 | Autoresearch round 1 (expert reuse T>1) finished; **independent re-verification** | `tiles_mma`: T=8 1.64×, T=16 1.87×, T=32 2.36× vs mlx-lm (real traces) |
| 00:55 | Speculative decoding e2e (Qwen3-0.6B draft) | +16–22% vs stock; spec decoding itself slower than plain on this MoE |
| 01:05 | "M4 boost?" tests: battery vs AC, 10-minute sustained run | ratio unchanged (1.169 → 1.167); no throttling; other chips untested |
| 01:20 | Autoresearch round 2 (T=1 decode) | +1.5%; **slope timing**: kernels at 95–100% of measured ~255 GB/s |
| 01:45 | Production dispatcher `metal_q4_moe`; multi-token e2e test | plain decode **+9.2%** (99.9 tok/s) |
| 01:55 | Found the spec-decoding pitfall: diverging text changes draft acceptance | built `--verify-chunks` (acceptance-free) |
| 02:00 | Frozen quality corpus; JAX backend (`ragged_dot` + Pallas GMM, TPU-style scalar prefetch) | correct on CPU (interpret); ready for Colab TPU |
| 02:05 | Thesis drafts ch. 4–7 (Polish); figures from data (`moelab/figures.py`) | `docs/thesis/` |
| 02:30 | Autoresearch round 3 (tiles, 3 models) | +4%; verified on Qwen3-235B (clean held-out) and Mixtral (partially seen) |
| 02:40 | Mixtral puzzle solved | implicit SLC reuse when same-expert slots run concurrently (dispatch order) |
| 03:25 | Autoresearch round 4 (MXFP4, gpt-oss format) | score 1.40; exact FP4→half2 bit trick |
| 03:55 | gpt-oss-20b end-to-end (agent), **real-checkpoint bug found**: E8M0 ≥ 129 overflowed a half scale in the MMA path | fixed, regression test, honest re-measurement (−8–11% at T≥16) |
| 04:10 | Tests on **real checkpoint weights** (Qwen3 int4, gpt-oss MXFP4) | all regimes correct |
| 04:30 | Autoresearch round 5 (MXFP4 after fix) | fp32 per-group scale; second latent overflow (bytes ≥ 241) fixed; T=16 back to 2.00× |
| 04:35 | gpt-oss-20b e2e on the final kernel | plain **+7.9%**, verify-16 **+69.5%** |
| 04:40 | Second independent adversarial audit of all claims | no cheating; CI method, held-out labels and T=1 % corrected |
| 04:50 | Audit-2 fixes: hierarchical CI, fresh seeds, 8 windows, non-repeating slope, real-model T>1 test | point estimates hold; Qwen3 T=1 at 94% of floor |
| 05:00 | RQ1 crossover T=16..512 | wins up to T=256, parity at T=512; full-model 32-token chunk 1.86× (Qwen3), 1.52× (gpt-oss) |
| 05:10 | Thesis chapters 1–3 drafted (agent, 62 refs, ~45 verified) | `docs/thesis/` |
| 05:15 | 8-min thermal check on final dispatcher | ratio 1.209 → 1.211 |
| 05:20 | Tiny-scale int4 test written first: production MMA path failed it (latent half-precision hazard) | → round 6 |
| 05:30 | Autoresearch round 6: raw-code MMA + fp32 scale/bias, bf16 operands | exact for any scale; speed unchanged; full-model verify-16 +48%, chunk-32 +84% |
| 05:40 | Real routing traces for gpt-oss-20b | same ~45% adjacent-token locality as Qwen3 (new measurement) |

## What to look at first
1. `docs/atlas/index.html`, also published as a private artifact (link in the chat summary).
2. `README.md` "Headline result".
3. `docs/RESULTS.md`: the full record, including every correction.
4. `docs/thesis/`: drafts of chapters 1–7 (Polish), plus `fig/`.

## Decisions waiting for you
- Post the MLX issue #4509 comment? (draft: `docs/upstream/mlx_issue_4509_draft.md`)
- Log in to Modal (`.venv/bin/modal setup`) to unlock the NVIDIA half of the thesis.
- Find a second Apple chip (M1/M2/M3) to test portability.
- Make the repo public on GitHub, after a final review.
