# Thesis plan

**Title (PL):** Optymalizacja kerneli warstwy Mixture-of-Experts w inferencji modeli językowych na architekturach Apple Silicon i NVIDIA

## Research questions
- **RQ1 — Regime crossover.** For T = 1…4096, which strategy wins?
  - dense-masked
  - gather-GEMV (one simdgroup/warp per row)
  - sorted grouped GEMM

  Does the crossover T follow each architecture's ridge point? Hypothesis: an analytic model predicts it within ~20% on the M4 Pro and on NVIDIA.
- **RQ2 — Quantized decode on Metal.** How close can a fused int4 MoE GEMV get to 273 GB/s? Baseline: `gather_qmm` reaches ~48% of roofline at T=1. Compare with Marlin-MoE on NVIDIA.
- **RQ3 — Fusion value.** What does each fusion buy, per regime and per architecture?
  - gate+up+SwiGLU
  - down + combine
  - GPU-side align
- **RQ4 — Portability.** Do tile configs and kernel structure transfer between Metal and Triton? Do automatically found variants transfer?
- **RQ5 — Automated search (optional, high DeepMind value).** Grid vs BayesOpt vs an LLM mutator over the same evaluator, reporting:
  - fast_p
  - evaluations spent
  - reward hacks caught

## Chapters (~70 pp)
1. Introduction (5)
2. Background: MoE, routing, GPU architectures (SM vs Apple GPU, unified memory), roofline (15)
3. Related work (10)
4. Analytic model of the MoE layer (8)
5. Harness and kernels (15)
6. Evaluation, one section per RQ (15–20)
7. Threats to validity (5)
8. Conclusions (3)
- Appendix: reproducibility

## Timeline (6 months)
| month | work |
|---|---|
| 1 | harness ✅ (started), baselines, analytic model validation on M4 Pro |
| 2 | Metal kernels: fused fp16 GEMV, int4 fused GEMV, sorted grouped simdgroup_matrix GEMM |
| 3 | Triton/CUDA counterparts on Modal (L4 → H100), vLLM fused_moe + Marlin as baselines |
| 4 | Quant formats (int4, MXFP4), end-to-end decode with mlx-lm (Qwen3-30B-A3B), regime dispatcher |
| 5 | Full sweeps with statistics, profiling (Xcode capture, Nsight Compute), workshop paper draft |
| 6 | Writing, open-source release, blog post |

## Visibility for DeepMind
- **Public repo:** one command per figure, CI correctness tests.
- **Upstream PR to MLX:** a faster quantized MoE path, or tuning the `gather_qmm_rhs` thresholds.
- **Small JAX port:**
  - `jax.lax.ragged_dot` + a Pallas grouped GEMM run on Colab/Kaggle TPU
  - a portability section comparing TPU, H100 and M4
- **Write-up:** blog post in scaling-book style, then arXiv and a workshop (ES-FoMo @ ICML, MLSys workshops, EuroMLSys).

## Status (2026-11-07, morning)
| RQ | status | evidence |
|---|---|---|
| RQ1 regime crossover | **partial** | T-sweeps on 3 shapes; dispatcher picks the GEMV kernel at T=1 and tiles at T>1. Crossover vs `gather_qmm` sorted path (T≥64) not mapped yet |
| RQ2 quantized decode on Metal | **done (Apple)** | 95–100% of measured bandwidth at T=1; 1.24× vs mlx-lm on Qwen3; mechanism = K%512 cliff + fusion |
| RQ3 fusion value | **done (Apple)** | dispatch count matters little (GPU overlaps launches). Lane utilisation and reading each expert once matter a lot (T>1: up to 2.36×) |
| RQ4 portability | **started** | JAX/Pallas port correct on CPU (interpret mode). NVIDIA (Triton/Modal) and TPU runs not done; a second Apple chip not tested |
| RQ5 automated search | **done (3 rounds)** | autoresearch loops: round 1 found the MMA tile kernel; round 2 hit the bandwidth ceiling; round 3 running. Lessons on keep rules and noise |

New findings worth a section or a workshop paper:
1. Real-routing locality in Qwen3. Adjacent tokens share 43–47% of experts (7× random). U(T) is about half the uniform model.
2. Free-running speculative-decoding tok/s is confounded by acceptance. A verify benchmark fixes this.
3. Slope timing removes the fixed per-eval cost of MLX, which plain chained timing hides.

## Next concrete steps
1. **NVIDIA:**
   - `modal setup`, then deploy, then run `triton_fused_moe` (correctness first).
   - Port the expert-tile idea to Triton and compare against vLLM `fused_moe` and Marlin-MoE.
2. **TPU:**
   - Run `python -m moelab.jax_bench --no-interpret` on Colab or Kaggle TPU.
   - Compare against `tokamax.ragged_dot` / Megablox.
3. **Second Apple chip:** M1/M2/M3 from a friend or a cloud Mac. This decides whether the fusion half of the gain is M3/M4-specific.
4. **Mixtral puzzle:** our gain grows with T at d=4096/f=14336 without expert reuse. Test the hypothesis that same-expert slots run concurrently and hit the cache: compare one_hot vs uniform per-slot time.
5. **Upstream:**
   - Comment on MLX #4509 with the K%512 ablation (draft in `docs/upstream/`).
   - A minimal PR relaxing the `qmv_fast` alignment. This needs Xcode's Metal toolchain to build MLX.
6. **Writing:** chapters 4, 6 and 7 are drafted in `docs/thesis/`. Figures come from `python -m moelab.figures`.
