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

## Next concrete steps
1. Fused int4 Metal GEMV variant (fork `metal_simd_gemv` → read packed uint32 + scales/biases).
2. Sorted grouped GEMM on Metal with `simdgroup_matrix` for T ≥ 64.
3. `modal setup` + deploy; first L4 run of `triton_fused_moe`, to confirm it is correct before tuning.
4. Router/top-k + align kernels as separate benchmark stages (K1–K3).
5. End-to-end tokens/s: patch `SwitchGLU` in mlx-lm with the chosen variant.
