# Research synthesis (10 parallel literature scans, 2026-11-06)

Facts marked ⚠ came from search snippets / memory, not primary sources. Verify before citing.

## 1. Core model of the problem

SwiGLU MoE layer, hidden `d`, expert width `f`, `E` experts, top-`k`, `T` tokens, `b_w` bits/weight:

| quantity | formula |
|---|---|
| FLOPs | `6·d·f·T·k` |
| unique experts touched | `U(T) = E·(1 − (1 − k/E)^T)` (uniform routing; real routers are skewed → lower) |
| bytes | `U·3·d·f·b_w/8 + activations` |
| arithmetic intensity | `16·n̄ / b_w`, with `n̄ = T·k/U` tokens per active expert |
| decode floor | `t ≥ U·3·d·f·b_w/8 / BW` |

- Decode (n̄≈1) is purely bandwidth-bound on both platforms. Prefill is compute-bound only once `n̄ ≳ ridge·b_w/16`.
- Ridge points: M4 Pro ≈ 34 FLOP/B (≈9 TFLOPS fp16 / 273 GB/s, no 2× fp16 rate); H100 ≈ 295 (bf16).
- Example: DeepSeek-V3 on H100 at FP8 becomes compute-bound only at T ≈ 9.4k tokens per layer.
- Sparsity divides tokens per expert by E/k, so "prefill-like" batches still behave like decode until T ≫ E/k.

## 2. State of the art

**NVIDIA**
- vLLM `fused_moe` (Triton): `moe_align_block_size` + implicit gather + routed weight in epilogue; tile configs tuned per (E, N, dtype, GPU) as JSON.
- Marlin-MoE: int4 W4A16, near-ideal bandwidth at decode.
- DeepGEMM: FP8 grouped GEMM, contiguous (prefill) and masked (decode) layouts.
- CUTLASS grouped GEMM; FlashInfer `cutlass_fused_moe` (atomic, non-deterministic finalize); MegaBlocks (block-sparse, MLSys'23); ScatterMoE; SonicMoE (arXiv 2512.14080).
- Megakernels: MonoMoE ⚠(2609.04244), Alpha-MoE, FlashMoE (2506.04667).
- TRT-LLM "MoE as dense GEMM" (B200): dense-masked beats grouped at T≈64–208 despite ~97% wasted FLOPs. **A regime crossover we can measure on both platforms.**

**Apple**
- mlx-lm `SwitchGLU`: 3 separate `gather_mm`/`gather_qmm` calls; sorts tokens only when `indices.size ≥ 64`; combine is a separate op.
- MLX dispatch: `gather_qmm_rhs` only if `M==1 && B≥16 && sorted && B/E≥4`, with a `// TODO: Tune 16 and 4` in the source. `gather_qmv` below `vector_limit` (13–33).
- llama.cpp Metal: `map0` prepass builds per-expert tile lists, then `kernel_mul_mm_id` (NR0=64, NR1=32, simdgroup 8×8); `mul_mv_id` for decode. Uses tensor ops only on M5+.
- M4 Pro facts:
  - SIMD 32, 32 KB threadgroup memory, `simdgroup_matrix` 8×8, Dynamic Caching.
  - 273 GB/s. Our measured copy is 201 GB/s; `gather_mm` reaches 210 GB/s at T=1.
  - No FP8/FP4 hardware, no Neural Accelerators (those arrive with M5).

**Google** (career alignment)
- JAX `jax.lax.ragged_dot`; Megablox (Pallas GMM, MaxText).
- Tokamax (`tokamax.ragged_dot`, Mosaic GPU + TPU, has an autotuner) ⚠ early-stage.
- Pallas Triton backend is deprecated in favour of Mosaic GPU (Hopper+). jax-metal is effectively dead; there is **no Pallas on Apple**.
- AlphaEvolve: 23% speedup on Gemini Pallas matmul tiling, up to 32.5% on FlashAttention. "How to Scale Your Model" (scaling book) is the house style for rooflines.

## 3. Gaps → candidate contributions
1. **No paper compares MoE kernels on Apple GPU vs NVIDIA under a common roofline.** Closest:
   - TritonMoE (A100/MI300X), MonoMoE (H200)
   - BaseRT (M5 Neural Accelerators), NPUMoE (ANE)
   - The M4 Pro has none of the M5 accelerators → clean separation from BaseRT.
2. **Small-batch regime 1 < T < 64 on Apple:**
   - no sort, so shared experts are reloaded per token
   - untuned `gather_qmm_rhs` thresholds
   - `gather_qmm_q4` reaches only ~48% of roofline at T=1 (our measurement)
3. **Regime-adaptive dispatcher:** dense-masked vs gather-GEMV vs sorted grouped GEMM, chosen by an analytic model. Test the hypothesis that the crossover tracks the ridge point per architecture.
4. **Fusion on Metal:**
   - gate+up+SwiGLU in one kernel
   - down + weighted combine in the epilogue
   - GPU-side `map0` compaction instead of an `argsort` round-trip
5. **LLM/evolutionary kernel search for fused MoE on Metal.**
   - K-Search (BAIR, Jul 2026) lists "fused MoE on MLX" as future work ⚠.
   - Use the same harness as evaluator, with a correctness gate and a count of caught reward hacks. This is AlphaEvolve-flavoured, i.e. DeepMind-relevant.
6. **Tile-config portability:** does a config tuned on one architecture transfer to another (NVIDIA ↔ Metal)?

## 4. Methodology rules (implemented / to implement)
- [x] fp32/fp64 reference, rel-Frobenius error, NaN check, per-dtype tolerance
- [x] Edge cases: T=1, T not a multiple of the tile, one-hot routing (all tokens → same experts), Zipf skew
- [x] Fresh subprocess per job; inputs rotated every call (anti-memoization)
- [x] MLX: chained in-graph timing.
  - Each `mx.eval`+sync costs ~130 µs on M4 Pro, which would swamp decode kernels.
  - Per-call sync latency is reported separately as `sync_us`.
- [x] CUDA: events + device-wide sync before the end event (defeats the side-stream exploit, CUDA-L1)
- [x] Report % of roofline floor; a result above 100% means a bug
- [ ] ≥5 process launches × median/IQR + bootstrap CI for thesis figures
- [ ] Measured peaks (copy BW, big GEMM) next to datasheet peaks
- [ ] Thermal: AC power, High Power mode, `powermetrics` logging, interleaved run order
- [ ] NVIDIA: lock clocks (`nvidia-smi -lgc`), cold-L2 vs warm/CUDA-graph modes labelled
- [ ] Poison output buffers with NaN before each run (matters for custom kernels with `init_value`)
- [ ] Held-out shapes never used during tuning

## 5. Models
| role | model | MLX repo |
|---|---|---|
| tiny, correctness only | `hf-internal-testing/tiny-random-Qwen2MoeForCausalLM`, `yujiepan/deepseek-v3-tiny-random` | – |
| main | Qwen3-30B-A3B (128×8, d=2048, f=768) | `mlx-community/Qwen3-30B-A3B-4bit` (~17 GB) |
| contrast | gpt-oss-20b (MXFP4, 32×4) | `mlx-community/gpt-oss-20b-MXFP4-Q4` |
| contrast | Mixtral-8x7B (8×2, huge experts) | `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` (~26 GB) |
| cheap DeepSeek arch | DeepSeek-V2-Lite | `mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx` |

Shape-only: DeepSeek-V3, Kimi-K2, Qwen3-235B, gpt-oss-120b (see `moelab/shapes.py`).

## 6. Quant formats
- Tier 1:
  - fp16/bf16
  - int4 affine g64 (MLX native; Marlin on CUDA)
  - int8
  - MXFP4 g32 (gpt-oss; native on Blackwell, emulated on Apple/Hopper)
- Tier 2: FP8 block-128 (Hopper), NVFP4 (Blackwell only).
- Always report `b_w` including metadata (int4 g64 = 4.5 bits).

## 7. Key references
- Shazeer'17 · GShard · Switch · GLaM · ST-MoE · Expert Choice · Mixtral (2401.04088) · DeepSeekMoE (2401.06066) · DeepSeek-V3 (2412.19437)
- MegaBlocks (2211.15841) · Tutel (2206.03382) · DeepSpeed-MoE (2201.05596) · Fiddler (2402.07033) · MoE-Lightning (2411.11217) · Pre-gated MoE (2308.12066)
- Marlin (2408.11743) · FlashAttention (2205.14135) · Roofline (Williams'09)
- KernelBench (2502.10517) · AlphaEvolve (2506.13131) · CUDA-L1 (2507.14111)
- TritonMoE ⚠(2605.23911) · BaseRT ⚠(2607.19438) · TaxBreak ⚠(2603.12465)
- Scaling book: https://jax-ml.github.io/scaling-book/
- MLX custom kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- TRT-LLM dense MoE: https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html
- Tokamax: https://github.com/openxla/tokamax
