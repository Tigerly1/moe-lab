# Results log

> **Correction (2026-11-07, after independent audit):**
> - The layer-level "1.25×" below was inflated. Its baseline (`gather_qmm_q4.py`) ran SwiGLU as three uncompiled ops plus an extra cast; mlx-lm uses one compiled op.
> - The chained timing also decayed activations to exact zeros after about 3 calls.
> - Rigorous numbers are in "Audit-corrected results". The end-to-end +7.1% was always measured on the real mlx-lm path and stands.

Apple M4 Pro (20-core GPU, 48 GB), macOS, MLX 0.32.4, fp16 activations, int4 affine g64 weights (4.5 bits/weight including scales and biases).
"% roof" means the analytic floor, `U·3·d·f·4.5/8 / 273 GB/s`, divided by measured time. 273 GB/s is the datasheet figure; a measured copy only reaches ~201 GB/s.

## 2026-11-07: fused int4 decode kernel beats MLX `gather_qmm`

### Kernel journey (T=1 decode, uniform routing)
| variant | idea | Qwen3-30B-A3B | gpt-oss-20b | Mixtral-8x7B |
|---|---|---|---|---|
| `gather_qmm_q4` (MLX, baseline) | mlx-lm `SwitchGLU` path | 60% | 65% | 79% |
| `metal_q4_gemv` v1 | one simdgroup per row, gate+up fused, down loops over k serially | 44% | 62% | 65% |
| `metal_q4_gemv_v2` | 4 rows/simdgroup + prescaled-x nibble trick | 36% | 40% | 55% |
| `metal_q4_gemv_v3` | down parallel over (token, slot), fp16 `h`, tuned R | 72% | 72% | 82% |
| **`metal_q4_gemv_v31`** | v3 + 3 dispatches (fused combine kernel, int32 idx) | **74% (1.25×)** | **74% (1.14×)** | **83% (1.04×)** |

All variants pass the correctness gate: rel-Fro < 5e-3 against an fp32 reference built from the dequantized weights, with edge cases T=1, 7, 33, 64 one-hot and 100.

### What we learned (each point from a measurement, not a guess)
1. **Profile per kernel before optimizing.**
   - v1's gate_up already ran at 214 GB/s on Mixtral. The bottleneck was the down kernel (96–121 GB/s): it looped over the k experts serially, which gives low memory-level parallelism, and it read `h` in fp32.
2. **The "ALU-bound" hypothesis was wrong.**
   - v2's ALU savings made things slower.
   - A probe that removes the x loads, or the dot product, barely changed the time (61%→64%). At T=1 the kernel is bound by memory and latency, not ALU.
3. **Knob sensitivity is large.**
   - Over R_UP × R_DOWN × SG_PER_TG ∈ {1,2,4,8}³, worst vs best differs by 2×.
   - R_UP=1 is best on every shape. More rows per simdgroup hurts on M4 (likely occupancy).
4. **Dispatch cost matters for fine-grained experts.**
   - Each `mx.eval` plus sync costs about 130 µs on the host, so the harness times chained calls inside one graph.
   - Host-side graph build is only 6–8 µs per layer call.
   - The GPU-side cost of each dispatch is still significant. Going from 5 to 3 dispatches gave +3.5% on Qwen3.
5. **Small batch with skewed (Zipf) routing is unsolved.**
   - At 1 < T < 64, both MLX and v3 re-read an expert's weights once per (token, slot); MLX sorts only once there are 64 or more slots.
   - The expert-centric v4 (`metal_q4_expert_gemv`: each expert row read once per ≤8 tokens) is correct but slower, because per-token activation reloads dominate.
   - Open problem: stage the shared activations in threadgroup memory.

### Methodology changes triggered by these results
- `time_chained` now reports device time (eval) and host graph-build time (`host_us`) separately.
- Per-call sync latency is kept as `sync_us`.

## End-to-end (mlx-lm 0.31.3, `mlx-community/Qwen3-30B-A3B-4bit`, bf16, q4 g64)

`python -m moelab.e2e --tokens 256 --rounds 3`
- All 48 MoE blocks are patched; router and prefill are unchanged, and decode steps use `metal_q4_gemv_v31`.
- Stock and variant rounds alternate to spread thermal drift; round 0 is warmup.
- Laptop was **on battery** (94%). Re-run on AC + High Power mode for thesis numbers.

| path | decode tok/s (3 rounds) | median |
|---|---|---|
| stock mlx-lm (`gather_qmm`) | 85.30, 85.39, 84.95 | **85.3** |
| `metal_q4_gemv_v31` | 91.06, 91.34, 91.49 | **91.3 (+7.1%)** |

The whole-model speedup comes from the expert kernel alone (attention, norms, router and LM head are unchanged). At the layer level it is 1.25×.

### Quality: teacher-forced, 1003 tokens, decode path
`python -m moelab.e2e --quality`

| path | PPL |
|---|---|
| stock | 21.098 |
| variant | 21.102 |
| fp32 truth (same dequantized weights, fp32 expert math) | 21.184 |

| pair | top-1 agreement | mean \|ΔNLL\| |
|---|---|---|
| stock vs fp32 | 95.81% | 0.070 |
| variant vs fp32 | **96.51%** | 0.073 |
| stock vs variant | 95.91% | 0.071 |

- All pairs are equally far apart. This is the bf16 noise floor of the model: any change in reduction order diverges greedy decoding after ~20 tokens.
- The variant is as close to fp32 as stock is, so there is no quality loss.

## Audit-corrected results (2026-11-07)

**What changed in the method:**
- **Baseline:** `mlxlm_switchglu_q4.py`, the real `mlx_lm` SwitchGLU (compiled swiglu) plus the Qwen3 combine.
- **Activations:** a residual stream in chained timing (`x += y`), so activations stay O(1).
- **Runs:** interleaved A/B per process, 5 process launches × 6 rounds, paired bootstrap 95% CI (`moelab/compare.py`).
- **Conditions:** bf16, AC power.

**Speedup of v31 over mlx-lm, uniform routing** (`runs/cmp_v31_*.json`):

| shape | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| Qwen3-30B-A3B | **1.168** [1.163, 1.175] | 1.063 [1.032, 1.104] | 1.029 [1.002, 1.076] | 1.033 [1.026, 1.038] |
| gpt-oss-20b | 1.117 [1.101, 1.145] | 1.100 [1.094, 1.160] | 1.210 [1.152, 1.227] | 1.276 [1.270, 1.282] |
| Mixtral-8x7B | 1.029 [1.024, 1.034] | 1.127 [1.114, 1.193] | 1.352 [1.332, 1.406] | 1.416 [1.408, 1.427] |

- Accuracy against fp32 (bf16): v31 rel-err 3.5e-3, mlx-lm 6.9e-3. v31 is about 2× more accurate because it does the expert weighting and the k-sum in fp32.
- The e2e result (+7.1% tok/s) implies about 1.14× per layer, which agrees with the 1.17× above.
- **Open question:** why the gain *grows* with T for big experts (Mixtral 1.42× at T=8). Neither kernel shares weights across slots at T=8 with 16 slots. Hypothesis: MLX switches kernel family (`gather_qmv` → `gather_qmm`) somewhere in that range. Not yet verified.

### Mechanism: MLX's fast qmv path needs K % 512 == 0
MLX source (`quantized.cpp`, `qmv_fast_k_alignment` = 512 for 4-bit) routes K not divisible by 512 to the slower `qmv_impl`. Ablation that changes only the offending dimension (T=1, 3×6 rounds):

| shape | K not aligned | K aligned | gain explained by the slow-path fallback |
|---|---|---|---|
| Qwen3-like (down K 768 → 1024) | 1.168 | 1.097 | ~45% |
| gpt-oss-like (2880 → 3072) | 1.117 | 1.060 | ~50% |

So about half of the gain is MLX's alignment cliff, and about half is fusion and fewer dispatches (6 → 3 for the expert part). This is directly relevant to the open MLX issue #4509. PR #4598, a narrower fix, was rejected for dispatch complexity.

### Real routing (Qwen3-30B-A3B, 48 layers, ~5.4k tokens: English tech, Python code, Polish)
`moelab/routing_trace.py` → `runs/routing_qwen3.npz`, then `moelab/routing_stats.py`.

| T | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| uniform E(1-(1-k/E)^T) | 8 | 15.5 | 29.1 | 51.6 | 82.4 | 111.8 |
| T tokens from different sequences | 8 | 14.9 | 26.2 | 42.6 | 62.5 | 81.9 |
| T consecutive tokens (mean of 3 domains) | 8 | 12.5 | 18.5 | 27.9 | 38.2 | 50.3 |

- Adjacent tokens share **43–47%** of their top-8 experts, against 6.2% for random routing (7×).
- The top 10% of experts take 42–59% of selections; one layer reaches 87% for Polish.
- 12–28 experts per layer are never used in 2k tokens.
- **Implication:** at T=8 consecutive tokens (speculative-decoding verify), kernels read 64 expert slots when 28 would suffice, a ~2.3× traffic waste. That is the target of `autoresearch/`.
- With trace routing, v31 vs mlx-lm is 1.19× at T=2–4, 1.06× at T=8 and 1.01× at T=16 (`runs/cmp_v31_qwen3_trace.json`). Neither exploits reuse yet.

## Open problems / next steps
1. **Small-batch + skewed routing (1 < T < 64):** expert-centric reuse via threadgroup-staged activations (v4 is correct, but not fast yet).
2. **Regime dispatcher:** v31 for T ≤ 16, sorted `gather_qmm` above. The crossover sits around T≈16–32 on Qwen3 (RQ1).
3. **Statistics:** ≥5 process launches, bootstrap CI, AC power, `powermetrics` log.
4. **Same study on NVIDIA:** Marlin-MoE vs vLLM `fused_moe` on L4/H100 via Modal.
5. **Upstream:** propose the improvement to MLX's `gather_qmv` path for fine-grained MoE (f=768) as an MLX PR. This is a strong DeepMind-visibility artifact.
