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

## Night of 2026-11-07: autoresearch, slope timing, end-to-end (all on AC power)

### Autoresearch round 1: expert reuse at T>1 (`autoresearch/program.md`)
- **Setup:** a Karpathy-style agent loop. The agent edits one file; the evaluator is read-only (correctness gate plus interleaved A/B against real mlx-lm on real routing traces); each experiment is a git commit that is kept or discarded.
- **Agent's results:** 7 runs. Best kept score 1.286 (`metal_q4_tiles_b`). Top score 1.426 (`metal_q4_tiles_mma`), discarded only by a keep rule that used the noisy baseline CI (lesson: use the candidate's own CI).
- **Independent re-verification** (5 launches × 6 rounds, Qwen3 shape, bf16, `runs/cmp_metal_q4_tiles_*_qwen3_*.json`):

| T | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| tiles_mma, real traces | 1.160 | 1.215 | 1.363 | **1.642** | **1.872** | **2.361** |
| tiles_mma, uniform | 1.159 | 1.137 | 1.122 | 1.152 | 1.224 | 1.435 |
| tiles_b, real traces | 1.155 | 1.179 | 1.277 | 1.314 | 1.345 | 1.387 |

All 95% CIs are within ±1–3%. Correctness holds on 5 real shapes × {uniform, one_hot, zipf, trace} × T ∈ {1..64}, with worst rel-err 3.6e-3. `tiles_mma` requires d, f % 128 == 0; it refuses gpt-oss (2880) with an assert and never computes wrong.

**How tiles_mma works:**
1. A map prepass groups the (token, slot) pairs of one step by expert, in tiles of ≤8.
2. `simdgroup_matrix` 8×8 (half inputs, fp32 accumulation) dequantizes int4 exactly via `as_type<half2>(q | 0x6400) - 1024`.
3. 128-wide K superblocks give one uint4 weight load per lane.
4. Single-slot tiles fall back to the GEMV path.

### Autoresearch round 2: plain decode T=1 (`autoresearch/program_decode.md`)
- **Result:** +1.5% score, giving `metal_q4_decode_v4`. The down kernel spreads each row over LPR lanes with `simd_shuffle_xor`, so all 32 lanes are busy at K=768; the k slots are reduced in threadgroup memory, so the combine dispatch is gone.
- **Key finding:** the T=1 kernels are at the bandwidth ceiling (next section). A full single-dispatch fusion (atomic counters) was correct but slower.

### Slope timing (`moelab/slope.py`): the fixed eval cost was hiding in chained timing
- **Method:** per-call time = slope of total eval time vs chain length (10/20/40/80). The intercept is the fixed per-eval cost (130–340 µs), which plain chained timing had spread across the chain.
- **Measured read-only peak:** 240–250 GB/s (copy: 224 GB/s). Datasheet: 273 GB/s.

| T=1, bf16 | floor @250 GB/s | mlx-lm | v31 | v4 | speedup v4 |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 85.1 µs | 107.4 µs (79%) | 89.1 µs (96%) | 86.4 µs (98.5%) | **1.244×** |
| gpt-oss-20b | 224.2 µs | 248.9 µs (90%) | 221.5 µs (101%) | 225.8 µs (99%) | 1.10–1.12× |
| Mixtral-8x7B | 793.2 µs | 801.8 µs (99%) | 773.4 µs (103%) | 771.1 µs (103%) | 1.04× |

- Values above 100% mean the true read ceiling is about 255–260 GB/s.
- **Conclusion:** at T=1 our kernels read memory at the hardware limit. Further T=1 gains can only come from moving fewer bytes, e.g. lower-bit formats.
- Chained-timing ratios slightly *understated* the gains: 1.17× there vs 1.24× by slope for Qwen3.

### "Is it just an M4 boost?" (thermal/clock check)
- **Battery vs AC:** stock moves 85.3 → 89.9 tok/s, but the gain stays the same (+7.1% vs +7.0%).
- **Sustained load** (`moelab/sustained.py`): 10 minutes nonstop, 37,651 interleaved rounds, T=1. The ratio is 1.169 in the first quarter and 1.167 in the last; absolute times did not drift either (133 µs → 133 µs), so there was no throttling.
- **Verdict:** clock or thermal boost is ruled out on this machine. Whether the gain holds on other Apple GPUs (M1–M3, M5) is **untested**.
  - The K%512 half of the gain is chip-independent MLX code.
  - The fusion half may depend on M3/M4 Dynamic Caching.
  - M5 uses different MLX (NAX) paths.

### End-to-end with the dispatcher `metal_q4_moe` (v4 at T=1, tiles_mma/tiles_b at T>1)
Model: mlx-lm `Qwen3-30B-A3B-4bit`, AC power, 3 interleaved rounds.

| mode | stock | ours | gain |
|---|---|---|---|
| plain decode, 512 tokens | 91.5 tok/s | **99.9 tok/s** | **+9.2%** |
| verify step, chunk 2 (fixed text) | 102.6 | 110.2 | +7.4% |
| verify step, chunk 4 (= 3 drafts) | 124.4 | 148.6 | **+19.4%** |
| verify step, chunk 7 (= 6 drafts) | 128.4 | 167.6 | **+30.5%** |
| verify step, chunk 16 | 193.8 | 271.8 | **+40.3%** |
| speculative decode, 6 drafts (outputs identical 512/512) | 76.5 | 92.3 | +20.8% |

**Pitfall: free-running speculative tok/s is not a fair kernel metric.**
- With 3 drafts, the stock and kernel outputs diverged at token 22 (bf16 noise).
- The draft acceptance rate then differed (0.63 vs 0.54), so the kernel arm *looked* 6% slower while doing more verify steps.
- The verify benchmark (`e2e.py --verify-chunks`) feeds identical text in chunks and removes this confound. Use it for claims.

**Speculative decoding vs plain decoding:**
- On this MoE, speculative decoding with Qwen3-0.6B is **slower** than plain decoding for stock MLX (72–77 vs 91.5 tok/s).
- With our kernel it is roughly break-even (92.3 vs 99.9).
- Each verify step touches 20–30 experts instead of 8.

**Quality:**
- Teacher-forced over ~1000 tokens: PPL 21.17 (stock) vs 21.15 (ours) vs 21.02 (fp32 experts).
- All three pairs agree on the top-1 token 95.8–96.2% of the time, which is the bf16 noise floor.
- The corpus is now frozen at `moelab/data/quality_corpus.txt`.

### Real routing per layer (`runs/routing_qwen3_perlayer.json`)
- Adjacent-token expert overlap is 17% at layer 0 (random routing gives 6.2%) and 40–54% from about layer 5 on.
- U(8 consecutive tokens) is 40.6 at layer 0 and 25–29 elsewhere.
- This matches the Mixtral literature: the first layer routes almost randomly.

### Autoresearch round 3: T>1 tiles (`autoresearch/program_tiles.md`) → `metal_q4_tiles_v2`
- **Agent result:** score 1.264 → 1.316 (+4.1%). Three changes:
  - LPR down-projection lanes for single-slot tiles;
  - a scalar path for 2-slot tiles (MMA wins only for ≥3 slots);
  - a per-shape down K-split.
- **Agent's measured limit:** at T≥8 the MMA tiles are bound by MMA instruction count (removing the up-projection MMAs saves 27%; removing all dequant saves only 5%). Reusing weights raised the arithmetic intensity, so the kernel is now compute-bound.
- **Independent verification**, 5×6 against mlx-lm, including shapes never used for tuning:

| shape (routing) | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---|---|---|---|---|
| Qwen3-30B-A3B (real traces) | 1.251 | 1.422 | 1.721 | 1.982 | **2.443** |
| Qwen3-235B-A22B, held-out (uniform) | 1.050 | 1.084 | 1.132 | 1.266 | |
| Mixtral-8x7B, held-out (uniform) | 1.129 | 1.386 | **1.952** | | |
| OLMoE-1B-7B (uniform) | 1.055 | 1.168 | 1.215 | 1.359 | |

Correctness: worst rel-err 3.9e-3 over 5 shapes × 4 routings × T ∈ {1..64}.

### The Mixtral puzzle, solved: implicit cache reuse from dispatch order
- **The puzzle:** v31 (no tiling) gained up to 1.36× on Mixtral at T≥4, while matching MLX at T=1.
- **The test:** slope timing, T=4, with expert-distinct vs shared routing:

| routing | mlx-lm | v31 |
|---|---|---|
| 8 slots → 8 distinct experts | 3075 µs (258 GB/s) | 3054 µs (260 GB/s) |
| 8 slots → 2 experts (one_hot) | 3063 µs | **2243 µs** |

- **Interpretation:**
  - When every slot needs a different expert, both kernels run at the true DRAM ceiling of about 260 GB/s.
  - When slots share an expert, v31 reuses the weights through the cache (SLC) and MLX does not. v31's grid puts slots on the y axis, so slots of the same expert run concurrently. MLX `gather_qmv` uses grid (M, N/8, B) with the batch outermost.
  - Uniform routing at T=4 already has duplicates (U ≈ 5.5 of 8 slots), which explains the earlier gain.
  - `tiles_v2` makes the reuse explicit and reaches 1.95× at T=8.

### Final end-to-end (dispatcher = v4 at T=1 + tiles_v2 at T>1), Qwen3-30B-A3B-4bit, AC power

| mode | stock | ours | gain |
|---|---|---|---|
| plain decode, 512 tokens | 91.6 tok/s | **99.8 tok/s** | **+9.0%** |
| verify chunk 2 | 102.6 | 112.1 | +9.3% |
| verify chunk 4 (3 drafts) | 127.5 | 156.6 | **+22.9%** |
| verify chunk 7 (6 drafts) | 126.9 | 170.0 | **+34.0%** |
| verify chunk 16 | 192.6 | 279.2 | **+44.9%** |

### Autoresearch round 4: MXFP4 (gpt-oss expert format) → `metal_mxfp4_moe`

> **Correction (2026-11-07, 04:00):** the agent porting gpt-oss end-to-end found that the T≥8 MMA path stored the E8M0 scale (with the 2^14 fold) in a single half. That overflows to inf for scale bytes ≥ 129, which ~11% of the real gpt-oss down_proj groups have; random-normal lab weights never reach it.
> - **Fix:** split the scale into two exact half factors (`hs2`).
> - **New test:** `tests/test_mxfp4_large_scales.py` fails on the old kernel and passes on the new one.
> - **Effect on speed:** re-measured after the fix (5×6), the T≥8 speedups drop. Qwen3 traces T=8/16/32 go from 1.786/2.064/2.424 to **1.644/1.885/2.148**. gpt-oss T=8/16 go from 1.521/1.688 to **1.504/1.559**. T<8 is unaffected because it does not use the MMA path. Held-out shapes after the fix: gpt-oss-120b T=8/16 1.164/1.240 (unchanged), Mixtral T=8 **1.765** (was 1.923), DeepSeek-V2-Lite T=8/16 1.343/**1.384** (was 1.353/1.448). The table below shows the pre-fix numbers for T≥8 and is superseded by these.
- **Format:** MLX `mode="mxfp4"`. Each value is FP4 E2M1, with one E8M0 scale per 32 values and no bias, giving **4.25 bits/weight**. That is 6% fewer bytes than affine int4 g64.
- **Baseline:** the real mlx-lm SwitchGLU quantized to MXFP4. The candidate started as that same wrapper (score 1.000 ± 0.003; this doubles as the bench's sanity check).
- **Agent result:** score **1.398** after 7 bench runs.
- **Rule slip:** the agent kept one commit that regressed a point by more than 0.03. It reported this itself, and the final commit fixes it.

**Key tricks:**
1. **Exact FP4 decode of 2 values per bit-trick:** `as_type<half2>(((p&0x80008)<<12)|((p&0x70007)<<9))` = value·2^-14, subnormals included. The 2^14 is folded into the E8M0 scale. Verified against `mx.dequantize` with max diff 0.
2. **T=1:** a port of the int4 v4 GEMV.
3. **1<T<8:** slots of the same expert are paired inside the kernel (rank-parity scan), so each pair shares every weight decode.
4. **T≥8:** a map prepass, then groups of ≤8, with simdgroup 8×8 MMA for groups of ≥3. The k index is remapped inside 64-wide blocks, so d=2880 needs no tail path.

**Independent verification** (5×6, bf16). Correctness: worst rel-err 2.5e-3 over 7 shapes × 4 routings × 11 values of T.

| shape (MXFP4) | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---|---|---|---|---|---|
| gpt-oss-20b, uniform | 1.164 | 1.156 | 1.285 | 1.521 | 1.688 | |
| Qwen3-30B-A3B, real traces | **1.264** | 1.412 | 1.536 | 1.786 | 2.064 | **2.424** |
| gpt-oss-120b, uniform (held-out) | 1.168 | | 1.135 | 1.167 | 1.242 | |
| Mixtral-8x7B, uniform (held-out) | 1.045 | | 1.385 | 1.923 | | |
| DeepSeek-V2-Lite, uniform (held-out) | 1.184 | | 1.248 | 1.353 | 1.448 | |

**Agent's measured lessons:**
- **FP4 decode cost:** decoding FP4 costs about the same as decoding affine int4 (≈2.5 int ops + 1 cvt per weight, against int4's bias term and x prescale). Our MXFP4 GEMV is therefore faster than our int4 GEMV roughly in proportion to the bytes (4.25 vs 4.5 bits): Qwen3 T=1 106 µs vs 114 µs.
- **T>1 bottleneck:** at T>1 the kernel is ALU-bound, because repeated experts are served from cache. Sharing the decode between slots of the same expert is the main lever.
- **Failed:** lookup tables (threadgroup or `simd_shuffle`) were 1.6–2.2× slower; half2 math gave no gain.
- **Remaining headroom:** gpt-oss points are within ~4% of the memory floor. Qwen3 T=16 is ~45% above its floor, bound by MMA plus decode instruction count.

### End-to-end #2: gpt-oss-20b (`mlx-community/gpt-oss-20b-MXFP4-Q8`, MXFP4 experts, AC power)
Variant `metal_mxfp4_gptoss`: the `metal_mxfp4_moe` kernels plus per-expert biases and the clamped gpt-oss SwiGLU (alpha 1.702, limit 7, +1 on the linear branch). It is selected by the `ACT` template. `e2e.patch()` dispatches on model type.

Agent-measured numbers, independently re-run:

| mode | stock | ours | gain |
|---|---|---|---|
| plain decode, 512 tokens × 3 | 75.5 tok/s | 81.2 tok/s | **+7.6%** (agent: +8.0%) |
| verify chunk 1 | 67.9 | 72.4 | +6.6% |
| verify chunk 2 (agent) | 97.4 | 112.5 | +15.5% |
| verify chunk 4 | 125.9 | 161.6 | **+28.4%** |
| verify chunk 7 (agent) | 127.8 | 168.6 | +31.9% |
| verify chunk 16 | 174.8 | 287.7 | **+64.6%** |

**Quality:** teacher-forced on the frozen corpus (raw markdown, no harmony template, so the absolute PPL is high):

| path | PPL | top-1 agreement with fp32 |
|---|---|---|
| stock | 181.2 | 90.5% |
| ours | 179.6 | 92.3% |
| fp32 experts | 172.4 | — |

Our kernel is closer to fp32 than stock.

One layer-17 channel reaches about 40,000, which is close to the half range. The kernel matches it in fp32 (40037.586 vs 40037.590); the per-layer bf16 difference is one ULP.

### Autoresearch round 5: MXFP4 after the fix (`autoresearch/program_mxfp4_r5.md`)
**Two changes:**
1. **Scale applied in fp32.** The K index is remapped so that the 4 MMA steps of one 32-weight scale group are contiguous. The E8M0 scale is applied in fp32 to the per-group partial product (`C += scale * P`). The half-scale workaround (`hs2`) is deleted, so the code is simpler.
2. **Second latent bug found and fixed:** the 2^14 fold inside `e8s` overflowed for E8M0 bytes ≥ 241 (NaN in *every* path, T=1 included). Now `e8s = 2^(s-127)` and the 2^14 is applied once per output, so the kernel is exact for bytes 1..254. Real checkpoints only reach ~136.

**Agent's measurements:**
- Raw `simdgroup_matrix` 8×8 throughput on the M4 Pro is ~6.1 G MMA/s (≈6.2 TFLOPS), the same for half, bfloat and float.
- The MMA gate_up path runs at ~80% of that issue rate.
- Every point except Qwen3 T=16 is already at the bandwidth floor (237–255 GB/s).

**Remaining limit:** the T≥8 MMA path converts activations to half. |x| or |h| > 65504 would break it. A bfloat-fragment variant fixes that at −2.4% on Qwen3 T=16; it is not adopted, but it is saved.

**Independent verification** (5×6) after the merge. The gate includes the large-scale and real-checkpoint tests (268 tests pass).

| MXFP4 | r4 (buggy on real scales) | after hs2 fix | **r5** |
|---|---|---|---|
| Qwen3 traces T=1 / 4 / 8 / 16 / 32 | 1.26 / 1.54 / 1.79 / 2.06 / 2.42 | – / – / 1.64 / 1.89 / 2.15 | **1.27 / 1.53 / 1.73 / 2.00 / 2.34** |
| gpt-oss-20b T=1 / 8 / 16 | 1.16 / 1.52 / 1.69 | – / 1.50 / 1.56 | **1.16 / 1.50 / 1.60** |
| gpt-oss-20b e2e: plain / verify 4 / verify 16 | – | +7.6% / +28.4% / +64.6% | **+7.9% / +27.4% / +69.5%** |

## Audit 2 (2026-11-07, ~05:00): independent review of all night claims, and the fixes

The reviewer read code, re-ran tests and scripts, and spot-checked ~15 numbers against `runs/*.json`; all of them matched. **No cheating was found**: baselines are the real mlx-lm path, the variant skips no work, and verify_bench is fair. Problems found and how each was fixed:

1. **Trace-routing CIs were far too narrow.**
   - Rounds within one launch share the same routing windows, but the bootstrap treated them as independent.
   - **Fix:** `compare.py` now reports a *hierarchical* CI (resample launches, then rounds), uses `--windows 8` per launch, and takes `--seed-base 1000`, seeds the tuning agents never saw (they used 0–2).
   - Re-run (6 launches × 4 rounds × 8 windows): the point estimates hold, now with honest intervals.

| real traces, fresh seeds | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---|---|---|---|---|---|
| int4 dispatcher `metal_q4_moe` | 1.200 [1.19, 1.21] | 1.234 [1.22, 1.30] | 1.468 [1.42, 1.56] | **1.751 [1.63, 1.83]** | **2.019 [1.90, 2.13]** | **2.455 [2.40, 2.53]** |
| MXFP4 `metal_mxfp4_moe` (r5) | 1.257 [1.25, 1.27] | 1.392 [1.35, 1.48] | 1.520 [1.49, 1.59] | 1.764 [1.68, 1.80] | 2.010 [1.90, 2.08] | 2.366 [2.32, 2.42] |

2. **"Held-out" was overstated.**
   - Only **Qwen3-235B** is clean (it first appears in verification).
   - Mixtral was timed in round 2 (T=1) and studied in the "Mixtral puzzle". DeepSeek-V2-Lite was timed in rounds 2–3.
   - gpt-oss-120b has the same per-expert shape as gpt-oss-20b.
   - Fresh-seed results, uniform routing:

| shape | status | T=1 | T=4 | T=8 | T=16 |
|---|---|---|---|---|---|
| Qwen3-235B, MXFP4 | clean held-out | 1.070 | 1.103 | 1.109 | 1.227 |
| Qwen3-235B, int4 | clean held-out | 1.060 | 1.094 | 1.125 | 1.263 |
| Mixtral, MXFP4 | partially seen | 1.045 | 1.393 | 1.816 | – |
| DeepSeek-V2-Lite, MXFP4 | seen (int4 rounds) | 1.192 | 1.216 | 1.340 | 1.417 |

3. **README quoted a pre-fix MXFP4 number** (Mixtral 1.92). Corrected to the round-5, fresh-seed value of 1.82.
4. **T=1 "% of bandwidth" was inflated by cache reuse.**
   - The timing chain rotated only 3 inputs, so expert sets recurred every 3rd call.
   - **Fix:** `slope.py --windows 80` uses non-repeating routing. Output: `runs/slope_nonrepeating.txt`.
   - Qwen3 int4 v4: **93.7%** of the 250 GB/s floor (was reported as 98.5%); speedup 1.216×.
   - gpt-oss: 99.8%. Mixtral: 102.7%. Values above 100% are genuine and imply a true ceiling of ~257 GB/s.
   - MXFP4 at T=1: Qwen3 90.8% (1.28× vs mlx-lm MXFP4), gpt-oss 98.5% (1.18×), Mixtral 102.4% (1.05×).
5. **No real-model output check at T>1.**
   - **Fix:** `tests/test_real_model_multitoken.py` (opt-in, `MOE_SLOW=1`) runs both full models with chunks T=4/8/16 through the patched blocks.
   - Logits match stock: rel ≤ 0.08 per chunk (measured ~0.014–0.045), ≥85% top-1 agreement over 28 positions. Passes.
   - gpt-oss quality was re-measured on the round-5 kernel: identical (PPL 179.63 vs stock 181.17 vs fp32 172.39).
6. **Smaller items:**
   - bf16 test tolerance tightened from 3e-2 to 1e-2; all 268 tests still pass.
   - verify JSONs now record `max_t`.
   - Stale `hs2` docstring removed.
   - **Remaining limit:** the int4 `tiles_v2` MMA path forms dequantized weights in half. Scales below ~6e-8 would flush to zero; the real Qwen3 minimum is 8.6e-6, so this does not occur today.
7. **Framing:** the README now states that free-running speculative decoding is itself slower than plain decoding on this setup; the verify-step gain applies to the verify step only.

### RQ1 closed: crossover vs mlx-lm as T grows (`runs/crossover_q4_qwen3_*.json`, fresh seeds 2000+)
Layer level, int4 dispatcher vs the real mlx-lm path, Qwen3 shapes, 4 launches × 3 rounds × 4 windows, CI resampled by launch:

| T | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| real traces | 1.98 | **2.45** | 1.61 | 1.34 | 1.12 | **1.00** [1.00, 1.02] |
| uniform | 1.33 | 1.55 | **1.93** | 1.36 | 1.09 | **0.99** [0.99, 1.00] |

**Crossover:**
- Our kernels win up to T=256 and reach parity at T=512. That is where MLX's sorted grouped GEMM is efficient and the layer becomes compute-bound (the analytic ridge for Qwen3 int4 on the M4 Pro is T*≈157 under uniform routing).
- `e2e.MAX_T` now defaults to 256.

**Full model, fixed text in chunks** (short prefill / batch of tokens), Qwen3-30B-A3B-4bit, `runs/e2e_prefill_chunks_qwen3.json`:

| chunk | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| stock tok/s | 237.5 | 387.0 | 494.6 | 595.8 |
| ours tok/s | 441.3 | 528.6 | 593.5 | 644.0 |
| speedup | **1.86×** | 1.37× | 1.20× | 1.08× |

### gpt-oss-20b full model, fixed text in chunks (`runs/e2e_prefill_chunks_gptoss.json`)
| chunk | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| stock tok/s | 284.1 | 394.1 | 497.1 | 574.0 |
| ours tok/s | 432.7 | 504.0 | 554.2 | 584.7 |
| speedup | **1.52×** | 1.28× | 1.11× | 1.02× |

The crossover pattern is the same as Qwen3.

### Thermal check on the final dispatcher (`runs/sustained_T1_dispatcher.json`, 8 min nonstop, T=1)
- The ratio is 1.209 in the first quarter and 1.211 in the last.
- Absolute times rose ~2.5% on both sides (mild heating: stock 129.2 → 132.5 µs, ours 106.8 → 109.5 µs), so the ratio is unaffected.
- This addresses the audit-2 note that the earlier sustained run used `tiles_mma`.

### Real routing #2: gpt-oss-20b (E=32, top-4), the same three domains (`runs/routing_gptoss.npz`, `_stats.json`)
| T | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| uniform E(1-(1-k/E)^T) | 4 | 7.5 | 13.2 | 21.0 | 28.2 | 31.6 |
| T tokens from different sequences | 4 | 7.3 | 12.1 | 18.1 | 23.4 | 27.3 |
| T consecutive tokens (English / code / Polish) | 4 | 6.1–6.3 | 8.5–9.5 | 12.2–13.4 | 15.7–17.8 | 18.9–21.4 |

- **Locality:** adjacent tokens share 41–50% of experts, against 12.5% for random routing (3.3–4×).
- **Skew:** the top 10% of experts take 32–42% of selections. Unlike Qwen3, almost no experts go unused (0.1–3.2 of 32).
- **Generality (refined later):** Qwen3 (128×8) and gpt-oss (32×4) both show ~40–50% adjacent-token expert overlap. DeepSeek-V2-Lite shows only 18–29%; see real routing #3. At T=8, a per-slot kernel reads 32 expert slots where ~13 suffice (gpt-oss), or 64 where ~28 suffice (Qwen3). Expert-reuse kernels pay off on both.
- No published per-layer routing statistics for gpt-oss were found (research scan); this is a new measurement.

### Autoresearch round 6: int4 tiles exact for any affine scale (`autoresearch/program_r6.md`)
- **Test first:** `tests/test_int4_small_scales.py` uses weights × 1e-6 and x × 1e6.
  - It failed for the production MMA tile path at T≥3. Half-precision dequant flushed tiny scales, and half activations overflowed.
  - The real Qwen3 checkpoint (min scale 8.6e-6, max |x| 153) never triggered it.
- **Fix:**
  - The MMA runs on raw codes plus an exact offset (`q|0x6400` as half, `q|0x4300` as bfloat).
  - The K index is remapped per 64-wide scale group.
  - `C += s*P + (b − OFF*s)*Σx` in fp32.
  - MMA operands are bfloat for bf16 inputs.
- **Speed:** unchanged within noise (agent bench 1.3140 → 1.3102, CI 0.04).
  - Fresh-seed check (seeds 3000+, 6×4×8): Qwen3 traces T=1/4/8/16/32 = 1.208 / 1.483 / 1.598 [1.57, 1.75] / 2.046 / 2.412.
  - Full model (Qwen3): verify chunk 4 / 16 / 32 = **1.23× / 1.48× / 1.84×**.
- **Historical kernel:** `metal_q4_tiles_mma` (round 1) is xfail on the tiny-scale test, with the reason documented. Production does not use it.
- **Not adopted:** a faster variant (16 rows per threadgroup, +0.7%) fell below the keep threshold. It is saved on branch `autoresearch/int4-scales-r6-mr2`.

### `moelab/generate.py`: use the kernels in mlx-lm
- `python -m moelab.generate --model <repo> --prompt ... [--compare]` picks the kernel from the checkpoint format (Qwen3-MoE affine int4 → `metal_q4_moe`; gpt-oss MXFP4 → `metal_mxfp4_gptoss`).
- **Example (Qwen3-30B-A3B-4bit, single run after warmup):** prompt ~30 tokens 171.6 → 213.8 tok/s, decode 94.3 → 103.1 tok/s.
- **Limitation: cold start.** The custom Metal kernels are JIT-compiled on first use, adding about 0.4 s to the first request. The first unwarmed run measured 56 tok/s prompt throughput. Every benchmark in this document warms up first.

## Final reproduction run (2026-11-07 05:39–05:47, `scripts/reproduce_headline.sh`, seeds 5000+, AC)
All headline numbers were reproduced from scratch in one run. Output: `runs/repro/` and `runs/repro_log.txt`.

| check | result |
|---|---|
| tests incl. full-model multi-token (`MOE_SLOW=1`) | **327 passed**, 3 xfailed (historical kernel) |
| int4 dispatcher, Qwen3 traces, T=1/8/16/32 | 1.206 / 1.719 / 2.049 / 2.448 |
| MXFP4, Qwen3 traces, T=1/8/16/32 | 1.262 / 1.734 / 2.068 / 2.404 |
| int4, Qwen3-235B (clean held-out), T=1/16 | 1.065 / 1.262 |
| T=1 slope, non-repeating routing (Qwen3) | mlx-lm 76.7% / v4 93.4% of floor, 1.218× |
| Qwen3 e2e: plain / verify 4 / 16 / 32 | +9.0% / +22.8% / +48.4% / +83.5% |
| gpt-oss e2e: plain / verify 4 / 16 | +8.0% / +27.4% / +69.6% |

### End-to-end #3: DeepSeek-V2-Lite (`mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx`, int4 g64, AC)
- **Architecture:** 64 routed experts, top-6, plus 2 shared experts, d=2048, f=1408, MLA attention, first layer dense.
- **Patch:** `e2e._patch_deepseek` keeps the stock MoEGate (greedy or group-limited top-k, routed scaling) and the stock shared experts. Only the routed SwitchGLU goes to `metal_q4_moe`.
- **Tests:** tiny-model logit tests for both top-k methods (`tests/test_deepseek_patch.py`); full-model multi-token test (`MOE_SLOW=1`) passes.

| mode | stock | ours | gain |
|---|---|---|---|
| plain decode, 512 max tokens × 3 (outputs identical, 225/225 tokens to EOS) | 126.9 tok/s | 135.9 tok/s | **+7.1%** |
| verify chunk 1 | 102.0 | 108.3 | +6.2% |
| verify chunk 4 | 184.8 | 210.0 | +13.7% |
| verify chunk 16 | 257.4 | 343.6 | **+33.5%** |
| verify chunk 32 | 324.8 | 531.0 | **+63.5%** |

**Quality (teacher-forced, frozen corpus):**

| path | PPL | top-1 agreement with fp32 |
|---|---|---|
| stock | 41.35 | 97.70% |
| ours | 41.40 | 97.87% |
| fp32 experts | 41.26 | — |

**Three architectures with end-to-end gains:**
- Qwen3-MoE: 128×8, no shared experts.
- gpt-oss: 32×4, MXFP4, biases, clamped SwiGLU.
- DeepSeek-V2: 64×6 + 2 shared, MLA.

Plain decode gains +7–9% on all three; verify/batch steps of 16–32 tokens gain +33–84%.

### Real routing #3: DeepSeek-V2-Lite (E=64 routed + 2 shared, top-6) (`runs/routing_dsv2.npz`)
| T | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| uniform | 6 | 11.4 | 20.8 | 34.9 | 50.8 | 61.3 |
| different sequences | 6 | 11.4 | 20.4 | 33.5 | 48.2 | 58.8 |
| consecutive tokens | 6 | 10.4–10.8 | 17.3–18.3 | 27.0–29.5 | 38.5–41.4 | 50.8–52.5 |

- **Weaker locality than the other two models:** adjacent-token overlap is 18–29%, against 9.4% for random routing (2–3×, compared with 4–7× for Qwen3 and 3–4× for gpt-oss). Expert popularity is nearly uniform (entropy 0.95–0.97, top-10% share 22–24%, no unused experts).
- **Correction of an earlier generalisation:** "~45% locality is a property of trained MoE routers" does not hold. Locality is model-dependent.
  - Plausible causes: shared experts absorb common tokens, plus DeepSeek's strong load balancing. Both are hypotheses, not tested.
- **This matches the e2e results:** the DeepSeek verify-step gain (+34% at 16 tokens) is the smallest of the three models (Qwen3 +48%, gpt-oss +70%). The expert-reuse benefit scales with locality, as the model in chapter 4 predicts (traffic ∝ U(T)).

### Decode gain vs context length (Qwen3-30B-A3B-4bit, `e2e.py --context N`, 256 tokens × 2 rounds)
| prompt length | ~30 | 1,024 | 4,096 | 8,192 |
|---|---|---|---|---|
| stock tok/s | 91.5 | 83.5 | 74.5 | 57.2 |
| ours tok/s | 99.8 | 90.3 | 79.9 | 59.6 |
| gain | +9.0% | +8.1% | +7.2% | **+4.3%** |

This is Amdahl's law. With longer contexts, attention (KV-cache reads) takes a growing share of each decode step, so a faster MoE layer matters less. The MoE kernels themselves are unchanged. Long-context serving would also need attention work, which is out of scope here.

## Open problems / next steps
1. ~~Small-batch + skewed routing~~ → solved by `metal_q4_tiles_mma` (round 1). Remaining: gpt-oss (d % 128 ≠ 0) tail path, T=2 with uniform routing (round 3 running).
2. **Regime dispatcher:** v31 for T ≤ 16, sorted `gather_qmm` above. The crossover sits around T≈16–32 on Qwen3 (RQ1).
3. **Statistics:** ≥5 process launches, bootstrap CI, AC power, `powermetrics` log.
4. **Same study on NVIDIA:** Marlin-MoE vs vLLM `fused_moe` on L4/H100 via Modal.
5. **Upstream:** propose the improvement to MLX's `gather_qmv` path for fine-grained MoE (f=768) as an MLX PR. This is a strong DeepMind-visibility artifact.
