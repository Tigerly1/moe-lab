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

- **Weaker locality than the other two models:** adjacent-token overlap is 18–29%, against 9.4% for random routing (1.9–3.1×, compared with 6.9–7.6× for Qwen3 and 3.3–4.0× for gpt-oss). Expert popularity is nearly uniform (entropy 0.95–0.97, top-10% share 22–24%, no unused experts).
- **Correction of an earlier generalisation:** "~45% locality is a property of trained MoE routers" does not hold. Locality is model-dependent.
  - Plausible causes: shared experts absorb common tokens, plus DeepSeek's strong load balancing. Both are hypotheses, not tested.
- **Consistent with, but not proof of, a locality effect:** the DeepSeek verify-step gain at 16 tokens (+33.5%) is the smallest of the three models (Qwen3 +48%, gpt-oss +70%). At 32 tokens, though, DeepSeek (+63.5%) beats gpt-oss (+52%). Shapes, attention type (MLA) and quantization format also differ between the models, so "benefit scales with locality" remains a hypothesis (audit 3).

### Decode gain vs context length (Qwen3-30B-A3B-4bit, `e2e.py --context N`, 256 tokens × 2 rounds)
| prompt length | ~30 (reproduction run, 512 tok × 3) | 1,024 | 4,096 | 8,192 |
|---|---|---|---|---|
| stock tok/s | 91.5 | 83.5 | 74.5 | 57.2 |
| ours tok/s | 99.8 | 90.3 | 79.9 | 59.6 |
| gain | +9.0% | +8.1% | +7.2% | **+4.3%** |

The 1k/4k/8k points are single launches with 2 rounds each. The spread between rounds is under 0.4%.

This is Amdahl's law. With longer contexts, attention (KV-cache reads) takes a growing share of each decode step, so a faster MoE layer matters less. The MoE kernels themselves are unchanged. Long-context serving would also need attention work, which is out of scope here.

## Audit 3 (2026-11-07 ~06:10): review of everything added after audit 2, and the fixes
1. **Real bug in round 6:** fp32 activations on the T>1 MMA path lost ~10% accuracy silently.
   - Cause: the MMA used x rounded to half, while Σx came from exact fp32; the OFF=1024 cancellation amplified the difference.
   - bf16/fp16 are exact, and all models run in bf16, so no headline number changed.
   - **Fix:** the dispatcher sends fp32 at T>1 to the exact scalar tile path; `tiles_v2` now asserts bf16/fp16.
   - **New test** `tests/test_fp32_activations.py`: it fails on the old dispatcher (5 cases) and passes now.
2. **README mixed kernel versions:** the layer table, crossover and chunk-32 numbers predated round 6. All of them were re-measured on the final kernels:

   | | T=16 | 32 | 64 | 128 | 256 | 512 |
   |---|---|---|---|---|---|---|
   | crossover, traces (`runs/crossover_r6_q4_qwen3_trace.json`) | 1.95 | 2.43 | 1.53 | 1.29 | 1.10 | **0.98** [0.96, 0.99] |
   | crossover, uniform (`…_uniform.json`) | 1.27 | 1.53 | 1.87 | 1.36 | 1.09 | **0.99** [0.98, 0.99] |

   - At T=512 the kernel is now 1–2% *slower* than mlx-lm, so `MAX_T=256` stays: stock runs above 256.
   - Whole-model chunks (Qwen3): 32 → 1.83×, 64 → 1.37×, 128 → 1.19×, 256 → 1.09×.
   - gpt-oss: 32 → 1.52×, 64 → 1.28×.
   - The README layer table now uses the reproduction-run values.
3. **DeepSeek quality numbers were not saved, and the README overstated quality.**
   - `quality_check` now writes JSON (`runs/e2e_dsv2_quality.json`); the re-run reproduced the numbers exactly.
   - The README now says "PPL within 0.1%; top-1 agreement with fp32 as high or higher". On DeepSeek our PPL is 0.1% further from fp32.
   - The e2e JSON now records `tokens_identical` and `first_diff`: DeepSeek 225/225 (`runs/e2e_dsv2_plain_r6.json`, +7.2%).
4. **Over-claims softened:**
   - "locality explains the smaller gain" is now a hypothesis;
   - the "~45%" locality figure is qualified per model;
   - the context table header is corrected;
   - rounding is fixed (+33.5% / +63.5%, 93%).
5. **Guard added:** the DeepSeek patch asserts that `sharding_group` is None, since tensor-parallel is not supported.

The audit verified as sound:
- the round-6 kernel maths for bf16/fp16;
- the xfail of the historical kernel;
- the DeepSeek patch matches stock;
- the DeepSeek and context numbers match their JSONs;
- the routing-trace claims;
- the generate CLI;
- the reproduction outputs.

## Final reproduction #2 on the final code (after audit 3), 2026-11-07 06:22–06:33
`scripts/reproduce_headline.sh` was extended to cover DeepSeek, gpt-oss chunk-32 and the crossover. Output: `runs/repro/`, `runs/repro_log.txt`.

| check | result |
|---|---|
| tests incl. full-model multi-token (3 models) | **346 passed**, 3 xfailed |
| int4 Qwen3 traces T=1/8/16/32 | 1.200 / 1.724 / 2.054 / 2.455 |
| MXFP4 Qwen3 traces T=1/8/16/32 | 1.264 / 1.736 / 2.060 / 2.404 |
| int4 Qwen3-235B T=1/16 | 1.068 / 1.265 |
| T=1 slope (Qwen3) | mlx-lm 76.4% / v4 93.4% of floor, 1.222× |
| Qwen3 e2e plain / verify 4 / 16 / 32 | +8.6% / +22.5% / +48.3% / +83.4% |
| gpt-oss e2e plain / verify 4 / 16 / 32 | +8.1% / +27.4% / +69.4% / +52.4% |
| DeepSeek e2e plain (identical 225/225) / verify 4 / 16 / 32 | +7.3% / +13.6% / +33.4% / +63.3% |
| crossover uniform T=64/256/512 | 1.867 / 1.087 / 0.990 |

## Round 7 (2026-11-07, daytime): wider and narrower testing, a fused router, and an audit

Goal: find wrong or slower results outside the shapes the kernels were tuned on, then look for the next real gain.
Rule used throughout: a quick screen only nominates cases. A case counts only after a re-measurement with `moelab.compare` (3 process launches, CI by launch, fresh seeds). An adversarial read-only audit (fresh agent) reviewed the first version of this section; its findings and fixes are at the end.

### Harness changes
- **GPU contention guard** (`moelab/guard.py`):
  - Every `compare`, `slope`, `e2e`, `sweep` and `grid` run brackets itself with a 512 MB read-bandwidth probe.
  - The probe is compared with the 90th percentile of the last 40 probes for the same (chip, power source); below 92% the run is flagged CONTENDED.
  - With fewer than 5 probes in history the verdict is "calibrating". The v1 design used the best probe ever seen; the audit noted that this drifts upward with noise.
  - Why: a T=1 slope run with Chrome and the display active gave 80% of floor instead of 93%, while the A/B ratio barely moved (1.196× vs 1.222×). Ratios are robust to contention; absolute "% of floor" numbers are not.
  - Check: a concurrent matmul loop drops the probe from ~236 to 8 GB/s and the guard fires.
  - Known weakness: the history is a sliding window, so long sustained contention becomes the norm after ~7 runs (documented in `tests/test_guard.py`).
- **Narrow edge cases** (`tests/test_edge_cases.py`):
  - Coverage:
    - T at every dispatch boundary (1, 2, 3, 4, 5, 7, 8, 9, 15, 17, 31, 63, 65, 127, 129, 257);
    - d or f not a multiple of 128, plus k=1, k=E and E=1;
    - zero or unnormalized combine weights (an all-zero row must give exactly zero);
    - every token on the same experts;
    - slot-order invariance;
    - batch consistency (token t inside T=16 matches T=1);
    - 100× activations.
  - Each case runs twice: with the shipped dispatch, and with the fallback disabled. In the second mode the test fails if the fallback runs, so the custom kernels are exercised at every T. (Audit: in the first version many cases silently reached the fallback.)
  - Mutation check: swapping one expert of one token raises the error 6–15× over the clean run, 2–5× over the tolerance.
- **Wide random sweep** (`moelab/sweep.py`): 60 random shapes per run (d 512–5120, f 256–4096, E 8–256, k 1–8, T 1–128, bf16/fp16, uniform/zipf/one-hot routing), with correctness checks and a single-process A/B screen.
- **Targeted grid** (`moelab/grid.py`): real-model and small shapes × T chosen by load = T·k/E (average tokens per expert), with the **custom kernels forced** so the grid measures what it is meant to measure.

### Finding 1: correct everywhere
- 480 random cases (8 sweeps, 3 of them on the final code) and every edge case pass against the fp32 reference.
- Total suite after round 7: 648 passed + 3 xfailed with `MOE_SLOW=1`. This includes real-model multi-token logits, run with both router modes.

### Finding 2: what decides the winner at T>1 is load = T·k/E, not T
On the original dispatcher, 17/60 random int4 cases and 16/60 MXFP4 cases were below 1.0× in the screen. The grids explain why (`runs/grid_q4.json`, `grid_mxfp4.json`, `grid_small_*.json`, custom kernels forced, all guard ok):

| shape class | custom kernel wins up to | then |
|---|---|---|
| int4, d and f multiples of 128, d·f ≥ 1.5M (Qwen3-30B, DeepSeek-V2-Lite, OLMoE, Qwen3-235B) | load 24: 13 of 14 cells > 1 at load 12–24 (the exception: 0.957) | load 32: 0.92–1.06 |
| int4, smaller experts (d·f 0.13–1.05M) | load 8 (all 6 shapes) | load 12–16: 0.90–1.03, load 24: 0.82–0.97, load 32: 0.78–0.94 |
| int4, d=2880 (scalar tile path) | load 4 | load 6: 0.93, load 8–32: 0.71 → 0.44 |
| MXFP4, d·f ≥ 1.5M | load 24: 15 of 18 cells > 1 (the rest 0.95, 0.97, 1.00) | load 32: 0.88–1.11 |
| MXFP4, smaller experts | load 8 | load 16: 0.96–1.18 |

- **Policy** (`metal_q4_moe.custom_wins`, `metal_mxfp4_moe.load_max`, gpt-oss patch in `e2e.py`): above the per-class threshold, the dispatchers call `moelab/fallback.py:gather_moe`. It is a copy of mlx-lm's own SwitchGLU path, and `tests/test_fallback.py` checks it is bit-identical to the real module. Above the threshold the result therefore *is* the baseline.
- A first version used load 8 everywhere; the audit showed it threw away measured wins (up to 1.23× on Qwen3-235B). The thresholds were refitted.
- **Rigorous check at the boundaries** (`runs/r7_policy/*.json`, seeds 9100+, all guard ok):

| case | load | result |
|---|---|---|
| Qwen3-30B int4, T=192/256/384 | 12/16/24 | 1.070 / 1.089 / 1.054 |
| Qwen3-30B int4, T=512 | 32 → fallback | 1.000 |
| Qwen3-235B int4, T=256/384 | 16/24 | 1.124 / 1.100 |
| gpt-oss int4, T=32 / 48 | 4 / 6 → fallback | 1.345 / 1.000 |
| gpt-oss MXFP4, T=96/128/192 | 12/16/24 | 1.078 / 1.132 / 1.064 |
| small MXFP4 (d768 f512 E64 k8), T=64/96/128 | 8 / 12, 16 → fallback | 1.232 / 1.001 / 0.996 (with threshold 16: 0.989 at T=128, so lowered to 8) |
| small int4 (d1024 f512 E32 k4), T=64/96 | 8 / 12 → fallback | 1.224 / 1.001 |

- **Before vs after on the gpt-oss int4 shape** (`runs/crossover_r7_q4_gptoss.json`, guard ok): T=64 went 0.71 → 1.00, T=128 0.51 → 0.98 [0.98, 1.02].
- **Headline numbers are unaffected:** every headline cell runs at load ≤ 4, where all policy versions dispatch the same kernel.

### Finding 3: final sweeps and the remaining losses
On the final code (`runs/sweep_*_final*.json`):

| run | guard | correct | median | min | below 1.0 |
|---|---|---|---|---|---|
| q4, seed 11 | **CONTENDED (87%)** | 60/60 | 1.239 | 0.762 | 18 |
| mxfp4, seed 12 | ok | 60/60 | 1.257 | 0.943 | 15 |
| q4, held-out seed 31 | ok | 60/60 | 1.307 | 0.550 | 9 |

- **Every screen value below 0.95 was re-measured with compare:**
  - **False alarms of the screen:** 0.550 (d2048 f512 E8 k2, T=1) became **1.313** [1.29, 1.33]. Earlier, 0.51 / 0.78 / 0.88 became 1.04–1.22 (`runs/t1_*.json`, `runs/recheck_*.json`; 3 of those 11 rechecks ran CONTENDED: `t1_d2048f2048E32k8` at 73%, plus two at 91–92%).
  - **Real, small losses on tiny experts:**
    - d512 f1536 E32 k1, T=3: **0.929** [0.92, 0.95];
    - d768 f256 E64 k4, T=128: **0.953** [0.95, 0.96].
    - These experts have d·f of 0.2–0.8M, 2–8× smaller than the smallest model studied (Qwen3, 1.57M). The thresholds were not tuned to individual points.
  - **Real, large loss: saturated skew.** When all k experts of a step get every token (synthetic one-hot: d2560 f256 E64 k8, T=32), the policy sees load 4 while the true load is 32, and the custom kernel runs at **0.758** [0.757, 0.765] (`runs/r7_policy/q4_onehot.json`). The other one-hot screens give 0.76–0.85 (int4) and 0.94–0.96 (MXFP4).
  - **Fallback cases:** most remaining flags are 0.98–1.00. The fallback equals the baseline, so this is noise: a recheck of 0.970 became 1.030 [0.98, 1.06].
- **Do real routers saturate like that?** In real traces the busiest expert in a T=32 window gets a median of 26 (Qwen3), 22 (gpt-oss) or 13 (DeepSeek) tokens, but the mean per active expert is only 5.3, 6.6 and 3.7. The real-trace layer speedups at T=32 (2.45×) already include this skew. Saturating all k experts at once was not seen.
- **Observation: one hot expert per layer, per document.** In each layer, one expert is chosen by a median 68% (Qwen3), 54% (gpt-oss) and 31% (DeepSeek) of a document's tokens; uniform routing would give 6–12%.
  - The hot expert depends on the document: in 98% (Qwen3) and 92% (gpt-oss) of layers, the three documents have different hot experts.
  - This is the main source of weight reuse at T>1.
  - Caveat: 3 documents, ~5k tokens. "Super expert" work exists in the literature, so this is not claimed as new.

### Finding 4: a fused router tail, +5–6 points end to end
- **Cost of the router:** in Qwen3 (48 layers, T=1, `runs/router_cost.py`) it costs **1.49 ms per token**. The gate GEMV is only 0.26 ms; the rest is 4–5 tiny dispatches per layer (softmax, argpartition, take, renormalize).
- **Kernel** (`moelab/router.py`): one simdgroup per token computes the top-k over the logits plus the weights.
  - Weights: softmax over the top-k (Qwen3 `norm_topk_prob`, and gpt-oss, which applies softmax to the top-k values), or full-softmax probabilities (DeepSeek-V2 greedy, with `routed_scaling_factor`).
  - Any E ≤ 1024. Ties go to the lowest id.
  - NaN logits count as −∞. With fewer than k finite logits the remaining slots get unused ids and weight 0 (audit: the first version wrote out of bounds there).
  - DeepSeek-V3 gates (sigmoid + correction bias) stay stock.
  - Opt-in: `e2e.py --fused-router`; on by default in `moelab/generate.py`.
- **Semantics change, measured** (`runs/router_ties.json`, corrected in round 8 by `runs/r8/router_fix_qwen3_moe.json`):
  - On 512 real tokens × 48 layers, stock and the fused kernel pick different experts in **9.00%** of token-layers.
  - Round-7 claim, **withdrawn**: that stock's bf16-rounded softmax decides these arbitrarily and the fused kernel "picks the larger logit".
  - Round 8 checked stock with its own selection code (`argpartition(g, kth=-k)[..., -k:]`). Stock always returns a valid top-8 by logit (0 of 24,576 wrong). Every difference is an exact tie of the **bf16 gate logits** themselves: two experts with the same score, so either choice is equally correct.
  - The fused kernel breaks ties by lowest id, stock by its partition order; an fp32 softmax cannot change this. That tie-breaking difference is why greedy outputs diverge from stock at token 2.
  - The first round-8 check also used the wrong (DeepSeek-style) selection form for Qwen3; it is superseded.
- **End to end** (`runs/r7/*.json`, same session, 512 tokens × 3 rounds, all guard ok; stock is unmodified mlx-lm in every row):

| model | our kernels | kernels + fused router | verify 4 / 16 / 32, kernels + router |
|---|---|---|---|
| Qwen3-30B-A3B | +11.4% | **+17.3%** | 1.26 / 1.47 / 1.79 (kernels only: 1.23 / 1.48 / 1.83) |
| gpt-oss-20b | +7.7% | **+9.3%** | 1.30 / 1.71 / 1.54 |
| DeepSeek-V2-Lite | +7.7% | **+12.1%** (output still identical, 225/225) | 1.18 / 1.33 / 1.63 |

  - The router helps at T=1. At verify steps its cost is spread over the chunk and the difference is within run-to-run drift (both arms moved ~4% between runs).
  - Today's stock Qwen3 ran at 84 tok/s vs 91.6 at night, with the guard ok. The kernels-only gain is therefore higher today (+11.4%) than the night's +8.6–9.0%; the cause is not known. Only same-session ratios are compared here; the README headline keeps the night numbers.
- **Quality** (`runs/r7/*_quality_fused.json`, teacher-forced, frozen corpus):
  - The fp32 truth uses the **stock** router. The audit found the first version circular, because its truth also used the fused router.
  - A second truth with exact routing is reported as `fp32_exact`.

| model | PPL stock / ours (kernels + router) / fp32 (stock routing) / fp32_exact | top-1 vs fp32: stock / ours | mean abs ΔNLL vs fp32: stock / ours |
|---|---|---|---|
| Qwen3 | 30.80 / 30.73 / 30.90 / 30.63 | 97.00% / 96.35% | 0.074 / 0.087 |
| gpt-oss | 181.2 / 177.3 / 172.4 / 169.2 | 90.47% / 91.46% | 0.143 / 0.129 |
| DeepSeek | 41.35 / 41.39 / 41.26 / 41.32 | 97.70% / 97.70% | 0.031 / 0.032 |

  - Honest reading for Qwen3: the fused router changes the tie decisions. Against the stock-routed truth it is slightly further away (0.087 vs 0.074); against the exact-routed truth it is closer (0.075 vs 0.086). Perplexities are within 0.9% of each other, so this is a different, not a better, model.
  - On gpt-oss, ours is closer to the truth than stock. On DeepSeek they are equal.

### Finding 5: long context
Decode time breakdown (`moelab/breakdown.py`, Qwen3, real KV cache; "no-moe" = MoE block returns zeros; guard ok):

| context | stock ms/token | ours | no-moe | MoE stock → ours | MoE share (stock) | e2e |
|---|---|---|---|---|---|---|
| 128 | 13.35 | 12.60 | 5.58 | 7.77 → 7.02 (1.11×) | 58% | 1.059× |
| 1k | 14.78 | 13.56 | 6.66 | 8.12 → 6.90 (1.18×) | 55% | 1.090× |
| 4k | 16.56 | 15.45 | 8.38 | 8.17 → 7.07 (1.16×) | 49% | 1.071× |
| 8k | 20.21 | 19.47 | 12.06 | 8.15 → 7.41 (1.10×) | 40% | 1.038× |
| 16k | 24.39 | 24.11 | 16.61 | 7.79 → 7.50 (1.04×) | 32% | 1.012× |

- Attention grows from 5.6 to 16.6 ms (Amdahl).
- The MoE-only gain also falls from 1k on (1.18 → 1.04; the 128 point, 1.11, does not fit a monotone trend), although MoE work does not depend on context.
- **Tested and rejected:** layer-level chains with 4–34 MB of streaming reads per layer (`runs/kv_interleave.py`), or with real SDPA over a 1k–16k KV cache (`runs/sdpa_interleave.py`), between MoE calls keep the gain at 1.25–1.38×. So it is not simple memory interference.
- **Partial clue:** with SDPA in the chain, our MoE time rises ~0.7 ms at 16k (4.83 → 5.57), stock's ~0.4 ms.
- **Open:** CPU-encode/GPU overlap in the real model, or attribution error of the "no-moe" subtraction. Round-to-round spread is 0.1–0.3 ms.

### Audit of round 7 (fresh agent, read-only) and what was done
| # | finding | fix |
|---|---|---|
| 1 | Fused DeepSeek path also caught `deepseek_v3` (no `topk_method`, different gate) | restricted to `deepseek_v2` |
| 2 | Router kernel wrote out of bounds when fewer than k finite logits | NaN → −∞, unused ids with weight 0; tests with NaN/−∞ rows |
| 3 | Load-8 policy discarded measured wins; small-expert rule rested on contended screens | clean grids incl. small shapes, refitted thresholds, boundary rechecks |
| 4 | Contended runs not disclosed | flagged in the tables above |
| 5 | "medians 1.05–1.08×" misquoted | replaced by the measured values |
| 6 | 0.73× one-hot number had no file | re-measured: `runs/r7_policy/q4_onehot.json`, 0.758 |
| 7 | Router quality comparison circular; no file for the 9% tie claim | stock-router truth + `fp32_exact`; `runs/router_ties.json` |
| 8 | Edge tests reached the fallback, not the custom kernels | forced-custom mode that fails if the fallback runs |
| 9 | Weak fallback, DeepSeek and router tests | bit-identity vs SwitchGLU; routed_scaling_factor 2.5 and a call assertion; bf16-path tie test; E up to 1000, k=32 |
| 10 | Guard's running maximum drifts | p90 of recent history per (chip, power) |
| 11 | Stale comment in `e2e.py`, non-monotone long-context wording | fixed |

## Round 8 (2026-11-07, afternoon): when does it help, checked against the latest mlx-lm

Method:
- A ranked backlog (`docs/RESEARCH_BACKLOG.md`, expected value = P × impact ÷ cost, with kill criteria).
- 4 research agents, one narrow question each, web only, with key claims re-checked by me (`docs/related_work_r8.md`).
- GPU jobs run strictly one at a time.

### Concurrent requests: the main practical win
`moelab/batch_bench.py` drives mlx-lm's `BatchGenerator` (what `mlx_lm.server` uses for parallel requests) with B different prompts, greedy, no stop tokens, 128 tokens. It reports aggregate tok/s, with interleaved stock/ours rounds.

| B | Qwen3-30B-A3B | gpt-oss-20b | DeepSeek-V2-Lite |
|---|---|---|---|
| 1 | 1.16 | 1.09 | 1.12 |
| 4 | 1.22 | 1.24 | 1.15 |
| 8 | 1.23 | 1.43 | 1.27 |
| 16 | 1.33 | 1.55 | 1.27 |
| 32 | **1.59** (222 → 352 tok/s) | **1.60** (224 → 359) | **1.54** (312 → 481) |

- Setup: kernels + fused router, mlx-lm 0.32.2, guard ok (`runs/r8/latest/*_batch_fused.json`).
- On mlx-lm 0.31.3, 2 launches each, the ratios agree within ±0.05 (`runs/r8/batch_*_L1/L2.json`).
- Kernels only (no router): the same at B ≥ 4; the router matters only at B ≤ 2.
- **Correctness in batch:** one batched decode step against the fp32-expert truth (`runs/r8/batch_truth_check.json`).
  - At B=32, stock's worst row deviates 0.194 (rel.), ours 0.068; argmax agreement is 93.8% for both.
  - At B=8, both have 87.5%: nearly tied logits flip under bf16.
  - The early divergence of greedy streams at B ≥ 8 is therefore model sensitivity, not a kernel bug.
- **Positioning** (agent C, `docs/related_work_r8.md`):
  - No published B=1–32 MoE batched sweep exists for MLX/Metal.
  - Grouping tokens by expert is standard on NVIDIA (vLLM fused_moe; llama.cpp #12612 map0 + mul_mm_id). The contribution is bringing it to MLX with measured serving gains.
  - The baseline includes MLX's merged gather_qmm improvement (#4572, in 0.32.3+).

### Baseline is current
Everything above was re-measured on **mlx-lm 0.32.2** (latest; Python 3.14 venv `.venv-latest`) with MLX 0.32.4 (latest), guard ok:
- single stream: Qwen3 1.090 (kernels) / 1.166 (+router), gpt-oss 1.075 / 1.101, DeepSeek 1.073 / 1.119;
- patch tests on real models pass on both mlx-lm versions (119 + 648).
- The MoE path and the router code are unchanged between 0.31.3 and 0.32.2; `generate.py` changed a lot (~1000 lines).

### Hypotheses tested and rejected
1. **"Decode is CPU-bound, and fewer ops is why we win."** Rejected (`runs/r8/cpu*`).
   - 4 busy CPU threads change nothing.
   - 8 busy threads slow both arms and shrink our gain (1.09 → 1.01 kernels, 1.16 → 1.07 fused).
   - Host graph-build cost per token is 0.32 ms for ours vs 0.49 ms for stock (`runs/r8/host_overhead.json`), so caching it is not worth it (backlog #6 killed).
2. **"The MoE gain shrinks with context."** Not reproduced.
   - Two clean launches (`runs/r8/longctx_*_clean*.json`) give a constant MoE-only gain: kernels 1.14–1.17× at 1k and 1.13–1.14× at 16k; fused 1.27× at both.
   - The e2e gain falls only by Amdahl: fused 1.13× at 1k, 1.07× at 16k.
   - The round-7 single run (1.04× at 16k) is not reproducible. The guard, which measures bandwidth only, did not flag whatever disturbed it.
   - Side result: forcing MLX's command-buffer policy changes the picture a lot. One buffer per step: ours 0.93× at 16k. Commit after every op: 1.01–1.03× e2e. So the gain depends on MLX's default commit policy (50 ops / 50 MB on `applegpu_g16s`).
   - Raising `MLX_MAX_MB_PER_BUFFER` does not speed up stock (the confound raised in mlx-lm #956), so it does not explain our gain.
3. **"mlx-lm's bf16 router softmax picks wrong experts in 9% of cases."** Withdrawn (see the Round 7 correction).
   - Stock always picks a valid top-k by logit (0 of 24,576 wrong); the differences are exact bf16 *logit* ties.
   - The planned upstream issue is dropped (backlog #2 killed).

### Mistakes made in this round (and caught)
- A router check used the DeepSeek selection form for Qwen3 (superseded).
- zsh did not split `$VAR` argument strings twice: gpt-oss runs did not start, and a long-context queue ran nothing. Every queue is now a bash script.
- A `set -e` script stopped after a test failure caused by a renamed class in mlx-lm 0.32 (`gpt_oss.SwiGLU` → `SwigluOAI`). The test was fixed and the runs repeated.

### Still needs the user
- a second chip (M2 16 GB: `scripts/portability_m2.sh`);
- `modal setup` for NVIDIA;
- Xcode (Metal compiler) for an MLX upstream kernel PR.

## Open problems / next steps
1. ~~Small-batch + skewed routing~~ → solved by `metal_q4_tiles_mma` (round 1). Remaining: gpt-oss (d % 128 ≠ 0) tail path, T=2 with uniform routing (round 3 running).
2. **Regime dispatcher:** v31 for T ≤ 16, sorted `gather_qmm` above. The crossover sits around T≈16–32 on Qwen3 (RQ1).
3. **Statistics:** ≥5 process launches, bootstrap CI, AC power, `powermetrics` log.
4. **Same study on NVIDIA:** Marlin-MoE vs vLLM `fused_moe` on L4/H100 via Modal.
5. **Upstream:** propose the improvement to MLX's `gather_qmv` path for fine-grained MoE (f=768) as an MLX PR. This is a strong DeepMind-visibility artifact.
