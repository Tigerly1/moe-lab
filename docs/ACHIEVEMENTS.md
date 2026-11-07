# Achievements log

Verified results only: re-measured (≥2 process launches or `moelab.compare`), guard ok. Newest first.
Hardware: Apple M4 Pro, 48 GB, unless stated. Details and caveats: `docs/RESULTS.md`.

## 2026-11-07 — upstream: PRs to ml-explore/mlx-lm

- **PR 1 opened:** [ml-explore/mlx-lm#2043](https://github.com/ml-explore/mlx-lm/pull/2043), "server: reject empty prompts instead of ending the generation thread".
  - The bug: one empty prompt kills the batched server for every client.
  - The new test fails on main and passes with the fix; all 44 server tests pass.
- **PR 2 ready, not opened:** "SwitchGLU: grouped-expert Metal kernels for small batches of 4-bit experts".
  - Branch `Tigerly1/mlx-lm:moe-grouped-expert-kernels`, body in `docs/upstream/PR2_body_final.md`.
  - The full mlx-lm suite passes (411 tests); a gradient-through-custom_function test covers LoRA training.
  - **mlx-lm's own `mlx_lm benchmark`** (main vs branch, 2 passes), B=32:

    | model | speedup |
    |---|---|
    | Gemma-4 | **1.71×** |
    | Qwen3-30B | 1.56× |
    | DeepSeek-V2-Lite | 1.55× |
    | GLM-4.7-Flash | 1.40× |
    | LFM2.5 | 1.38× |

    B=1 is 1.03–1.08× (`runs/r9/upstream_mlx_lm_benchmark.log`).
  - GitHub refused to create it right after PR 1 (`CreatePullRequest` permission error; REST 404). This is probably a limit for first-time contributors. Open it once #2043 is reviewed, or from the web link.
- **Measurement pitfall caught before publishing:** the first A/B run `cd`'d into the branch checkout, so `python -m` loaded the branch in both arms (identical numbers). Fixed by running from a neutral directory and printing which code is imported.

## 2026-11-07 — round 9c: one patch for every MoE in mlx-lm; Gemma-4 / Qwen3.6 on a real server

- **Universal patch** (`moelab/universal.py`): replaces `SwitchGLU` itself. Each (token, expert) slot is a row with k=1, and at T=1 the k-slot GEMV runs without its k-reduction (`run_slots`, v4 `NORED`). Routing, weighting and shared experts stay the model's own code. `moelab.generate` / `moelab.serve` select it automatically for any 4-bit g64 SwitchGLU model without a dedicated patch.
- **Exactness:** with the bit-identical fallback as the kernel, logits equal stock on tiny models of 6 families with different routers: Qwen3-MoE, Qwen3.5, Gemma-4, LFM2-MoE, GLM-4.7-Flash, DeepSeek-V2. In total 778 tests pass.
- **Cost vs a dedicated patch** (Qwen3, kernels only, no fused router): 1.075× / 1.076× vs 1.086×.

**New families,** single stream (`--chat`, 2 launches) and concurrent requests (BatchGenerator):

| model | 1 chat | B=8 | B=32 |
|---|---|---|---|
| LFM2.5-8B-A1B 4-bit | 1.040× / 1.038× | 1.25× / 1.31× | **1.49× / 1.52×** (556 → 846 tok/s) |
| GLM-4.7-Flash 4-bit | 1.033× / 1.034× | 1.12× | **1.27×** (180 → 229 tok/s) |

- **Quality** on model-native text: PPL within 1%, top-1 agreement within noise. Two contended GLM runs were discarded and re-measured.

**Real `mlx_lm.server`** (`moelab.serve`, 256 tokens per reply, 2 rounds):

| model | 1 client | 8 clients | 32 clients |
|---|---|---|---|
| Gemma-4-26B-A4B | 1.11× | 1.18× | **1.43×** (152 → 218 tok/s; median reply 53 → 38 s) |
| Qwen3.6-35B-A3B | 1.07× | 1.08× | 1.13× (179 → 203 tok/s; 38 → 35 s) |

On the server, Qwen3.6 gains less than in BatchGenerator (1.24×) because a larger share of server time is prefill, which our kernels do not touch.

## 2026-11-07 — round 9b: MMA tiles on every 4-bit shape, Gemma-4 at 32 requests 1.25× → 1.70×

- **The cause.** The MMA expert tiles were gated on D, F % 128. Gemma-4 (F=704) therefore fell to scalar tiles, which are 2× slower than stock at load 4. The MMA kernels only need D, F % 64, which always holds with group-64 quantization.
- **Verification:** correct on F=704, 192 and D=320 at T=3–64; 530 kernel tests pass.
- **Layer level** (grid, Gemma-4 shape): T=32 goes from 1.28× to **1.66×**, and T=64 reaches 1.90×.
- **Load policy re-measured** (`runs/r9/grid_q4_mma64.json`): the "≤ 4 for non-%128" special case is gone. Gemma crosses at load 24–32, Qwen3.6 at load 8–12, which matches the existing rule.
- **Random-shape sweep:** 60 cases, all correct, median 1.21×, min 0.986×. The guard flagged contention at 92%, so this counts as a screen only.
- **Gemma-4 BatchGenerator,** 2 launches: B=8 **1.25×**, B=32 **1.70×** (232 → 394 tok/s).
- **Batched accuracy vs fp32** (`scripts/r9_batch_truth.py`): median logit error is ours 0.013 / 0.021 / 0.024 vs stock 0.028 / 0.027 / 0.025 at B=1 / 8 / 32. Argmax agreement is equal within one sequence.

## 2026-11-07 — round 9: the MoE models people actually run locally

**Choosing targets from data.** We ranked the 400 most-downloaded mlx-community repos and looked at the MoE models among them.
- The two most-downloaded MoE families had no moe-lab support:
  - Qwen3.5/3.6-35B-A3B (`qwen3_5_moe`): ~87k downloads;
  - Gemma-4-26B-A4B: ~35k downloads.
- Support existed only for gpt-oss, Qwen3-MoE and DeepSeek-V2.

**New support:**
- **Gemma-4:** int4 kernels gain a GeGLU activation (template `ACT`, tanh-GELU exactly as `nn.gelu_approx`) on every path: T=1 GEMV, scalar tiles, MMA tiles, and the fallback. The fused router is wired too (top-k → softmax → per-expert scale).
- **Qwen3.5/3.6:** gated shared expert added.

**Single-stream decode** (`--chat` prompt, 512 tokens, 2 launches):

| model | kernels | kernels + fused router |
|---|---|---|
| Gemma-4-26B-A4B 4-bit | 1.085× / 1.086× | **1.103× / 1.103×** (72.3 → 79.8 tok/s) |
| Qwen3.6-35B-A3B 4-bit | 1.031× / 1.029× | **1.106× / 1.105×** (83.8 → 92.7 tok/s) |

For Qwen3.6 the fused router matters: with E=256, the stock softmax plus argpartition is a large share of a step.

**Concurrent requests** (BatchGenerator, kernels + fused router, 2 launches, `runs/r9/batch_*`):

| model | B=1 | B=8 | B=32 |
|---|---|---|---|
| Gemma-4-26B-A4B | 1.106× / 1.109× | 1.139× / 1.140× | **1.251× / 1.254×** (230 → 290 tok/s) |
| Qwen3.6-35B-A3B | 1.105× / 1.103× | 1.095× / 1.093× | **1.240× / 1.242×** (289 → 358 tok/s) |

These gains are smaller than the 1.5–1.6× on the older models. The cause is structural, and the layer level shows it (compare, T=32: Qwen3.6 1.21×, Gemma-4 1.28×):
- **Qwen3.6:** with E=256 there is about 1 token per expert at B=32, so little weight reuse.
- **Gemma-4:** F=704 is not a multiple of 128, so T>1 ran on the scalar tiles instead of MMA. Fixed in round 9b: 1.70×.

**Accuracy** (`scripts/r9_layer_error.py`): per layer on real activations, against fp32 truth on the same inputs.

| model | our error | stock error | our layer closer to fp32 |
|---|---|---|---|
| Gemma-4 | 2.6e-3 | 5.1e-3 | 100% of 1,920 calls |
| Qwen3.6 | 3.2e-3 | 5.6e-3 | 100% of 2,560 calls |

- **Teacher-forced test** on model-native text, generated by the fp32 arm: PPL is equal within 1% across all arms. Top-1 agreement differences are within the noise of one 400-token text.

**Corrected harness pitfalls** (they would have produced false conclusions):
- **Raw-text perplexity on Gemma-4-IT:** PPL is ~300 on the corpus, which makes argmax agreement meaningless. We now use model-native text (`--quality-text`, `scripts/r9_native_text.py`).
- **Raw prompt on Gemma-4-IT:** the model falls into a "-----" loop and amplifies any perturbation, so all three arms disagree. We now use a chat prompt (`--chat`).
- **Tiny random models:** logits tolerance tests on them are meaningless, since even fp32 experts differ from stock by 4–14%. The patch tests now require exact equality with the bit-identical fallback, and mutation-checked: removing the shared expert fails.

**Ruled out (measured, not worth doing):**
- **8-bit and 6-bit kernels.** At T=1, stock already reads at 229–245 GB/s in 8-bit and 210–243 GB/s in 6-bit, the same level as our 4-bit kernel (239–255). The weights differ per call, so the SLC cache does not help (`scripts/r9_bits_probe.py`). The best case is a few percent.
- **Prefill MoE:** MoE is 55% of a Qwen3 4k-token prefill. But the layer already runs at 4.0 TFLOP/s against a 4.8 TFLOP/s dense-qmm ceiling for the same work, so the best case is ~1.10× prefill. Routing skew changes nothing (real vs random states: 1859 vs 1862 ms).
- **Further single-stream gains on Qwen3:** we are at ~83% of the bytes/bandwidth ceiling (105 of ~126 tok/s). The rest is non-MoE (attention, lm_head, dispatch).

## Earlier (rounds 1–8, summary)
- **Concurrent serving:** real `mlx_lm.server` with 32 clients is 1.32–1.38× faster on Qwen3-30B-A3B, gpt-oss-20b and DeepSeek-V2-Lite. On Qwen3 it is 1.48× faster than llama.cpp.
- **Single stream:** Qwen3 1.17×, gpt-oss 1.10×, DeepSeek-V2-Lite 1.12×. T=1 kernels reach 93–100% of the bandwidth floor.
- **Load policy:** load = T·k/E decides kernel vs fallback; the fallback is bit-identical to mlx-lm.
- **Withdrawn after checking:** "router picks 9% wrong", "gain vanishes at long context", "decode is CPU-bound".
