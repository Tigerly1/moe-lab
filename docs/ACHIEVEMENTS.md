# Achievements log

Verified results only: re-measured (≥2 process launches or `moelab.compare`), guard ok. Newest first.
Entries that do not meet this rule are marked in place (see the 2026-11-10 verification pass).
Hardware: Apple M4 Pro, 48 GB, unless stated. Details and caveats: `docs/RESULTS.md`.

## 2026-11-12 — #2044 reduced to the batch path (commit `61accb9`)

- **Removed:** the T = 1 kernels (gate+up GEMV, down GEMV). One token now takes the generic path; upstream #2053 covers it.
- **Moved:** `mlx_lm/models/switch_kernels.py` → `mlx_lm/kernels/switch_glu_batch.py`, entry point `switch_glu_by_expert`; tests in `tests/test_switch_glu_batch.py`. PR size 742 → 637 added lines, 5 files.
- **Checked:** `tests.test_switch_glu_batch` + `tests.test_kernels` 14 tests OK; `tests.test_models` OK (skipped=1). Full suite not rerun. black / isort / ruff at the pinned versions pass.
- **Composition with #2053, by an actual merge** (local branch `compose-2053`, not pushed, **not run**): 3 files conflict in 5 hunks, all adjacent one-line additions (imports, `__all__`, the `metal_act` comment). After the merge `combine` handles T = 1 and falls through to `__call__`, where the batch hook sits, for T > 1.
- **Not re-measured:** the B=8 / B=32 numbers are from before this commit. The T > 1 kernel code is unchanged, so they should hold; B=1 generation is now equal to main by construction. Prefill at B=1 (T = 128) still takes the kernels.
- The PR checkout moved from temporary storage (files were being purged) to a clone next to moe-lab.

## 2026-11-12 — upstream moved: maintainer PRs #2053 and #2054 cover the single-token path and the router

No measurement of ours in this entry. Read from the PR pages and diffs on 2026-11-12.

| PR (ml-explore/mlx-lm) | author, opened | what | their numbers (M5 Ultra, B=1, tok/s) |
|---|---|---|---|
| #2053 | Pierre Ablin, 2026-11-12 00:23 UTC | `SwitchGLU.combine`: one token, two kernels (gate+up+GLU+routing weight; down + weighted sum over experts). Affine 2/3/4/5/6/8-bit, group size 32/64/128, bf16/fp16. 39 model call sites | Qwen3-30B 156.3 → 175.4, DeepSeek-V2-Lite 240.0 → 270.5, Gemma-4-26B 156.7 → 171.2, GLM-4.7-Flash 110.1 → 120.5 |
| #2054 (draft) | same, 00:54 UTC | `softmax_topk`: softmax + top-k + renormalize in one kernel (ties: lowest id); `expert_routing` extended to DeepSeek-V3 groups; 34 MoE blocks | Qwen3-30B 156.3 → 166.1 (1.06×), LFM2.5 328.1 → 388.4 (1.18×), Ling-mini 330.8 → 412.3 (1.25×) |

Consequences:
- **Fused router port: dropped.** #2054 is the same kernel as `moelab/router.py` (same algorithm, same tie rule), wired into more models. Their 1.06× on Qwen3 agrees with our stored 1.17× vs 1.09–1.11×. A second PR from us would be a duplicate. Backlog #7 closed.
- **Our T = 1 path in #2044 is now redundant upstream.** #2053 covers more formats and also fuses the weighted sum, which ours leaves to three small ops. Not compared on the M4 Pro: `scripts/r14_upstream_t1.py` (upstream kernels on a synthetic layer) is written but **not run**; it executes code from the upstream PRs, which the user runs by hand.
- **The T > 1 path of #2044 is not covered by either PR.** #2053 states it targets a single token and names #2044 as the small-batch counterpart. `combine` falls back to `SwitchGLU.__call__` for T > 1, which is where our hook sits, so the two compose: #2053 at T = 1, #2044 at T > 1. Both PRs add the same `metal_act` tag (0 = SwiGLU, 1 = GeGLU).
- **Policy problem with #2044.** `CONTRIBUTING.md` of mlx-lm (since 2026-08-21) prohibits AI-written PR descriptions and requires an AI usage disclosure. The description of #2044 was written by Claude. It has to be rewritten by the author, with the disclosure line. Missed at opening (2026-11-07) and at the update (2026-11-11).

## 2026-11-11 — round 13: cost of one small Metal dispatch (probe)

Question: where can a larger M4 win still come from? Synthetic chain, no model (`scripts/r13_dispatch_probe.py`): 48 "layers", each one 4-bit GEMV (2048 → 4096, 4.7 MB) plus n tiny ops on a 2048-vector. Evidence: `runs/r13/dispatch_probe_*.json`. M4 Pro on AC, one process per setting, no bandwidth guard.

| command buffer (ops) | µs per `rms_norm` | µs per `add` | 48 GEMVs alone |
|---|---|---|---|
| 50 (default), launch 1 / 2 | 2.72 / 2.70 | 1.69 / 1.70 | 1.55 / 1.58 ms |
| 25 | 2.72 | 1.72 | 1.54 ms |
| 200 | 2.72 | 1.61 | 1.67 ms |
| 1000 | 3.22 | 2.18 | 1.68 ms |

- **Measured:** a small op costs 1.7–2.7 µs in a decode-like chain, linear in the number of ops (n = 0–32 per layer).
- **Measured:** larger command buffers do not help here (200: same slope, GEMVs 8% slower; 1000: worse). Backlog #14 is negative on this proxy; it was not run on a real model.
- **Estimate, not measured:** a Qwen3 decode layer issues about 21–31 small ops (counted from the model code), so 48 layers spend roughly 2–3 ms of an 11.4–12.6 ms token on them. Removing ~9 per layer by fusion would save about 1 ms (~8%). This is the lead for backlog #16; no e2e number exists yet.
- **Side observation:** one 4.7 MB GEMV takes 32 µs in the chain (146 GB/s effective, against 224–244 probed), so small dense matrices also pay a fixed cost per call.

### Round 13b: which small ops of a Qwen3-MoE layer are worth fusing (mostly: none)

Synthetic model with the real Qwen3-30B-A3B layer dimensions, 6 layers, 16 experts, random weights (~320 MB, `scripts/r13_layer_ablation.py`), single-stream decode with KV cache, baseline = mlx-lm with the #2044 expert kernels. 5 process launches (`runs/r13/layer_ablation_{1..5}.json`), best of 5–7 rounds each. Baseline: 304–308 µs per layer. "(bound)" variants change the output and only show the most a fused kernel could save.

| change | saved per layer, µs (launches) | share of the layer |
|---|---|---|
| both residual adds removed (bound) | −0.2 / 1.0 / 1.1 / 1.0 / 0.1 | ≤ 0.4% |
| q_norm and k_norm removed (bound) | 2.5 / 4.9 / 0.9 / 2.4 / 0.2 | ≤ 1.6% |
| router tail removed: softmax, argpartition, take, sum, divide (bound) | 25.8 / 21.1 / 23.0 / 21.2 / 21.0 | 7–8% |
| router tail fused (`moelab/router.py`), exact up to tie order | 8.3 / 7.7 / 5.8 / 6.1 / 3.8 | 1–3% |
| same, kernel writes uint32 ids and bf16 weights (no casts) | 8.2 / 4.9 (2 launches) | 2–3% |

- **Killed: backlog #16** (fusing add + norm, q/k norm + RoPE). These ops cost ≤ 2% of a layer together; the kill criterion was 3% e2e. The 1.7–2.7 µs per op from the first probe does not carry over to the real layer.
- **The router tail is the only small-op item that matters:** 21–26 µs per layer. The existing fused router recovers 4–8 µs of it on this model (E = 16). On the real Qwen3 (E = 128) the stored e2e runs show kernels + fused router at 1.17× against 1.09–1.11× for kernels only. The fused router is not part of mlx-lm#2044.
- **A custom `mx.fast.metal_kernel` call costs the same as a built-in op:** 2.71–2.75 µs against 2.70–2.77 µs for `rms_norm` (`dispatch_probe_custom_{1,2}.json`).
- **Per `mx.eval`:** 124–227 µs (`gemv_chain_length.json`, chain lengths 6–48). In decode that is about 2% of a token.
- **Stock dense GEMV:** 212–228 GB/s marginal on the fast path (`gemv_fixed_cost_{1,2}.json`). K = 2880 and 5760 (not multiples of 512) run at 103–105 and 168 GB/s effective: the known K % 512 cliff (backlog #10).

**Layer budget, single stream (estimate from these runs):** a Qwen3 layer reads 31.8 MB of weights, 141 µs at 225 GB/s, which is 54% of the 262 µs measured in the real model. The rest is spread over the router tail (21–26 µs), about five serial weight kernels with a few µs of ramp each, ~10 small ops, attention over the KV cache and the per-token sync. No single remaining item is above 8%.

## 2026-11-11 — audit of the open PR ml-explore/mlx-lm#2044

Question: is the published PR (head `ea3da29`) still correct, and does it make sense on other Apple GPUs (M5, M6)?
Evidence: `runs/r12/pr2044_audit.log`, scripts next to it. Measured on the M4 Pro only.

**Holds (M4 Pro):**
- Kernel path vs stock vs fp32 truth: 1,136 synthetic cases (8 shapes, 4 routings, bf16/fp16), 0 bad. Worst error vs fp32: ours 5.0e-3, stock 7.9e-3. Shape and dtype equal stock.
- Real model (LFM2.5-8B, chat prompt): logits error vs fp32 is 4.7e-2 on the kernel path and 6.2e-2 on stock; generated text identical.
- Raw prompt: the kernel path gives the same text as fp32; bf16 stock is the arm that diverges.
- mlx-lm unit suite: 411 tests on the head, 413 with the fix.

**Wrong in the published PR:**
- **Zero tokens:** the branch raises (Metal library does not build); stock returns an empty array.
- **Top-k > 32 at T = 1:** threadgroup limit exceeded (k = 40). No current model has k > 8.
- **Mixed precision** (fp32 scales, bf16 input): the branch returns bf16, stock fp32. Converted models do not have this mix.
- **PR text:** "prefill unchanged within noise" is false at B=1: prefill is 8–20% faster (unchanged at B=8/32).
- **M5 and later:** from 4 tokens per expert MLX runs `gather_qmm` on the NAX kernels. The published gate (load ≤ 8 or ≤ 24) replaces that path without any measurement against it.

**Fix (commit `42e5d30`, pushed to the PR branch on 2026-11-11; `runs/r12/pr2044_fix.patch`):**
- Fall back on zero tokens, k > 8 and scale dtype ≠ input dtype.
- On GPU generation ≥ 17 use the kernels only at load < 4, where stock still runs `gather_qmv`. M1–M4 gate unchanged.
- Decode at B=32 stays on the kernels for Qwen3, Gemma-4, GLM (load 2) and DeepSeek (3). LFM2.5 (load 4) and the B=1 prefill fall back there.
- PR description replaced (`runs/r12/pr2044_body.md`). It now states why the PR should work on M5 / M6 (same `gather_qmv` baseline below 4, fallback from 4) and lists three checks for someone with that hardware: the kernel unit test, the B=8/B=32 benchmark, and whether the limit of 4 can be raised.

**Not known:**
- Any speed number on M1–M3, base/Max/Ultra, M5 or M6. The M5+ gate is a safety limit, not a measured crossover.
- Upstream now has `mlx_lm/kernels/` (one fused Metal kernel per file, with its own GPU-generation check). The PR module sits in `mlx_lm/models/`; moving it is offered in the PR text, not done.

**Corrected in this log:** the 2026-11-10 pass changed the upstream B=1 range to 1.03–1.09×. The log gives 1.033–1.084×, so 1.03–1.08× was right.

**Mistake caught during the audit:** I first narrowed the M4 gate to load ≤ 8 from three grid shapes (1.03–1.09× at load 12–24). The stored benchmark shows a real gain there for larger experts (DeepSeek B=1 prefill 1.15× at load 12, LFM2.5 1.08× at load 16), so the measured gate was restored.

## 2026-11-10 — verification pass over this log

Independent re-check of the entries below: both test environments, the stored result files, and fresh runs.
Evidence: `runs/r11/` (`verify_kernels.log`, `verify_e2e.log`, `verify_tests.log`, 4 JSONs). Many fresh runs were flagged contended by the guard (antivirus scanner, VM), so for those only the ratio counts.

**Wrong, now corrected:**
- **Round 11 "removing all decode work gains ~3%".** That ablation covered only the MMA path, which carries 37% of the weight bytes on Qwen3 at T=32 (3% at T=8). With decode removed on all three paths the gain is **4–5% at T=32** (table below). The kill of repacking / LUT decoding stands.
- **Round 11 "73% at T=32, 97% at T=8, the 27% gap".** These were single `moelab.slope` launches. Five launches give **69–81% at T=32** and **74–97% at T=8** (floor at an assumed 250 GB/s; probes read 224–244).
- **Round 11 "next suspect: the 1-slot GEMV path".** It is not: that path runs at ~94% of the floor (per-tile table below).

**Weakened:**
- Qwen3.6 layer speedup at T=32: log said 1.21×; two fresh compares give 1.15× and 1.16× (both contended).
- Gemma-4 layer speedup came from a single-launch grid (1.66× / 1.90×); `moelab.compare` gives 1.63× [1.62–1.67] at T=32 and 1.88× [1.87–1.89] at T=64 (guard ok).
- "1.48× faster than llama.cpp" holds only at 32 clients. At 8 clients it is 0.99×, at 1 client 1.05× (`runs/r8/`).
- Server table, Gemma-4 at 32 clients: 1.43× is the mean of two rounds at 1.32× and 1.54×. Qwen3.6: 1.09–1.18×. The server JSONs store no guard status.
- GLM-4.7-Flash batch numbers (1.12× / 1.27×) come from one launch.
- Test count: 778 is the `.venv-latest` (mlx-lm 0.32.2) number. On the pinned `.venv` (0.31.3) 5 `lfm2_moe` tests failed on a fixture kwarg that only newer mlx-lm has, and `pytest tests` failed at collection without jax. Both fixed: now 786 pass on `.venv` and 778 on `.venv-latest` (`runs/r11/verify_tests.log`).

**Confirmed:**

| claim | log | re-check |
|---|---|---|
| server, 32 clients, Qwen3 / gpt-oss / DeepSeek | 1.32–1.38× | 1.38× / 1.38× / 1.32× (stored JSON) |
| single stream, Qwen3 / gpt-oss / DeepSeek | 1.17× / 1.10× / 1.12× | 1.17× / 1.10× / 1.12× (stored); Qwen3 fresh 1.18× |
| Qwen3 BatchGenerator, B=32 | 1.54–1.60× | 1.60× fresh (B=1 1.18×, B=8 1.21×) |
| Gemma-4 BatchGenerator, B=8 / B=32 | 1.25× / 1.70× | 1.25× / 1.69× fresh (219 → 371 tok/s) |
| Qwen3.6 single chat, fused router | 1.106× | 1.16× fresh (contended; the log value stays) |
| layer, DeepSeek-V2-Lite, T=1 / 8 / 32 | — | 1.17× / 1.33× / 1.66× (guard ok) |
| layer, Qwen3, T=1 / 8 / 32 | — | 1.17–1.21× / 1.11× / 1.53–1.54× (contended) |
| T=1 share of the bandwidth floor | 93–100% | Qwen3 94.5%, DeepSeek 93.5% (guard ok; gpt-oss not re-run) |

**Decode ablation on all paths** (`scripts/r11_ablations.py`, 4 launches × 4 rounds, speedup of "no decode" over the unchanged kernel):

| shape | T=8 | T=16 | T=32 | guard |
|---|---|---|---|---|
| Qwen3-30B | 1.039× [1.02–1.06] | 1.034× [1.03–1.08] | 1.054× [1.04–1.06] | contended (90%) |
| DeepSeek-V2-Lite | — | — | 1.046× [1.04–1.05] | ok |

"No decode" (`nda`) keeps one xor per weight and the multiply-accumulate. A harsher variant (`ndb`) that also lets the compiler drop the per-weight multiplies on the GEMV paths gives 1.07–1.09× at T=32; that is an upper bound, not a decode cost.

**Efficiency by tile size** (`scripts/r11_tile_efficiency.py`, Qwen3 shape; routing built so every touched expert serves exactly n tokens; % of the bytes floor at the probed 230 GB/s; guard ok):

| tokens per expert | kernel path | stock | ours | ours, no decode |
|---|---|---|---|---|
| 1 | 1-slot GEMV | 91% | **94%** | 96% |
| 2 | 2-slot GEMV | 60% | **66%** | 68% |
| 3 | 8×8 MMA | 40% | **60%** | 62% |
| 4 | 8×8 MMA | 30% | **59%** | 61% |
| 8 | 8×8 MMA | 15% | **55%** | 57% |

- The T=32 gap sits in the 2-slot path (~66%) and the MMA path (55–60%). The 1-slot path is bandwidth-bound.
- Decode costs 2–5% on every path.
- On the 2-slot path, `ndb` reaches 83%: dropping the per-weight multiplies gains ~26%. That path is bound by multiply-accumulate work, which makes it the best lead for the T=32 gap.

**Not re-measured in this pass:** LFM2.5 and GLM-4.7 runs, gpt-oss MXFP4, the accuracy tables, the Gemma-4 / Qwen3.6 server runs, the expert-cache simulation, the prefill and 8-/6-bit probes.

## 2026-11-10 — round 11: where the T=32 time goes (weight decoding is not it)

Kernel-only harness (`moelab.compare`, `moelab.slope`), Qwen3-30B shape, bf16, uniform routing, M4 Pro on battery (guard ok).
Ablations are timing-only copies of `metal_q4_tiles_v2` with deliberately wrong outputs.

| ablation (MMA path) | T=8 | T=16 | T=32 |
|---|---|---|---|
| no bit shuffle (`pg`/`pu`/`pw` repack) | 1.00–1.02× | 1.00–1.02× | 1.00× |
| no shuffle and no dequant (`q\|0x6400` → `P^s`) | 1.00–1.04× | 1.01–1.03× | **1.027×** |
| +256 empty threadgroup rows | — | — | 0.98× (CI 0.91–1.01) |

- **Killed: weight repacking and LUT/T-MAC decoding.** Removing the decode work on the MMA path gains ~3% at T=32; on all paths it is 4–5% (verification pass above). No decode scheme can recover the gap to the floor, which is 19–31% at T=32 across launches (`moelab.slope`; the first single launch read 73% of the floor at T=32 and 97% at T=8).
- **Not the cause: empty threadgroups.** Non-leader slots launch threadgroups that exit at once; 256 extra empty rows cost ≤2%.
- **Where the gap sits:** per-stage timing (`scratchpad stages.py`, noisy) puts gate_up at ~70% of its own floor at T=32 (~87% at T=8). Down is ~80%. Next suspects: the 2-slot GEMV path and the 8×8 MMA tiles (the 1-slot path was ruled out by the verification pass: ~94% of the floor).
- This corrects round 10's "ALU-bound on weight decoding" wording: the path is not bandwidth-bound at T=32, but decode ALU is not the limiter.

## 2026-11-08 — round 10: T=32 search (killed), expert-cache feasibility, MXFP4 for mlx-lm

**T=32 tile search** (background agent, kill criterion ≥ 5%): **killed.**
- The best change, a scalar N-slot down path for tiles of 2–3 slots, gives only +2–3%, and the gemma4 / qwen3.6 CIs include 1.0. It is not merged.
- **Key finding:** with 3–8 slots per tile, the 8×8 MMA path sits at ~60–65% of the bytes floor for any n, so it is ALU-bound by weight decode, not memory-bound. Removing the activation loads gains only 6%, and removing the dequant only 10%. (Later re-measured: 55–60% of the floor, and decode is 2–5%, so "bound by weight decode" is wrong; see round 11 and the verification pass.)
- Dead hypotheses: compacting the tile list (empty threadgroups are cheap), down SGK/LPR tuning, fusing the combine kernel (≤ 1%).
- The agent's worktree branch `worktree-agent-a571ee52ffbaad921` holds the N-slot down path.

**Expert cache for models larger than RAM** (simulated on real Qwen3 routing): a cache of 30% of experts gives 86% LRU hits and 94% optimal (Belady). The user chose to stay on kernels, so this is backlog #12.

**Speculative decoding for single-stream MoE: killed.** Qwen3.5/3.6 recurrent caches are not trimmable in mlx-lm.

**gpt-oss MXFP4 for mlx-lm** (follow-up to #2044):
- On real gpt-oss-20b weights, per-slot outputs (what `SwitchGLU` returns) cost almost nothing vs our dedicated k-summed path: layer speedup vs stock 1.09× (T=1), 1.52× (T=8), 1.84× (T=32).
- Branch `Tigerly1/mlx-lm:moe-kernels-mxfp4` adds MXFP4 pair/group kernels with biases and SwigluOAI to `switch_kernels.py`. 414 mlx-lm tests pass, including the new MXFP4 match/gradient/fallback tests.
- The `mlx_lm benchmark` A/B is **not yet valid**: it ran on battery while the antivirus scanner used 356% CPU, and main itself varied 2× between passes. It needs a re-run on AC.

## 2026-11-07 — upstream: PRs to ml-explore/mlx-lm

- **2026-11-08 update:** PR 1 (#2043) was closed to make room. **PR 2 is open as [ml-explore/mlx-lm#2044](https://github.com/ml-explore/mlx-lm/pull/2044).** The empty-prompt fix stays on the fork (`server-reject-empty-prompt`) for later.
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
- **Exactness:** with the bit-identical fallback as the kernel, logits equal stock on tiny models of 6 families with different routers: Qwen3-MoE, Qwen3.5, Gemma-4, LFM2-MoE, GLM-4.7-Flash, DeepSeek-V2. In total 778 tests pass (`.venv-latest`, mlx-lm 0.32.2).
- **Cost vs a dedicated patch** (Qwen3, kernels only, no fused router): 1.075× / 1.076× vs 1.086×.

**New families,** single stream (`--chat`, 2 launches) and concurrent requests (BatchGenerator):

| model | 1 chat | B=8 | B=32 |
|---|---|---|---|
| LFM2.5-8B-A1B 4-bit | 1.040× / 1.038× | 1.25× / 1.31× | **1.49× / 1.52×** (556 → 846 tok/s) |
| GLM-4.7-Flash 4-bit | 1.033× / 1.034× | 1.12× (1 launch) | **1.27×** (180 → 229 tok/s, 1 launch) |

- **Quality** on model-native text: PPL within 1%, top-1 agreement within noise. Two contended GLM runs were discarded and re-measured.

**Real `mlx_lm.server`** (`moelab.serve`, 256 tokens per reply, 2 rounds):

| model | 1 client | 8 clients | 32 clients |
|---|---|---|---|
| Gemma-4-26B-A4B | 1.11× | 1.18× | **1.43×** (152 → 218 tok/s; median reply 53 → 38 s; rounds 1.32× and 1.54×) |
| Qwen3.6-35B-A3B | 1.07× | 1.08× | 1.13× (179 → 203 tok/s; 38 → 35 s; rounds 1.09× and 1.18×) |

The server result files carry no guard status.

On the server, Qwen3.6 gains less than in BatchGenerator (1.24×) because a larger share of server time is prefill, which our kernels do not touch.

## 2026-11-07 — round 9b: MMA tiles on every 4-bit shape, Gemma-4 at 32 requests 1.25× → 1.70×

- **The cause.** The MMA expert tiles were gated on D, F % 128. Gemma-4 (F=704) therefore fell to scalar tiles, which are 2× slower than stock at load 4. The MMA kernels only need D, F % 64, which always holds with group-64 quantization.
- **Verification:** correct on F=704, 192 and D=320 at T=3–64; 530 kernel tests pass.
- **Layer level** (Gemma-4 shape): T=32 goes from 1.28× to **1.63×** [1.62–1.67], and T=64 reaches 1.88× (`moelab.compare`, 3 launches, guard ok; the first single-launch grid read 1.66× / 1.90×).
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

These gains are smaller than the 1.5–1.6× on the older models. The cause is structural, and the layer level shows it (compare, T=32: Qwen3.6 1.15–1.21×, Gemma-4 1.28×):
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
- **Concurrent serving:** real `mlx_lm.server` with 32 clients is 1.32–1.38× faster on Qwen3-30B-A3B, gpt-oss-20b and DeepSeek-V2-Lite. On Qwen3 it is 1.48× faster than llama.cpp at 32 clients (0.99× at 8 clients, 1.05× at 1 client).
- **Single stream:** Qwen3 1.17×, gpt-oss 1.10×, DeepSeek-V2-Lite 1.12×. T=1 kernels reach 93–100% of the bandwidth floor (re-measured: Qwen3 94.5%, DeepSeek 93.5%).
- **Load policy:** load = T·k/E decides kernel vs fallback; the fallback is bit-identical to mlx-lm.
- **Withdrawn after checking:** "router picks 9% wrong", "gain vanishes at long context", "decode is CPU-bound".
