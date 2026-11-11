# Research backlog (round 8+), ranked by expected value

Rules:
- Expected value = P(success) × impact ÷ cost.
- Nothing starts without a falsifiable claim, a kill criterion and the file that will prove it.
- Agents get one narrow question each and never touch the GPU; GPU work runs serially, one job at a time.
- A result counts only after an independent re-measurement (`moelab.compare`, or ≥2 process launches for e2e).

| # | claim to establish | P | impact | cost | owner | kill criterion | evidence file | status |
|---|---|---|---|---|---|---|---|---|
| 1 | Concurrent requests (mlx-lm BatchGenerator, B=8–32) get 1.2–1.6× throughput on 3 MoE models | 0.9 | high: the practical use case | 1 h GPU | me | <1.1× at B=32 on 2 of 3 models | `runs/r8/batch_*` | running |
| 2 | mlx-lm's bf16 router softmax decides ~9% of Qwen3 expert picks by arbitrary ties; a one-line fp32 fix removes them at ~0 cost | 0.8 | high: an upstream fix for every MLX user of 7 MoE families | 0.5 h GPU + draft | me (+ agent B for prior art) | the fix changes cost >5% or other models show <1% ties | `runs/r8/router_fix_*.json` | **killed**: stock always picks a valid top-k by logit (0/24,576 wrong on Qwen3, 9/13,312 on DeepSeek, which are not prob ties); the 9% differences are exact bf16 *logit* ties no softmax precision can fix; the fp32 fix costs +0.14 ms on Qwen3. No upstream issue. |
| 3 | The "hot expert per document" observation is known or new; decide the wording and citations | 0.95 | medium: honesty of the thesis | agent | agent A | none (pure check) | `docs/related_work_r8.md` | **done**: partly known (locality, domain skew); the single per-document hot expert quantification appears unpublished; wording fixed |
| 4 | Position the batched result against llama.cpp Metal / NVIDIA small-batch MoE | 0.9 | medium: tells whether #1 is notable | agent | agent C | none | agent report (in `docs/related_work_r8.md` after verification) | **done**: no published B=1–32 MoE batched sweep on Apple; technique (grouping by expert) is standard on GPUs (vLLM fused_moe, llama.cpp map0+mul_mm_id) → claim = bringing it to MLX with measured serving gains |
| 5 | Explain why our MoE time grows with context (MLX Metal scheduling) | 0.4 | low–medium: closes an open thesis question | agent + ≤1 h GPU tests it proposes | agent D, then me | no proposed test separates the mechanisms | agent report + `runs/r8/longctx_*` | running |
| 6 | Host-side cost per custom-kernel call (Python template/dispatch) is a measurable part of single-stream time; caching it gains ≥2% e2e | 0.5 | medium: single-stream gain | 1 h | me | <1% of step time is host overhead | `runs/r8/host_overhead.json` | **killed**: graph build per token is 0.32 ms ours vs 0.49 ms stock (48 layers), <4% of a step and largely overlapped |
| 7 | Fusing the router tail into the gate GEMV (one dispatch) adds ≥2% e2e on Qwen3 | 0.35 | medium | 3–4 h | me | <1% in isolated chain | — | idea |
| 8 | Same gains on a second Apple chip (M2, 16 GB) | 0.7 | high for generality | user runs `scripts/portability_m2.sh` | user | — | `runs/portability_*` | waiting on user |
| 9 | NVIDIA counterpart (title promises it) | 0.6 | high for the thesis | needs `modal setup` | user, then me | — | — | waiting on user |
| 11 | Baseline is current: re-run headline on latest mlx-lm 0.32.2 (MLX 0.32.4 already includes gather_qmm PR #4572) | 0.9 | high: honesty | 1 h GPU | me | — | `runs/r8/latest/` | queued |
| 12 | **MoE larger than RAM on a Mac** (e.g. Qwen3-235B, 132 GB, on 48 GB): an in-RAM (layer, expert) cache plus SSD streaming makes it usable (~4 tok/s) where plain loading thrashes | 0.5 | very high for local users | weeks | me | an LRU/predictive cache on a real run gives < 2 tok/s or loses to macOS's own paging | `runs/r10/expert_cache_sim.json` | **simulated:** real Qwen3 traces, a cache of 30% of experts gives 86% LRU hits / 94% optimal (Belady), and 40% gives 93% / 97%. Prior art: Mixtral offloading (Eliseev & Mazur 2023), LLM in a flash (Apple 2023), MoE-Infinity. |
| 13 | Speculative decoding for single-stream MoE | — | — | — | — | — | — | **killed:** Qwen3.5/3.6 GatedDeltaNet caches are not trimmable in mlx-lm (native MTP is open PR #990, 1.6k lines, conflicting). On Qwen3 a 0.6B draft already loses (92 vs 105 tok/s). |
| 14 | MLX command-buffer size (`MLX_MAX_OPS_PER_BUFFER` / `_MB_`) reduces dispatch bubbles in single-stream decode by ≥ 3% e2e | 0.3 | medium, generic | 0.5 h GPU | me | < 2% on Qwen3 and Qwen3.6 | `runs/r10/cmdbuf_*` | **negative on the synthetic proxy** (`runs/r13/`: 200 ops no gain, 1000 worse); not run on a real model |
| 15 | Small-op dispatch overhead is the largest remaining single-stream cost on M4: ≥ 2 µs per small dispatch inside a decode-like chain (a Qwen3 layer issues ~21–31 of them, counted from the model code) | 0.6 | high: decides #16 and #14 | 2 min GPU, ~250 MB, no model | me | < 1 µs per dispatch | `runs/r13/dispatch_probe_*.json` | **holds** (2026-11-11): 2.7 µs per `rms_norm`, 1.7 µs per `add`, two launches, linear in the op count |
| 16 | Fusing the small ops of a Qwen3-family decode layer (q/k norm + RoPE, add + norm, combine + add) gains ≥ 5% single-stream e2e. Upstream ships such kernels in `mlx_lm/kernels/` for other models (muse_spark, flint), not for Qwen3 / Gemma-4 / DeepSeek / LFM2 / GLM | 0.4 | high: single stream is the common local use | 1–2 days + 0.5 h GPU | me | #15 killed, or < 3% e2e on Qwen3 | — | next: #15 holds; estimate ~1 ms (~8%) of a Qwen3 token, unmeasured |
| 10 | Upstream MLX kernel fix for the K%512 cliff | 0.3 (merge) | very high if merged | needs Xcode / Metal compiler | user (install), then me | maintainers reject the approach (cf. PR #4598) | — | blocked |

Not doing (low expected value):
- more kernel micro-tuning at T=1 (already at 93–100% of the bandwidth floor);
- a crossover push beyond load 24 (the fallback is already parity);
- a multi-row kernel for the dense projections at batch 2–12: MLX already has it (`affine_qmv_mma`, from 4 rows on M3/M4 when N·K ≥ 2M; read in `quantized.cpp`, 2026-11-11);
- a speculative-decoding product (on this setup, free-running spec decoding with a 0.6B draft is slower than plain decode).
