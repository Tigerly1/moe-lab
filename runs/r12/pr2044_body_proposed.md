## What
Adds `mlx_lm/models/switch_kernels.py`, plus a 5-line hook at the top of `SwitchGLU.__call__`. For quantized experts at small batch, it replaces the per-slot `gather_qmm` calls:
- **T = 1:** gate and up are computed in one GEMV pass. The down projection runs over (slot, row block), so all k experts are in flight at once.
- **T > 1:** slots are grouped by expert into tiles of ≤ 8 and multiplied with simdgroup MMA. Each expert's weights are read and dequantized once per tile instead of once per (token, slot).

The output has the same per-slot shape as `SwitchGLU`, so no model code changes. The only exception is a `metal_act` tag on `SwiGLU` and on Gemma-4's `GeGLU`.

## When it applies
All of these must hold; otherwise the existing path runs unchanged:
- affine 4-bit, group size 64, no bias;
- bf16 or fp16 activations, scales in the same dtype;
- SwiGLU or tanh-GELU GLU;
- top-k ≤ 8;
- Metal;
- average tokens per expert T·k/E ≤ 8 (small experts) or ≤ 24 (D·F ≥ 1.5M). These thresholds come from crossover measurements against `gather_qmm` on M4 Pro;
- on M5 and later: T·k/E < 4 only. From 4, MLX runs `gather_qmm` on the NAX kernels, which I could not measure against (see Hardware scope).

Gradients: the op is an `mx.custom_function` whose VJP is the `gather_qmm` math, so LoRA training through `SwitchGLU` is unaffected (`test_gradient_matches_generic_path`).

## Results (M4 Pro 48 GB, `mlx_lm benchmark -p 128 -g 256 -n 3`, main vs this branch, alternating, 2 passes)
| model | B=1 | B=8 | B=32 |
|---|---|---|---|
| Qwen3-30B-A3B-4bit | 91 → 98 (1.08×) | 166 → 200 (1.20×) | 220 → 345 (1.56×) |
| gemma-4-26b-a4b-it-4bit | 78 → 83 (1.05×) | 152 → 195 (1.29×) | 222 → 380 (1.71×) |
| LFM2.5-8B-A1B-MLX-4bit | 178 → 188 (1.06×) | 332 → 442 (1.33×) | 625 → 864 (1.38×) |
| DeepSeek-V2-Lite-Chat-4bit-mlx | 125 → 132 (1.05×) | 210 → 266 (1.27×) | 304 → 471 (1.55×) |
| GLM-4.7-Flash-4bit | 68 → 70 (1.03×) | 132 → 178 (1.35×) | 178 → 249 (1.40×) |

generation tok/s, mean of 2 passes (each `-n 3`). Prompt (prefill) speed: 8–20% faster at B=1 (e.g. Qwen3 505 → 602 tok/s), unchanged (±1%) at B=8 and B=32.

## Hardware scope
- Everything above was measured on one machine: M4 Pro (`applegpu_g16s`), macOS 26.
- M1–M4: same code path as measured (`gather_qmv` / `gather_qmm_rhs` without NAX in MLX). Not measured on M1–M3 or on base/Max/Ultra variants; the gain depends on memory bandwidth and GPU core count.
- M5 and later: the kernels only replace the `gather_qmv` path (T·k/E < 4). The B=32 rows for Qwen3, Gemma-4, GLM and DeepSeek fall in that range (2–3 tokens per expert); LFM2.5 at B=32 is at 4 and stays on `gather_qmm`, as does the 128-token prefill. Not measured there; a run on an M5/M6 would be welcome.
- Threadgroups are at most 256 threads; `simdgroup_matrix` is available on all Apple Silicon GPUs.

## Scope and alternatives
- **Where the gain is:** concurrent requests (`mlx_lm.server` with `--decode-concurrency`, `batch_generate`). A single stream gains only 3–8%.
- **Why not in MLX core:** the main gain comes from grouping slots by expert across tokens. That needs the routing of the whole step, which `gather_qmm` does not see as a unit. A core `gather_qmm` change would be the cleaner long-term home; this PR is self-contained, opt-out by format, and easy to remove if that happens. Happy to move it (for example into `mlx_lm/kernels/` next to the other fused kernels), split it (e.g. the T > 1 path only), or put it behind a flag if you prefer.

## Accuracy
- Per-slot outputs vs an fp32 reference on the dequantized weights: rel. error < 1e-2 (bf16/fp16, T = 1–17, F = 192/384/704).
- On real models, the per-layer error vs fp32 is about half of `gather_qmm`'s (the kernels accumulate in fp32).

## Tests
- `tests/test_switch_kernels.py`: reference match, gradient, fallbacks, zero tokens, NAX gate.
- The full suite passes: `python -m unittest discover tests`, 413 tests OK. Formatted with the pinned pre-commit hooks.

Methodology and wider measurements (several launches, CIs, bandwidth guard): https://github.com/Tigerly1/moe-lab

🤖 Generated with [Claude Code](https://claude.com/claude-code)
