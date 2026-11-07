# PR 2 → ml-explore/mlx-lm: grouped-expert Metal kernels for small-batch MoE decode (plan)

**What:** `SwitchGLU` dispatches to custom `mx.fast.metal_kernel` expert kernels when all of these hold:
- weights are 4-bit affine, group 64, no bias;
- activations are bf16/fp16;
- activation is SwiGLU or GeGLU;
- load T·k/E ≤ 8–24.

Otherwise it uses the existing path, unchanged. The return value is the same per-slot `[..., k, D]`, so the routing and weighting code of every model stays as it is (moe-lab `universal.py` design).

Precedent for custom kernels in mlx-lm: #2027 (DSA sparse prefill), the gated delta kernels.

**Evidence to include** (M4 Pro, ≥2 launches, bandwidth guard; from moe-lab `docs/ACHIEVEMENTS.md`):
- BatchGenerator, B=32: Qwen3-30B-A3B 1.59×, gpt-oss-20b 1.60× (gpt-oss is MXFP4, so it is not in this PR), DeepSeek-V2-Lite 1.54×, Gemma-4-26B 1.70×, LFM2.5-8B 1.52×, GLM-4.7-Flash 1.27×, Qwen3.6-35B 1.24×.
- Single stream: 1.03–1.09× (kernels only, no fused router).
- Accuracy: per-layer error vs fp32 is 2× lower than `gather_qmm`; exact-glue tests.

**Risks / reviewer objections to pre-empt:**
- Maintainers may prefer this in MLX core (`gather_qmm` / `gather_qmv`); cf. #952 ("not the right way") and mlx#4598, which was not merged.
- Code size (~700 lines Metal).
- Benchmark noise claims (#1676 walked back); our numbers come with launches, CIs and guard.

**Needed before opening:**
- run the mlx-lm test suite with the change;
- benchmark with mlx-lm's own `mlx_lm.benchmark` / server.
