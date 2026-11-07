# DRAFT comment for ml-explore/mlx#4509 (not posted; the user decides)

**Title context:** "gather_qmm at M=1 streams 4-bit experts at ~half bandwidth"

---

Data from an M4 Pro (20-core GPU), MLX 0.32.4, mlx-lm 0.31.3, bf16, 4-bit affine g64. Qwen3-30B-A3B expert shapes: d=2048, f=768, E=128, top-8.

**1. Alignment cliff in `gather_qmv`.**
- `qmv_fast` needs `K % 512 == 0` for 4-bit (`qmv_fast_k_alignment`).
- The Qwen3 down projection has K=768, so it falls back to `qmv_impl`. So do all of gpt-oss-20b's projections (K=2880).
- We compared a fused custom kernel against the stock SwitchGLU path, then changed only the offending dimension:

| shape | speedup vs stock at T=1 |
|---|---|
| Qwen3 shape (K=768) | 1.168× [1.163, 1.175] |
| same, f=1024 (aligned) | 1.097× |
| gpt-oss shape (K=2880) | 1.117× |
| same, 3072 (aligned) | 1.060× |

Roughly half of the gap is the alignment fallback. One-row-per-simdgroup with `uint4` loads handles any K % 32 == 0 at full speed. Relaxing the alignment, or adding a partial-K tail to `qmv_fast_impl`, looks like a small, general fix.

**2. Decode with T>1 (speculative verify, batching) re-reads experts once per slot.**
- With real Qwen3 routing, T consecutive tokens touch U(8) ≈ 28 and U(16) ≈ 38 unique experts. That is 64 and 128 slots respectively.
- At those sizes `gather_qmm` does not sort (`indices.size >= 64` in mlx-lm), and `gather_qmm_rhs` needs `B/E >= 4`.
- An expert-tiled kernel (map prepass, then tiles of ≤8 slots per expert, then `simdgroup_matrix` 8×8 with an exact `0x6400` half dequant) gives:

| T | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| speedup | 1.22× | 1.36× | 1.64× | 1.87× | 2.36× |

- End to end (mlx-lm `stream_generate`, Qwen3-30B-A3B-4bit, AC power):
  - plain decode: 89.9 → 96.1 tok/s (+7.0%)
  - speculative decoding with Qwen3-0.6B draft, 6 draft tokens: 63.5 → 77.2 tok/s (+21.6%)
  - Teacher-forced perplexity is unchanged.

Code, harness and methodology: <link to public repo once released>. Happy to turn (1) into a PR against `qmv_fast_impl` with tests for non-aligned K, if that direction is welcome.

---

Notes for the author before posting:
- Re-run on at least one other chip (M1/M2/M3; borrow a Mac or use cloud Mac minis). The fusion half of the gain may be M3/M4-specific because of Dynamic Caching.
- PR #4598 was rejected for dispatch complexity, so keep (1) minimal.
- Make the repo public first.
