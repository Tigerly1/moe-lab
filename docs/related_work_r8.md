# Related work checks (round 8)

## Hot expert per document (backlog #3)
Checked by an agent; the two key papers were then re-opened by me (abstract plus the model table of 2505.16056).

| work | what it shows | relation to ours | verified by me |
|---|---|---|---|
| Su et al., *Unveiling Super Experts in MoE LLMs*, arXiv:2507.23279, ICLR 2026 (per arXiv page) | a few experts with extreme down_proj outliers → massive activations, attention sinks; "model-specific, data-agnostic"; Qwen3-30B-A3B: 3 of 6,144 | different phenomenon (ours is data-dependent, routing frequency); cite to contrast | abstract |
| Liang et al., *Not All Models Suit Expert Offloading: On Local Routing Consistency of MoE Models*, arXiv:2505.16056 | segment-level routing consistency (SRP, SCH) across 20 MoEs incl. "Qwen3" (variant unstated) and DeepSeek-V2-Lite, no gpt-oss; shared experts lower consistency; domain-specialized experts drive it | closest prior work; measures groups of experts, not one dominant expert | abstract + Table 1 |
| Jiang et al., *Mixtral of Experts*, arXiv:2401.04088 | first-choice expert repeats on 24–28% of consecutive tokens (random 12.5%) | prior evidence of temporal locality (top-1, not top-k: not directly comparable to our 41–50% top-k overlap) | agent only |
| Qiu et al., *Demons in the Detail*, ACL 2025, arXiv:2501.11873 | global-batch load balancing increases domain specialization; used in Qwen3-MoE | likely mechanism for Qwen3's strongest concentration | agent only |
| MoE-Infinity 2401.14361, ProMoE 2410.22134, HOBBIT 2411.01433, Pre-gated MoE 2308.12066, ExpertFlow 2410.17954 | offloading/caching/prefetching that exploits reuse across decode steps; ExpertFlow groups tokens with similar predicted routes | system-level uses of locality; our use is on-chip weight reuse at small T | agent only |
| MoESD 2505.19645, MoE-Spec 2602.16052 | speculative decoding with sparse MoE; the union of experts in verification grows with draft size | context for verify steps | agent only |

**Verdict:** temporal locality and domain skew are known. What appears unpublished: per layer and per document, one document-specific expert takes a median 68% / 54% / 31% of tokens (Qwen3-30B-A3B / gpt-oss-20b / DeepSeek-V2-Lite), measured on 3 documents (~5k tokens). Claim only this quantification and the kernel use, never "first to observe locality".

**Thesis wording (Polish draft goes in rozdział 4):** "Consistent with prior reports of temporal routing locality [Mixtral; Liang et al. 2025] and domain-driven expert specialization [Qiu et al. 2025], within a single document routing in each layer concentrates on one document-specific expert (median 68%/54%/31% of tokens; uniform 6–12%). These are not the data-agnostic Super Experts of Su et al. (2025): they change between documents in 98% of Qwen3 layers."

## Router precision (backlog #2), agent B; the mlx-lm part re-checked by me on mlx-lm 0.32.2
- **fp32 softmax and top-k in other frameworks:**
  - HF transformers qwen3_moe: `softmax(router_logits, dtype=torch.float)`;
  - vLLM `topk_softmax` CUDA kernel: float softmax, lower index wins ties;
  - SGLang `moe_fused_gate`: fp32, ranks on logits (same design as our fused kernel);
  - llama.cpp: f32 logits → `ggml_soft_max` → `argsort_top_k`;
  - TensorRT-LLM: float softmax, smaller index wins ties.
  
  Links with commit hashes are in the agent report (session log); to be added to the thesis bibliography.
- **mlx-lm 0.32.2 (verified locally):**
  - dbrx, deepseek_v3 and glm4_moe cast gates to fp32;
  - qwen3_moe, qwen2_moe, qwen3_next, hunyuan, deepseek_v2 and kimi_linear call `softmax(gates, precise=True)` on bf16 and get bf16 probabilities.
- **Existing MLX issues:** none about router precision or ties. Related: mlx-lm #1280 (temp=0 output differs across Apple chips on Qwen3.6-35B-A3B, not diagnosed).
- **Independent support (agent only):** vllm.cpp PR #2556 found 32.9% of Qwen-MoE prefill top-k cut-offs were exact bf16 ties.
- **Training-side precedent:** Switch Transformer §2.4 "selective precision" (fp32 router); ST-MoE (router z-loss). No inference-time router-precision paper found.
- **Positioning of the batched result (agent C):**
  - no published B=1–32 MoE batched sweep for MLX/Metal;
  - grouping tokens by expert is standard on GPUs (vLLM fused_moe #2542, llama.cpp #12612 map0 + mul_mm_id, PyTorch column-major MoE blog);
  - the claim is therefore the technique on a unified-memory Metal stack with measured serving gains.
  - Must cite: mlx #4509, #4598 (rejected), #4572 (merged, included in MLX 0.32.3+, so in our 0.32.4 baseline), discussion #3801, mlx-lm #626 (server batching), #956 (gate/up fusion; `MLX_MAX_MB_PER_BUFFER` confound noted there).

**Round-8 correction:** the measurement in `runs/r8/router_fix_*.json` shows mlx-lm's bf16 probabilities do not cause wrong expert picks. The differing picks are exact ties of the bf16 gate logits, which every framework has: they also compute the gate in bf16, and only the softmax in fp32. The router-precision issue idea is dropped. The vLLM/TRT-LLM "lower index wins" convention remains relevant only for deterministic tie-breaking.
