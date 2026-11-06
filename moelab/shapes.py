"""MoE layer shapes from real models (per-layer, routed experts only).

d = hidden size, f = per-expert intermediate size, E = experts, k = top-k.
Sources: HF config.json of each model (checked 2026-11-06).
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class MoEShape:
    name: str
    d: int
    f: int
    E: int
    k: int
    shared: int = 0  # number of shared experts (not part of routed kernel)

    def to_dict(self):
        return asdict(self)


SHAPES = {s.name: s for s in [
    # tiny: correctness + fast iteration
    MoEShape("tiny", d=256, f=512, E=8, k=2),
    # mid: fit on M4 Pro 48GB
    MoEShape("olmoe-1b-7b", d=2048, f=1024, E=64, k=8),
    MoEShape("qwen3-30b-a3b", d=2048, f=768, E=128, k=8),
    MoEShape("gpt-oss-20b", d=2880, f=2880, E=32, k=4),
    MoEShape("deepseek-v2-lite", d=2048, f=1408, E=64, k=6, shared=2),
    MoEShape("mixtral-8x7b", d=4096, f=14336, E=8, k=2),
    # big: shape-only benchmarks
    MoEShape("qwen3-235b-a22b", d=4096, f=1536, E=128, k=8),
    MoEShape("gpt-oss-120b", d=2880, f=2880, E=128, k=4),
    MoEShape("deepseek-v3", d=7168, f=2048, E=256, k=8, shared=1),
    MoEShape("kimi-k2", d=7168, f=2048, E=384, k=8, shared=1),
    # ablations: change only the dimension that hits MLX's qmv_fast K % 512 requirement
    MoEShape("abl-qwen3-f1024", d=2048, f=1024, E=128, k=8),   # down K=1024 aligned (qwen3: 768 not)
    MoEShape("abl-gptoss-3072", d=3072, f=3072, E=32, k=4),    # all K aligned (gpt-oss: 2880 not)
]}

TOKEN_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
