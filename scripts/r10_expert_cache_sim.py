"""Round 10 feasibility: an in-RAM expert cache for MoE models larger than RAM (experts streamed from SSD otherwise).
Replays real Qwen3-30B routing traces (runs/routing_qwen3.npz: 48 layers x tokens x top-8, three domains)
through a cache of (layer, expert) entries holding a fraction f of all experts, and reports the hit rate:
  - LRU (implementable), - Belady/MIN (optimal offline policy: upper bound for any cache).
Decode order: token by token, layers 0..47 in order. Domains run back to back (a topic switch mid-stream).

    python3 scripts/r10_expert_cache_sim.py
"""
import heapq
import json
from collections import OrderedDict, defaultdict

import numpy as np

z = np.load("runs/routing_qwen3.npz")
E = int(z["E"])
doms = ["english_tech", "python_code", "polish"]
seq = []
for d in doms:
    tr = z[d]                                   # [L, T, k]
    L, T, k = tr.shape
    for t in range(T):
        for l in range(L):
            for e in tr[l, t]:
                seq.append(l * E + int(e))
N = L * E
seq = np.array(seq)


def lru(cap):
    c, hits = OrderedDict(), 0
    for x in seq:
        if x in c:
            hits += 1
            c.move_to_end(x)
        else:
            c[x] = None
            if len(c) > cap:
                c.popitem(last=False)
    return hits / len(seq)


def belady(cap):
    nxt = np.empty(len(seq), dtype=np.int64)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], 1 << 62)
        last[seq[i]] = i
    inc, heap, hits = {}, [], 0                 # inc: item -> its next use; heap: (-next_use, item) lazy
    for i, x in enumerate(seq):
        if x in inc:
            hits += 1
        elif len(inc) >= cap:
            while True:
                nu, y = heapq.heappop(heap)
                if inc.get(y) == -nu:
                    del inc[y]
                    break
        inc[x] = nxt[i]
        heapq.heappush(heap, (-nxt[i], x))
    return hits / len(seq)


res = {"accesses": int(len(seq)), "experts_total": N, "rows": []}
print(f"{len(seq)} expert reads over {N} experts ({L} layers x {E}); per-token unique reads = {L * k}")
for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    cap = int(f * N)
    a, b = lru(cap), belady(cap)
    res["rows"].append({"fraction": f, "lru_hit": a, "belady_hit": b})
    print(f"cache {int(f * 100):3d}% of experts: LRU hit {100 * a:5.1f}%   optimal (Belady) {100 * b:5.1f}%", flush=True)
json.dump(res, open("runs/r10/expert_cache_sim.json", "w"), indent=1)
