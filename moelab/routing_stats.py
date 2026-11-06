"""Statistics of recorded routing traces (moelab/routing_trace.py output).

    .venv/bin/python -m moelab.routing_stats runs/routing_qwen3.npz

U(T): unique experts touched by T tokens, per layer, averaged.
  consecutive = T adjacent tokens of one sequence (speculative-decoding verify / chunked prefill)
  batched     = T tokens from different sequences/positions (multi-user batch decode)
  uniform     = E(1-(1-k/E)^T)
Prints JSON-able dict too (saved next to input as *_stats.json).
"""
import json
import sys

import numpy as np

TS = [1, 2, 4, 8, 16, 32, 64]


def unique_count(inds_layer, rows):  # inds_layer [L, k]
    return len(np.unique(inds_layer[rows]))


def stats(path, seed=0):
    z = np.load(path)
    E = int(z["E"])
    doms = [k for k in z.files if k != "E"]
    k = z[doms[0]].shape[2]
    rng = np.random.default_rng(seed)
    out = {"E": E, "k": k, "T": TS, "uniform": [E * (1 - (1 - k / E) ** T) for T in TS], "domains": {}}
    allpool = [(d, l) for d in doms for l in range(z[d].shape[1])]
    batched = []
    for T in TS:
        vals = []
        for _ in range(200):
            pick = rng.choice(len(allpool), T, replace=False)
            for layer in range(0, z[doms[0]].shape[0], 4):
                rows = np.concatenate([z[allpool[i][0]][layer, allpool[i][1]][None] for i in pick])
                vals.append(len(np.unique(rows)))
        batched.append(float(np.mean(vals)))
    out["batched"] = batched
    for d in doms:
        inds = z[d]                                   # [layers, L, k]
        nl, L, _ = inds.shape
        cons = []
        for T in TS:
            starts = rng.integers(0, L - T, 200)
            cons.append(float(np.mean([unique_count(inds[l], slice(s, s + T)) for l in range(nl) for s in starts[:25]])))
        counts = np.stack([np.bincount(inds[l].ravel(), minlength=E) for l in range(nl)])   # [layers, E]
        p = counts / counts.sum(1, keepdims=True)
        ent = -(p * np.log(p + 1e-12)).sum(1) / np.log(E)
        top10 = np.sort(p, 1)[:, ::-1][:, : E // 10].sum(1)
        dead = (counts == 0).sum(1)
        overlap = np.mean([len(np.intersect1d(inds[l, t], inds[l, t - 1])) / k
                           for l in range(nl) for t in range(1, L, 7)])
        out["domains"][d] = {
            "tokens": int(L), "consecutive": cons,
            "entropy_norm_mean": float(ent.mean()), "entropy_norm_min": float(ent.min()),
            "top10pct_share_mean": float(top10.mean()), "top10pct_share_max": float(top10.max()),
            "unused_experts_mean": float(dead.mean()),
            "adjacent_overlap": float(overlap), "random_overlap": k / E,
            "popularity_layer0": p[0].tolist(), "popularity_mid": p[nl // 2].tolist(),
        }
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/routing_qwen3.npz"
    s = stats(path)
    json.dump(s, open(path.replace(".npz", "_stats.json"), "w"))
    print(f"E={s['E']} k={s['k']}")
    print("T          " + " ".join(f"{t:6d}" for t in s["T"]))
    print("uniform    " + " ".join(f"{u:6.1f}" for u in s["uniform"]))
    print("batched    " + " ".join(f"{u:6.1f}" for u in s["batched"]))
    for d, v in s["domains"].items():
        print(f"{d[:10]:10s} " + " ".join(f"{u:6.1f}" for u in v["consecutive"]))
    for d, v in s["domains"].items():
        print(f"{d:13s} entropy {v['entropy_norm_mean']:.3f} (min {v['entropy_norm_min']:.3f}) "
              f"top10%-share {v['top10pct_share_mean']:.2f} (max {v['top10pct_share_max']:.2f}) "
              f"unused {v['unused_experts_mean']:.1f} adj-overlap {v['adjacent_overlap']:.3f} vs random {v['random_overlap']:.3f}")


if __name__ == "__main__":
    main()
