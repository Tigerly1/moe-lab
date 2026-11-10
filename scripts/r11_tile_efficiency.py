"""Round 11: efficiency vs tile size. Routing is built so that every touched expert serves exactly n tokens.
Per-call device time (slope fit, residual add removed) against the bytes floor at the bandwidth probed in-process.

    PYTHONPATH=. .venv/bin/python scripts/r11_tile_efficiency.py qwen3-30b-a3b 1x16,2x16,3x8,4x8,8x4 \\
        moelab/variants/mlx/mlxlm_switchglu_q4.py moelab/variants/mlx/metal_q4_tiles_v2.py

Cases are n x groups: groups of n tokens share k experts, groups use disjoint experts (T = n * groups)."""
import sys, numpy as np, mlx.core as mx
from moelab import mlx_backend as be, roofline
from moelab.slope import fit
from moelab.shapes import SHAPES
from moelab.guard import probe, verdict, report

def routing(n, groups, E, k, rng):
    ex = rng.permutation(E)[:groups * k].reshape(groups, k)
    idx = np.repeat(ex, n, axis=0)
    idx = idx[rng.permutation(len(idx))]
    idx = np.stack([r[rng.permutation(k)] for r in idx]).astype(np.int32)
    w = rng.random(idx.shape).astype(np.float32); w /= w.sum(1, keepdims=True)
    return idx, w

sh = SHAPES[sys.argv[1]]
cases = [tuple(map(int, c.split("x"))) for c in sys.argv[2].split(",")]   # n x groups
variants = sys.argv[3:]
w = be.make_weights(sh, "bfloat16")
mods = [be.load_variant(v) for v in variants]
sts = [m.prepare(*w) for m in mods]; mx.eval(sts)
probes = [probe()]
bw = probes[0]
print(f"{sh.name} probe {bw:.0f} GB/s")
for n, groups in cases:
    T = n * groups
    inp = []
    for s in range(80):
        rng = np.random.default_rng(5000 + s)
        idx, wt = routing(n, groups, sh.E, sh.k, rng)
        x = mx.array(rng.standard_normal((T, sh.d)).astype(np.float32)).astype(mx.bfloat16)
        inp.append((x, mx.array(idx), mx.array(wt)))
    mx.eval(inp)
    cnt = np.bincount(np.array(inp[0][1]).ravel(), minlength=sh.E); assert set(cnt[cnt > 0]) == {n}
    res, _ = fit(lambda x, i, ww: mx.zeros_like(x), inp)
    U = groups * sh.k
    floor = roofline.bytes_moved(T, sh.d, sh.f, sh.E, sh.k, 4.5, unique=U) / (bw * 1e9) * 1e6
    row = [1e9] * len(mods)
    for _ in range(2):   # two interleaved passes, keep the faster one per variant
        for j, (m, st) in enumerate(zip(mods, sts)):
            s, _ = fit(lambda *z: m.run(st, *z), inp)
            row[j] = min(row[j], s - res)
    probes.append(probe())
    print(f"n={n} T={T:2d} experts={U:3d} floor {floor:6.0f}us | " + " | ".join(
        f"{v.split('/')[-1][:-3]} {t:6.0f}us {100 * floor / t:5.1f}%" for v, t in zip(variants, row))
        + f" | probe {probes[-1]:.0f}", flush=True)
print(report(verdict(probes, update=False)))
