"""Benchmark the JAX MoE paths on whatever device JAX sees (CPU here; TPU/GPU on Colab/Kaggle).

Colab (TPU runtime):
    !pip -q install -U "jax[tpu]"           # or use the preinstalled jax
    !git clone <repo> && cd moe-lab
    !python -m moelab.jax_bench --shape qwen3-30b-a3b --tokens 1 4 16 64 --no-interpret
Prints per-call median ms for: dense masked einsum, XLA ragged_dot, Pallas grouped GEMM.
Correctness is checked against ragged_dot (fp32 accumulate) before timing.
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import jax_moe
from .problem import make_routing
from .shapes import SHAPES


def dense(x, wg, wu, wd, idx, wts):
    T, E = x.shape[0], wg.shape[0]
    m = jnp.zeros((T, E), x.dtype).at[jnp.arange(T)[:, None], idx].add(wts.astype(x.dtype))
    g = jnp.einsum("td,efd->etf", x, wg); u = jnp.einsum("td,efd->etf", x, wu)
    return jnp.einsum("te,etd->td", m, jnp.einsum("etf,edf->etd", jax.nn.silu(g) * u, wd))


def bench(f, args, reps=20):
    out = f(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(f(*args)); ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts)) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="tiny"); ap.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--no-interpret", action="store_true")
    ap.add_argument("--BM", type=int, default=8)
    a = ap.parse_args()
    sh, dt = SHAPES[a.shape], jnp.dtype(a.dtype)
    key = jax.random.PRNGKey(0); k1, k2, k3 = jax.random.split(key, 3)
    wg = (jax.random.normal(k1, (sh.E, sh.f, sh.d)) * sh.d ** -0.5).astype(dt)
    wu = (jax.random.normal(k2, (sh.E, sh.f, sh.d)) * sh.d ** -0.5).astype(dt)
    wd = (jax.random.normal(k3, (sh.E, sh.d, sh.f)) * sh.f ** -0.5).astype(dt)
    print(f"device {jax.devices()[0]} | shape {sh} | {a.dtype}")
    for T in a.tokens:
        rng = np.random.default_rng(T)
        x = jnp.asarray(rng.standard_normal((T, sh.d)), dt)
        idx, wts = make_routing(T, sh.E, sh.k, "uniform", rng)
        idx, wts = jnp.asarray(idx), jnp.asarray(wts)
        ref, t_r = bench(jax.jit(jax_moe.moe_ragged), (x, wg, wu, wd, idx, wts))
        pal = lambda *z: jax_moe.moe_pallas(*z, BM=a.BM, interpret=not a.no_interpret)  # noqa: E731
        y_p, t_p = bench(pal, (x, wg, wu, wd, idx, wts), reps=5)
        res = [f"ragged_dot {t_r:8.3f} ms", f"pallas {t_p:8.3f} ms"]
        if sh.E * T * sh.f * sh.d < 4e9:
            _, t_d = bench(jax.jit(dense), (x, wg, wu, wd, idx, wts)); res.insert(0, f"dense {t_d:8.3f} ms")
        err = float(jnp.linalg.norm((y_p - ref).astype(jnp.float32)) / jnp.linalg.norm(ref.astype(jnp.float32)))
        print(f"T={T:4d}  " + "  ".join(res) + f"  pallas rel err {err:.1e}")


if __name__ == "__main__":
    main()
