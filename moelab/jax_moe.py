"""MoE expert layer in Google's stack (JAX / XLA / Pallas). Same contract as the other backends:
    y = moe(x [T,d], w_gate [E,f,d], w_up [E,f,d], w_down [E,d,f], idx [T,k], wts [T,k])

1. moe_ragged:  sort slots by expert -> jax.lax.ragged_dot (XLA grouped matmul) -> unsort, combine.
2. moe_pallas:  Megablox-style grouped GEMM in Pallas. Rows sorted by expert and padded per expert
   to a multiple of BM (like vLLM moe_align_block_size); each row tile's expert id is passed by
   scalar prefetch so the weight BlockSpec index_map selects that expert's slab (the TPU pattern).
   Runs in interpret mode on CPU here; on TPU/GPU set interpret=False (not tested on hardware yet).
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _sort_slots(idx, E):
    flat = idx.reshape(-1)
    order = jnp.argsort(flat, stable=True)
    return order, flat[order], jnp.bincount(flat, length=E).astype(jnp.int32)


def moe_ragged(x, wg, wu, wd, idx, wts):
    T, k = idx.shape
    E = wg.shape[0]
    order, _, sizes = _sort_slots(idx, E)
    xs = x[order // k]
    g = jax.lax.ragged_dot(xs, jnp.swapaxes(wg, 1, 2), sizes)
    u = jax.lax.ragged_dot(xs, jnp.swapaxes(wu, 1, 2), sizes)
    o = jax.lax.ragged_dot(jax.nn.silu(g) * u, jnp.swapaxes(wd, 1, 2), sizes)
    y = jnp.zeros((T * k, x.shape[1]), o.dtype).at[order].set(o)
    return (y.reshape(T, k, -1) * wts[..., None].astype(o.dtype)).sum(1)


def align(idx, E, BM):
    """Host-side (numpy) metadata: padded row ids sorted by expert, expert per row tile."""
    flat = np.asarray(idx).reshape(-1)
    n = flat.size
    rows, tile_e = [], []
    for e in range(E):
        r = np.nonzero(flat == e)[0]
        if r.size == 0:
            continue
        pad = (-r.size) % BM
        rows.extend(r.tolist() + [n] * pad)          # n = sentinel (padding row)
        tile_e.extend([e] * ((r.size + pad) // BM))
    return np.asarray(rows, np.int32), np.asarray(tile_e, np.int32)


def _gmm_kernel(tile_e_ref, a_ref, w_ref, o_ref):
    o_ref[...] = jnp.dot(a_ref[...], w_ref[0], preferred_element_type=jnp.float32).astype(o_ref.dtype)


def gmm_pallas(a, w, tile_e, BM, BN, interpret=True):
    """a [M_pad, K] rows sorted+padded by expert; w [E, K, N]; tile_e [M_pad/BM] expert of each row tile."""
    M, K = a.shape
    _, _, N = w.shape
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(M // BM, N // BN),
        in_specs=[pl.BlockSpec((BM, K), lambda i, j, te: (i, 0)),
                  pl.BlockSpec((1, K, BN), lambda i, j, te: (te[i], 0, j))],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j, te: (i, j)),
    )
    return pl.pallas_call(_gmm_kernel, grid_spec=grid_spec,
                          out_shape=jax.ShapeDtypeStruct((M, N), a.dtype), interpret=interpret)(tile_e, a, w)


def moe_pallas(x, wg, wu, wd, idx, wts, BM=8, BN=None, interpret=True):
    T, k = idx.shape
    E, f, d = wg.shape
    rows, tile_e = align(idx, E, BM)
    rows_j, tile_j = jnp.asarray(rows), jnp.asarray(tile_e)
    xpad = jnp.concatenate([x, jnp.zeros((1, d), x.dtype)])          # sentinel row -> zeros
    a = xpad[jnp.where(rows_j < T * k, rows_j // k, T)]               # token of each slot; padding -> zero row
    wgu = jnp.concatenate([jnp.swapaxes(wg, 1, 2), jnp.swapaxes(wu, 1, 2)], axis=2)  # [E, d, 2f]
    gu = gmm_pallas(a, wgu, tile_j, BM, BN or 2 * f, interpret)
    h = (jax.nn.silu(gu[:, :f].astype(jnp.float32)) * gu[:, f:].astype(jnp.float32)).astype(x.dtype)
    o = gmm_pallas(h, jnp.swapaxes(wd, 1, 2), tile_j, BM, BN or d, interpret)        # [M_pad, d]
    valid = rows_j < T * k
    y = jnp.zeros((T * k + 1, d), jnp.float32).at[jnp.where(valid, rows_j, T * k)].set(o.astype(jnp.float32))[:T * k]
    return (y.reshape(T, k, d) * wts[..., None]).sum(1).astype(x.dtype)
