"""READ-ONLY evaluator for the autoresearch loop (do not edit; agents edit only
moelab/variants/mlx/ar_expert.py).

    .venv/bin/python autoresearch/bench.py > run.log 2>&1; grep "^score:\\|^status:" run.log

1. gate: unit tests for ar_expert (fp16+bf16, edge cases) + real Qwen3 shape correctness
   (bf16, uniform/trace/one_hot, T in 1,2,3,8,16) vs fp32 reference: rel_fro < 1e-2.
2. speed: interleaved A/B vs the REAL mlx-lm path (mlxlm_switchglu_q4), Qwen3-30B-A3B shape,
   bf16, real routing traces, T in {1,2,4,8,16}; 3 process launches x 6 rounds.
3. score = geometric mean of speedups over T. Also prints per-T speedups and a noise estimate.
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv/bin/python")
CAND = "moelab/variants/mlx/ar_expert.py"
BASE = "moelab/variants/mlx/mlxlm_switchglu_q4.py"
TOKENS = [1, 2, 4, 8, 16]


def gate():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "tests/test_mlx_variants.py", "-k", "ar_expert"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-2000:])
        return "unit tests failed"
    code = f"""
import sys; sys.path.insert(0, '.')
import mlx.core as mx
from moelab import mlx_backend as be
from moelab.shapes import SHAPES
sh = SHAPES['qwen3-30b-a3b']; w = be.make_weights(sh, 'bfloat16', seed=7)
m = be.load_variant('{CAND}'); st = m.prepare(*w); rw = m.dequant(st)
worst = 0.0
for routing in ['uniform', 'trace', 'one_hot']:
    for T in [1, 2, 3, 8, 16]:
        x, idx, wts = be.make_inputs(sh, T, 'bfloat16', routing, seed=T + 11)
        e = be.errors(m.run(st, x, idx, wts), be.reference(*rw, x, idx, wts))
        assert e['finite'], (routing, T, e)
        worst = max(worst, e['rel_fro'])
print('worst_rel_err', worst)
assert worst < 1e-2, worst
"""
    r = subprocess.run([PY, "-c", code], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return "real-shape correctness failed"
    return None


def main():
    err = gate()
    if err:
        print("status: incorrect")
        print("reason:", err)
        return
    out = os.path.join(ROOT, "runs/ar_last.json")
    r = subprocess.run([PY, "-m", "moelab.compare", BASE, CAND, "--shape", "qwen3-30b-a3b", "--dtype", "bfloat16",
                        "--routing", "trace", "--tokens", *map(str, TOKENS), "--launches", "3", "--rounds", "6",
                        "--out", out], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        print("status: crash")
        return
    print(r.stdout.strip())
    s = json.load(open(out))["summary"]
    sp = [x["speedup_B_over_A"] for x in s]
    width = max(x["ci95"][1] - x["ci95"][0] for x in s)
    print("per_T: " + " ".join(f"T{x['T']}={x['speedup_B_over_A']:.3f}" for x in s))
    print(f"max_ci_width: {width:.3f}")
    print(f"score: {float(np.exp(np.mean(np.log(sp)))):.4f}")
    print("status: ok")


if __name__ == "__main__":
    sys.exit(main())
