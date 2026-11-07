"""READ-ONLY evaluator, round 3 (T>1 expert-reuse kernels). Agents edit only moelab/variants/mlx/ar_tiles.py.

    .venv/bin/python autoresearch/bench_tiles.py > run.log 2>&1; grep "^score:\\|^status:\\|^per_point:\\|^ci:" run.log

gate: unit tests for ar_tiles + bf16 real-shape correctness (qwen3/olmoe/dsv2lite; T 1,2,3,8,16,32; uniform, one_hot, trace).
speed: interleaved A/B vs real mlx-lm path; 3 launches x 6 rounds per config:
  qwen3-30b-a3b trace T 2 4 8 16 | olmoe-1b-7b uniform T 2 4 8 | deepseek-v2-lite uniform T 2 4 8
score = geometric mean of the 10 speedups. ci = mean 95% CI half-width.
"""
import json, os, subprocess, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv/bin/python")
CAND, BASE = "moelab/variants/mlx/ar_tiles.py", "moelab/variants/mlx/mlxlm_switchglu_q4.py"
CONFIGS = [("qwen3-30b-a3b", "trace", [2, 4, 8, 16]), ("olmoe-1b-7b", "uniform", [2, 4, 8]), ("deepseek-v2-lite", "uniform", [2, 4, 8])]

def gate():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "tests/test_mlx_variants.py", "tests/test_int4_small_scales.py", "-k", "ar_tiles"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:]); return "unit tests failed"
    code = f"""
import sys; sys.path.insert(0, '.')
from moelab import mlx_backend as be
from moelab.shapes import SHAPES
worst = 0.0
for s in ['qwen3-30b-a3b', 'olmoe-1b-7b', 'deepseek-v2-lite']:
    sh = SHAPES[s]; w = be.make_weights(sh, 'bfloat16', seed=9)
    m = be.load_variant('{CAND}'); st = m.prepare(*w); rw = m.dequant(st)
    for routing in ['uniform', 'one_hot'] + (['trace'] if s.startswith('qwen3') else []):
        for T in [1, 2, 3, 8, 16, 32]:
            x, i, wt = be.make_inputs(sh, T, 'bfloat16', routing, seed=T + 17)
            e = be.errors(m.run(st, x, i, wt), be.reference(*rw, x, i, wt))
            assert e['finite'], (s, routing, T)
            worst = max(worst, e['rel_fro'])
    del w, st, rw
print('worst_rel_err', worst)
assert worst < 1e-2, worst
"""
    r = subprocess.run([PY, "-c", code], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode:
        print(r.stderr[-2000:]); return "real-shape correctness failed"

def main():
    err = gate()
    if err:
        print("status: incorrect"); print("reason:", err); return
    pts, hws = [], []
    for shape, routing, Ts in CONFIGS:
        out = os.path.join(ROOT, f"runs/art_{shape}.json")
        r = subprocess.run([PY, "-m", "moelab.compare", BASE, CAND, "--shape", shape, "--dtype", "bfloat16", "--routing", routing,
                            "--tokens", *map(str, Ts), "--launches", "3", "--rounds", "6", "--out", out], cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-3000:]); print("status: crash"); return
        for x in json.load(open(out))["summary"]:
            pts.append((f"{shape.split('-')[0]}T{x['T']}", x["speedup_B_over_A"])); hws.append((x["ci95"][1] - x["ci95"][0]) / 2)
    print("per_point: " + " ".join(f"{n}={v:.3f}" for n, v in pts))
    print(f"ci: {float(np.mean(hws)):.4f}")
    print(f"score: {float(np.exp(np.mean(np.log([v for _, v in pts])))):.4f}")
    print("status: ok")

if __name__ == "__main__":
    sys.exit(main())
