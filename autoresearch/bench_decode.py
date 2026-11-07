"""READ-ONLY evaluator, round 2 (plain decode, T=1). Agents edit only moelab/variants/mlx/ar_decode.py.

    .venv/bin/python autoresearch/bench_decode.py > run.log 2>&1; grep "^score:\\|^status:\\|^per_shape:\\|^ci:" run.log

gate: unit tests for ar_decode + real-shape bf16 correctness (5 shapes, T in 1,2,3,8, uniform/one_hot).
speed: interleaved A/B vs real mlx-lm path at T=1 on 5 shapes, 3 launches x 6 rounds each.
score = geometric mean speedup over shapes. ci: half-width of each shape's 95% CI.
"""
import json, os, subprocess, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv/bin/python")
CAND, BASE = "moelab/variants/mlx/ar_decode.py", "moelab/variants/mlx/mlxlm_switchglu_q4.py"
SHAPES = ["qwen3-30b-a3b", "olmoe-1b-7b", "deepseek-v2-lite", "gpt-oss-20b", "mixtral-8x7b"]

def gate():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "tests/test_mlx_variants.py", "-k", "ar_decode"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:]); return "unit tests failed"
    code = f"""
import sys; sys.path.insert(0, '.')
from moelab import mlx_backend as be
from moelab.shapes import SHAPES
worst = 0.0
for s in {SHAPES!r}:
    sh = SHAPES[s]; w = be.make_weights(sh, 'bfloat16', seed=5)
    m = be.load_variant('{CAND}'); st = m.prepare(*w); rw = m.dequant(st)
    for routing in ['uniform', 'one_hot']:
        for T in [1, 2, 3, 8]:
            x, i, wt = be.make_inputs(sh, T, 'bfloat16', routing, seed=T + 3)
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
    sp, hw = [], []
    for s in SHAPES:
        out = os.path.join(ROOT, f"runs/ard_{s}.json")
        r = subprocess.run([PY, "-m", "moelab.compare", BASE, CAND, "--shape", s, "--dtype", "bfloat16", "--tokens", "1",
                            "--launches", "3", "--rounds", "6", "--out", out], cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-3000:]); print("status: crash"); return
        x = json.load(open(out))["summary"][0]
        sp.append(x["speedup_B_over_A"]); hw.append((x["ci95"][1] - x["ci95"][0]) / 2)
    print("per_shape: " + " ".join(f"{s}={v:.3f}" for s, v in zip(SHAPES, sp)))
    print("ci: " + " ".join(f"{h:.3f}" for h in hw))
    print(f"score: {float(np.exp(np.mean(np.log(sp)))):.4f}")
    print("status: ok")

if __name__ == "__main__":
    sys.exit(main())
