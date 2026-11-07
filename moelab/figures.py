"""Thesis figures from measured data (one command, no GPU):

    .venv/bin/python -m moelab.figures      # -> docs/thesis/fig/*.pdf (+ .png previews)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs/thesis/fig")
R = lambda p: json.load(open(os.path.join(ROOT, "runs", p)))  # noqa: E731
INK, MUTED, ACC, WARM, BAD = "#16202c", "#7a8591", "#1f5fbf", "#c8641c", "#c23b3b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e4e7ea", "grid.linewidth": 0.6, "legend.frameon": False})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=150)
    plt.close(fig)
    print("wrote", name)


def speedup_vs_T():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for f, lab, col, ls in [("cmp_tiles_v2_qwen3-30b-a3b_trace.json", "tiles_v2 (runda 3), ślady", INK, "-"),
                            ("cmp_metal_q4_tiles_mma_qwen3_trace.json", "kafle + MMA (runda 1), ślady rzeczywiste", ACC, "-"),
                            ("cmp_metal_q4_tiles_mma_qwen3_uniform.json", "kafle + MMA, routing jednostajny", ACC, "--"),
                            ("cmp_metal_q4_tiles_b_qwen3_trace.json", "kafle skalarne, ślady", WARM, "-"),
                            ("cmp_v31_qwen3_trace.json", "v31 (GEMV na slot), ślady", MUTED, ":")]:
        s = R(f)["summary"]
        T = [x["T"] for x in s]; y = [x["speedup_B_over_A"] for x in s]
        lo = [x["ci95"][0] for x in s]; hi = [x["ci95"][1] for x in s]
        ax.plot(T, y, ls, color=col, marker="o", ms=3, label=lab)
        ax.fill_between(T, lo, hi, color=col, alpha=0.15, lw=0)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xticks(T, [str(t) for t in T]); ax.set_xlabel("tokeny w kroku T"); ax.set_ylabel("przyspieszenie vs mlx-lm (×)")
    ax.set_title("Qwen3-30B-A3B, warstwa MoE, bf16 / int4", fontsize=10, color=INK)
    ax.legend(fontsize=8)
    save(fig, "speedup_vs_T")


def unique_experts():
    s = R("routing_qwen3_stats.json")
    T = s["T"]
    cons = np.mean([v["consecutive"] for v in s["domains"].values()], axis=0)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(T, [8 * t for t in T], "--", color=BAD, label="sloty T·k (czytane per-slot)")
    ax.plot(T, s["uniform"], ":", color=MUTED, label="model jednostajny")
    ax.plot(T, s["batched"], "-o", ms=3, color=WARM, label="tokeny z różnych sekwencji")
    ax.plot(T, cons, "-o", ms=3, color=ACC, label="kolejne tokeny (ślady)")
    ax.set_xscale("log", base=2); ax.set_xticks(T, [str(t) for t in T]); ax.set_ylim(0, 140); ax.set_xlabel("T"); ax.set_ylabel("unikalni eksperci U(T)")
    ax.legend(fontsize=8)
    save(fig, "unique_experts")


def per_layer():
    d = R("routing_qwen3_perlayer.json")
    ov = np.array(d["adj_overlap"]) * 100
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.bar(range(len(ov)), ov, color=[BAD] + [ACC] * (len(ov) - 1), width=0.8)
    ax.axhline(100 * 8 / 128, color=MUTED, ls="--", lw=1, label="routing losowy (6,25%)")
    ax.set_xlabel("warstwa"); ax.set_ylabel("% ekspertów wspólnych\nz poprzednim tokenem"); ax.legend(fontsize=8)
    save(fig, "per_layer_overlap")


def sustained():
    d = R("sustained_T1.json")["rows"]
    t = np.array([x["t_s"] for x in d]); r = np.array([x["ratio"] for x in d])
    k = 200; rm = np.convolve(r, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.plot(t, r, ",", color=MUTED, alpha=0.3)
    ax.plot(t[k - 1:], rm, color=ACC, lw=1.5, label="średnia krocząca (200 rund)")
    ax.set_ylim(1.0, 1.35); ax.set_xlabel("sekundy ciągłego obciążenia"); ax.set_ylabel("przyspieszenie (×)")
    ax.legend(fontsize=8)
    save(fig, "sustained_ratio")


def verify_bench():
    d = R("e2e_verify_bench_v2.json")
    c = [x["chunk"] for x in d]; s = [np.median(x["stock"]) for x in d]; v = [np.median(x["variant"]) for x in d]
    x = np.arange(len(c)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(x - w / 2, s, w, color=MUTED, label="mlx-lm")
    ax.bar(x + w / 2, v, w, color=ACC, label="nasz kernel")
    for i in range(len(c)):
        ax.text(x[i] + w / 2, v[i] + 3, f"+{100 * (v[i] / s[i] - 1):.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [str(k) for k in c]); ax.set_xlabel("tokenów na krok weryfikacji"); ax.set_ylabel("tokeny/s (cały model)")
    ax.legend(fontsize=8)
    save(fig, "verify_bench")


def ablation():
    pairs = [("Qwen3 (K=768)", "cmp_v31_qwen3-30b-a3b.json", "cmp_v31_abl-qwen3-f1024.json"),
             ("gpt-oss (K=2880)", "cmp_v31_gpt-oss-20b.json", "cmp_v31_abl-gptoss-3072.json")]
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    for i, (lab, a, b) in enumerate(pairs):
        ua = R(a)["summary"][0]["speedup_B_over_A"]; al = R(b)["summary"][0]["speedup_B_over_A"]
        ax.barh(i, (al - 1) * 100, color=ACC); ax.barh(i, (ua - al) * 100, left=(al - 1) * 100, color=WARM)
    ax.set_yticks(range(len(pairs)), [p[0] for p in pairs])
    ax.set_xlabel("przyspieszenie przy T=1 (%)")
    ax.legend(["fuzja + mniej dispatchy", "wolna ścieżka MLX (K % 512 ≠ 0)"], fontsize=8, loc="lower right")
    save(fig, "ablation_alignment")


def mxfp4():
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for f, lab, col in [("cmp_mxfp4_r5_qwen3_trace.json", "Qwen3-30B-A3B, ślady", ACC),
                        ("cmp_mxfp4_r5_gpt-oss-20b.json", "gpt-oss-20b, jednostajny", WARM)]:
        s = R(f)["summary"]
        T = [x["T"] for x in s]; y = [x["speedup_B_over_A"] for x in s]
        ax.plot(T, y, "-o", ms=3, color=col, label=lab)
        ax.fill_between(T, [x["ci95"][0] for x in s], [x["ci95"][1] for x in s], color=col, alpha=0.15, lw=0)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("tokeny w kroku T"); ax.set_ylabel("przyspieszenie vs mlx-lm MXFP4 (×)"); ax.legend(fontsize=8)
    save(fig, "mxfp4_speedup")


if __name__ == "__main__":
    for f in (speedup_vs_T, unique_experts, per_layer, sustained, verify_bench, ablation, mxfp4):
        f()
