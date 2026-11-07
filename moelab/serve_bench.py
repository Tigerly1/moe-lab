"""Real-server A/B (round 8): start `moelab.serve` (patched) or the same with MOELAB_STOCK=1, fire N concurrent
OpenAI /v1/completions requests (different prompts, greedy, fixed max_tokens) and measure aggregate completion
tokens per second and per-request latency, as a client sees them. Stock and patched alternate per round.

    .venv-latest/bin/python -m moelab.serve_bench --model mlx-community/Qwen3-30B-A3B-4bit --clients 1 8 32
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def start(model, port, stock, conc):
    env = dict(os.environ, MOELAB_STOCK="1" if stock else "")
    # server logs go to a file: an unread PIPE fills up (64 KB) and blocks the server (round-8 bug: timeouts at 32 clients)
    log = open(os.path.join(os.path.dirname(HERE), "runs", "r8", f"server_{'stock' if stock else 'ours'}_{port}.log"), "w")
    p = subprocess.Popen([sys.executable, "-m", "moelab.serve", "--model", model, "--port", str(port),
                          "--decode-concurrency", str(conc), "--prompt-concurrency", str(min(conc, 8))],
                         env=env, stdout=log, stderr=log, text=True, cwd=os.path.dirname(HERE))
    p.logpath = log.name
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read()
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError(open(p.logpath).read()[-2000:])
            time.sleep(1)
    raise RuntimeError("server did not start")


def request(port, model, prompt, max_tokens):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return time.perf_counter() - t0, r["usage"]["completion_tokens"], r["choices"][0]["text"]


def burst(port, model, prompts, max_tokens):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(len(prompts)) as ex:
        res = list(ex.map(lambda p: request(port, model, p, max_tokens), prompts))
    wall = time.perf_counter() - t0
    toks = sum(r[1] for r in res)
    return {"wall_s": wall, "tokens": toks, "agg_tps": toks / wall, "lat_median_s": float(np.median([r[0] for r in res])),
            "texts": [r[2] for r in res]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--clients", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    words = open(os.path.join(HERE, "data/quality_corpus.txt")).read().split()
    conc = max(a.clients)
    rows = {}
    for r in range(a.rounds):
        for stock in ((True, False) if r % 2 == 0 else (False, True)):
            p = start(a.model, a.port, stock, conc)
            try:
                burst(a.port, a.model, [" ".join(words[:60])], 16)                      # warmup (JIT, caches)
                for n in a.clients:
                    # 90-word windows, wrapped around the corpus (round-8 bug: windows past the end were EMPTY,
                    # and one empty prompt kills mlx_lm.server's generation thread for every client)
                    prompts = [" ".join((words * 3)[(i * 37) % len(words):(i * 37) % len(words) + 90]) for i in range(n)]
                    assert all(p.strip() for p in prompts)
                    res = burst(a.port, a.model, prompts, a.max_tokens)
                    rows.setdefault(n, {"stock": [], "ours": []})["stock" if stock else "ours"].append(res)
                    print(f"round {r} {'stock' if stock else 'ours ':5s} clients={n:3d}: {res['agg_tps']:7.1f} tok/s aggregate, "
                          f"median latency {res['lat_median_s']:.2f}s, {res['tokens']} tokens", flush=True)
            finally:
                p.terminate(); p.wait()
    summary = []
    for n, d in rows.items():
        s, o = np.median([x["agg_tps"] for x in d["stock"]]), np.median([x["agg_tps"] for x in d["ours"]])
        ls, lo = np.median([x["lat_median_s"] for x in d["stock"]]), np.median([x["lat_median_s"] for x in d["ours"]])
        summary.append({"clients": n, "stock_tps": float(s), "ours_tps": float(o), "speedup": float(o / s),
                        "stock_latency_s": float(ls), "ours_latency_s": float(lo)})
        print(f"clients={n:3d}: stock {s:7.1f} tok/s -> ours {o:7.1f} tok/s ({o / s:.3f}x); median latency {ls:.2f}s -> {lo:.2f}s")
    if a.out:
        json.dump({"args": vars(a), "summary": summary,
                   "rows": {n: {k: [{kk: vv for kk, vv in x.items() if kk != "texts"} for x in v] for k, v in d.items()} for n, d in rows.items()}},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
