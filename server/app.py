"""Kernel lab server. Run:  .venv/bin/uvicorn server.app:app --port 8765

Edits variant files on disk, runs each job in a fresh worker subprocess,
streams NDJSON results to the browser, appends them to runs/history.jsonl.
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from moelab import roofline  # noqa: E402
from moelab.shapes import SHAPES, TOKEN_SWEEP  # noqa: E402

VARIANT_DIRS = {"mlx": ROOT / "moelab/variants/mlx", "cuda": ROOT / "moelab/variants/cuda"}
HISTORY = ROOT / "runs/history.jsonl"
JOB_TIMEOUT_S = 300
NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")
app = FastAPI()
gpu_lock = asyncio.Lock()  # one benchmark at a time, never overlap on the GPU


def vpath(backend, name):
    if backend not in VARIANT_DIRS or not NAME_RE.match(name):
        raise HTTPException(400, "bad backend or name")
    return VARIANT_DIRS[backend] / f"{name}.py"


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/meta")
def meta():
    return {
        "shapes": {k: v.to_dict() for k, v in SHAPES.items()},
        "tokens": TOKEN_SWEEP, "hardware": roofline.HARDWARE,
        "variants": {b: sorted(p.stem for p in d.glob("*.py")) for b, d in VARIANT_DIRS.items()},
    }


@app.get("/api/variant/{backend}/{name}")
def get_variant(backend: str, name: str):
    p = vpath(backend, name)
    if not p.exists():
        raise HTTPException(404)
    return {"name": name, "source": p.read_text()}


class Source(BaseModel):
    source: str


@app.put("/api/variant/{backend}/{name}")
def put_variant(backend: str, name: str, body: Source):
    vpath(backend, name).write_text(body.source)
    return {"ok": True}


class Job(BaseModel):
    backend: str = "mlx"
    variants: list[str]
    shape: str
    tokens: list[int]
    dtype: str = "float16"
    routing: str = "uniform"
    hw: str = "m4-pro"
    iters: int = 30


async def run_local(job: dict):
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "moelab.worker", cwd=ROOT,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, start_new_session=True)
    proc.stdin.write(json.dumps(job).encode())
    proc.stdin.close()
    deadline = time.monotonic() + JOB_TIMEOUT_S
    try:
        while True:
            line = await asyncio.wait_for(proc.stdout.readline(), max(1, deadline - time.monotonic()))
            if not line:
                break
            yield line.decode()
    except asyncio.TimeoutError:
        proc.kill()
        yield json.dumps({"type": "error", "error": f"timeout after {JOB_TIMEOUT_S}s (kernel hang?)"}) + "\n"
    code = await proc.wait()
    if code != 0:
        err = (await proc.stderr.read()).decode()[-3000:]
        yield json.dumps({"type": "error", "error": f"worker exited {code}", "trace": err}) + "\n"


async def run_remote_cuda(job: dict):
    """NVIDIA runs via Modal (see moelab/remote_modal.py). Requires `modal setup` once."""
    try:
        import modal
        fn = modal.Function.from_name("moe-lab", "run_job")
        sources = {n: vpath("cuda", n).read_text() for n in job["variants"]}
        async for line in fn.remote_gen.aio(job, sources):
            yield line
    except Exception as ex:
        yield json.dumps({"type": "error", "error": f"remote CUDA failed: {ex}"}) + "\n"


@app.post("/api/run")
async def run(job: Job):
    j = job.model_dump()
    if j["backend"] == "mlx":
        j["variants"] = [str(vpath("mlx", n)) for n in job.variants]
    run_id = time.strftime("%Y%m%d-%H%M%S")

    async def stream():
        async with gpu_lock:
            gen = run_local(j) if j["backend"] == "mlx" else run_remote_cuda(j)
            with HISTORY.open("a") as hist:
                async for line in gen:
                    rec = json.loads(line)
                    rec.update(run_id=run_id, shape=job.shape, dtype=job.dtype, routing=job.routing, backend=job.backend)
                    if rec["type"] == "result":
                        hist.write(json.dumps(rec) + "\n")
                    yield json.dumps(rec) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/history")
def history(limit: int = 2000):
    if not HISTORY.exists():
        return []
    return [json.loads(l) for l in HISTORY.read_text().splitlines()[-limit:]]
