"""Remote NVIDIA runner on Modal. One-time setup:
    .venv/bin/modal setup                      # login (opens browser)
    .venv/bin/modal deploy moelab/remote_modal.py
Then pick "CUDA / Triton (Modal)" in the lab UI. GPU type: MOE_LAB_GPU env at deploy (default L4).
The lab sends the CURRENT editor source of each variant, so no redeploy after edits.
"""
import json
import os
import subprocess
import sys
import tempfile

import modal

GPU = os.environ.get("MOE_LAB_GPU", "L4")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "triton", "numpy")
         .add_local_python_source("moelab"))
app = modal.App("moe-lab")


@app.function(gpu=GPU, image=image, timeout=900, scaledown_window=600)
def run_job(job: dict, sources: dict):
    tmp = tempfile.mkdtemp()
    paths = []
    for name, src in sources.items():
        p = os.path.join(tmp, f"{name}.py")
        open(p, "w").write(src)
        paths.append(p)
    job = {**job, "variants": paths, "backend": "cuda"}
    proc = subprocess.Popen([sys.executable, "-m", "moelab.worker"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc.stdin.write(json.dumps(job))
    proc.stdin.close()
    for line in proc.stdout:
        yield line
    if proc.wait() != 0:
        yield json.dumps({"type": "error", "error": "worker crashed", "trace": proc.stderr.read()[-3000:]}) + "\n"
