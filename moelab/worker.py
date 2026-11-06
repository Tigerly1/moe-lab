"""Isolated benchmark worker. Reads a job (JSON) from stdin, prints JSON lines.

Job: {"backend": "mlx", "variants": [path, ...], "shape": name | {d,f,E,k},
      "tokens": [...], "dtype": "float16", "routing": "uniform", "iters": 30}
Run fresh per job, so edited kernels never hit a stale in-process cache
and a crashing kernel cannot take the server down.
"""
import json
import os
import platform
import sys
import traceback

from .shapes import SHAPES, MoEShape


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    job = json.loads(sys.stdin.read())
    sh = job["shape"]
    shape = SHAPES[sh] if isinstance(sh, str) else MoEShape(name="custom", **sh)
    if job.get("backend", "mlx") == "mlx":
        import mlx.core as mx
        from . import mlx_backend as be
        emit({"type": "env", "device": mx.device_info().get("device_name"), "mlx": mx.__version__,
              "python": platform.python_version(), "shape": shape.to_dict()})
        weights = be.make_weights(shape, job.get("dtype", "float16"))
    else:
        from . import torch_backend as be
        emit({"type": "env", **be.env_info(), "shape": shape.to_dict()})
        weights = be.make_weights(shape, job.get("dtype", "float16"))
    for path in job["variants"]:
        try:
            mod = be.load_variant(path)
            mod.META["name"] = os.path.splitext(os.path.basename(path))[0]  # forks stay distinct
            emit({"type": "start", "variant": mod.META["name"], "path": path})
            be.bench_variant(mod, shape, job["tokens"], dtype=job.get("dtype", "float16"),
                             routing=job.get("routing", "uniform"), hw=job.get("hw", "m4-pro"),
                             iters=job.get("iters", 30), weights=weights, emit=emit)
        except Exception as ex:  # compile error, shape assert, ... -> report, keep going
            emit({"type": "error", "path": path, "error": f"{type(ex).__name__}: {ex}",
                  "trace": traceback.format_exc(limit=4)})
    emit({"type": "done"})


if __name__ == "__main__":
    main()
