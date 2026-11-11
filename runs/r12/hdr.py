import json, struct, sys, glob, os, collections
for d in sys.argv[1:]:
    c = collections.Counter()
    for f in sorted(glob.glob(os.path.join(d, "snapshots", "*", "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        for k, v in h.items():
            if k == "__metadata__": continue
            if any(s in k for s in ("gate_proj", "up_proj", "down_proj", "switch_mlp", "experts")) and k.split(".")[-1] in ("scales", "biases", "weight"):
                c[(k.split(".")[-1], v["dtype"])] += 1
    print(os.path.basename(d).replace("models--mlx-community--", ""), dict(c))
