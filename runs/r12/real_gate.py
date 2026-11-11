"""Does the patched gate still take the kernel path on a real converted model? Counts calls; one small model."""
import collections, sys, mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models import switch_kernels as sk
hits = collections.Counter(); orig = sk.switch_glu
def counted(glu, x, indices):
    y = orig(glu, x, indices)
    hits[("kernel" if y is not None else "fallback", str(x.dtype), str(glu.gate_proj.scales.dtype), indices.size // indices.shape[-1])] += 1
    return y
sk.switch_glu = counted
model, tok = load(sys.argv[1])
out = generate(model, tok, prompt="The capital of France is", max_tokens=12, verbose=False)
print(repr(out[:80]))
for k, v in sorted(hits.items(), key=lambda kv: -kv[1]): print(v, k)
print("peak GB", round(mx.get_peak_memory() / 2**30, 1))
