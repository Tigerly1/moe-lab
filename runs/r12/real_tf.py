"""Teacher-forced decode, one token at a time with KV cache (T = 1 kernels). fp32 stock = truth."""
import sys, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models import switch_kernels as sk
model, tok = load(sys.argv[1])
ids = tok.encode("The capital of France is the capital of France is the capital of France")
K = sk._kernels
def steps(kern):
    sk._kernels = K if kern else None
    c = make_prompt_cache(model); out = []
    lg = model(mx.array(ids[:5])[None], cache=c)[:, -1]; out.append(lg.astype(mx.float32))
    for t in ids[5:]:
        lg = model(mx.array([[t]]), cache=c)[:, -1]; out.append(lg.astype(mx.float32))
    sk._kernels = K
    o = mx.concatenate(out); mx.eval(o); return o
bk, bs = steps(True), steps(False)
model.set_dtype(mx.float32)
fs = steps(False)
rel = lambda a, b: (mx.linalg.norm(a - b, axis=-1) / mx.linalg.norm(b, axis=-1)).tolist()
rk, rs = rel(bk, fs), rel(bs, fs)
srt = mx.sort(fs, axis=-1)
margin = (srt[:, -1] - srt[:, -2]).tolist()
d = lambda i: repr(tok.decode([i]))
print("pos  after        margin  err_kern err_stock  top1 truth / kernel / stock")
for i in range(fs.shape[0]):
    a = [mx.argmax(z[i]).item() for z in (fs, bk, bs)]
    ctx = d(ids[4 + i])
    print(f"{i:3d}  {ctx:12s} {margin[i]:6.2f}  {rk[i]:.2e} {rs[i]:.2e}  {d(a[0])} / {d(a[1])} / {d(a[2])}")
print("mean err kernel", f"{sum(rk)/len(rk):.2e}", "stock", f"{sum(rs)/len(rs):.2e}")
