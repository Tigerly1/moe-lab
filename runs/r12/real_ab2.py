"""Is the 4e-2 logit gap bf16 rounding? fp32 stock = truth; compare bf16 kernel, bf16 stock, fp32 kernel. Plus raw prompt text."""
import sys, mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models import switch_kernels as sk
model, tok = load(sys.argv[1])
msgs = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = mx.array(tok.encode(prompt))[None]
K = sk._kernels
def logits(kern):
    sk._kernels = K if kern else None
    lg = model(ids).astype(mx.float32); mx.eval(lg); sk._kernels = K
    return lg
def rel(a, b): return (mx.linalg.norm((a - b).flatten()) / mx.linalg.norm(b.flatten())).item()
raw = "The capital of France is"
txt = {}
for kern in (True, False):
    sk._kernels = K if kern else None
    txt[kern] = generate(model, tok, prompt=raw, max_tokens=12, verbose=False)
sk._kernels = K
print("raw prompt kernel:", repr(txt[True])); print("raw prompt stock :", repr(txt[False]))
bk, bs = logits(True), logits(False)
model.set_dtype(mx.float32)
fk, fs = logits(True), logits(False)
print(f"fp32 kernel vs fp32 stock {rel(fk, fs):.2e}")
print(f"bf16 kernel vs fp32 stock {rel(bk, fs):.2e}")
print(f"bf16 stock  vs fp32 stock {rel(bs, fs):.2e}")
am = lambda a: mx.argmax(a, -1)
print("argmax agree with fp32 stock: bf16 kernel", (am(bk) == am(fs)).sum().item(), "bf16 stock", (am(bs) == am(fs)).sum().item(), "fp32 kernel", (am(fk) == am(fs)).sum().item(), "of", ids.shape[1])
print("peak GB", round(mx.get_peak_memory() / 2**30, 1))
