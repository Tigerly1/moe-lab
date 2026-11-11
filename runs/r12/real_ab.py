"""Same model, same prompt: kernel path vs stock path (kernels off). Compares logits and greedy text."""
import sys, mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models import switch_kernels as sk
model, tok = load(sys.argv[1])
msgs = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = mx.array(tok.encode(prompt))[None]
def run():
    lg = model(ids).astype(mx.float32); mx.eval(lg)
    return lg, generate(model, tok, prompt=prompt, max_tokens=24, verbose=False)
lk, tk = run()
k, sk._kernels = sk._kernels, None
ls, ts = run()
sk._kernels = k
d = (mx.linalg.norm((lk - ls).flatten()) / mx.linalg.norm(ls.flatten())).item()
print("prompt tokens", ids.shape[1], "logits rel diff kernel vs stock", f"{d:.2e}", "argmax agree", f"{(mx.argmax(lk, -1) == mx.argmax(ls, -1)).astype(mx.float32).mean().item():.3f}")
print("kernel:", repr(tk[:120])); print("stock :", repr(ts[:120]))
print("peak GB", round(mx.get_peak_memory() / 2**30, 1))
