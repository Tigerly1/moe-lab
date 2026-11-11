import sys, mlx.core as mx
from mlx_lm import load, generate
model, tok = load(sys.argv[1])
model.set_dtype(mx.float32)
print("fp32 stock raw prompt:", repr(generate(model, tok, prompt="The capital of France is", max_tokens=12, verbose=False)))
