"""Round 11: timing-only ablations of metal_q4_tiles_v2 (outputs are deliberately wrong).

    .venv/bin/python scripts/r11_ablations.py <out dir>
    .venv/bin/python -m moelab.compare <out>/tv2_base.py <out>/tv2_nda.py --tokens 8 16 32 --launches 4 --rounds 4

base    unchanged copy (own kernel names, so both arms compile separately)
noshuf  no pg/pu/pw bit shuffle on the MMA path
nodeq   noshuf + no dequant on the MMA path only (tiles of >= 3 slots)
nda     nodeq + no decode on the 1-/2-slot GEMV paths (one xor per weight; the multiply-accumulate stays)
ndb     nodeq + one bit-cast per 32-bit word on the GEMV paths; the value no longer depends on the weight index,
        so the compiler can also drop the per-weight multiplies: an upper bound, not a decode cost
"""
import re
import sys

src = open("moelab/variants/mlx/metal_q4_tiles_v2.py").read()
out = sys.argv[1]


def w(name, s):
    open(f"{out}/tv2_{name}.py", "w").write(s.replace("mq4tv3_", f"mq4{name}_"))


w("base", src)
shuf = src
for v, a in (("pg", "wg"), ("pu", "wu"), ("pw", "w")):
    shuf = re.sub(rf"uint {v}\[2\] = \{{.*?\}};", f"uint {v}[2] = {{{a}.x, {a}.y}};", shuf)
assert shuf.count("{wg.x, wg.y}") == 1 and shuf.count("{w.x, w.y}") == 1
w("noshuf", shuf)
deq = shuf.replace("return as_type<half2>(((P >> (4 * s)) & 0x000F000Fu) | 0x64006400u);", "return as_type<half2>(P ^ s);")
deq = deq.replace("return as_type<vec<bfloat, 2>>(((P >> (4 * s)) & 0x000F000Fu) | 0x43004300u);",
                  "return as_type<vec<bfloat, 2>>(P ^ s);")
assert deq.count("(P ^ s)") == 2
w("nodeq", deq)
a1 = "acc += xs[j * 8 + i] * float(word & (0xFu << (4 * i)));"
a2 = "float q = float(word & (0xFu << (4 * i)));"
assert deq.count(a1) == 1 and deq.count(a2) == 1
w("nda", deq.replace(a1, "acc += xs[j * 8 + i] * as_type<float>(word ^ i);").replace(a2, "float q = as_type<float>(word ^ i);"))
cast = "as_type<float>((word & 0x007FFFFFu) | 0x3F800000u)"
w("ndb", deq.replace(a1, f"acc += xs[j * 8 + i] * {cast};").replace(a2, f"float q = {cast};"))
