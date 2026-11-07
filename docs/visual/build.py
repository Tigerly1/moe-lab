"""Builds the visual explainer page (inline SVG charts drawn to scale from measured data)."""
import json
import sys

REPO = sys.argv[1]
OUT = sys.argv[2]

# ---------- data (all measured; sources in captions) ----------
floor = {  # floor_probe: % of bytes floor, distinct uniform routing per call (us)
    "Qwen3-30B": {"T": [1, 2, 4, 8, 16, 32], "ours": [86, 85, 89, 100, 93, 75], "stock": [77, 80, 81, 83, 68, 50]},
}
grid = json.load(open(f"{REPO}/runs/r9/grid_q4_mma64.json"))["rows"]
gemma_grid = [(r["load"], r["speedup"]) for r in grid if r["shape"] == "d2816f704E128k8"]
qwen_grid = [(r["load"], r["speedup"]) for r in grid if r["shape"] == "d2048f768E128k8"]
upstream = [  # mlx_lm benchmark main vs PR #2044, mean of 2 passes
    ("Gemma-4-26B", 1.05, 1.29, 1.71), ("Qwen3-30B", 1.08, 1.20, 1.56), ("DeepSeek-V2-Lite", 1.05, 1.27, 1.55),
    ("GLM-4.7-Flash", 1.03, 1.35, 1.40), ("LFM2.5-8B", 1.06, 1.33, 1.38)]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


# ---------- figure 1: where time goes per token (memory vs compute) ----------
def fig_wall():
    W, H, x0, x1 = 680, 170, 150, 640
    ms_mem, ms_cmp = 7.9, 0.5   # 1.9 GB / 240 GB/s ; ~3.6 GFLOP-ish active compute at several TFLOP/s
    sc = (x1 - x0) / 9.0
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Czas na token: pamięć 7,9 ms, obliczenia około 0,5 ms">']
    for v in range(0, 10, 2):
        x = x0 + v * sc
        s.append(f'<line x1="{x:.0f}" y1="22" x2="{x:.0f}" y2="128" class="grid"/>'
                 f'<text x="{x:.0f}" y="146" class="tick" text-anchor="middle">{v} ms</text>')
    rows = [("Czytanie wag z pamięci", ms_mem, "mem", "1,9 GB ÷ 240 GB/s = 7,9 ms"),
            ("Obliczenia", ms_cmp, "cmp", "≈ 0,5 ms")]
    for i, (lab, v, cls, note) in enumerate(rows):
        y = 34 + i * 50
        s.append(f'<text x="{x0 - 10}" y="{y + 17}" class="lab" text-anchor="end">{lab}</text>'
                 f'<rect x="{x0}" y="{y}" width="{v * sc:.1f}" height="26" rx="3" class="{cls}"/>'
                 f'<text x="{x0 + v * sc + 8:.1f}" y="{y + 17}" class="val">{note}</text>')
    s.append(f'<text x="{x0}" y="{H - 4}" class="cap">Qwen3-30B-A3B, 4 bity, jeden token. Sufit: 1 / 7,9 ms ≈ 126 tok/s.</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- figure 2: MoE layer ----------
def fig_moe():
    W, H = 680, 250
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Warstwa MoE: router wybiera 8 ze 128 ekspertów">']
    s.append('<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
             '<path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>')
    s.append('<rect x="14" y="100" width="92" height="44" rx="6" class="box"/>'
             '<text x="60" y="127" class="boxt" text-anchor="middle">token</text>')
    s.append('<rect x="146" y="100" width="92" height="44" rx="6" class="box acc"/>'
             '<text x="192" y="127" class="boxt" text-anchor="middle">router</text>')
    s.append('<line x1="106" y1="122" x2="144" y2="122" class="edge" marker-end="url(#ar)"/>')
    cols, rows_, cw, ch, gx, gy = 16, 8, 21, 21, 290, 30
    chosen = {3, 17, 30, 44, 61, 78, 99, 115}
    for i in range(cols * rows_):
        r, c = divmod(i, cols)
        x, y = gx + c * (cw + 3), gy + r * (ch + 3)
        s.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="3" class="{"exp on" if i in chosen else "exp"}"/>')
    for i in sorted(chosen)[:8]:
        r, c = divmod(i, cols)
        x, y = gx + c * (cw + 3) + cw / 2, gy + r * (ch + 3) + ch / 2
        s.append(f'<line x1="238" y1="122" x2="{x:.0f}" y2="{y:.0f}" class="edge faint"/>')
    s.append(f'<text x="{gx}" y="{gy + rows_ * (ch + 3) + 16}" class="cap">128 ekspertów, każdy = 3 macierze '
             f'(gate, up: 2048×768; down: 768×2048). Router wybiera k = 8 (na pomarańczowo).</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- figure 3: stock vs grouped reads ----------
def fig_reads():
    W, H = 680, 300
    toks = ["A", "B", "C", "D"]
    picks = {"A": [1, 4], "B": [1, 2], "C": [1, 5], "D": [2, 4]}   # expert ids 0..5
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Stock czyta eksperta przy każdej parze, my raz na grupę">']
    s.append('<defs><marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
             '<path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>')
    for panel, (ox, title) in enumerate([(10, "Stock mlx-lm: odczyt na każdą parę"), (350, "Nasz kernel: odczyt raz na eksperta")]):
        s.append(f'<text x="{ox}" y="22" class="h">{title}</text>')
        for i, t in enumerate(toks):
            y = 50 + i * 52
            s.append(f'<rect x="{ox}" y="{y}" width="40" height="34" rx="5" class="box"/>'
                     f'<text x="{ox + 20}" y="{y + 22}" class="boxt" text-anchor="middle">{t}</text>')
        for e in range(6):
            y = 40 + e * 37
            used = any(e in v for v in picks.values())
            s.append(f'<rect x="{ox + 230}" y="{y}" width="64" height="28" rx="5" class="{"exp on" if used else "exp"}"/>'
                     f'<text x="{ox + 262}" y="{y + 19}" class="boxt sm" text-anchor="middle">E{e}</text>')
        reads = 0
        if panel == 0:
            for i, t in enumerate(toks):
                for e in picks[t]:
                    y1, y2 = 50 + i * 52 + 17, 40 + e * 37 + 14
                    s.append(f'<line x1="{ox + 40}" y1="{y1}" x2="{ox + 228}" y2="{y2}" class="edge mem" marker-end="url(#ar2)"/>')
                    reads += 1
        else:
            for e in sorted({e for v in picks.values() for e in v}):
                users = [i for i, t in enumerate(toks) if e in picks[t]]
                y2 = 40 + e * 37 + 14
                bx = ox + 150
                s.append(f'<line x1="{ox + 228}" y1="{y2}" x2="{bx + 6}" y2="{y2}" class="edge mem thick" marker-end="url(#ar2)"/>')
                for i in users:
                    y1 = 50 + i * 52 + 17
                    s.append(f'<line x1="{bx}" y1="{y2}" x2="{ox + 42}" y2="{y1}" class="edge faint"/>')
                reads += 1
        s.append(f'<text x="{ox}" y="{H - 12}" class="big">{reads} odczytów wag</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- figure 3b: picks vs distinct experts (pigeonhole) ----------
def fig_overlap():
    W, H, x0, x1, y0, y1 = 680, 300, 60, 650, 240, 24
    E, K = 128, 8
    Bs = [1, 2, 4, 8, 16, 32]
    picks = [b * K for b in Bs]
    uniq = [E * (1 - (1 - K / E) ** b) for b in Bs]
    xs = {b: x0 + (x1 - x0) * i / (len(Bs) - 1) for i, b in enumerate(Bs)}
    ys = lambda v: y0 - (y0 - y1) * v / 256  # noqa: E731
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Wybory ekspertów a liczba różnych ekspertów przy losowym routingu">']
    for v in (0, 64, 128, 192, 256):
        s.append(f'<line x1="{x0}" y1="{ys(v):.0f}" x2="{x1}" y2="{ys(v):.0f}" class="{"grid strong" if v == 128 else "grid"}"/>'
                 f'<text x="{x0 - 8}" y="{ys(v) + 4:.0f}" class="tick" text-anchor="end">{v}</text>')
    s.append(f'<text x="{x0 + 4}" y="{ys(128) - 6:.0f}" class="tick">wszystkich ekspertów: 128</text>')
    for b, x in xs.items():
        s.append(f'<text x="{x:.0f}" y="{y0 + 18}" class="tick" text-anchor="middle">{b}</text>')
    s.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y0 + 36}" class="tick" text-anchor="middle">zapytań naraz</text>')
    for b, pk, u in zip(Bs, picks, uniq):
        x = xs[b]
        s.append(f'<rect x="{x - 15:.0f}" y="{ys(pk):.1f}" width="30" height="{y0 - ys(pk):.1f}" rx="2" class="b1"/>'
                 f'<rect x="{x - 9:.0f}" y="{ys(u):.1f}" width="18" height="{y0 - ys(u):.1f}" rx="2" class="mem"/>')
        if b >= 8:
            s.append(f'<text x="{x:.0f}" y="{ys(pk) - 6:.0f}" class="val" text-anchor="middle">{pk} / {u:.0f}</text>')
    s.append(f'<rect x="{x0}" y="{H - 16}" width="12" height="10" rx="2" class="b1"/><text x="{x0 + 18}" y="{H - 7}" class="tick">wybory (zapytania × 8)</text>'
             f'<rect x="{x0 + 190}" y="{H - 16}" width="12" height="10" rx="2" class="mem"/><text x="{x0 + 208}" y="{H - 7}" class="tick">różni eksperci, routing losowy (do przeczytania)</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- figure 4: % of floor vs T ----------
def fig_floor():
    W, H, x0, x1, y0, y1 = 680, 300, 60, 650, 250, 30
    d = floor["Qwen3-30B"]
    xs = {T: x0 + (x1 - x0) * i / (len(d["T"]) - 1) for i, T in enumerate(d["T"])}
    ys = lambda p: y0 - (y0 - y1) * p / 100  # noqa: E731
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Procent sufitu przepustowości w funkcji liczby tokenów">']
    for p in (0, 25, 50, 75, 100):
        s.append(f'<line x1="{x0}" y1="{ys(p):.0f}" x2="{x1}" y2="{ys(p):.0f}" class="grid"/>'
                 f'<text x="{x0 - 8}" y="{ys(p) + 4:.0f}" class="tick" text-anchor="end">{p}%</text>')
    for T, x in xs.items():
        s.append(f'<text x="{x:.0f}" y="{y0 + 20}" class="tick" text-anchor="middle">T={T}</text>')
    for key, cls in (("stock", "lstock"), ("ours", "lours")):
        pts = " ".join(f"{xs[T]:.1f},{ys(v):.1f}" for T, v in zip(d["T"], d[key]))
        s.append(f'<polyline points="{pts}" class="{cls}"/>')
        for T, v in zip(d["T"], d[key]):
            s.append(f'<circle cx="{xs[T]:.1f}" cy="{ys(v):.1f}" r="4" class="{cls}d"/>')
        s.append(f'<text x="{x1 + 2}" y="{ys(d[key][-1]) + (16 if key == "stock" else -10):.0f}" class="val" text-anchor="end">'
                 f'{"stock" if key == "stock" else "nasz kernel"} {d[key][-1]}%</text>')
    s.append(f'<text x="{x0}" y="{H - 6}" class="cap">Warstwa MoE Qwen3-30B, routing inny w każdym wywołaniu. 100% = czas potrzebny na samo przeczytanie unikalnych ekspertów przy 240 GB/s.</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- figure 5: dispatcher: speedup vs load ----------
def fig_load():
    W, H, x0, x1, y0, y1 = 680, 290, 60, 650, 240, 24
    loads = sorted({l for l, _ in gemma_grid})
    lx = lambda l: x0 + (x1 - x0) * loads.index(l) / (len(loads) - 1)  # noqa: E731
    ys = lambda v: y0 - (y0 - y1) * (v - 0.8) / (2.0 - 0.8)  # noqa: E731
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Przyspieszenie warstwy w funkcji obciążenia na eksperta">']
    for v in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
        s.append(f'<line x1="{x0}" y1="{ys(v):.0f}" x2="{x1}" y2="{ys(v):.0f}" class="{"grid strong" if v == 1.0 else "grid"}"/>'
                 f'<text x="{x0 - 8}" y="{ys(v) + 4:.0f}" class="tick" text-anchor="end">{v:.1f}×</text>')
    for l in loads:
        s.append(f'<text x="{lx(l):.0f}" y="{y0 + 18}" class="tick" text-anchor="middle">{l:g}</text>')
    s.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y0 + 36}" class="tick" text-anchor="middle">obciążenie T·k/E (średnio tokenów na eksperta)</text>')
    thr = lx(24)
    s.append(f'<rect x="{thr:.0f}" y="{y1}" width="{x1 - thr:.0f}" height="{y0 - y1}" class="zone"/>'
             f'<text x="{thr + 6:.0f}" y="{y1 + 14}" class="val">ścieżka stock (bit-identyczna)</text>')
    for data, cls, name in ((gemma_grid, "lours", "Gemma-4"), (qwen_grid, "lalt", "Qwen3")):
        data = sorted(data)
        pts = " ".join(f"{lx(l):.1f},{ys(v):.1f}" for l, v in data)
        s.append(f'<polyline points="{pts}" class="{cls}"/>')
        for l, v in data:
            s.append(f'<circle cx="{lx(l):.1f}" cy="{ys(v):.1f}" r="3.5" class="{cls}d"/>')
        l, v = max(data, key=lambda t: t[1])
        s.append(f'<text x="{lx(l) + 8:.0f}" y="{ys(v) - 6:.0f}" class="val">{name} {v:.2f}×</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- figure 6: upstream benchmark bars ----------
def fig_upstream():
    W, x0, x1 = 680, 150, 640
    rh, H = 46, 46 * len(upstream) + 70
    sc = lambda v: x0 + (x1 - x0) * (v - 1.0) / 0.8  # noqa: E731
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Przyspieszenie w benchmarku mlx-lm dla 1, 8 i 32 zapytań">']
    for v in (1.0, 1.2, 1.4, 1.6, 1.8):
        s.append(f'<line x1="{sc(v):.0f}" y1="14" x2="{sc(v):.0f}" y2="{H - 46}" class="grid"/>'
                 f'<text x="{sc(v):.0f}" y="{H - 30}" class="tick" text-anchor="middle">{v:.1f}×</text>')
    for i, (name, b1, b8, b32) in enumerate(upstream):
        y = 18 + i * rh
        s.append(f'<text x="{x0 - 10}" y="{y + 22}" class="lab" text-anchor="end">{name}</text>')
        for j, (v, cls) in enumerate(((b1, "b1"), (b8, "b8"), (b32, "b32"))):
            yy = y + j * 12
            s.append(f'<rect x="{x0}" y="{yy}" width="{max(1.5, sc(v) - x0):.1f}" height="10" rx="2" class="{cls}"/>')
        s.append(f'<text x="{sc(b32) + 6:.0f}" y="{y + 33}" class="val">{b32:.2f}×</text>')
    lx = x0
    for lab, cls in (("1 rozmowa", "b1"), ("8 naraz", "b8"), ("32 naraz", "b32")):
        s.append(f'<rect x="{lx}" y="{H - 16}" width="12" height="10" rx="2" class="{cls}"/>'
                 f'<text x="{lx + 18}" y="{H - 7}" class="tick">{lab}</text>')
        lx += 110
    s.append("</svg>")
    return "".join(s)


# ---------- figure 7: universal patch ----------
def fig_universal():
    W, H = 680, 200
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Uniwersalna łatka podmienia tylko SwitchGLU">']
    s.append('<defs><marker id="ar3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
             '<path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>')
    boxes = [(14, "router modelu", "box"), (184, "SwitchGLU", "box swap"), (354, "ważenie i suma", "box"), (524, "eksperci współdzieleni", "box")]
    for x, lab, cls in boxes:
        s.append(f'<rect x="{x}" y="60" width="142" height="54" rx="7" class="{cls}"/>'
                 f'<text x="{x + 71}" y="92" class="boxt sm" text-anchor="middle">{lab}</text>')
    for x in (156, 326, 496):
        s.append(f'<line x1="{x}" y1="87" x2="{x + 26}" y2="87" class="edge" marker-end="url(#ar3)"/>')
    s.append('<text x="255" y="44" class="val" text-anchor="middle">tylko to podmieniamy</text>'
             '<text x="85" y="140" class="tick" text-anchor="middle">bez zmian</text>'
             '<text x="425" y="140" class="tick" text-anchor="middle">bez zmian</text>'
             '<text x="595" y="140" class="tick" text-anchor="middle">bez zmian</text>')
    s.append('<text x="14" y="180" class="cap">Każda rodzina ma inny router (softmax, sigmoid, bias, korekty). Wspólne jest tylko SwitchGLU, więc jedna podmiana działa wszędzie.</text>')
    s.append("</svg>")
    return "".join(s)


html = open(f"{sys.argv[3]}").read()
for k, f in {"FIG_WALL": fig_wall, "FIG_MOE": fig_moe, "FIG_READS": fig_reads, "FIG_OVERLAP": fig_overlap, "FIG_FLOOR": fig_floor,
             "FIG_LOAD": fig_load, "FIG_UP": fig_upstream, "FIG_UNI": fig_universal}.items():
    html = html.replace("{{" + k + "}}", f())
assert "{{" not in html
open(OUT, "w").write(html)
print("ok", len(html))
