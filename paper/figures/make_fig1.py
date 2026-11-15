"""Figure 1: method and study design, drawn as a vector figure.

Illustrative data only (smooth random fields stand in for recordings).
Usage: python make_fig1.py <output directory>
"""
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from scipy.ndimage import gaussian_filter

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8984", "#b9b8b2"
PANEL = "#f6f5f1"
MASK = "#2a2f8f"
CM = 1 / 2.54
W, H = 155.0, 149.0  # mm
mpl.rcParams.update({"font.family": "Liberation Sans", "font.size": 6.2, "pdf.fonttype": 42, "text.color": INK,
                     "mathtext.fontset": "dejavusans"})
fig = plt.figure(figsize=(W / 10 * CM, H / 10 * CM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
rng = np.random.default_rng(7)


# ------------------------------------------------------------------ helpers
def T(x, y, s, size=6.2, weight="normal", color=INK, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, **kw)


def arrow(x0, y0, x1, y1, color=INK, lw=1.0, ms=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0))


def rbox(x, y, w, h, fc="white", ec=LINE, lw=0.8, r=2.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))


def inset(x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.5)
        s.set_color(INK2)
    return a


def field(n_rows, n_cols, sy=1.2, sx=2.5, seed=0):
    r = np.random.default_rng(seed)
    f = gaussian_filter(r.normal(size=(n_rows, n_cols)), (sy, sx))
    return (f - f.min()) / (f.max() - f.min())


def speech_mask(n, seed=3):
    m = np.zeros(n)
    r = np.random.default_rng(seed)
    t = 4
    while t < n - 6:
        L = r.integers(7, 12)
        m[t:t + L] = 1
        t += L + r.integers(5, 9)
    return m


def network(cx, cy, color, scale=1.0, dashed=False, lock=False):
    layers = [3, 4, 3]
    xs = [cx - 4.2 * scale, cx, cx + 4.2 * scale]
    pts = []
    for lx, n in zip(xs, layers):
        ys = np.linspace(cy - 3.3 * scale * (n - 1) / 3, cy + 3.3 * scale * (n - 1) / 3, n)
        pts.append([(lx, yy) for yy in ys])
    for a_, b_ in zip(pts[:-1], pts[1:]):
        for p in a_:
            for q in b_:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=0.45, alpha=0.75, ls="--" if dashed else "-")
    for layer in pts:
        for p in layer:
            ax.add_patch(Ellipse(p, 1.9 * scale, 1.9 * scale, fc=color, ec="white", lw=0.4, zorder=5))
    if lock:
        ax.add_patch(Rectangle((cx + 4.6 * scale, cy - 4.6 * scale), 2.4, 2.0, fc=INK2, ec=INK2, zorder=6))
        ax.add_patch(Ellipse((cx + 5.8 * scale, cy - 2.6 * scale), 1.6, 2.0, fc="none", ec=INK2, lw=0.7, zorder=6))


def heat(a, data, cmap="viridis"):
    a.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest", origin="lower")


def mask_grid(x, y, w, h, nr, nc, masked):
    cw, ch = w / nc, h / nr
    for i in range(nr):
        for j in range(nc):
            fc = MASK if masked[i, j] else "white"
            ax.add_patch(Rectangle((x + j * cw, y + i * ch), cw, ch, fc=fc, ec=LINE, lw=0.35))


def label(x, y, s, title):
    T(x, y, s, size=9, weight="bold", ha="left")
    T(x + 6, y, title, size=7.6, weight="bold", ha="left")


# ------------------------------------------------------------------ data stand-ins
NT = 64
hg = field(18, NT, 1.0, 1.8, seed=1)
sp = speech_mask(NT)
hg = np.clip(hg * 0.8 + 0.35 * sp[None, :] * field(18, NT, 2, 1, seed=4), 0, 1)
X = np.concatenate([np.roll(hg, k, axis=1) for k in range(-4, 5)], axis=0)  # 9 lagged copies
mel = field(23, NT, 1.5, 1.2, seed=5) * 0.45
for b in range(23):
    mel[b] += sp * (0.75 - 0.025 * b) * (0.7 + 0.3 * np.sin(np.arange(NT) / 2.0 + b / 3.0))
mel = gaussian_filter(mel, (0.8, 0.6))
pred = gaussian_filter(mel, (1.6, 1.4)) * 0.85 + 0.12 * field(23, NT, 1.5, 2, seed=9)

# ================================================================== (a)
label(2, 145.6, "(a)", "From sEEG to model inputs and targets")
bx, by = 16, 123
ax.add_patch(Ellipse((bx, by), 22, 25, fc="#efeee9", ec=INK2, lw=0.8))
ax.plot([bx, bx], [by - 12.5, by + 12.5], color=LINE, lw=0.6)
for k, (y0, ang) in enumerate([(by + 6.5, 8), (by + 0.5, -4), (by - 6, 6)]):
    x0, x1 = bx - 13.0, bx - 1.5
    yy0, yy1 = y0, y0 - (x1 - x0) * np.tan(np.radians(ang))
    ax.plot([x0, x1], [yy0, yy1], color=INK, lw=0.9)
    for t in np.linspace(0.3, 1.0, 5):
        ax.add_patch(Ellipse((x0 + t * (x1 - x0), yy0 + t * (yy1 - yy0)), 1.15, 1.15, fc=INK, ec="none", zorder=4))
for k, (y0, ang) in enumerate([(by + 3.5, -6), (by - 3.5, 5)]):
    x0, x1 = bx + 13.0, bx + 2.0
    yy0, yy1 = y0, y0 + (x0 - x1) * np.tan(np.radians(ang))
    ax.plot([x0, x1], [yy0, yy1], color=INK, lw=0.9)
    for t in np.linspace(0.3, 1.0, 5):
        ax.add_patch(Ellipse((x0 + t * (x1 - x0), yy0 + t * (yy1 - yy0)), 1.15, 1.15, fc=INK, ec="none", zorder=4))
T(bx, 140.0, "sEEG depth electrodes", weight="bold")
T(bx, 105.3, "30 participants\n54–127 contacts each\n(3,248 in total)", size=5.8, color=INK2, linespacing=1.15)

arrow(29, 124, 32.5, 124)
a = inset(34, 113, 18, 21)
heat(a, hg)
T(43, 140.0, "High-gamma", weight="bold")
T(43, 136.8, "70–170 Hz, 10-ms frames", size=5.6, color=INK2)
T(32.9, 123.5, "contacts", size=5.2, color=INK2, rotation=90)
T(43, 111.0, "time", size=5.2, color=INK2)

arrow(54, 124, 57.5, 124)
cols = ["#d9d3f5", "#c7bff0", "#b5aceb", "#a399e6", "#9186e0", "#7f73db", "#6d60d6", "#5b4dd1", "#493ac9"]
for i, c in enumerate(cols):
    y0 = 113.0 + i * 2.25
    ax.add_patch(Polygon([(61, y0), (71, y0), (74.2, y0 + 4.0), (64.2, y0 + 4.0)], closed=True, fc=c, ec="white", lw=0.5))
T(67.6, 140.0, "9 lagged copies", weight="bold")
T(67.6, 136.8, "±200 ms in 50-ms steps", size=5.6, color=INK2)
T(60.0, 114.5, "−200", size=5.0, color=INK2, ha="right")
T(60.0, 132.5, "+200", size=5.0, color=INK2, ha="right")

T(77.5, 124, "=", size=11)
a = inset(81, 111, 13, 23.5)
heat(a, X)
T(87.5, 140.0, "Input", weight="bold")
T(87.5, 136.8, r"$X \in \mathbb{R}^{T \times 9C}$", size=6.2)
T(95.6, 123.5, "9C features", size=5.2, color=INK2, rotation=90)

T(129.5, 140.0, "Target from the synchronised audio", weight="bold")
wave = gaussian_filter(rng.normal(size=400), 1.2) * np.repeat(0.15 + sp, 400 // NT + 1)[:400]
ax.plot(np.linspace(107, 152, 400), 134.5 + 2.0 * wave / np.abs(wave).max(), color=INK2, lw=0.5)
arrow(129.5, 132.2, 129.5, 130.4, lw=0.8, ms=6)
a = inset(107, 115.5, 45, 14.5)
heat(a, mel, cmap="magma")
T(105.4, 122.8, "23 Mel bands", size=5.2, color=INK2, rotation=90)
a = inset(107, 112.8, 45, 2.0)
a.imshow(sp[None, :], aspect="auto", cmap=mpl.colors.ListedColormap(["#e9e8e2", ORANGE]), interpolation="nearest")
T(129.5, 110.4, "log-Mel spectrogram Y; speech / silence labels", size=5.6, color=INK2)

py = 101.5
T(34, py + 2.0, "Session split in time", size=6.0, weight="bold", ha="left")
x0, tw = 69, 83
for a_, b_, s_, fc in ((0, 0.70, "training 70%", "#e4e2db"), (0.70, 0.85, "validation", "#f1f0eb"),
                       (0.85, 1.0, "test", "#d4d2ca")):
    ax.add_patch(Rectangle((x0 + a_ * tw, py), (b_ - a_) * tw, 4.0, fc=fc, ec=INK2, lw=0.5))
    T(x0 + (a_ + b_) / 2 * tw, py + 2.0, s_, size=5.6)
for i, f in enumerate((1.0, 0.5, 0.25, 0.10, 0.05)):
    yy = py - 1.3 - i * 1.2
    ax.add_patch(Rectangle((x0, yy - 0.42), 0.70 * tw * f, 0.84, fc=VIOLET, alpha=0.30 + 0.14 * i, lw=0))
T(34, py - 3.3, "nested label budgets, 100 → 5%", size=5.6, color=INK2, ha="left")
T(76.5, py - 6.1, "5% = 10.5 s (word tasks), 21 s (sentences)", size=5.4, color=INK2, ha="left")
ax.plot([4, 151], [92.6, 92.6], color=LINE, lw=0.6)

# ================================================================== (b)
label(2, 88.5, "(b)", "Masked pretraining on unlabelled sEEG (no audio)")
a = inset(5, 55, 13, 24)
heat(a, X)
r2 = np.random.default_rng(11)
for i in range(0, 162, 9):
    for j in range(0, 64, 4):
        if r2.random() < 0.5:
            a.add_patch(Rectangle((j - 0.5, i - 0.5), 4, 9, fc="white", ec="none", alpha=0.95))
T(11.5, 83.2, "Unlabelled input", weight="bold")
T(11.5, 52.4, "50% of values hidden", size=5.4, color=INK2)
arrow(19.3, 67, 22.8, 67)

rbox(23.5, 50.2, 52, 35.3, fc=PANEL)
T(49.5, 82.5, "Masks", weight="bold")
gx = [27, 44, 61]
names = [("Random values", "default"), ("Time blocks", "ablation"), ("Whole contacts", "ablation")]
m1 = rng.random((9, 8)) < 0.5
m2 = np.zeros((9, 8), bool)
m2[:, 3] = True
m2[:, 4] = True
m3 = np.zeros((9, 8), bool)
m3[3:6, :] = True
for (nm, tag), g, mm in zip(names, gx, (m1, m2, m3)):
    mask_grid(g, 58.0, 12.5, 14.5, 9, 8, mm)
    T(g + 6.25, 78.2, nm, size=5.8)
    T(g + 6.25, 75.6, tag, size=5.3, color=INK2)
ax.add_patch(Rectangle((27, 52.6), 2.0, 2.0, fc="white", ec=LINE, lw=0.4))
T(29.9, 53.6, "kept", size=5.4, ha="left")
ax.add_patch(Rectangle((37, 52.6), 2.0, 2.0, fc=MASK, ec=LINE, lw=0.4))
T(39.9, 53.6, "hidden", size=5.4, ha="left")
T(51.5, 53.6, "ratios 25–75% in ablation", size=5.2, color=INK2, ha="left")

arrow(76.2, 67, 79.6, 67)
rbox(80, 54.5, 23.5, 26, fc="white", ec=INK2)
T(91.75, 77.6, "Encoder", weight="bold")
ax.add_patch(Rectangle((81.8, 71.2), 1.9, 1.9, fc=BLUE, lw=0))
T(84.6, 72.15, "Transformer", size=5.8, ha="left")
T(84.6, 69.5, "4 layers, d = 128", size=5.2, color=INK2, ha="left")
ax.add_patch(Rectangle((81.8, 63.6), 1.9, 1.9, fc=ORANGE, lw=0))
T(84.6, 64.55, "or GRU", size=5.8, ha="left")
T(84.6, 61.9, "2 × 128, bidirectional", size=5.2, color=INK2, ha="left")
T(91.75, 57.0, "learned mask token", size=5.2, color=INK2)
arrow(104, 67, 106.6, 67)
rbox(107, 60, 10.5, 14, fc="white", ec=INK2)
T(112.25, 69.6, "Recon-", size=5.6)
T(112.25, 67.0, "struction", size=5.6)
T(112.25, 64.4, "head", size=5.6)
arrow(118, 67, 120.6, 67)
a = inset(120.5, 55, 12, 24)
heat(a, X)
T(126.5, 83.2, "Restored", weight="bold")
T(126.5, 51.6, "loss on hidden\nvalues only", size=5.3, color=INK2, linespacing=1.1)

rbox(134.6, 50.2, 19.6, 35.3, fc=PANEL)
T(144.4, 82.5, "Pretraining data", weight="bold", size=5.9)
rows_b = [("own session", VIOLET), ("12.5–50% of it", VIOLET), ("labelled windows", VIOLET),
          ("placebo (shuffled)", MUTED), ("other patients", AQUA)]
for i, (s_, c) in enumerate(rows_b):
    yy = 77.0 - i * 5.5
    ax.add_patch(Ellipse((136.4, yy), 1.7, 1.7, fc=c, ec="none"))
    T(137.9, yy, s_, size=5.3, ha="left")
ax.plot([4, 151], [47.0, 47.0], color=LINE, lw=0.6)

# ================================================================== (c)
label(2, 43.0, "(c)", "Fine-tuning on few labels")
init = [("Own-patient pretrained", VIOLET, False), ("Placebo-pretrained", MUTED, True),
        ("Other patients' data", AQUA, False), ("Random initialisation", "#c9c8c2", False),
        ("BrainBERT (frozen)", "#0f8a5f", False)]
rbox(2.5, 2.5, 41.5, 36.5, fc=PANEL)
T(23.25, 36.0, "Starting point", weight="bold")
for i, (s_, c, d) in enumerate(init):
    yy = 30.6 - i * 6.3
    network(10.0, yy, c, scale=0.52, dashed=d)
    T(16.8, yy, s_, size=5.8, ha="left")

arrow(44.8, 24, 48.6, 24)
dx0, dw = 49.2, 45.5
rbox(dx0, 13.2, dw, 24.3, fc="white", ec=INK2)
T(dx0 + dw / 2, 34.6, "Decoder", weight="bold")
T(dx0 + dw / 2, 31.4, "plain fine-tuning or LP-FT", size=5.4, color=INK2)
rbox(dx0 + 2.5, 22.6, dw - 5, 6.0, fc="#eef4fc", ec=BLUE, lw=0.6, r=1.2)
T(dx0 + dw / 2, 25.6, "encoder: Transformer or GRU", size=5.6)
rbox(dx0 + 2.5, 15.2, dw - 5, 6.0, fc="#f3f2ee", ec=INK2, lw=0.6, r=1.2)
T(dx0 + dw / 2, 18.2, "Mel head (23 bands)", size=5.6)
# labelled windows feeding the decoder
cx = dx0 + dw / 2
widths = [9.0, 6.4, 4.5, 2.8, 2.0]
xs0 = cx - (sum(widths) + 4 * 1.2) / 2
xx = xs0
for i, wdt in enumerate(widths):
    ax.add_patch(Rectangle((xx, 6.6), wdt, 1.6, fc=VIOLET, alpha=0.30 + 0.14 * i, lw=0))
    xx += wdt + 1.2
arrow(cx, 8.9, cx, 12.9, lw=0.9, ms=6)
T(cx, 4.2, "labelled windows (with audio): 100 → 5% of training", size=5.4, color=INK2)

arrow(95.4, 24, 99.0, 24)
a = inset(99.5, 15.0, 23, 17)
heat(a, pred, cmap="magma")
T(111.0, 35.6, r"Predicted $\hat{Y}$", weight="bold")
ax.add_patch(FancyArrowPatch((123.6, 23.5), (129.4, 23.5), arrowstyle="<|-|>", mutation_scale=6, lw=0.8, color=INK2))
T(126.5, 26.6, "MSE", size=5.8, weight="bold", color=INK2)
a = inset(130.5, 15.0, 23, 17)
heat(a, mel, cmap="magma")
T(142.0, 35.6, r"Target $Y$", weight="bold")
T(126.5, 11.0, "loss on the labelled windows; early stopping\nand learning rate chosen on validation",
  size=5.4, color=INK2, linespacing=1.15)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig1_study_design.{ext}"), dpi=220)
print("ok")
