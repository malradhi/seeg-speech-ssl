"""Figures 2-6 of the JNE manuscript from the participant-level results.

Usage: python make_figures.py <directory holding the phase bundles> <output directory>
Error bars are 95% bootstrap confidence intervals of the participant mean (10,000 resamples),
or of the median where stated.
"""
from __future__ import annotations

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figdata import FR, LOW, S3, S5, ci95, ewide, iwide, label_ratio, load, wide  # noqa: E402

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# ---- style: validated categorical slots (light), neutral ink for references -------------
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
CM = 1 / 2.54
W_FULL = 15.5 * CM
mpl.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "lines.markersize": 4.2, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
})
BUDGET_LABEL = "Labelled training data (%)"
XT = [0.05, 0.1, 0.25, 0.5, 1.0]
XTL = ["5", "10", "25", "50", "100"]

D = load(ROOT)
print("participants:", len(D.pids))


def budget_axis(ax, fracs=XT):
    ax.set_xscale("log")
    ax.set_xticks(fracs)
    ax.set_xticklabels([XTL[XT.index(f)] for f in fracs])
    ax.minorticks_off()
    ax.set_xlim(min(fracs) / 1.25, max(fracs) * 1.25)
    ax.set_xlabel(BUDGET_LABEL)
    ax.grid(axis="x", visible=False)


def curve(ax, W, fracs, color, ls="-", marker="o", mfc=None, label=None, shift=1.0, lw=1.4, err=True):
    m = [W[f].mean() for f in fracs]
    lo, hi = zip(*[ci95(W[f].dropna().values) for f in fracs])
    x = np.array(fracs) * shift
    yerr = [np.array(m) - lo, np.array(hi) - m] if err else None
    ax.errorbar(x, m, yerr=yerr, color=color, ls=ls, lw=lw, marker=marker,
                mfc=mfc if mfc is not None else color, mec=color, mew=0.9, elinewidth=0.7, capsize=0, label=label,
                zorder=3)
    return m


def panel_label(ax, s, x=-0.16):
    ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


# =========================== Figure 2: learning curves ===================================
SERIES2 = [("sup7", BLUE, "--", "o", "white", "Transformer, supervised", 0.94),
           ("placebo7", BLUE, ":", "^", "white", "Transformer, placebo pretraining", 0.97),
           ("ssl7", BLUE, "-", "o", None, "Transformer, own-patient pretraining", 1.0),
           ("gru7", ORANGE, "--", "s", "white", "GRU, supervised", 1.03),
           ("gru_ssl7", ORANGE, "-", "s", None, "GRU, own-patient pretraining", 1.06),
           ("ridge", INK2, "-", "D", "white", "Ridge regression", 1.09)]
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 6.4 * CM))
for ax, metric, title in zip(axes, ("r_mean", "r_speech"), ("Overall r (all test frames)", "Within-speech r")):
    for c, col, ls, mk, mfc, lab, sh in SERIES2:
        curve(ax, wide(D, metric, c), XT, col, ls, mk, mfc, lab, 1.0, lw=1.1 if c == "placebo7" else 1.3, err=False)
    budget_axis(ax)
    ax.set_title(title, loc="left")
    ax.set_ylabel("Pearson r")
axes[0].set_ylim(0.49, 0.68)
axes[1].set_ylim(0.105, 0.195)
panel_label(axes[0], "(a)")
panel_label(axes[1], "(b)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.17), handlelength=2.6, columnspacing=2.0)
fig.tight_layout(w_pad=2.5)
save(fig, "fig2_learning_curves")

# =========================== Figure 3: per-participant gains, label efficiency ===========
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 6.0 * CM), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
rng = np.random.default_rng(3)
for i, f in enumerate(XT[::-1]):
    for j, (a, b, col, mk) in enumerate((("ssl7", "sup7", BLUE, "o"), ("gru_ssl7", "gru7", ORANGE, "s"))):
        d = (wide(D, "r_mean", a)[f] - wide(D, "r_mean", b)[f]).dropna().values
        x0 = i + (-0.18 if j == 0 else 0.18)
        ax.scatter(x0 + rng.uniform(-0.07, 0.07, len(d)), d, s=7, color=col, alpha=0.45, lw=0, zorder=2)
        lo, hi = ci95(d)
        ax.errorbar(x0 + 0.15 * (1 if j else -1), d.mean(), yerr=[[d.mean() - lo], [hi - d.mean()]], color=col,
                    marker=mk, ms=5, mec=INK, mew=0.6, elinewidth=1.6, capsize=0, zorder=4)
ax.axhline(0, color=INK2, lw=0.7, zorder=1)
ax.set_xticks(range(5))
ax.set_xticklabels(XTL[::-1])
ax.set_xlabel(BUDGET_LABEL)
ax.set_ylabel("Gain in overall r\n(pretrained − supervised)")
ax.grid(axis="x", visible=False)
ax.set_title("Gain from own-patient pretraining, per participant", loc="left")
ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="", label="Transformer"),
                   Line2D([], [], color=ORANGE, marker="s", ls="", label="GRU")], loc="upper left", ncol=2,
          handletextpad=0.3, columnspacing=1.0)
ax.set_ylim(-0.065, 0.105)
panel_label(ax, "(a)", -0.12)

ax = axes[1]
spec3 = [("sup7", "ssl7", BLUE, "o", None, "Transformer, whole session", -0.27),
         ("sup7", "ssl_self", BLUE, "o", "white", "Transformer, labelled windows only", -0.09),
         ("gru7", "gru_ssl7", ORANGE, "s", None, "GRU, whole session", 0.09),
         ("gru7", "gru_ssl_self", ORANGE, "s", "white", "GRU, labelled windows only", 0.27)]
for base, own, col, mk, mfc, lab, dx in spec3:
    for i, f in enumerate(LOW):
        r = label_ratio(D, base, own, f)
        med = np.median(r)
        lo, hi = ci95(r, fn=np.median)
        ax.errorbar(i + dx, med, yerr=[[med - lo], [hi - med]], color=col, marker=mk, mfc=mfc or col, mec=col,
                    mew=0.9, ms=4.5, elinewidth=1.0, capsize=0, label=lab if i == 0 else None, zorder=3)
ax.axhline(1, color=INK2, lw=0.7)
ax.set_xticks(range(3))
ax.set_xticklabels(["25", "10", "5"])
ax.set_xlabel(BUDGET_LABEL)
ax.set_ylabel("Label-equivalent factor\n(median, × budget)")
ax.grid(axis="x", visible=False)
ax.set_ylim(0.9, 2.05)
ax.set_title("Labels saved", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, handletextpad=0.3, columnspacing=0.8,
          fontsize=6.5)
panel_label(ax, "(b)", -0.2)
fig.tight_layout(w_pad=2.0)
save(fig, "fig3_gains_label_efficiency")

# =========================== Figure 4: data amount, other patients, BrainBERT ============
fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 7.4 * CM), gridspec_kw={"width_ratios": [1.2, 1, 1.1]})
ax = axes[0]
pos = np.arange(5)
for arch, base, full, pre, col, mk in (("Transformer", "sup7", "ssl7", "ssl", BLUE, "o"),
                                      ("GRU", "gru7", "gru_ssl7", "gru_ssl", ORANGE, "s")):
    for f, ls, mfc in ((0.1, "-", None), (0.05, "--", "white")):
        b = wide(D, "r_mean", base, S3)[f]
        conds = [f"{pre}_self"] + [f"{pre}_u{u}_7" for u in ("125", "250", "500")] + [full]
        ds = [(wide(D, "r_mean", c, S3)[f] - b).dropna().values for c in conds]
        m = [d.mean() for d in ds]
        lo, hi = zip(*[ci95(d) for d in ds])
        off = (-0.06 if arch == "Transformer" else 0.06) + (-0.02 if f == 0.1 else 0.02)
        ax.errorbar(pos + off, m, yerr=[np.array(m) - lo, np.array(hi) - m], color=col, ls=ls, marker=mk,
                    mfc=mfc or col, mec=col, mew=0.9, elinewidth=0.6, capsize=0, lw=1.2,
                    label=f"{arch}, {int(f * 100)}% labels")
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xticks(pos)
ax.set_xticklabels(["labelled\nwindows", "12.5", "25", "50", "100"])
ax.set_xlabel("Unlabelled recording used for pretraining (%)")
ax.set_ylabel("Gain in overall r")
ax.grid(axis="x", visible=False)
ax.set_title("Amount of own recording", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, handlelength=2.2, fontsize=6.3,
          columnspacing=0.6)
ax.set_ylim(-0.03, 0.05)
panel_label(ax, "(a)", -0.25)

ax = axes[1]
ramp = {0.25: "#86b6ef", 0.1: "#3987e5", 0.05: "#184f95"}
xs = np.array([0, 1, 2, 3])
for f in LOW:
    b = wide(D, "r_mean", "sup_g4", S3)[f]
    ds = [(wide(D, "r_mean", c, S3)[f] - b).dropna().values for c in ("ssl_g4", "scale5_g4", "scale15_g4", "pooled_g4")]
    m = [d.mean() for d in ds]
    lo, hi = zip(*[ci95(d) for d in ds])
    sh = {0.25: -0.08, 0.1: 0.0, 0.05: 0.08}[f]
    ax.errorbar(xs + sh, m, yerr=[np.array(m) - lo, np.array(hi) - m], color=ramp[f], marker="o", lw=1.2,
                elinewidth=0.6, capsize=0, label=f"{int(f * 100)}% labels", zorder=3)
    d = (wide(D, "r_mean", "loo_adapt_g4", S3)[f] - b).dropna().values
    lo, hi = ci95(d)
    ax.errorbar(4.5 + sh, d.mean(), yerr=[[d.mean() - lo], [hi - d.mean()]], color=ramp[f], marker="D", mfc="white",
                mew=0.9, elinewidth=0.6, capsize=0, ls="", zorder=3)
ax.axhline(0, color=INK2, lw=0.7)
ax.axvline(3.75, color=GRID, lw=0.8)
ax.set_xticks([0, 1, 2, 3, 4.5])
ax.set_xticklabels(["1\n(own)", "5", "15", "30", "29 others\n+ adapt."])
ax.set_xlim(-0.4, 5.1)
ax.set_xlabel("Patients in pretraining")
ax.grid(axis="x", visible=False)
ax.set_title("Other patients' data", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=1, fontsize=6.3, handlelength=1.6)
ax.set_ylabel("Gain in overall r")
ax.set_ylim(-0.005, 0.052)
panel_label(ax, "(b)", -0.18)

ax = axes[2]
for c, col, ls, mk, mfc, lab, sh in (("gru_ssl7", ORANGE, "-", "s", None, "Own data: GRU, pretrained", 1.0),
                                     ("probe_ssl", BLUE, "--", "o", "white", "Own data: ridge on frozen encoder", 1.04),
                                     ("bb_best_gru", AQUA, "-", "^", None, "BrainBERT: GRU readout", 0.96),
                                     ("bb_best_ridge", AQUA, "--", "^", "white", "BrainBERT: ridge readout", 1.08)):
    curve(ax, wide(D, "r_mean", c), XT, col, ls, mk, mfc, lab, sh, lw=1.2)
budget_axis(ax)
ax.set_ylabel("Overall r")
ax.set_title("BrainBERT", loc="left")
ax.set_ylim(0.2, 0.7)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.42), ncol=1, fontsize=6.3, handlelength=2.4)
panel_label(ax, "(c)", -0.22)
fig.tight_layout(w_pad=1.2)
save(fig, "fig4_data_patients_brainbert")

# =========================== Figure 5: what is decoded ==================================
fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 6.6 * CM), gridspec_kw={"width_ratios": [1.0, 1.0, 0.85]})
ax = axes[0]
ev = D.ev
tr = ev[(ev.condition == "timing_reference") & (ev.source == "eval")].groupby("participant").r_mean.mean()
best = wide(D, "r_mean", "gru_ssl7")[1.0]
tr = tr.reindex(best.index)
ax.plot([0.3, 1.0], [0.3, 1.0], color=MUTED, lw=0.7, ls="--", zorder=1)
ax.scatter(best, tr, s=12, color=INK, lw=0, alpha=0.8, zorder=3)
ax.set_xlim(0.35, 0.95)
ax.set_ylim(0.35, 0.95)
ax.set_xlabel("Best decoder\noverall r")
ax.set_ylabel("Timing reference\noverall r")
ax.text(0.93, 0.38, f"{int((tr > best).sum())} of {len(best)} participants\nabove the diagonal", va="bottom",
        ha="right", fontsize=6.5, color=INK2)
ax.set_title("Timing alone", loc="left")
panel_label(ax, "(a)", -0.3)

ax = axes[1]
for metric, col, lab in (("r_loud", VIOLET, "Loudness"), ("r_shape", AQUA, "Spectral shape")):
    W = ewide(D, metric, "lpft_gru_ssl", "eval4b")
    C = ewide(D, metric + "_chance_p95", "lpft_gru_ssl", "eval4b")
    curve(ax, W, XT, col, "-", "o", None, lab, 1.0 if metric == "r_loud" else 1.04, lw=1.3)
    ax.plot(XT, [C[f].mean() for f in XT], color=col, ls=":", lw=1.1, label=f"{lab}, chance (95th pct.)")
budget_axis(ax)
ax.set_ylabel("Within-speech r")
ax.set_title("Loudness vs shape", loc="left")
ax.set_ylim(0.0, 0.31)
h, l = ax.get_legend_handles_labels()
order = [l.index(x) for x in ("Loudness", "Loudness, chance (95th pct.)", "Spectral shape",
                              "Spectral shape, chance (95th pct.)")]
ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1,
          fontsize=6.3, handlelength=2.0)
panel_label(ax, "(b)", -0.25)

ax = axes[2]
rows = [("True spectrogram", "true_spectrogram", "eval", None),
        ("Pretrained GRU, LP-FT", "lpft_gru_ssl", "eval4b", 1.0),
        ("Ridge regression", "ridge", "eval", 1.0),
        ("Pretrained GRU", "gru_ssl", "eval4b", 1.0),
        ("Pretrained Transformer", "ssl", "eval", 1.0),
        ("Timing reference", "timing_reference", "eval", None),
        ("Training mean", "train_mean", "eval", None),
        ("Shifted truth", "shifted_truth", "eval", None)]
for i, (lab, c, src, f) in enumerate(rows):
    W = ewide(D, "estoi", c, src, [1])
    v = W[1.0 if f else -1.0].dropna().values
    lo, hi = ci95(v)
    ref = f is None
    ax.errorbar(v.mean(), i, xerr=[[v.mean() - lo], [hi - v.mean()]], color=MUTED if ref else ORANGE, marker="o",
                mfc="white" if ref else ORANGE, mew=0.9, ms=4, elinewidth=0.9, capsize=0, zorder=3)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
ax.invert_yaxis()
ax.set_xlim(-0.05, 0.8)
ax.axvline(0, color=INK2, lw=0.6)
ax.set_xlabel("ESTOI (100% labels)")
ax.grid(axis="y", visible=False)
ax.set_title("Intelligibility", loc="left")
panel_label(ax, "(c)", -0.85)
fig.tight_layout(w_pad=1.0)
save(fig, "fig5_what_is_decoded")

# =========================== Figure 6: identification ===================================
fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 5.9 * CM), gridspec_kw={"width_ratios": [1.25, 1, 1]})
ax = axes[0]
for c, col, ls, mk, mfc, lab, sh in (("gru", ORANGE, "--", "s", "white", "GRU, supervised", 0.94),
                                     ("gru_ssl", ORANGE, "-", "s", None, "GRU, pretrained", 0.98),
                                     ("lpft_gru_ssl", VIOLET, "-", "o", None, "GRU, pretrained, LP-FT", 1.02),
                                     ("ridge", INK2, "-", "D", "white", "Ridge regression", 1.06),
                                     ("bb_best_gru", AQUA, "-", "^", None, "BrainBERT, GRU readout", 1.10)):
    curve(ax, iwide(D, "full_pairwise", c), XT, col, ls, mk, mfc, lab, 1.0, lw=1.2, err=False)
ax.axhline(0.5, color=INK, lw=0.8, ls=":")
ax.text(1.15, 0.502, "chance", fontsize=6.5, color=INK2, va="bottom", ha="right")
budget_axis(ax)
ax.set_ylabel("Pairwise identification accuracy")
ax.set_title("Which segment was spoken?", loc="left")
ax.set_ylim(0.49, 0.64)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=6.3, handlelength=2.2,
          columnspacing=0.8)
panel_label(ax, "(a)", -0.2)

ax = axes[1]
for task, ls, mk, lab in (("sentences", "-", "o", "Sentences"), ("unique_words", "--", "s", "Unique words"),
                          ("repeated_words", ":", "^", "Repeated words")):
    ps = list(D.task_of[D.task_of == task].index)
    W = iwide(D, "full_pairwise", "gru_ssl", ps)
    m = curve(ax, W, XT, INK2, ls, mk, "white" if task != "sentences" else INK2, lab, 1.0, lw=1.2)
    ax.text(1.32, m[-1], lab, fontsize=6.4, va="center", color=INK2)
ax.axhline(0.5, color=INK, lw=0.8, ls=":")
budget_axis(ax)
ax.set_xlim(0.04, 1.25)
ax.set_title("By task (GRU, pretrained)", loc="left")
ax.set_ylim(0.46, 0.70)
panel_label(ax, "(b)", -0.24)

ax = axes[2]
for kind, ls, mk, lab in (("full_pairwise", "-", "o", "Full spectrogram"), ("shape_pairwise", "--", "s", "Shape only"),
                          ("loudness_pairwise", ":", "^", "Loudness only")):
    W = iwide(D, kind, "lpft_gru_ssl")
    m = curve(ax, W, XT, VIOLET, ls, mk, "white" if kind != "full_pairwise" else VIOLET, lab, 1.0, lw=1.2)
ax.axhline(0.5, color=INK, lw=0.8, ls=":")
budget_axis(ax)
ax.set_title("What carries it (LP-FT)", loc="left")
ax.set_ylim(0.49, 0.64)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1, fontsize=6.3, handlelength=2.4)
panel_label(ax, "(c)", -0.24)
fig.tight_layout(w_pad=1.0)
save(fig, "fig6_identification")
print("done")
