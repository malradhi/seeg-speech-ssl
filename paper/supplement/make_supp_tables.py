"""Generate the tables of the supplementary material from the saved statistical outputs.

Usage: python make_supp_tables.py <stats dir> <phase-bundle dir> <out .tex>
Every number is read from the analysis outputs (stats/*.json) or recomputed from the per-run
results exactly as in the pre-specified analyses (../figures/figdata.py).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "figures"))
from figdata import FR, S3, S5, ewide, iwide, load, wide  # noqa: E402

ST, ROOT, OUT = sys.argv[1:4]
J = {k: json.load(open(os.path.join(ST, f"phase{k}_stats.json"))) for k in ("1", "2", "3", "4", "4b", "5")}
J29 = {k: json.load(open(os.path.join(ST, f"phase{k}_stats_n29.json"))) for k in ("2", "3", "4", "4b", "5")}
D = load(ROOT)
BUD = {"1.0": "100", "0.5": "50", "0.25": "25", "0.1": "10", "0.05": "5"}
out = []
w = out.append


def sgn(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+.{nd}f}"
    return s.replace("-", "$-$")


def d100(x):
    """A difference in r (or in accuracy) in hundredths, one decimal, as in table 2 of the article."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    v = round(100 * x, 1)
    if v == 0:
        return "0.0"
    return f"{v:+.1f}".replace("-", "$-$")


def num(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if round(x, nd) == 0:
        x = 0.0
    return f"{x:.{nd}f}".replace("-", "$-$")


def pv(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p >= 0.995:
        return "1"
    if p >= 0.001:
        return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    e = int(math.floor(math.log10(p)))
    m = p / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"


def pvc(p):
    if p >= 0.995:
        return "1"
    if p >= 0.001:
        return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m}e$-${int(e[1:])}"


def esc(s):
    return s.replace("%", "\\%").replace("_", "\\_")


def res(phase, key, metric, n29=False):
    src = (J29 if n29 else J)[phase]
    if phase == "1":
        return src[metric][key]["by_fraction"]
    if phase == "5":
        return src["results"][key] if metric == "r_mean" else None
    return src["results"][metric][key]


def cell(r):
    if r is None:
        return "-- & -- & --"
    ci = r.get("ci95") or [float("nan")] * 2
    return f"{d100(r['delta'])} [{d100(ci[0])}, {d100(ci[1])}] & {r['n_pos']}/{r['n']} & {pv(r.get('p_holm', r.get('p')))}"


def test_block(label, phase, key, note=None, n29=False, metrics=("r_mean", "r_speech")):
    rm = res(phase, key, metrics[0], n29)
    rs = res(phase, key, metrics[1], n29) if metrics[1] else None
    w(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{label}}}{(' ' + note) if note else ''}}} \\\\*")
    for f in ("1.0", "0.5", "0.25", "0.1", "0.05"):
        if f not in rm:
            continue
        a = cell(rm[f])
        b = cell(rs.get(f)) if rs is not None else "-- & -- & --"
        w(f"{BUD[f]}\\% & {a} & {b} \\\\")


def longtable_tests(caption, label, blocks, n29=False):
    w("\\begin{small}")
    w("\\setlength{\\tabcolsep}{3.2pt}")
    w("\\begin{longtable}{@{}rlrrlrr@{}}")
    w(f"\\caption{{{caption}}}\\label{{{label}}}\\\\")
    hdr = ("\\toprule & \\multicolumn{3}{c}{Overall r} & \\multicolumn{3}{c}{Within-speech r} \\\\ "
           "\\cmidrule(lr){2-4}\\cmidrule(l){5-7} Labels & $\\Delta$ [95\\% CI] & $n_+$ & $p_{\\mathrm{Holm}}$ & "
           "$\\Delta$ [95\\% CI] & $n_+$ & $p_{\\mathrm{Holm}}$ \\\\ \\midrule")
    w(hdr + "\\endfirsthead")
    w("\\multicolumn{7}{@{}l}{\\textit{(continued)}} \\\\ " + hdr + "\\endhead")
    w("\\bottomrule\\endfoot")
    for b in blocks:
        test_block(*b, n29=n29)
    w("\\end{longtable}")
    w("\\end{small}")


# =============================================================================== S3 tables
w("% ---- Table: mean scores of every decoder")
rows = [("Transformer, supervised", "sup7"), ("Transformer, placebo pretraining", "placebo7"),
        ("Transformer, own-patient pretraining", "ssl7"), ("Transformer, self-pretraining", "ssl_self"),
        ("Transformer, pretraining, LP-FT", "lpft_ssl7"), ("GRU, supervised", "gru7"),
        ("GRU, placebo pretraining", "gru_placebo7"), ("GRU, own-patient pretraining", "gru_ssl7"),
        ("GRU, self-pretraining", "gru_ssl_self"), ("GRU, pretraining, LP-FT", "lpft_gru_ssl7"),
        ("Ridge regression", "ridge"), ("Ridge on frozen pretrained Transformer", "probe_ssl"),
        ("Ridge on frozen pretrained GRU", "probe_gru_ssl"), ("BrainBERT, ridge readout", "bb_best_ridge"),
        ("BrainBERT, GRU readout", "bb_best_gru")]
w("\\begin{table}[p]\\centering\\small\\setlength{\\tabcolsep}{3.3pt}")
w("\\caption{Mean overall r and within-speech r on the test partition of every per-patient decoder (30 participants, "
  "five seeds averaged within participant; seven learning rates chosen on validation for the deep decoders). "
  "Self-pretraining was run at 25\\%, 10\\% and 5\\% labels only. Speech-timing reference: overall r 0.837.}")
w("\\label{tab:s-means}")
w("\\begin{tabular}{@{}lrrrrrrrrrr@{}}\\toprule")
w(" & \\multicolumn{5}{c}{Overall r} & \\multicolumn{5}{c}{Within-speech r} \\\\ \\cmidrule(lr){2-6}\\cmidrule(l){7-11}")
w("Decoder & 100\\% & 50\\% & 25\\% & 10\\% & 5\\% & 100\\% & 50\\% & 25\\% & 10\\% & 5\\% \\\\ \\midrule")
for lab, c in rows:
    a = wide(D, "r_mean", c)
    b = wide(D, "r_speech", c)
    va = [num(a[f].mean()) if f in a and a[f].notna().any() else "--" for f in FR]
    vb = [num(b[f].mean()) if f in b and b[f].notna().any() else "--" for f in FR]
    w(f"{lab} & " + " & ".join(va + vb) + " \\\\")
w("\\midrule \\multicolumn{11}{@{}l}{\\textit{Cross-patient comparisons (four learning rates; seeds 1--3 unless stated)}} \\\\")
for lab, c, sd in (("Transformer, supervised", "sup_g4", S3), ("Own-patient pretraining", "ssl_g4", S3),
                   ("Pooled, 5 patients", "scale5_g4", S3), ("Pooled, 15 patients", "scale15_g4", S3),
                   ("Pooled, 30 patients (five seeds)", "pooled_g4", S5),
                   ("Leave-one-out, adapted", "loo_adapt_g4", S3), ("Leave-one-out, not adapted", "loo_noadapt_g4", S3)):
    a = wide(D, "r_mean", c, sd)
    b = wide(D, "r_speech", c, sd)
    va = [num(a[f].mean()) if f in a and a[f].notna().any() else "--" for f in FR]
    vb = [num(b[f].mean()) if f in b and b[f].notna().any() else "--" for f in FR]
    w(f"{lab} & " + " & ".join(va + vb) + " \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

# by task group
w("% ---- Table: by task group")
w("\\begin{table}[p]\\centering\\small\\setlength{\\tabcolsep}{4pt}")
w("\\caption{Mean overall r and within-speech r by task group (10 participants each; seven learning rates).}")
w("\\label{tab:s-tasks}")
w("\\begin{tabular}{@{}llrrrrrrrrrr@{}}\\toprule")
w(" & & \\multicolumn{5}{c}{Overall r} & \\multicolumn{5}{c}{Within-speech r} \\\\ \\cmidrule(lr){3-7}\\cmidrule(l){8-12}")
w("Task & Decoder & 100\\% & 50\\% & 25\\% & 10\\% & 5\\% & 100\\% & 50\\% & 25\\% & 10\\% & 5\\% \\\\ \\midrule")
for t, tl in (("unique_words", "Unique words"), ("repeated_words", "Repeated words"), ("sentences", "Sentences")):
    ps = list(D.task_of[D.task_of == t].index)
    for i, (lab, c) in enumerate((("Transformer, supervised", "sup7"), ("Transformer, pretrained", "ssl7"),
                                  ("GRU, supervised", "gru7"), ("GRU, pretrained", "gru_ssl7"),
                                  ("GRU, pretrained, LP-FT", "lpft_gru_ssl7"), ("Ridge regression", "ridge"))):
        a = wide(D, "r_mean", c, pids=ps)
        b = wide(D, "r_speech", c, pids=ps)
        w(f"{tl if i == 0 else ''} & {lab} & " + " & ".join([num(a[f].mean()) for f in FR] + [num(b[f].mean()) for f in FR])
          + " \\\\")
    w("\\midrule" if t != "sentences" else "")
w("\\bottomrule\\end{tabular}\\end{table}")

# =============================================================================== tests
w("% ---- Long table of tests: own-patient pretraining")
longtable_tests(
    "Own-patient pretraining: mean paired differences (A minus B) over 30 participants in hundredths of r "
    "($\\Delta r \\times 100$, as in table~2 of the article; +1.3 is a difference of +0.013 in correlation), with the "
    "bootstrap 95\\% confidence interval in the same units, the number of participants with a positive difference ($n_+$) "
    "and the Holm-corrected two-sided Wilcoxon $p$ within the family shown (budgets listed). Seven learning rates unless stated.",
    "tab:s-tests-own",
    [("Transformer: pretrained vs supervised (H1, H12)", "4b", "R2_ssl7_vs_sup7_all"),
     ("Transformer: pretrained vs placebo (H2)", "4b", "R2_ssl7_vs_placebo7_all"),
     ("Transformer: pretrained vs supervised, four learning rates (H1 as first run)", "1", "H1_ssl_vs_sup"),
     ("Transformer: pretrained vs placebo, four learning rates (H2 as first run)", "1", "H2_ssl_vs_placebo"),
     ("GRU: pretrained vs supervised (H10)", "4b", "R2_H10_gru_ssl_vs_gru_7"),
     ("GRU: pretrained vs supervised, all budgets", "4b", "R2_gru_ssl7_vs_gru7_all"),
     ("GRU: pretrained vs placebo (H10b)", "4b", "R2_H10b_gru_ssl_vs_gru_placebo_7"),
     ("GRU pretrained vs Transformer pretrained", "4b", "R2_gru_ssl7_vs_ssl7_all"),
     ("Transformer: self-pretrained vs supervised (H17)", "4b", "H17_ssl_self_vs_sup7"),
     ("GRU: self-pretrained vs supervised (H17)", "4b", "H17_gru_ssl_self_vs_gru7"),
     ("Transformer: whole-session vs self-pretraining", "4b", "ssl7_vs_ssl_self"),
     ("GRU: whole-session vs self-pretraining", "4b", "gru_ssl7_vs_gru_ssl_self"),
     ("Ridge on frozen pretrained vs randomly initialised Transformer (H9)", "2", "H9_probe_ssl_vs_random"),
     ("Ridge on frozen pretrained vs placebo Transformer (H9)", "2", "H9_probe_ssl_vs_placebo"),
     ("Ridge on frozen pretrained Transformer vs ridge on input features", "2", "probe_ssl_vs_ridge"),
     ("Ridge on frozen pretrained vs randomly initialised GRU (H13)", "3", "H13_probe_gru_ssl_vs_random"),
     ("Ridge on frozen pretrained vs placebo GRU (H13)", "3", "H13_probe_gru_ssl_vs_placebo"),
     ("Transformer pretrained vs ridge regression, four learning rates (H4)", "1", "H4_ssl_vs_ridge")])

w("% ---- tests: other patients and BrainBERT")
longtable_tests(
    "Other patients' data and BrainBERT: paired differences as in table~\\ref{tab:s-tests-own}. Cross-patient rows use "
    "four learning rates for both arms; leave-one-out, number-of-patients and large-model rows use seeds 1--3. For H7 "
    "the non-inferiority criterion is the one-sided 95\\% lower bound against the margin of $-0.5$ ($-0.005$ in r; "
    "table~\\ref{tab:s-noninf}).",
    "tab:s-tests-cross",
    [("Pooled (30 patients) vs own-patient (H5)", "2", "H5_pooled_vs_ssl"),
     ("Pooled (30 patients) vs supervised (H5)", "2", "H5_pooled_vs_sup"),
     ("Pooled, 5 patients vs own-patient (H6)", "2", "H6_scale5_vs_ssl"),
     ("Pooled, 15 patients vs own-patient (H6)", "2", "H6_scale15_vs_ssl"),
     ("Leave-one-out, adapted vs own-patient (H7)", "2", "H7_looadapt_vs_ssl"),
     ("Leave-one-out, adapted vs supervised (H7)", "2", "H7_looadapt_vs_sup"),
     ("Leave-one-out, not adapted vs supervised", "2", "loonoadapt_vs_sup_all"),
     ("Leave-one-out: adapted vs not adapted", "2", "looadapt_vs_loonoadapt"),
     ("Large model: pooled vs own-patient (H11)", "3", "H11_pooledL_vs_sslL"),
     ("Large vs base model, own-patient pretraining", "3", "sslL_vs_ssl"),
     ("Large vs base model, pooled pretraining", "3", "pooledL_vs_pooled"),
     ("Own-patient GRU vs BrainBERT GRU, best $k$ (H14a, R1)", "4b", "R1_gru_ssl7_vs_bb_best_gru_all"),
     ("Ridge on own-patient encoder vs BrainBERT ridge, best $k$ (H14b, R1)", "4b", "R1_probe_ssl_vs_bb_best_ridge_all"),
     ("Own-patient GRU vs BrainBERT GRU, $k = 9$, six learning rates (H14a as first run)", "4", "H14a_gru_ssl_vs_bb_gru"),
     ("Ridge on own-patient encoder vs BrainBERT ridge, $k = 9$ (H14b as first run)", "4", "H14b_probe_ssl_vs_bb_ridge"),
     ("BrainBERT ridge (best $k$) vs ridge on high-gamma features", "4b", "bb_best_ridge_vs_ridge"),
     ("BrainBERT GRU (best $k$) vs GRU on high-gamma features", "4b", "bb_best_gru_vs_gru7"),
     ("BrainBERT GRU: 64 vs 9 components", "4b", "bb64_gru_vs_bb_gru7"),
     ("BrainBERT ridge: 64 vs 9 components", "4b", "bb64_ridge_vs_bb_ridge")])

w("% ---- tests: fine-tuning")
longtable_tests(
    "Fine-tuning: paired differences as in table~\\ref{tab:s-tests-own}; seven learning rates (H16 as first run used six; "
    "table~\\ref{tab:s-lr}).",
    "tab:s-tests-ft",
    [("Transformer: LP-FT vs plain fine-tuning (H16)", "4b", "R2_H16_lpft_ssl_vs_ssl_7"),
     ("GRU: LP-FT vs plain fine-tuning (H16)", "4b", "R2_H16_lpft_gru_ssl_vs_gru_ssl_7"),
     ("GRU pretrained with LP-FT vs supervised GRU", "4b", "R2_lpft_gru_ssl7_vs_gru7_all"),
     ("GRU pretrained with LP-FT vs ridge regression", "4b", "R2_lpft_gru_ssl7_vs_ridge_all")])

# Page tests and non-inferiority
w("% ---- Page tests")
w("\\begin{table}[htbp]\\centering\\small")
w("\\caption{Page's trend tests ($N = 30$). Order: the conditions listed, in the predicted order of increasing gain.}")
w("\\label{tab:s-page}")
w("\\begin{tabular}{@{}llrr@{}}\\toprule Test & Order & $L$ & $p$ \\\\ \\midrule")
pg = J["4b"]["results"]["r_mean"]
w(f"H12: Transformer gain grows as labels decrease (seven rates) & 100\\% $\\to$ 5\\% & "
  f"{J['3']['results']['r_mean']['page_ssl_g7_minus_sup_g7']['L']:.0f} & {pv(J['3']['results']['r_mean']['page_ssl_g7_minus_sup_g7']['p'])} \\\\")
w(f"H1: same, four learning rates & 100\\% $\\to$ 5\\% & {J['2']['results']['r_mean']['H1_page_ssl_vs_sup_g4']['L']:.0f} & "
  f"{pv(J['2']['results']['r_mean']['H1_page_ssl_vs_sup_g4']['p'])} \\\\")
w(f"GRU gain grows as labels decrease & 100\\% $\\to$ 5\\% & {J['3']['results']['r_mean']['page_gru_ssl_minus_gru']['L']:.0f} & "
  f"{pv(J['3']['results']['r_mean']['page_gru_ssl_minus_gru']['p'])} \\\\")
for f in ("0.25", "0.1", "0.05"):
    x = J["2"]["results"]["r_mean"]["H6_page"][f]
    w(f"H6: gain grows with patients, {BUD[f]}\\% labels & 1, 5, 15, 30 & {x['L']:.0f} & {pv(x['p'])} \\\\")
for f in ("0.25", "0.1", "0.05"):
    x = J["2"]["results"]["r_mean"]["H6_page_reverse_posthoc"][f]
    w(f"H6 reversed (post hoc), {BUD[f]}\\% labels & 30, 15, 5, 1 & {x['L']:.0f} & {pv(x['p'])} \\\\")
for arch, lab in (("transformer", "Transformer"), ("gru", "GRU")):
    for f in ("0.1", "0.05"):
        x = J["4b"]["results"]["r_mean"]["R2_H15_dose"][arch][f]
        w(f"H15: {lab} gain grows with unlabelled data, {BUD[f]}\\% labels & 0, 12.5, 25, 50, 100\\% & {x['L']:.0f} & {pv(x['p'])} \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

w("% ---- non-inferiority")
w("\\begin{table}[htbp]\\centering\\small")
w("\\caption{H7 non-inferiority of leave-one-out pretraining with adaptation against own-patient pretraining (overall r; "
  "seeds 1--3), in hundredths of r as in table~\\ref{tab:s-tests-own}; margin $-0.5$ ($-0.005$ in r). Non-inferior if the "
  "one-sided 95\\% lower bound exceeds the margin.}")
w("\\label{tab:s-noninf}")
w("\\begin{tabular}{@{}lrrl@{}}\\toprule Labels & $\\Delta$ & Lower 95\\% bound & Non-inferior \\\\ \\midrule")
for f in ("0.25", "0.1", "0.05"):
    x = J["2"]["results"]["r_mean"]["H7_looadapt_vs_ssl"][f]
    lb = x.get("lower95_onesided")
    w(f"{BUD[f]}\\% & {d100(x['delta'])} & {d100(lb)} & {'yes' if lb is not None and lb > -0.005 else 'no'} \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

# =============================================================================== dose, label efficiency
w("% ---- dose")
w("\\begin{table}[htbp]\\centering\\small")
w("\\caption{Gain in overall r over supervised training of the same architecture by the amount of the participant's own "
  "unlabelled data used for pretraining, in hundredths of r as in table~\\ref{tab:s-tests-own} (seeds 1--3; seven learning "
  "rates). Labelled windows = self-pretraining.}")
w("\\label{tab:s-dose}")
w("\\begin{tabular}{@{}llrrrrr@{}}\\toprule Decoder & Labels & Labelled windows & 12.5\\% & 25\\% & 50\\% & 100\\% \\\\ \\midrule")
for arch, lab in (("transformer", "Transformer"), ("gru", "GRU")):
    for f in ("0.1", "0.05"):
        g = J["4b"]["results"]["r_mean"]["dose_with_self_seven_rates"][arch][f]["gain_over_base"]
        w(f"{lab} & {BUD[f]}\\% & " + " & ".join(d100(v) for v in g[1:]) + " \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

w("% ---- label efficiency")
w("\\begin{table}[htbp]\\centering\\small")
w("\\caption{Label-equivalent factor (median over participants, bootstrap 95\\% confidence interval): the supervised budget "
  "that reaches the pretrained decoder's score, divided by the budget used. Descriptive (specified after Phases 1--4).}")
w("\\label{tab:s-labeleq}")
w("\\begin{tabular}{@{}llrrr@{}}\\toprule Metric & Decoder & 25\\% & 10\\% & 5\\% \\\\ \\midrule")
for met, ml in (("r_mean", "Overall r"), ("r_speech", "Within-speech r")):
    E = J["4b"]["E6_label_efficiency"][met]
    for k, kl in (("transformer", "Transformer, whole session"), ("gru", "GRU, whole session"),
                  ("transformer_self", "Transformer, labelled windows"), ("gru_self", "GRU, labelled windows")):
        if k not in E:
            continue
        cells = []
        for f in ("0.25", "0.1", "0.05"):
            x = E[k][f]
            cells.append(f"{x['median_ratio']:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]")
        w(f"{ml if k == 'transformer' else ''} & {kl} & " + " & ".join(cells) + " \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

# =============================================================================== masking
w("% ---- masking")
M = J["2"]["results"]["r_mean"]["masking"]
w("\\begin{table}[htbp]\\centering\\small")
w("\\caption{Masking ablation (Transformer, four learning rates, seeds 1--3): mean overall r of each mask and its difference "
  "from the default mask (random coordinates, 50\\%) in hundredths of r ($\\Delta r \\times 100$), with Holm-corrected $p$. "
  "Friedman tests across masks: "
  f"$p = {M['1.0']['friedman'][1]:.2f}$ with all labels and $p$ = {pv(M['0.1']['friedman'][1])} with 10\\% labels.}}")
w("\\label{tab:s-mask}")
w("\\begin{tabular}{@{}lrrrrrr@{}}\\toprule & \\multicolumn{3}{c}{100\\% labels} & \\multicolumn{3}{c}{10\\% labels} \\\\ "
  "\\cmidrule(lr){2-4}\\cmidrule(l){5-7} Mask & mean r & $\\Delta$ & $p_{\\mathrm{Holm}}$ & mean r & $\\Delta$ & $p_{\\mathrm{Holm}}$ \\\\ \\midrule")
names = {"mask_random25": "Random, 25\\%", "mask_random75": "Random, 75\\%", "mask_time25": "Time blocks, 25\\%",
         "mask_time50": "Time blocks, 50\\%", "mask_time75": "Time blocks, 75\\%", "mask_contact25": "Contacts, 25\\%",
         "mask_contact50": "Contacts, 50\\%", "mask_contact75": "Contacts, 75\\%", "mask_lagcons25": "All lags of a value, 25\\%",
         "mask_lagcons50": "All lags of a value, 50\\%", "mask_lagcons75": "All lags of a value, 75\\%"}
for k, lab in names.items():
    a, b = M["1.0"]["vs_default"].get(k), M["0.1"]["vs_default"].get(k)
    if a is None or b is None:
        continue
    w(f"{lab} & {num(a['mean_a'])} & {d100(a['delta'])} & {pv(a['p_holm'])} & {num(b['mean_a'])} & {d100(b['delta'])} & "
      f"{pv(b['p_holm'])} \\\\")
a0 = list(M["1.0"]["vs_default"].values())[0]["mean_b"]
b0 = list(M["0.1"]["vs_default"].values())[0]["mean_b"]
w(f"Random, 50\\% (default) & {num(a0)} & & & {num(b0)} & & \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

# =============================================================================== BrainBERT
w("% ---- BrainBERT")
V = J["4b"]["bb_variance_kept"]
K = J["4b"]["bb_k_chosen"]
w("\\begin{table}[htbp]\\centering\\small")
w("\\caption{BrainBERT readouts. Top: share of each contact's embedding variance kept by $k$ principal components (minimum, "
  "median and maximum over participants). Bottom: mean overall r of each readout with $k$ fixed, and the number of runs "
  "(of 150 per budget) in which validation chose each $k$.}")
w("\\label{tab:s-bb}")
w("\\begin{tabular}{@{}lrrrrr@{}}\\toprule")
w("$k$ & 9 & 16 & 32 & 64 & \\\\ \\midrule")
w("Variance kept & " + " & ".join(f"{V[k][1]:.2f} ({V[k][0]:.2f}--{V[k][2]:.2f})" for k in ("9", "16", "32", "64")) + " & \\\\")
w("\\midrule Overall r & 100\\% & 50\\% & 25\\% & 10\\% & 5\\% \\\\ \\midrule")
lin = D.sel
for c, lab in (("bb_ridge", "Ridge, $k = 9$"), ("bb16_ridge", "Ridge, $k = 16$"), ("bb32_ridge", "Ridge, $k = 32$"),
               ("bb64_ridge", "Ridge, $k = 64$")):
    x = lin[(lin.condition == c)]
    W = x.groupby(["participant", "fraction"]).r_mean.mean().unstack("fraction")
    w(f"{lab} & " + " & ".join(num(W[f].mean()) if f in W else "--" for f in FR) + " \\\\")
for c, lab in (("bb_best_ridge", "Ridge, best $k$"), ("bb_best_gru", "GRU, best $k$ (9 or 64)")):
    W = wide(D, "r_mean", c)
    w(f"{lab} & " + " & ".join(num(W[f].mean()) for f in FR) + " \\\\")
w("\\midrule Runs choosing $k$ & 100\\% & 50\\% & 25\\% & 10\\% & 5\\% \\\\ \\midrule")
for kk in ("9", "16", "32", "64"):
    w(f"Ridge, $k = {kk}$ & " + " & ".join(str(K['bb_best_ridge'][f].get(kk, 0)) for f in BUD) + " \\\\")
for kk in ("9", "64"):
    w(f"GRU, $k = {kk}$ & " + " & ".join(str(K['bb_best_gru'][f].get(kk, 0)) for f in BUD) + " \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

# =============================================================================== learning rates
w("% ---- learning rates")
SH = J["4b"]["share_top_lr"]
labs = {"sup7": "Transformer, supervised", "placebo7": "Transformer, placebo", "ssl7": "Transformer, pretrained",
        "ssl_self": "Transformer, self-pretrained", "lpft_ssl7": "Transformer, LP-FT", "gru7": "GRU, supervised",
        "gru_placebo7": "GRU, placebo", "gru_ssl7": "GRU, pretrained", "gru_ssl_self": "GRU, self-pretrained",
        "lpft_gru_ssl7": "GRU, LP-FT", "bb_gru7": "BrainBERT GRU, $k = 9$", "bb64_gru7": "BrainBERT GRU, $k = 64$"}
w("\\begin{table}[htbp]\\centering\\small")
w("\\caption{Learning-rate grid. Top: share of runs selecting the largest rate, $10^{-2}$, by decoder and budget. Bottom: "
  "the pre-specified tests whose conclusion (Holm-corrected $p < 0.05$ or not) differs between six and seven learning rates, "
  "with $\\Delta$ in hundredths of r; all other tests agree.}")
w("\\label{tab:s-lr}")
w("\\begin{tabular}{@{}lrrrrr@{}}\\toprule Decoder & 100\\% & 50\\% & 25\\% & 10\\% & 5\\% \\\\ \\midrule")
for k, lab in labs.items():
    if k in SH:
        w(f"{lab} & " + " & ".join(f"{SH[k].get(f, float('nan')):.2f}" if f in SH[k] else "--" for f in BUD) + " \\\\")
w("\\midrule \\multicolumn{6}{@{}l}{Test (metric, labels): $\\Delta$ and $p_{\\mathrm{Holm}}$ with six $\\to$ seven rates} \\\\ \\midrule")
tl = {"R2_H10_gru_ssl_vs_gru_7": "GRU pretrained vs supervised (H10)",
      "R2_H16_lpft_gru_ssl_vs_gru_ssl_7": "GRU LP-FT vs fine-tuning (H16)"}
for met, key, f, d6, p6, d7, p7 in J["4b"]["R2_flips"]:
    ml = "overall r" if met == "r_mean" else "within-speech r"
    w(f"\\multicolumn{{6}}{{@{{}}l}}{{{tl.get(key, esc(key))}, {ml}, {BUD[f]}\\%: {d100(d6)}, $p$ = {pv(p6)} $\\to$ "
      f"{d100(d7)}, $p$ = {pv(p7)}}} \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

# =============================================================================== contamination
w("% ---- contamination")
C = J["2"]["contamination"]
w("\\begin{table}[htbp]\\centering\\small\\setlength{\\tabcolsep}{4pt}")
w(f"\\caption{{Acoustic contamination audit per participant: number of contacts, largest and median contamination index, and "
  f"flagged contacts ($q < 0.05$ and index $\\ge 0.01$). Positive controls: audio injected into one contact per participant at "
  f"$-30$ and $-20$~dB was detected in {C['n_detected']} of {C['n_inject']} cases (median index of the injected contact "
  f"{C['median_inject_index']['-30']:.3f} and {C['median_inject_index']['-20']:.3f}). Total: {C['n_channels_flagged']} of "
  f"{C['n_channels_total']} contacts flagged.}}")
w("\\label{tab:s-contam}")
w("\\begin{tabular}{@{}lrrrr@{\\hspace{1.2em}}lrrrr@{}}\\toprule")
w("P & Contacts & Max index & Median & Flagged & P & Contacts & Max index & Median & Flagged \\\\ \\midrule")
pp = C["per_participant"]
half = (len(pp) + 1) // 2
for i in range(half):
    a = pp[i]
    left = f"{a['participant']} & {a['n_channels']} & {a['max_index'] + 0.0 if round(a['max_index'], 4) != 0 else 0.0:.4f} & {a['median_index'] + 0.0 if round(a['median_index'], 4) != 0 else 0.0:.4f} & {a['n_flagged']}"
    if i + half < len(pp):
        b = pp[i + half]
        right = f"{b['participant']} & {b['n_channels']} & {b['max_index'] + 0.0 if round(b['max_index'], 4) != 0 else 0.0:.4f} & {b['median_index'] + 0.0 if round(b['median_index'], 4) != 0 else 0.0:.4f} & {b['n_flagged']}"
    else:
        right = "& & & &"
    w((left + " & " + right).replace("-", "$-$") + " \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

w("% ---- without P30")
longtable_tests(
    "Key comparisons repeated without P30 ($N = 29$), as in tables~\\ref{tab:s-tests-own}--\\ref{tab:s-tests-ft} "
    "(differences in hundredths of r).",
    "tab:s-n29",
    [("Transformer: pretrained vs supervised", "4b", "R2_ssl7_vs_sup7_all"),
     ("Transformer: pretrained vs placebo", "4b", "R2_ssl7_vs_placebo7_all"),
     ("GRU: pretrained vs supervised (H10)", "4b", "R2_H10_gru_ssl_vs_gru_7"),
     ("Transformer: self-pretrained vs supervised (H17)", "4b", "H17_ssl_self_vs_sup7"),
     ("Pooled (30 patients) vs own-patient (H5)", "2", "H5_pooled_vs_ssl"),
     ("Leave-one-out, adapted vs own-patient (H7)", "2", "H7_looadapt_vs_ssl"),
     ("Own-patient GRU vs BrainBERT GRU (R1)", "4b", "R1_gru_ssl7_vs_bb_best_gru"),
     ("Transformer: LP-FT vs fine-tuning (H16)", "4b", "R2_H16_lpft_ssl_vs_ssl_7"),
     ("GRU: LP-FT vs fine-tuning (H16)", "4b", "R2_H16_lpft_gru_ssl_vs_gru_ssl_7")], n29=True)

# =============================================================================== evaluation suite
w("% ---- evaluation: references")
R = J["3"]["eval"]["E2_refs"]
w("\\begin{table}[htbp]\\centering\\small")
w("\\caption{Reference predictions on the test partition (participant means). Within-speech r is undefined for predictions "
  "that are constant within speech. ESTOI and STOI: seed 1, Griffin--Lim resynthesis with independent phases.}")
w("\\label{tab:s-refs}")
w("\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule Reference & Overall r & Within-speech r & Loudness r & Shape r & MCD (dB) & ESTOI & STOI \\\\ \\midrule")
for k, lab in (("true_spectrogram", "True spectrogram"), ("timing_reference", "Speech-timing reference"),
               ("train_mean", "Training-mean spectrum"), ("shifted_truth", "Shifted true spectrogram")):
    vals = [R["r_mean"][k], R["r_speech"][k], R["r_loud"][k], R["r_shape"][k], R["mcd_speech_db"][k], R["estoi"][k], R["stoi"][k]]
    vals = [None if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in vals]
    vals = [0.0 if (v is not None and abs(v) < 1e-9) else v for v in vals]
    w(f"{lab} & " + " & ".join(num(v, 3) if i != 4 else num(v, 2) for i, v in enumerate(vals)) + " \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

w("% ---- evaluation: decoders")


def esrc(c):
    return "eval4b" if c in ("gru_ssl", "lpft_gru_ssl", "lpft_ssl", "gru", "bb_gru", "ssl_self", "gru_ssl_self") else "eval"


w("\\begin{table}[htbp]\\centering\\small\\setlength{\\tabcolsep}{3.5pt}")
w("\\caption{Evaluation suite (seeds 1--3; ESTOI and STOI seed 1): within-speech loudness r and spectral-shape r with their "
  "circular-shift chance levels (95th percentile), mel-cepstral distortion within speech, ESTOI and STOI, at 100\\%, 10\\% and "
  "5\\% labels.}")
w("\\label{tab:s-eval}")
w("\\begin{tabular}{@{}llrrrrrrr@{}}\\toprule Decoder & Labels & Loudness r & chance & Shape r & chance & MCD (dB) & ESTOI & STOI \\\\ \\midrule")
for c, lab in (("sup", "Transformer, supervised"), ("ssl", "Transformer, pretrained"), ("gru", "GRU, supervised"),
               ("gru_ssl", "GRU, pretrained"), ("lpft_gru_ssl", "GRU, pretrained, LP-FT"), ("ridge", "Ridge regression"),
               ("bb_gru", "BrainBERT GRU, $k = 9$")):
    s = esrc(c)
    L, Lc = ewide(D, "r_loud", c, s), ewide(D, "r_loud_chance_p95", c, s)
    Sh, Sc = ewide(D, "r_shape", c, s), ewide(D, "r_shape_chance_p95", c, s)
    M_ = ewide(D, "mcd_speech_db", c, s)
    E1, S1 = ewide(D, "estoi", c, s, [1]), ewide(D, "stoi", c, s, [1])
    for i, f in enumerate((1.0, 0.1, 0.05)):
        g = lambda W: num(W[f].mean()) if f in W and W[f].notna().any() else "--"  # noqa: E731
        mm = num(M_[f].mean(), 2) if f in M_ and M_[f].notna().any() else "--"
        w(f"{lab if i == 0 else ''} & {BUD[str(f)]}\\% & {g(L)} & {g(Lc)} & {g(Sh)} & {g(Sc)} & {mm} & {g(E1)} & {g(S1)} \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

# =============================================================================== identification
w("% ---- identification")
I5 = J["5"]["results"]
w("\\begin{table}[htbp]\\centering\\small\\setlength{\\tabcolsep}{4pt}")
w("\\caption{Identification: pairwise accuracy (chance 0.5), with the one-sided Wilcoxon $p$ against chance, Holm-corrected "
  "across budgets, in brackets; and top-1 accuracy (chance: one over the number of segments, mean 0.057). Five seeds.}")
w("\\label{tab:s-ident}")
w("\\begin{tabular}{@{}lrrrrr@{}}\\toprule Pairwise accuracy & 100\\% & 50\\% & 25\\% & 10\\% & 5\\% \\\\ \\midrule")
DEC = (("sup", "Transformer, supervised"), ("ssl", "Transformer, pretrained"), ("ssl_self", "Transformer, self-pretrained"),
       ("lpft_ssl", "Transformer, LP-FT"), ("gru", "GRU, supervised"), ("gru_ssl", "GRU, pretrained"),
       ("gru_ssl_self", "GRU, self-pretrained"), ("lpft_gru_ssl", "GRU, LP-FT"), ("ridge", "Ridge regression"),
       ("probe_ssl", "Ridge on frozen encoder"), ("bb_best_ridge", "BrainBERT, ridge"), ("bb_best_gru", "BrainBERT, GRU"))
for c, lab in DEC:
    A = I5["all_decoders_full"][c]
    w(f"{lab} & " + " & ".join(f"{A[f]['mean']:.3f} ({pvc(A[f]['p_holm'])})" if f in A else "--" for f in BUD) + " \\\\")
ref = I5["references"]
w(f"Shifted true spectrogram & \\multicolumn{{5}}{{l}}{{{ref['shifted_truth']['full_pairwise']:.3f} "
  f"($p$ = {pv(ref['shifted_truth_vs_chance']['-1.0']['p'])})}} \\\\")
w("\\midrule Top-1 accuracy & 100\\% & 50\\% & 25\\% & 10\\% & 5\\% \\\\ \\midrule")
for c, lab in DEC:
    T = I5["all_decoders_top1"][c]
    w(f"{lab} & " + " & ".join(f"{T[f]['mean']:.3f}" if f in T else "--" for f in BUD) + " \\\\")
w(f"Shifted true spectrogram & \\multicolumn{{5}}{{l}}{{{ref['shifted_truth']['full_top1']:.3f}}} \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

w("\\begin{table}[htbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}")
w("\\caption{What carries identification, task groups and sensitivity analyses: pairwise accuracy (Holm-corrected $p$ against "
  "0.5). Loudness: frame means only; shape: each frame minus its mean. Task groups: pretrained GRU, 10 participants each. "
  "Sensitivity: pretrained GRU with 16 or 64 resampled frames, or stricter segments (gaps under 100~ms closed, at least "
  "300~ms long).}")
w("\\label{tab:s-ident2}")
w("\\begin{tabular}{@{}llrrrrr@{}}\\toprule Analysis & Decoder or group & 100\\% & 50\\% & 25\\% & 10\\% & 5\\% \\\\ \\midrule")
for kind, kl in (("loudness", "Loudness only"), ("shape", "Shape only")):
    for c, lab in (("gru_ssl", "GRU, pretrained"), ("lpft_gru_ssl", "GRU, LP-FT"), ("ridge", "Ridge regression")):
        A = I5["E7"][kind][c]
        w(f"{kl if c == 'gru_ssl' else ''} & {lab} & " + " & ".join(f"{A[f]['mean']:.3f} ({pvc(A[f]['p_holm'])})" for f in BUD) + " \\\\")
w("\\midrule")
for t, tl in (("unique_words", "Unique words"), ("repeated_words", "Repeated words"), ("sentences", "Sentences")):
    A = I5["cohorts"][t]["gru_ssl_vs_chance"]
    w(f"{'Task group' if t == 'unique_words' else ''} & {tl} & " + " & ".join(f"{A[f]['mean']:.3f} ({pvc(A[f]['p_holm'])})" for f in BUD) + " \\\\")
w("\\midrule")
for v, vl in (("k16", "16 frames"), ("k64", "64 frames"), ("strict", "Stricter segments")):
    A = I5["sensitivity"][v]["H18"]
    w(f"{'Sensitivity' if v == 'k16' else ''} & {vl} & " + " & ".join(f"{A[f]['mean']:.3f} ({pvc(A[f]['p_holm'])})" for f in BUD) + " \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

w("\\begin{small}")
w("\\begin{longtable}{@{}rlrr@{}}")
w("\\caption{Identification comparisons: mean paired differences in pairwise accuracy (A minus B) in percentage points "
  "(accuracy $\\times$ 100, as in table~2 of the article), with bootstrap 95\\% confidence interval in the same units, $n_+$ "
  "and Holm-corrected two-sided Wilcoxon $p$ across the budgets listed. Five seeds.}"
  "\\label{tab:s-ident-tests}\\\\")
hdr1 = "\\toprule Labels & $\\Delta$ [95\\% CI] & $n_+$ & $p_{\\mathrm{Holm}}$ \\\\ \\midrule"
w(hdr1 + "\\endfirsthead")
w("\\multicolumn{4}{@{}l}{\\textit{(continued)}} \\\\ " + hdr1 + "\\endhead")
w("\\bottomrule\\endfoot")
for lab, key in (("H19: GRU pretrained vs supervised", "H19_gru_ssl_vs_gru"),
                 ("H19: Transformer pretrained vs supervised", "H19_ssl_vs_sup"),
                 ("GRU: LP-FT vs plain fine-tuning", "cmp_lpft_gru_ssl_vs_gru_ssl_all"),
                 ("Transformer: LP-FT vs plain fine-tuning", "cmp_lpft_ssl_vs_ssl_all"),
                 ("GRU pretrained vs BrainBERT GRU", "cmp_gru_ssl_vs_bb_best_gru_all"),
                 ("GRU pretrained with LP-FT vs ridge regression", "cmp_lpft_gru_ssl_vs_ridge_all")):
    r = J["5"]["results"][key]
    w(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textbf{{{lab}}}}} \\\\*")
    for f in ("1.0", "0.5", "0.25", "0.1", "0.05"):
        if f in r:
            w(f"{BUD[f]}\\% & {cell(r[f])} \\\\")
w("\\end{longtable}")
w("\\end{small}")

# =============================================================================== moderators
w("% ---- moderators")
M1 = J["4"]["M1"]["tests"]
ml = {"n_contacts": "Number of contacts", "grey_share": "Share of contacts in grey matter",
      "n_speech_cortex": "Contacts in speech-related cortex", "label_seconds_10": "Labelled seconds (10\\% budget)",
      "sup_score_10": "Supervised score (10\\% budget)"}
gl = {"ssl_minus_sup_10": "Transformer, overall r", "gru_ssl_minus_gru_10": "GRU, overall r",
      "ssl_minus_sup_10_speech": "Transformer, within-speech r", "gru_ssl_minus_gru_10_speech": "GRU, within-speech r"}
w("\\begin{table}[htbp]\\centering\\small")
w("\\caption{Exploratory moderator analysis (M1): Spearman correlation of each participant's gain from pretraining at 10\\% "
  "labels with participant characteristics, with Benjamini--Hochberg $q$ across all tests. Speech-related cortex: superior "
  "temporal, pre- and postcentral, inferior frontal, insular and supramarginal contacts.}")
w("\\label{tab:s-mod}")
w("\\begin{tabular}{@{}llrrr@{}}\\toprule Gain & Moderator & $\\rho$ & $p$ & $q$ \\\\ \\midrule")
for g, m_, rho, p, q in M1:
    w(f"{gl.get(g, esc(g))} & {ml.get(m_, esc(m_))} & {sgn(rho, 2)} & {pv(p)} & {pv(q)} \\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

open(OUT, "w").write("\n".join(out) + "\n")
print("wrote", OUT, len(out), "lines")
