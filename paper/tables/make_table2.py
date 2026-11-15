"""Table 2 of the article (key pre-specified comparisons) from the saved statistical outputs.

Usage: python make_table2.py <stats dir> [out .tex]

Each entry is the mean paired difference over the participants (A minus B) in hundredths of r
(difference x 100), followed by the number of participants with a positive difference; entries
significant after Holm correction across the budgets of their family (p < 0.05) are bold.
Identification rows give pairwise accuracy x 100 (for "vs chance": accuracy minus 0.5).
"""
import json
import os
import sys

ST = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
J = {k: json.load(open(os.path.join(ST, f"phase{k}_stats.json"))) for k in ("2", "3", "4b", "5")}
LOW = ("0.25", "0.1", "0.05")


def res(phase, key, metric):
    if phase == "5":
        return J["5"]["results"][key] if metric == "r_mean" else None
    return J[phase]["results"][metric][key]


def entry(r):
    if r is None:
        return " "
    v = round(100 * r["delta"], 1)
    s = "0.0" if v == 0 else f"{v:+.1f}".replace("-", "$-$")
    if r.get("p_holm", r.get("p")) < 0.05:
        s = f"\\textbf{{{s}}}"
    return f"{s}\\,{{\\scriptsize({r['n_pos']})}}"


ROWS = [
    ("Own-patient pretraining", None),
    ("Transformer: pretrained vs supervised", ("4b", "R2_ssl7_vs_sup7_all")),
    ("Transformer: pretrained vs placebo", ("4b", "R2_ssl7_vs_placebo7_all")),
    ("GRU: pretrained vs supervised", ("4b", "R2_H10_gru_ssl_vs_gru_7")),
    ("GRU: pretrained vs placebo", ("4b", "R2_H10b_gru_ssl_vs_gru_placebo_7")),
    ("Transformer: self-pretrained vs supervised", ("4b", "H17_ssl_self_vs_sup7")),
    ("GRU: self-pretrained vs supervised", ("4b", "H17_gru_ssl_self_vs_gru7")),
    ("Other patients' data", None),
    ("Pooled (30 patients) vs own-patient", ("2", "H5_pooled_vs_ssl")),
    ("Leave-one-out, adapted vs own-patient$^{\\mathrm{a}}$", ("2", "H7_looadapt_vs_ssl")),
    ("Large model: pooled vs own-patient", ("3", "H11_pooledL_vs_sslL")),
    ("Own-patient GRU vs BrainBERT GRU", ("4b", "R1_gru_ssl7_vs_bb_best_gru_all")),
    ("Fine-tuning", None),
    ("Transformer: LP-FT vs fine-tuning", ("4b", "R2_H16_lpft_ssl_vs_ssl_7")),
    ("GRU: LP-FT vs fine-tuning", ("4b", "R2_H16_lpft_gru_ssl_vs_gru_ssl_7")),
    ("Identification (pairwise accuracy $\\times$ 100)", None),
    ("Pretrained GRU vs chance (0.5)", ("5", "H18_gru_ssl_full")),
    ("GRU: pretrained vs supervised", ("5", "H19_gru_ssl_vs_gru")),
    ("Transformer: pretrained vs supervised", ("5", "H19_ssl_vs_sup")),
    ("GRU: LP-FT vs fine-tuning", ("5", "cmp_lpft_gru_ssl_vs_gru_ssl_all")),
]

lines = []
for label, src in ROWS:
    if src is None:
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\")
        continue
    a, b = res(*src, "r_mean"), res(*src, "r_speech")
    cells = [entry(a[f]) for f in LOW] + [entry(b[f]) if b is not None else " " for f in LOW]
    lines.append(f"{label} & " + " & ".join(cells) + " \\\\")

text = "\n".join(lines) + "\n"
if OUT:
    open(OUT, "w").write(text)
else:
    sys.stdout.write(text)
