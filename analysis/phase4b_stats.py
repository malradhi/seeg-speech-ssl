"""Pre-specified Phase 4b analysis (docs/PHASE4B_DESIGN.md).

Usage: python -I phase4b_stats.py <p1_dir> <p2_dir> <p3_dir> <p4_dir> <p4b_dir> <out_json> [exclude_pid ...]

Seven-rate grid (1e-5 ... 1e-2) for every base-size per-patient decoder; runs of the same condition
are pooled across phases. Learning rate (and, for BrainBERT, the number of components) chosen per
(participant, condition, fraction, seed) on validation MSE. Participant score = mean over seeds;
comparisons with a three-seed condition use seeds 1-3 of both arms. Wilcoxon signed-rank, two-sided,
Holm across the budgets of a family.
"""
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats

P1, P2, P3, P4, P4B, OUT = sys.argv[1:7]
EXCL = sys.argv[7:]
FR = [1.0, 0.5, 0.25, 0.1, 0.05]
LOW = [0.25, 0.1, 0.05]
DF = [0.1, 0.05]
G4 = [3e-5, 1e-4, 3e-4, 1e-3]
G6 = [1e-5] + G4 + [3e-3]
G7 = G6 + [1e-2]
S3, S5 = [1, 2, 3], [1, 2, 3, 4, 5]
TASKS = ["unique_words", "repeated_words", "sentences"]

df = pd.concat([pd.read_csv(os.path.join(p, f"phase{i}_results.csv"))
                for i, p in (("1", P1), ("2", P2), ("3", P3), ("4", P4), ("4b", P4B))], ignore_index=True)
df = df[~df.participant.isin(EXCL)]
task_of = df.dropna(subset=["task"]).groupby("participant").task.first()
PIDS = sorted(task_of.index)
ft = df[(df.kind == "finetune") & (df.protocol == "new")].copy()
KEY = ["participant", "condition", "fraction", "seed"]


def inset(lr, grid):
    return np.isclose(lr.values[:, None], np.array(grid)[None, :]).any(1)


def select(frame, name):
    s = frame.loc[frame.groupby(KEY).val_mse.idxmin()].copy()
    s["source_condition"] = s["condition"]
    s["condition"] = name
    return s


SPECS = [("sup", G4, "sup"), ("ssl", G4, "ssl"), ("sup", G7, "sup7"), ("ssl", G7, "ssl7"),
         ("placebo", G6, "placebo6"), ("placebo", G7, "placebo7"),
         ("gru", G6, "gru6"), ("gru", G7, "gru7"), ("gru_ssl", G6, "gru_ssl6"), ("gru_ssl", G7, "gru_ssl7"),
         ("gru_placebo", G6, "gru_placebo6"), ("gru_placebo", G7, "gru_placebo7"),
         ("lpft_ssl", G6, "lpft_ssl6"), ("lpft_ssl", G7, "lpft_ssl7"), ("ssl", G6, "ssl6"),
         ("lpft_gru_ssl", G6, "lpft_gru_ssl6"), ("lpft_gru_ssl", G7, "lpft_gru_ssl7"),
         ("bb_gru", G6, "bb_gru6"), ("bb_gru", G7, "bb_gru7"), ("bb64_gru", G7, "bb64_gru7"),
         ("ssl_self", G7, "ssl_self"), ("gru_ssl_self", G7, "gru_ssl_self")]
for u in ("125", "250", "500"):
    SPECS += [(f"ssl_u{u}", G4, f"ssl_u{u}_4"), (f"ssl_u{u}", G7, f"ssl_u{u}_7"),
              (f"gru_ssl_u{u}", G6, f"gru_ssl_u{u}_6"), (f"gru_ssl_u{u}", G7, f"gru_ssl_u{u}_7")]
parts = [select(ft[(ft.condition == c) & inset(ft.lr, g)], n) for c, g, n in SPECS]
# BrainBERT, best readout capacity chosen on validation
parts.append(select(ft[ft.condition.isin(["bb_gru", "bb64_gru"]) & inset(ft.lr, G7)].assign(condition="x"), "bb_best_gru"))
lin = df[df.kind == "linear"]
parts.append(select(lin[lin.condition.isin(["bb_ridge", "bb16_ridge", "bb32_ridge", "bb64_ridge"])].assign(condition="x"),
                    "bb_best_ridge"))
SELFT = pd.concat(parts, ignore_index=True)
SEL = pd.concat([SELFT, df[df.kind.isin(["linear", "probe"])]], ignore_index=True)
# the source condition of a pooled selection was overwritten by "x"; recover k from the run id
SELFT["bb_k"] = np.where(SELFT.run_id.str.contains("bb64_"), 64, np.where(SELFT.run_id.str.contains("bb32_"), 32,
                         np.where(SELFT.run_id.str.contains("bb16_"), 16, 9)))


def wide(metric, cond, seeds=S5, src=None):
    src = SEL if src is None else src
    x = src[(src.condition == cond) & src.seed.isin(seeds)]
    return x.groupby(["participant", "fraction"])[metric].mean().unstack("fraction").reindex(PIDS)


def boot(x, n=10000, seed=1, fn=np.mean):
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    return fn(np.asarray(x)[idx], axis=1)


def paired(a, b):
    d = (a - b).dropna()
    w = stats.wilcoxon(d)
    nz = d[d != 0]
    rk = stats.rankdata(np.abs(nz))
    bs = boot(d.values)
    return {"mean_a": float(a[d.index].mean()), "mean_b": float(b[d.index].mean()), "delta": float(d.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_pos": int((d > 0).sum()), "n": int(len(d)), "p": float(w.pvalue),
            "rank_biserial": float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()) if rk.sum() else 0.0}


def holm(ps):
    ps = np.asarray(ps, float)
    order, m, adj, run = np.argsort(ps), len(ps), np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj.tolist()


def fam(a, b, metric, fracs=FR, seeds=S5, pids=None, src=None):
    A, B = wide(metric, a, seeds, src), wide(metric, b, seeds, src)
    if pids is not None:
        A, B = A.loc[pids], B.loc[pids]
    res = {f: paired(A[f], B[f]) for f in fracs}
    for f, q in zip(fracs, holm([res[f]["p"] for f in fracs])):
        res[f]["p_holm"] = q
    return {str(f): v for f, v in res.items()}


def dose(metric, base, arms, full, f, seeds=S3):
    conds = [base] + arms + [full]
    M = pd.concat([wide(metric, c, seeds)[f] for c in conds], axis=1, keys=conds).dropna()
    t = stats.page_trend_test(M.values, predicted_ranks=list(range(1, len(conds) + 1)))
    return {"conds": conds, "means": [float(v) for v in M.mean()],
            "gain_over_base": [float(v - M[base].mean()) for v in M.mean()],
            "L": float(t.statistic), "p": float(t.pvalue), "n": int(len(M))}


out = {"n_participants": len(PIDS), "excluded": EXCL,
       "provenance_p4b": sorted(pd.read_csv(os.path.join(P4B, "phase4b_results.csv")).code_hash.unique().tolist())}
R = {}
for metric in ("r_mean", "r_speech"):
    r = {}
    # ---- H17 self-pretraining
    r["H17_ssl_self_vs_sup7"] = fam("ssl_self", "sup7", metric, LOW)
    r["H17_gru_ssl_self_vs_gru7"] = fam("gru_ssl_self", "gru7", metric, LOW)
    r["ssl7_vs_ssl_self"] = fam("ssl7", "ssl_self", metric, LOW)
    r["gru_ssl7_vs_gru_ssl_self"] = fam("gru_ssl7", "gru_ssl_self", metric, LOW)
    r["gru_ssl_self_vs_ssl_self"] = fam("gru_ssl_self", "ssl_self", metric, LOW)
    r["gru_ssl_self_vs_sup7"] = fam("gru_ssl_self", "sup7", metric, LOW)
    r["gru_ssl_self_vs_ridge"] = fam("gru_ssl_self", "ridge", metric, LOW)
    ext = {}
    for arch, base, pre, full in (("transformer", "sup7", "ssl_u", "ssl7"), ("gru", "gru7", "gru_ssl_u", "gru_ssl7")):
        selfc = "ssl_self" if arch == "transformer" else "gru_ssl_self"
        ext[arch] = {str(f): dose(metric, base, [selfc] + [f"{pre}{u}_7" for u in ("125", "250", "500")], full, f)
                     for f in DF}
    r["dose_with_self_seven_rates"] = ext
    # ---- R1 BrainBERT readout capacity
    r["R1_probe_ssl_vs_bb_best_ridge"] = fam("probe_ssl", "bb_best_ridge", metric, LOW)
    r["R1_gru_ssl7_vs_bb_best_gru"] = fam("gru_ssl7", "bb_best_gru", metric, LOW)
    r["R1_probe_ssl_vs_bb_best_ridge_all"] = fam("probe_ssl", "bb_best_ridge", metric)
    r["R1_gru_ssl7_vs_bb_best_gru_all"] = fam("gru_ssl7", "bb_best_gru", metric)
    r["bb_best_ridge_vs_ridge"] = fam("bb_best_ridge", "ridge", metric)
    r["bb_best_gru_vs_gru7"] = fam("bb_best_gru", "gru7", metric)
    r["bb64_gru_vs_bb_gru7"] = fam("bb64_gru7", "bb_gru7", metric)
    r["bb64_ridge_vs_bb_ridge"] = fam("bb64_ridge", "bb_ridge", metric)
    # generous bound: best BrainBERT readout per participant and budget chosen on the TEST score
    bbw = [wide(metric, c) for c in ("bb_ridge", "bb16_ridge", "bb32_ridge", "bb64_ridge", "bb_gru7", "bb64_gru7")]
    oracle = pd.concat(bbw).groupby(level=0).max().reindex(PIDS)
    ora = {}
    for name, own in (("gru_ssl7", wide(metric, "gru_ssl7")), ("probe_ssl", wide(metric, "probe_ssl"))):
        res = {f: paired(own[f], oracle[f]) for f in FR}
        for f, q in zip(FR, holm([res[f]["p"] for f in FR])):
            res[f]["p_holm"] = q
        ora[name] = {str(f): v for f, v in res.items()}
    r["oracle_bb_vs_own"] = ora
    # ---- R2 seven-rate recomputation (and the six-rate version for comparison)
    for tag, sfx_g, sfx_t in (("7", "7", "7"), ("6", "6", "")):
        r[f"R2_H10_gru_ssl_vs_gru_{tag}"] = fam(f"gru_ssl{sfx_g}", f"gru{sfx_g}", metric, LOW)
        r[f"R2_H10b_gru_ssl_vs_gru_placebo_{tag}"] = fam(f"gru_ssl{sfx_g}", f"gru_placebo{sfx_g}", metric, LOW)
        r[f"R2_H14a_gru_ssl_vs_bb_gru_{tag}"] = fam(f"gru_ssl{sfx_g}", f"bb_gru{sfx_g}", metric, LOW)
        r[f"R2_H16_lpft_ssl_vs_ssl_{tag}"] = fam(f"lpft_ssl{sfx_g}", f"ssl{sfx_g}", metric, LOW)
        r[f"R2_H16_lpft_gru_ssl_vs_gru_ssl_{tag}"] = fam(f"lpft_gru_ssl{sfx_g}", f"gru_ssl{sfx_g}", metric, LOW)
    r["R2_gru_ssl7_vs_ssl7_all"] = fam("gru_ssl7", "ssl7", metric)
    r["R2_ssl7_vs_placebo7_all"] = fam("ssl7", "placebo7", metric)
    r["R2_gru_ssl7_vs_gru7_all"] = fam("gru_ssl7", "gru7", metric)
    r["R2_lpft_gru_ssl7_vs_gru7_all"] = fam("lpft_gru_ssl7", "gru7", metric)
    r["R2_lpft_gru_ssl7_vs_ridge_all"] = fam("lpft_gru_ssl7", "ridge", metric)
    r["R2_ssl7_vs_sup7_all"] = fam("ssl7", "sup7", metric)
    d15 = {}
    for arch, base, pre, full, g in (("transformer", "sup7", "ssl_u", "ssl7", "_7"), ("gru", "gru7", "gru_ssl_u", "gru_ssl7", "_7"),
                                     ("transformer_6", "sup", "ssl_u", "ssl", "_4"), ("gru_6", "gru6", "gru_ssl_u", "gru_ssl6", "_6")):
        d15[arch] = {str(f): dose(metric, base, [f"{pre}{u}{g}" for u in ("125", "250", "500")], full, f) for f in DF}
    r["R2_H15_dose"] = d15
    coh = {}
    for tname in TASKS:
        ps = list(task_of[task_of == tname].index)
        coh[tname] = {"ssl_self_vs_sup7": fam("ssl_self", "sup7", metric, LOW, S5, ps),
                      "gru_ssl_self_vs_gru7": fam("gru_ssl_self", "gru7", metric, LOW, S5, ps)}
    r["cohorts"] = coh
    R[metric] = r
out["results"] = R

# conclusions that change between the six- and seven-rate grids
flips = []
for metric in ("r_mean", "r_speech"):
    for k6 in [k for k in R[metric] if k.startswith("R2_") and k.endswith("_6")]:
        k7 = k6[:-2] + "_7"
        for f in R[metric][k6]:
            a, b = R[metric][k6][f], R[metric][k7][f]
            if (a["p_holm"] < 0.05) != (b["p_holm"] < 0.05) or np.sign(a["delta"]) != np.sign(b["delta"]):
                flips.append([metric, k7, f, a["delta"], a["p_holm"], b["delta"], b["p_holm"]])
    for arch in ("transformer", "gru"):
        for f in map(str, DF):
            a, b = R[metric]["R2_H15_dose"][arch + "_6"][f], R[metric]["R2_H15_dose"][arch][f]
            if (a["p"] < 0.05) != (b["p"] < 0.05):
                flips.append([metric, f"H15_{arch}", f, a["p"], a["p"], b["p"], b["p"]])
out["R2_flips"] = flips

TAB = {}
TCONDS = ["sup7", "ssl7", "placebo7", "gru7", "gru_ssl7", "gru_placebo7", "lpft_ssl7", "lpft_gru_ssl7", "ssl_self",
          "gru_ssl_self", "ridge", "probe_ssl", "bb_ridge", "bb16_ridge", "bb32_ridge", "bb64_ridge", "bb_best_ridge",
          "bb_gru7", "bb64_gru7", "bb_best_gru"]
for m in ("r_mean", "r_speech", "mcd_speech_db"):
    TAB[m] = {}
    for c in TCONDS:
        W = wide(m, c)
        TAB[m][c] = {str(f): float(W[f].mean()) for f in FR if f in W and W[f].notna().any()}
out["table"] = TAB
EDGE = ["placebo7", "lpft_ssl7", "gru7", "gru_ssl7", "gru_placebo7", "lpft_gru_ssl7", "bb_gru7", "bb64_gru7",
        "ssl_self", "gru_ssl_self", "sup7", "ssl7"] + [f"ssl_u{u}_7" for u in ("125", "250", "500")] + \
       [f"gru_ssl_u{u}_7" for u in ("125", "250", "500")]
out["share_top_lr"] = {c: {str(f): float(np.isclose(SELFT[(SELFT.condition == c) & (SELFT.fraction == f)].lr, 1e-2).mean())
                           for f in FR if ((SELFT.condition == c) & (SELFT.fraction == f)).any()} for c in EDGE}
bk = SELFT[SELFT.condition.isin(["bb_best_ridge", "bb_best_gru"])]
out["bb_k_chosen"] = {c: {str(f): bk[(bk.condition == c) & (bk.fraction == f)].bb_k.value_counts().sort_index().to_dict()
                          for f in FR} for c in ("bb_best_ridge", "bb_best_gru")}
bbv = {}
for f in sorted(glob(os.path.join(P4B, "brainbert_k64", "P*.json"))):
    j = json.load(open(f))
    if j["participant"] not in EXCL:
        bbv[j["participant"]] = {k: v["mean"] for k, v in j["pca_explained_at"].items()}
V = pd.DataFrame(bbv).T
out["bb_variance_kept"] = {k: [float(V[k].min()), float(V[k].median()), float(V[k].max())] for k in V}
pre = df[(df.kind == "pretrain") & df.condition.isin(["ssl_self", "gru_ssl_self"])]
out["self_pretrain_seconds"] = pre.groupby(["condition", "fraction"]).unlabelled_seconds.median().to_dict()
out["self_pretrain_seconds"] = {f"{k[0]}@{k[1]}": v for k, v in out["self_pretrain_seconds"].items()}

# ---------------- E6 label efficiency (descriptive)
def equivalent(sup_curve, score):
    """Supervised budget (fraction) with the same score: linear interpolation of score against
    log2(fraction) over the monotone envelope of the supervised curve."""
    fr = np.array(sorted(FR))
    s = np.maximum.accumulate(np.array([sup_curve[f] for f in fr]))
    if score >= s[-1]:
        return 1.0, "ge"
    if score <= s[0]:
        return float(fr[0]), "le"
    lf = np.interp(score, s, np.log2(fr))
    return float(2 ** lf), "in"


E6 = {}
for metric in ("r_mean", "r_speech"):
    E6[metric] = {}
    for arch, base, own in (("transformer", "sup7", "ssl7"), ("gru", "gru7", "gru_ssl7"),
                            ("transformer_self", "sup7", "ssl_self"), ("gru_self", "gru7", "gru_ssl_self")):
        B, O = wide(metric, base), wide(metric, own)
        res = {}
        for f in LOW:
            ratios, flags = [], []
            for p in PIDS:
                if pd.isna(O.loc[p, f]) or B.loc[p].isna().any():
                    continue
                eq, flag = equivalent(B.loc[p], O.loc[p, f])
                ratios.append(eq / f)
                flags.append(flag)
            ratios = np.array(ratios)
            bs = boot(ratios, fn=np.median)
            res[str(f)] = {"median_ratio": float(np.median(ratios)), "ci95": [float(np.percentile(bs, 2.5)),
                                                                             float(np.percentile(bs, 97.5))],
                           "n_ge_100": int(sum(x == "ge" for x in flags)), "n_le_5": int(sum(x == "le" for x in flags)),
                           "n": len(ratios)}
        E6[metric][arch] = res
out["E6_label_efficiency"] = E6

# ---------------- E5 evaluation suite (eval4b; Phase 3 eval for sup/ssl/ridge/probe)
rows = []
REFS = ("timing_reference", "train_mean", "shifted_truth", "true_spectrogram")
for src in (os.path.join(P4B, "eval4b"), os.path.join(P3, "eval")):
    for f in sorted(glob(os.path.join(src, "P*.json"))):
        j = json.load(open(f))
        if j["participant"] in EXCL:
            continue
        for row in j["rows"]:
            if src.endswith(os.sep + "eval") and (row["condition"] in REFS or row["condition"] in ("gru", "gru_ssl")):
                continue
            rows.append(row)
E = pd.DataFrame(rows)
E["fr"] = E.fraction.fillna(-1.0)


def ewide(metric, cond, seeds=S3):
    x = E[(E.condition == cond) & (E.seed.isin(seeds) | E.seed.isna())]
    return x.groupby(["participant", "fr"])[metric].mean().unstack("fr").reindex(PIDS)


NEW = ["gru", "gru_ssl", "lpft_ssl", "lpft_gru_ssl", "bb_gru", "bb64_gru", "bb64_ridge", "ssl_self", "gru_ssl_self"]
ev = {}
for m in ("r_speech", "r_loud", "r_shape"):
    ev[f"E5_{m}"] = {}
    for c in NEW:
        obs, ch = ewide(m, c), ewide(f"{m}_chance_p95", c)
        fr_c = [f for f in FR if f in obs and obs[f].notna().any()]
        res = {}
        for f in fr_c:
            d = (obs[f] - ch[f]).dropna()
            w = stats.wilcoxon(d, alternative="greater")
            res[str(f)] = {"mean_r": float(obs[f].mean()), "chance": float(ch[f].mean()), "n_pos": int((d > 0).sum()),
                           "p": float(w.pvalue)}
        for f, q in zip(fr_c, holm([res[str(f)]["p"] for f in fr_c])):
            res[str(f)]["p_holm"] = q
        ev[f"E5_{m}"][c] = res
for m in ("estoi", "stoi"):
    floor = ewide(m, "shifted_truth", [1])[-1.0]
    res = {}
    for f in FR:
        rr = {}
        for c in NEW + ["sup", "ssl", "ridge", "probe_ssl"]:
            v = ewide(m, c, [1])
            if f not in v or v[f].isna().all():
                continue
            d = (v[f] - floor).dropna()
            w = stats.wilcoxon(d, alternative="greater")
            rr[c] = {"mean": float(v[f].mean()), "delta": float(d.mean()), "n_pos": int((d > 0).sum()), "p": float(w.pvalue)}
        res[str(f)] = rr
    ev[f"E5_{m}"] = res
ev["contrasts"] = {}
for m in ("r_shape", "r_loud", "r_speech", "estoi", "mcd_speech_db"):
    for a, b in (("ssl_self", "sup"), ("gru_ssl_self", "gru"), ("ssl", "ssl_self"), ("gru_ssl", "gru_ssl_self"),
                 ("lpft_gru_ssl", "gru_ssl"), ("lpft_ssl", "ssl"), ("lpft_gru_ssl", "ridge"), ("gru_ssl", "gru")):
        seeds = [1] if m == "estoi" else S3
        A, B = ewide(m, a, seeds), ewide(m, b, seeds)
        fr_c = [f for f in FR if f in A and f in B and A[f].notna().any() and B[f].notna().any()]
        res = {str(f): paired(A[f], B[f]) for f in fr_c}
        for f, q in zip(fr_c, holm([res[str(f)]["p"] for f in fr_c])):
            res[str(f)]["p_holm"] = q
        ev["contrasts"][f"{m}:{a}-{b}"] = res
out["eval"] = ev
json.dump(out, open(OUT, "w"), indent=1, default=float)


# ---------------- printout
def show(metric, name, rr):
    print(f"\n{metric} | {name}")
    for f, x in rr.items():
        print(f"  {float(f):4.2f}: {x['mean_a']:.4f} vs {x['mean_b']:.4f} d={x['delta']:+.4f} "
              f"CI[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}] {x['n_pos']:2d}/{x['n']} rb={x['rank_biserial']:+.2f} "
              f"p={x['p']:.3g} holm={x['p_holm']:.3g}")


print("N =", len(PIDS), "excluded", EXCL, out["provenance_p4b"])
for metric in ("r_mean", "r_speech"):
    for name, v in R[metric].items():
        if name in ("cohorts", "dose_with_self_seven_rates", "R2_H15_dose", "oracle_bb_vs_own"):
            continue
        show(metric, name, v)
    for name, v in R[metric]["oracle_bb_vs_own"].items():
        show(metric, f"ORACLE best BrainBERT on test vs {name}", v)
    for key in ("dose_with_self_seven_rates", "R2_H15_dose"):
        for arch, res in R[metric][key].items():
            for f, x in res.items():
                print(f"{metric} {key} {arch} @{f}: {x['conds']} gains {np.round(x['gain_over_base'], 4).tolist()} "
                      f"Page p={x['p']:.3g}")
    print(f"\n{metric} COHORTS:")
    for t in TASKS:
        for name, v in R[metric]["cohorts"][t].items():
            print(f"  {t:15s} {name:22s}", " ".join(f"{float(f):.2f}:{x['delta']:+.4f}({x['n_pos']},{x['p']:.2f})"
                                                     for f, x in v.items()))
print("\nR2 flips (metric, test, budget, d6, p6, d7, p7):")
for x in flips:
    print("  ", x)
print("\nTABLE")
for m, t in TAB.items():
    print(m)
    for c, v in t.items():
        print(f"  {c:14s}", " ".join(f"{float(f):.2f}:{x:.4f}" for f, x in v.items()))
print("\nshare at 1e-2:", json.dumps({c: {f: round(v, 2) for f, v in d.items()} for c, d in out["share_top_lr"].items()}))
print("BrainBERT k chosen:", json.dumps(out["bb_k_chosen"]))
print("variance kept [min, median, max]:", json.dumps(out["bb_variance_kept"]))
print("self-pretraining seconds (median):", out["self_pretrain_seconds"])
print("\nE6 label efficiency (median ratio of equivalent supervised budget to the budget):")
for metric, d in E6.items():
    for arch, res in d.items():
        print(f"  {metric:8s} {arch:17s}", " ".join(f"{f}: x{x['median_ratio']:.2f} [{x['ci95'][0]:.2f},{x['ci95'][1]:.2f}] "
                                                  f"(>=100%: {x['n_ge_100']}, <=5%: {x['n_le_5']})" for f, x in res.items()))
for m in ("r_speech", "r_loud", "r_shape"):
    print(f"\nE5 {m} (mean | chance | n above | Holm p)")
    for c, res in ev[f"E5_{m}"].items():
        print(f"  {c:13s}", " ".join(f"{float(f):.2f}:{x['mean_r']:.3f}|{x['chance']:.3f}|{x['n_pos']}|{x['p_holm']:.2g}"
                                     for f, x in res.items()))
for m in ("estoi", "stoi"):
    print(f"\nE5 {m} (mean, n above floor, p):")
    for f, rr in ev[f"E5_{m}"].items():
        print(f"  {f}:", {c: (round(x["mean"], 3), x["n_pos"], round(x["p"], 3)) for c, x in rr.items()})
print("\ncontrasts (delta, n_pos, Holm p):")
for k, v in ev["contrasts"].items():
    print(" ", k, {f: (round(x["delta"], 4), x["n_pos"], round(x["p_holm"], 4)) for f, x in v.items()})
