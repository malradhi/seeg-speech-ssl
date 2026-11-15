"""Pre-specified Phase 4 analysis (docs/PHASE4_DESIGN.md).

Usage: python -I phase4_stats.py <p1_dir> <p2_dir> <p3_dir> <p4_dir> <out_json> [exclude_pid ...]

Selection: best learning rate per (participant, condition, fraction, seed) on validation MSE
over the grid named in the design (gru: six rates over Phases 2-3; sup/ssl: seven rates over
Phases 1-3; large and pooled models: four rates). Participant score = mean over seeds;
comparisons with a three-seed condition use seeds 1-3 of both arms.
"""
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats

P1, P2, P3, P4, OUT = sys.argv[1:6]
EXCL = sys.argv[6:]
FR = [1.0, 0.5, 0.25, 0.1, 0.05]
LOW = [0.25, 0.1, 0.05]
LFR = [1.0, 0.25, 0.1, 0.05]
G4 = [3e-5, 1e-4, 3e-4, 1e-3]
G6 = [1e-5] + G4 + [3e-3]
S3, S5 = [1, 2, 3], [1, 2, 3, 4, 5]
TASKS = ["unique_words", "repeated_words", "sentences"]

df = pd.concat([pd.read_csv(os.path.join(p, f"phase{i}_results.csv")) for i, p in ((1, P1), (2, P2), (3, P3), (4, P4))],
               ignore_index=True)
df = df[~df.participant.isin(EXCL)]
task_of = df.dropna(subset=["task"]).groupby("participant").task.first()
PIDS = sorted(task_of.index)
ft = df[(df.kind == "finetune") & (df.protocol == "new")].copy()
KEY = ["participant", "condition", "fraction", "seed"]


def inset(lr, grid):
    return np.isclose(lr.values[:, None], np.array(grid)[None, :]).any(1)


def select(frame, name=None):
    s = frame.loc[frame.groupby(KEY).val_mse.idxmin()].copy()
    if name:
        s["condition"] = name
    return s


parts = []
for c, grid, name in ([("sup", G4, "sup"), ("ssl", G4, "ssl"), ("placebo", G4, "placebo"),
                      ("sup", G6, "sup_g6"), ("ssl", G6, "ssl_g6"), ("placebo", G6, "placebo_g6"),
                      ("sup", G6 + [1e-2], "sup_g7"), ("ssl", G6 + [1e-2], "ssl_g7"),
                      ("gru", G4, "gru_g4"), ("gru", G6, "gru"), ("gru_ssl", G6, "gru_ssl"),
                      ("gru_placebo", G6, "gru_placebo"), ("pooled", G4, "pooled"), ("loo_adapt", G4, "loo_adapt"),
                      ("sup_L", G4, "sup_L"), ("ssl_L", G4, "ssl_L"), ("pooled_L", G4, "pooled_L"),
                      ("bb_gru", G6, "bb_gru"), ("lpft_ssl", G6, "lpft_ssl"), ("lpft_gru_ssl", G6, "lpft_gru_ssl")]
                     + [(f"ssl_u{u}", G4, f"ssl_u{u}") for u in ("125", "250", "500")]
                     + [(f"gru_ssl_u{u}", G6, f"gru_ssl_u{u}") for u in ("125", "250", "500")]):
    x = ft[(ft.condition == c) & inset(ft.lr, grid)]
    parts.append(select(x, name))
SELFT = pd.concat(parts, ignore_index=True)
SEL = pd.concat([SELFT, df[df.kind.isin(["linear", "probe"])]], ignore_index=True)


def wide(metric, cond, seeds=S5, src=None):
    src = SEL if src is None else src
    x = src[(src.condition == cond) & src.seed.isin(seeds)]
    return x.groupby(["participant", "fraction"])[metric].mean().unstack("fraction").reindex(PIDS)


def boot(x, n=10000, seed=1):
    return np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)


def paired(a, b, alternative="two-sided"):
    d = (a - b).dropna()
    w = stats.wilcoxon(d, alternative=alternative)
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


def fam(a, b, metric, fracs=FR, seeds=S5, pids=None, sign=1):
    A, B = wide(metric, a, seeds), wide(metric, b, seeds)
    if pids is not None:
        A, B = A.loc[pids], B.loc[pids]
    if sign < 0:
        A, B = -A, -B
    res = {f: paired(A[f], B[f]) for f in fracs}
    for f, q in zip(fracs, holm([res[f]["p"] for f in fracs])):
        res[f]["p_holm"] = q
    return {str(f): v for f, v in res.items()}



out = {"n_participants": len(PIDS), "excluded": EXCL,
       "provenance_p4": sorted(pd.read_csv(os.path.join(P4, "phase4_results.csv")).code_hash.unique().tolist())}
DF = [0.1, 0.05]
R = {}
for metric in ("r_mean", "r_speech"):
    r = {}
    # H14 published encoder
    r["H14a_gru_ssl_vs_bb_gru"] = fam("gru_ssl", "bb_gru", metric, LOW)
    r["H14b_probe_ssl_vs_bb_ridge"] = fam("probe_ssl", "bb_ridge", metric, LOW)
    r["gru_ssl_vs_bb_gru_all"] = fam("gru_ssl", "bb_gru", metric)
    r["probe_ssl_vs_bb_ridge_all"] = fam("probe_ssl", "bb_ridge", metric)
    r["bb_ridge_vs_ridge"] = fam("bb_ridge", "ridge", metric)
    r["bb_gru_vs_gru"] = fam("bb_gru", "gru", metric)
    r["bb_gru_vs_sup"] = fam("bb_gru", "sup", metric)
    # H15 dose-response (seeds 1-3)
    dose = {}
    for arch, base, full, pre in (("transformer", "sup", "ssl", "ssl_u"), ("gru", "gru", "gru_ssl", "gru_ssl_u")):
        conds = [base, pre + "125", pre + "250", pre + "500", full]
        res = {}
        for f in DF:
            M = pd.concat([wide(metric, c, S3)[f] for c in conds], axis=1, keys=conds).dropna()
            t = stats.page_trend_test(M.values, predicted_ranks=[1, 2, 3, 4, 5])
            res[str(f)] = {"conds": conds, "means": [float(v) for v in M.mean()],
                           "gain_over_base": [float(v - M[base].mean()) for v in M.mean()],
                           "L": float(t.statistic), "p": float(t.pvalue), "n": int(len(M)),
                           "full_vs_u125": paired(M[full], M[pre + "125"]),
                           "u125_vs_base": paired(M[pre + "125"], M[base])}
        dose[arch] = res
    r["H15_dose"] = dose
    # probes along the dose (u=0: random encoder)
    pd_ = {}
    for f in DF:
        conds = ["probe_random", "probe_ssl_u125", "probe_ssl_u250", "probe_ssl_u500", "probe_ssl"]
        M = pd.concat([wide(metric, c, S3)[f] for c in conds], axis=1, keys=conds).dropna()
        t = stats.page_trend_test(M.values, predicted_ranks=[1, 2, 3, 4, 5])
        pd_[str(f)] = {"means": [float(v) for v in M.mean()], "L": float(t.statistic), "p": float(t.pvalue)}
    r["probe_dose_transformer"] = pd_
    # H16 LP-FT
    r["H16_lpft_ssl_vs_ssl"] = fam("lpft_ssl", "ssl_g6", metric, LOW)
    r["H16_lpft_gru_ssl_vs_gru_ssl"] = fam("lpft_gru_ssl", "gru_ssl", metric, LOW)
    r["lpft_ssl_vs_ssl_all"] = fam("lpft_ssl", "ssl_g6", metric)
    r["lpft_gru_ssl_vs_gru_ssl_all"] = fam("lpft_gru_ssl", "gru_ssl", metric)
    r["lpft_ssl_vs_probe_ssl"] = fam("lpft_ssl", "probe_ssl", metric)
    r["lpft_gru_ssl_vs_probe_ssl"] = fam("lpft_gru_ssl", "probe_ssl", metric)
    r["lpft_gru_ssl_vs_ridge"] = fam("lpft_gru_ssl", "ridge", metric)
    r["lpft_gru_ssl_vs_lpft_ssl"] = fam("lpft_gru_ssl", "lpft_ssl", metric)
    r["lpft_gru_ssl_vs_gru"] = fam("lpft_gru_ssl", "gru", metric)
    r["lpft_ssl_vs_sup_g7"] = fam("lpft_ssl", "sup_g7", metric)
    D = (wide(metric, "lpft_gru_ssl") - wide(metric, "gru"))[FR].dropna()
    t = stats.page_trend_test(D.values, predicted_ranks=[1, 2, 3, 4, 5])
    r["page_lpft_gru_ssl_minus_gru"] = {"L": float(t.statistic), "p": float(t.pvalue)}
    coh = {}
    for tname in TASKS:
        ps = list(task_of[task_of == tname].index)
        coh[tname] = {"gru_ssl_vs_bb_gru": fam("gru_ssl", "bb_gru", metric, FR, S5, ps),
                      "lpft_gru_ssl_vs_gru_ssl": fam("lpft_gru_ssl", "gru_ssl", metric, FR, S5, ps)}
    r["cohorts"] = coh
    R[metric] = r
out["results"] = R
TAB = {}
for m in ("r_mean", "r_speech", "mcd_speech_db"):
    TAB[m] = {}
    for c in ["sup", "ssl", "ssl_g6", "gru", "gru_ssl", "ridge", "probe_ssl", "probe_random", "bb_ridge", "bb_gru",
              "lpft_ssl", "lpft_gru_ssl"]:
        W = wide(m, c)
        TAB[m][c] = {str(f): float(W[f].mean()) for f in FR if f in W and W[f].notna().any()}
    for c in ["ssl_u125", "ssl_u250", "ssl_u500", "gru_ssl_u125", "gru_ssl_u250", "gru_ssl_u500",
              "probe_ssl_u125", "probe_ssl_u250", "probe_ssl_u500"]:
        W = wide(m, c, S3)
        TAB[m][c] = {str(f): float(W[f].mean()) for f in DF if f in W}
out["table"] = TAB
out["lr_choice"] = {c: {str(f): SELFT[(SELFT.condition == c) & (SELFT.fraction == f)].lr.value_counts().sort_index()
                        .rename(lambda v: f"{v:g}").to_dict() for f in FR} for c in ("bb_gru", "lpft_ssl", "lpft_gru_ssl")}
lp = df[df.condition.isin(["lpft_ssl", "lpft_gru_ssl"]) & (df.kind == "finetune")]
if "lp_steps_run" in lp:
    out["lp_steps_run"] = lp.groupby("condition").lp_steps_run.describe().to_dict()
pre = df[(df.kind == "pretrain") & df.condition.str.contains("_u", na=False)]
if "unlabelled_seconds" in pre:
    out["dose_unlabelled_seconds"] = pre.groupby("condition").unlabelled_seconds.median().to_dict()

# ---------------- evaluation (eval4 + Phase 3 eval for the plain fine-tuned models)
rows = []
for src in (os.path.join(P4, "eval4"), os.path.join(P3, "eval")):
    for f in sorted(glob(os.path.join(src, "P*.json"))):
        j = json.load(open(f))
        if j["participant"] in EXCL:
            continue
        for row in j["rows"]:
            if src.endswith("eval") and row["condition"] in ("timing_reference", "train_mean", "shifted_truth", "true_spectrogram"):
                continue  # references once (eval4 holds them too)
            rows.append(row)
E = pd.DataFrame(rows)
E["fr"] = E.fraction.fillna(-1.0)


def ewide(metric, cond, seeds=S3):
    x = E[(E.condition == cond) & (E.seed.isin(seeds) | E.seed.isna())]
    return x.groupby(["participant", "fr"])[metric].mean().unstack("fr").reindex(PIDS)


NEW = ["lpft_ssl", "lpft_gru_ssl", "bb_ridge", "bb_gru"]
ev = {}
for m in ("r_speech", "r_loud", "r_shape"):
    ev[f"E4_{m}"] = {}
    for c in NEW:
        obs, ch = ewide(m, c), ewide(f"{m}_chance_p95", c)
        res = {}
        for f in FR:
            d = (obs[f] - ch[f]).dropna()
            w = stats.wilcoxon(d, alternative="greater")
            res[str(f)] = {"mean_r": float(obs[f].mean()), "chance": float(ch[f].mean()), "n_pos": int((d > 0).sum()),
                           "p": float(w.pvalue)}
        for f, q in zip(FR, holm([res[str(f)]["p"] for f in FR])):
            res[str(f)]["p_holm"] = q
        x = E[E.condition == c]
        sh = (x[f"{m}_p"] < 0.05).groupby(x.fr).mean()
        for f in FR:
            res[str(f)]["share_runs_p05"] = float(sh.get(f, np.nan))
        ev[f"E4_{m}"][c] = res
for m in ("estoi", "stoi"):
    floor = ewide(m, "shifted_truth", [1])[-1.0]
    res = {}
    for f in FR:
        rr = {}
        for c in NEW + ["ssl", "gru_ssl", "ridge", "probe_ssl"]:
            v = ewide(m, c, [1])[f]
            d = (v - floor).dropna()
            w = stats.wilcoxon(d, alternative="greater")
            rr[c] = {"mean": float(v.mean()), "delta": float(d.mean()), "n_pos": int((d > 0).sum()), "p": float(w.pvalue)}
        for c, q in zip(NEW, holm([rr[c]["p"] for c in NEW])):
            rr[c]["p_holm"] = q
        res[str(f)] = rr
    ev[f"E4_{m}"] = res
ev["contrasts"] = {}
for m in ("r_shape", "r_loud", "r_speech", "estoi"):
    for a, b in (("lpft_ssl", "ssl"), ("lpft_gru_ssl", "gru_ssl"), ("lpft_gru_ssl", "ridge"), ("lpft_gru_ssl", "probe_ssl")):
        seeds = [1] if m == "estoi" else S3
        A, B = ewide(m, a, seeds), ewide(m, b, seeds)
        res = {str(f): paired(A[f], B[f]) for f in FR}
        for f, q in zip(FR, holm([res[str(f)]["p"] for f in FR])):
            res[str(f)]["p_holm"] = q
        ev["contrasts"][f"{m}:{a}-{b}"] = res
out["eval"] = ev

# ---------------- M1 moderators (exploratory)
import re
SPEECH = re.compile(r"temp_sup|temporal_sup|precentral|postcentral|S_central|front_inf|insula|insular|Supramar|transv|Plan_tempo|Plan_polar", re.I)
anat = pd.concat([pd.read_csv(f) for f in sorted(glob(os.path.join(P4, "anatomy", "P*.csv")))], ignore_index=True)
anat = anat[anat.participant.isin(PIDS)]
anat["PTD"] = pd.to_numeric(anat.PTD, errors="coerce")
mod = pd.DataFrame(index=PIDS)
mod["n_contacts"] = anat.groupby("participant").size()
mod["grey_share"] = anat.groupby("participant").PTD.apply(lambda v: float((v > 0).mean()))
mod["n_speech_cortex"] = anat.groupby("participant").location.apply(lambda v: int(v.astype(str).str.contains(SPEECH).sum()))
ls = df[(df.kind == "finetune") & (df.condition == "sup") & (df.fraction == 0.1)].groupby("participant").label_seconds.first()
mod["label_seconds_10"] = ls
mod["sup_score_10"] = wide("r_mean", "sup")[0.1]
mod["gru_score_10"] = wide("r_mean", "gru")[0.1]
gains = {"ssl_minus_sup_10": (wide("r_mean", "ssl") - wide("r_mean", "sup"))[0.1],
         "gru_ssl_minus_gru_10": (wide("r_mean", "gru_ssl") - wide("r_mean", "gru"))[0.1],
         "ssl_minus_sup_10_speech": (wide("r_speech", "ssl") - wide("r_speech", "sup"))[0.1],
         "gru_ssl_minus_gru_10_speech": (wide("r_speech", "gru_ssl") - wide("r_speech", "gru"))[0.1]}
tests = []
for g, gv in gains.items():
    for cov in ["n_contacts", "grey_share", "n_speech_cortex", "label_seconds_10", "sup_score_10"]:
        rho, p = stats.spearmanr(gv.reindex(PIDS), mod[cov], nan_policy="omit")
        tests.append([g, cov, float(rho), float(p)])
ps = np.array([t[3] for t in tests])
order = np.argsort(ps); m_ = len(ps); q = np.empty(m_); prev = 1.0
for rank, i in enumerate(order[::-1]):
    k = m_ - rank
    prev = min(prev, ps[i] * m_ / k); q[i] = prev
for t, qq in zip(tests, q):
    t.append(float(qq))
out["M1"] = {"tests": tests, "moderators": mod.describe().to_dict()}
json.dump(out, open(OUT, "w"), indent=1, default=float)


def show(metric, name, rr):
    print(f"\n{metric} | {name}")
    for f, x in rr.items():
        print(f"  {float(f):4.2f}: {x['mean_a']:.4f} vs {x['mean_b']:.4f} d={x['delta']:+.4f} "
              f"CI[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}] {x['n_pos']:2d}/{x['n']} rb={x['rank_biserial']:+.2f} "
              f"p={x['p']:.3g} holm={x['p_holm']:.3g}")


print("N =", len(PIDS), "excluded", EXCL, out["provenance_p4"])
for metric in ("r_mean", "r_speech"):
    for name, v in R[metric].items():
        if name in ("cohorts", "H15_dose", "probe_dose_transformer") or name.startswith("page"):
            continue
        show(metric, name, v)
    print(f"\n{metric} page lpft_gru_ssl - gru:", R[metric]["page_lpft_gru_ssl_minus_gru"])
    for arch, res in R[metric]["H15_dose"].items():
        for f, x in res.items():
            print(f"\n{metric} DOSE {arch} @{f}: gains {np.round(x['gain_over_base'], 4).tolist()} Page L={x['L']:.1f} p={x['p']:.4g}; "
                  f"100 vs 12.5: d={x['full_vs_u125']['delta']:+.4f} p={x['full_vs_u125']['p']:.3g} ({x['full_vs_u125']['n_pos']}/30); "
                  f"12.5 vs base: d={x['u125_vs_base']['delta']:+.4f} p={x['u125_vs_base']['p']:.3g} ({x['u125_vs_base']['n_pos']}/30)")
    print(f"{metric} probe dose:", {f: (np.round(v['means'], 4).tolist(), round(v['p'], 5)) for f, v in R[metric]["probe_dose_transformer"].items()})
    print(f"\n{metric} COHORTS:")
    for t in TASKS:
        for name, v in R[metric]["cohorts"][t].items():
            print(f"  {t:15s} {name:24s}", " ".join(f"{float(f):.2f}:{x['delta']:+.4f}({x['n_pos']},{x['p']:.2f})" for f, x in v.items()))
print("\nTABLE")
for m, t in TAB.items():
    print(m)
    for c, v in t.items():
        print(f"  {c:16s}", " ".join(f"{float(f):.2f}:{x:.4f}" for f, x in v.items()))
print("\nLR:", json.dumps(out["lr_choice"]))
print("LP steps:", json.dumps(out.get("lp_steps_run")))
print("dose seconds:", out.get("dose_unlabelled_seconds"))
for m in ("r_speech", "r_loud", "r_shape"):
    print(f"\nE4 {m}")
    for c, res in ev[f"E4_{m}"].items():
        print(f"  {c:13s}", " ".join(f"{float(f):.2f}:{x['mean_r']:.3f}|{x['chance']:.3f}|{x['n_pos']}|{x['p_holm']:.2g}|{x['share_runs_p05']:.2f}" for f, x in res.items()))
for m in ("estoi", "stoi"):
    print(f"\nE4 {m}:")
    for f, rr in ev[f"E4_{m}"].items():
        print(f"  {f}:", {c: (round(x["mean"], 3), x["n_pos"], round(x.get("p_holm", x["p"]), 3)) for c, x in rr.items()})
print("\ncontrasts:")
for k, v in ev["contrasts"].items():
    print(" ", k, {f: (round(x["delta"], 4), x["n_pos"], round(x["p_holm"], 4)) for f, x in v.items()})
print("\nM1 (gain, covariate, rho, p, q):")
for t in out["M1"]["tests"]:
    print("  ", t[0], t[1], round(t[2], 3), round(t[3], 4), round(t[4], 3))
