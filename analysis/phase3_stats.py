"""Pre-specified Phase 3 analysis (docs/PHASE3_DESIGN.md).

Usage: python -I stats3.py <p1_dir> <p2_dir> <p3_dir> <out_json> [exclude_pid ...]

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

P1, P2, P3, OUT = sys.argv[1:5]
EXCL = sys.argv[5:]
FR = [1.0, 0.5, 0.25, 0.1, 0.05]
LOW = [0.25, 0.1, 0.05]
LFR = [1.0, 0.25, 0.1, 0.05]
G4 = [3e-5, 1e-4, 3e-4, 1e-3]
G6 = [1e-5] + G4 + [3e-3]
S3, S5 = [1, 2, 3], [1, 2, 3, 4, 5]
TASKS = ["unique_words", "repeated_words", "sentences"]

df = pd.concat([pd.read_csv(os.path.join(p, f"phase{i}_results.csv")) for i, p in ((1, P1), (2, P2), (3, P3))],
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
for c, grid, name in [("sup", G4, "sup"), ("ssl", G4, "ssl"), ("placebo", G4, "placebo"),
                      ("sup", G6, "sup_g6"), ("ssl", G6, "ssl_g6"), ("placebo", G6, "placebo_g6"),
                      ("sup", G6 + [1e-2], "sup_g7"), ("ssl", G6 + [1e-2], "ssl_g7"),
                      ("gru", G4, "gru_g4"), ("gru", G6, "gru"), ("gru_ssl", G6, "gru_ssl"),
                      ("gru_placebo", G6, "gru_placebo"), ("pooled", G4, "pooled"), ("loo_adapt", G4, "loo_adapt"),
                      ("sup_L", G4, "sup_L"), ("ssl_L", G4, "ssl_L"), ("pooled_L", G4, "pooled_L")]:
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
       "provenance_p3": {"code": sorted(pd.read_csv(os.path.join(P3, "phase3_results.csv")).code_hash.unique().tolist())}}
TAB = {}
for m in ("r_mean", "r_speech", "mcd_speech_db"):
    TAB[m] = {}
    for c in ["sup", "ssl", "sup_g7", "ssl_g7", "gru", "gru_ssl", "gru_placebo", "ridge", "pooled", "sup_L", "ssl_L",
              "pooled_L", "probe_ssl", "probe_gru_random", "probe_gru_placebo", "probe_gru_ssl"]:
        seeds = S3 if c.endswith("_L") else S5
        W = wide(m, c, seeds)
        TAB[m][c] = {str(f): float(W[f].mean()) for f in FR if f in W and W[f].notna().any()}
out["table"] = TAB

R = {}
for metric in ("r_mean", "r_speech"):
    r = {}
    r["H10_gru_ssl_vs_gru"] = fam("gru_ssl", "gru", metric, LOW)
    r["H10b_gru_ssl_vs_gru_placebo"] = fam("gru_ssl", "gru_placebo", metric, LOW)
    r["gru_ssl_vs_gru_all"] = fam("gru_ssl", "gru", metric)
    r["gru_placebo_vs_gru_all"] = fam("gru_placebo", "gru", metric)
    r["gru_ssl_vs_gru_placebo_all"] = fam("gru_ssl", "gru_placebo", metric)
    D = (wide(metric, "gru_ssl") - wide(metric, "gru"))[FR].dropna()
    t = stats.page_trend_test(D.values, predicted_ranks=[1, 2, 3, 4, 5])
    r["page_gru_ssl_minus_gru"] = {"L": float(t.statistic), "p": float(t.pvalue)}
    r["H11_pooledL_vs_sslL"] = fam("pooled_L", "ssl_L", metric, LOW, S3)
    r["pooledL_vs_sslL_all"] = fam("pooled_L", "ssl_L", metric, LFR, S3)
    r["pooledL_vs_pooled"] = fam("pooled_L", "pooled", metric, LFR, S3)
    r["sslL_vs_ssl"] = fam("ssl_L", "ssl", metric, LFR, S3)
    r["supL_vs_sup"] = fam("sup_L", "sup", metric, LFR, S3)
    r["sslL_vs_supL"] = fam("ssl_L", "sup_L", metric, LFR, S3)
    r["pooledL_vs_supL"] = fam("pooled_L", "sup_L", metric, LFR, S3)
    r["H12_ssl_g7_vs_sup_g7"] = fam("ssl_g7", "sup_g7", metric)
    r["sup_g7_vs_sup_g6"] = fam("sup_g7", "sup_g6", metric)
    r["ssl_g7_vs_gru"] = fam("ssl_g7", "gru", metric)
    r["gru_vs_gru_g4"] = fam("gru", "gru_g4", metric)
    r["gru_ssl_vs_ssl_g7"] = fam("gru_ssl", "ssl_g7", metric)
    r["gru_vs_ridge"] = fam("gru", "ridge", metric)
    r["gru_ssl_vs_ridge"] = fam("gru_ssl", "ridge", metric)
    D = (wide(metric, "ssl_g7") - wide(metric, "sup_g7"))[FR].dropna()
    t = stats.page_trend_test(D.values, predicted_ranks=[1, 2, 3, 4, 5])
    r["page_ssl_g7_minus_sup_g7"] = {"L": float(t.statistic), "p": float(t.pvalue)}
    r["H13_probe_gru_ssl_vs_random"] = fam("probe_gru_ssl", "probe_gru_random", metric)
    r["H13_probe_gru_ssl_vs_placebo"] = fam("probe_gru_ssl", "probe_gru_placebo", metric)
    r["probe_gru_placebo_vs_random"] = fam("probe_gru_placebo", "probe_gru_random", metric)
    r["probe_gru_ssl_vs_probe_ssl"] = fam("probe_gru_ssl", "probe_ssl", metric)
    r["probe_gru_ssl_vs_gru_ssl_ft"] = fam("probe_gru_ssl", "gru_ssl", metric)
    coh = {}
    for tname in TASKS:
        ps = list(task_of[task_of == tname].index)
        coh[tname] = {"gru_ssl_vs_gru": fam("gru_ssl", "gru", metric, FR, S5, ps),
                      "pooledL_vs_sslL": fam("pooled_L", "ssl_L", metric, LFR, S3, ps),
                      "ssl_g7_vs_sup_g7": fam("ssl_g7", "sup_g7", metric, FR, S5, ps)}
    r["cohorts"] = coh
    R[metric] = r
out["results"] = R
out["mcd_speech"] = {"gru_ssl_vs_gru": fam("gru_ssl", "gru", "mcd_speech_db", sign=-1),
                     "pooledL_vs_sslL": fam("pooled_L", "ssl_L", "mcd_speech_db", LFR, S3, sign=-1)}


def lrc(name):
    x = SELFT[SELFT.condition == name]
    return {str(f): x[x.fraction == f].lr.value_counts().sort_index().rename(lambda v: f"{v:g}").to_dict() for f in FR}


out["lr_choice"] = {c: lrc(c) for c in ("sup_g7", "ssl_g7", "gru", "gru_ssl", "gru_placebo", "sup_L", "ssl_L", "pooled_L")}

# pretraining curves
pre = df[df.kind.isin(["pretrain", "pretrain_multi"])]
out["pretrain_runs"] = pre.groupby("condition").size().to_dict()
runs = {}
for cond in ("gru_ssl", "gru_placebo", "ssl_L", "pooled_L"):
    curves = []
    for f in glob(os.path.join(P3, "runs", f"*__pretrain_{cond}__*.json")):
        j = json.load(open(f))
        if j.get("participant") in EXCL:
            continue
        c = (j.get("pretrain") or {}).get("loss_curve")
        if c:
            curves.append((c[0], c[-1]))
    if curves:
        a = np.array(curves)
        runs[cond] = {"n": len(a), "first_mean": float(a[:, 0].mean()), "last_mean": float(a[:, 1].mean()),
                      "last_range": [float(a[:, 1].min()), float(a[:, 1].max())]}
out["pretrain_loss"] = runs

# ---------------- evaluation suite
rows = []
for f in sorted(glob(os.path.join(P3, "eval", "P*.json"))):
    j = json.load(open(f))
    if j["participant"] in EXCL:
        continue
    rows += j["rows"]
E = pd.DataFrame(rows)
E["fr"] = E.fraction.fillna(-1.0)
out["eval_n"] = {"participants": int(E.participant.nunique()), "rows": int(len(E))}
EC = ["sup", "ssl", "gru", "gru_ssl", "pooled", "loo_adapt", "ridge", "probe_ssl"]
REFS = ["timing_reference", "train_mean", "shifted_truth", "true_spectrogram"]


def ewide(metric, cond, seeds=S3):
    x = E[(E.condition == cond) & (E.seed.isin(seeds) | E.seed.isna())]
    return x.groupby(["participant", "fr"])[metric].mean().unstack("fr").reindex(PIDS)


ev = {}
# E1: above chance (r - chance p95 > 0), one-sided, Holm across budgets within condition
for m in ("r_speech", "r_loud", "r_shape"):
    ev[f"E1_{m}"] = {}
    for c in EC:
        obs, ch = ewide(m, c), ewide(f"{m}_chance_p95", c)
        res = {}
        for f in FR:
            d = (obs[f] - ch[f]).dropna()
            w = stats.wilcoxon(d, alternative="greater")
            res[str(f)] = {"mean_r": float(obs[f].mean()), "mean_chance_p95": float(ch[f].mean()),
                           "delta": float(d.mean()), "n_pos": int((d > 0).sum()), "n": int(len(d)), "p": float(w.pvalue)}
        for f, q in zip(FR, holm([res[str(f)]["p"] for f in FR])):
            res[str(f)]["p_holm"] = q
        # share of runs with own p < 0.05
        x = E[(E.condition == c)]
        sh = (x[f"{m}_p"] < 0.05).groupby(x.fr).mean()
        for f in FR:
            res[str(f)]["share_runs_p05"] = float(sh.get(f, np.nan))
        ev[f"E1_{m}"][c] = res
# E2: descriptive means + exploratory contrasts on r_shape / r_loud
ev["E2_means"] = {m: {c: {str(f): float(ewide(m, c)[f].mean()) for f in FR} for c in EC} for m in ("r_loud", "r_shape", "r_speech")}
ev["E2_refs"] = {m: {c: float(ewide(m, c)[-1.0].mean()) for c in REFS if -1.0 in ewide(m, c)}
                 for m in ("r_loud", "r_shape", "r_speech", "r_mean", "mcd_speech_db", "estoi", "stoi")}
ev["E2_contrasts"] = {}
for m in ("r_shape", "r_loud"):
    for a, b in (("ssl", "sup"), ("ridge", "ssl"), ("gru_ssl", "gru"), ("probe_ssl", "ssl")):
        A, B = ewide(m, a), ewide(m, b)
        res = {str(f): paired(A[f], B[f]) for f in LOW}
        for f, q in zip(LOW, holm([res[str(f)]["p"] for f in LOW])):
            res[str(f)]["p_holm"] = q
        ev["E2_contrasts"][f"{m}:{a}-{b}"] = res
# E3: ESTOI (and STOI) above the shifted-truth floor, Holm across conditions at each budget (seed 1)
for m in ("estoi", "stoi"):
    floor = ewide(m, "shifted_truth", [1])[-1.0]
    res = {}
    for f in FR:
        rr = {}
        for c in EC:
            v = ewide(m, c, [1])[f]
            d = (v - floor).dropna()
            w = stats.wilcoxon(d, alternative="greater")
            rr[c] = {"mean": float(v.mean()), "floor": float(floor.mean()), "delta": float(d.mean()),
                     "n_pos": int((d > 0).sum()), "p": float(w.pvalue)}
        for c, q in zip(EC, holm([rr[c]["p"] for c in EC])):
            rr[c]["p_holm"] = q
        res[str(f)] = rr
    ev[f"E3_{m}"] = res
# also: ESTOI vs train-mean and vs timing reference for the best condition (descriptive)
pp = E[E.condition == "probe_ssl"]
ev["probe_check_max_abs_diff"] = float((pp.r_mean_here - pp.check_r_mean_phase2).abs().max())
out["eval"] = ev
json.dump(out, open(OUT, "w"), indent=1, default=float)


# ---------------- print
def show(metric, name, rr):
    print(f"\n{metric} | {name}")
    for f, x in rr.items():
        print(f"  {float(f):4.2f}: {x['mean_a']:.4f} vs {x['mean_b']:.4f} d={x['delta']:+.4f} "
              f"CI[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}] {x['n_pos']:2d}/{x['n']} rb={x['rank_biserial']:+.2f} "
              f"p={x['p']:.3g} holm={x['p_holm']:.3g}")


print("N =", len(PIDS), "excluded", EXCL, out["provenance_p3"])
for metric in ("r_mean", "r_speech"):
    for name, v in R[metric].items():
        if name == "cohorts":
            continue
        if name.startswith("page"):
            print(f"\n{metric} | {name}: {v}")
            continue
        show(metric, name, v)
    print(f"\n{metric} COHORTS:")
    for t in TASKS:
        for name, v in R[metric]["cohorts"][t].items():
            print(f"  {t:15s} {name:18s}", " ".join(f"{float(f):.2f}:{x['delta']:+.4f}({x['n_pos']},{x['p']:.2f})" for f, x in v.items()))
print("\nMCD speech:")
for k, v in out["mcd_speech"].items():
    print(" ", k, {f: (round(x["delta"], 3), x["n_pos"], round(x["p_holm"], 3)) for f, x in v.items()})
print("\nLR choice:", json.dumps(out["lr_choice"]))
print("\npretrain loss:", json.dumps(out["pretrain_loss"]))
print("\nTABLE")
for m, t in TAB.items():
    print(m)
    for c, v in t.items():
        print(f"  {c:18s}", " ".join(f"{float(f):.2f}:{x:.4f}" for f, x in v.items()))
print("\nEVAL", out["eval_n"], "probe check", ev["probe_check_max_abs_diff"])
for m in ("r_speech", "r_loud", "r_shape"):
    print(f"\nE1 {m} (mean r | chance p95 | delta | n_pos | holm | share runs p<.05)")
    for c, res in ev[f"E1_{m}"].items():
        print(f"  {c:10s}", " ".join(f"{float(f):.2f}:{x['mean_r']:.3f}|{x['mean_chance_p95']:.3f}|{x['delta']:+.3f}|{x['n_pos']}|{x['p_holm']:.2g}|{x['share_runs_p05']:.2f}" for f, x in res.items()))
print("\nE2 refs:", json.dumps(ev["E2_refs"], indent=0))
print("\nE2 contrasts:")
for k, v in ev["E2_contrasts"].items():
    print(" ", k, {f: (round(x["delta"], 4), x["n_pos"], round(x["p_holm"], 4)) for f, x in v.items()})
for m in ("estoi", "stoi"):
    print(f"\nE3 {m}:")
    for f, rr in ev[f"E3_{m}"].items():
        print(f"  {f}:", {c: (round(x["mean"], 3), round(x["delta"], 3), x["n_pos"], round(x["p_holm"], 3)) for c, x in rr.items()})
