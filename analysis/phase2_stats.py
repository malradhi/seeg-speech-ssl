"""Pre-specified Phase 2 analysis (docs/PHASE2_DESIGN.md).

Usage: python -I phase2_stats.py <phase1_bundle_dir> <phase2_bundle_dir> <out_json> [exclude_pid ...]

Selection rules (fixed before looking at Phase 2 results):
* deep fine-tunes: best LR per (participant, condition, fraction, seed) on validation MSE;
  "grid4" = Phase 1 grid (3e-5..1e-3, used by every Phase 2 condition),
  "grid6" = grid4 + 1e-5 and 3e-3 (sup/ssl/placebo only; H8).
* probes and ridge: alpha already chosen on validation inside each run.
* participant score = mean over seeds; comparisons with a 3-seed condition use seeds 1-3 of
  both arms (identical label subsets per seed).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

P1, P2, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EXCL = sys.argv[4:]
FR = [1.0, 0.5, 0.25, 0.1, 0.05]
LOW = [0.25, 0.1, 0.05]
GRID4 = [3e-5, 1e-4, 3e-4, 1e-3]
TASKS = ["unique_words", "repeated_words", "sentences"]
MET = ["r_mean", "r_speech", "mcd_db", "mcd_speech_db"]
S3, S5 = [1, 2, 3], [1, 2, 3, 4, 5]

d1 = pd.read_csv(os.path.join(P1, "phase1_results.csv"))
d2 = pd.read_csv(os.path.join(P2, "phase2_results.csv"))
df = pd.concat([d1, d2], ignore_index=True)
df = df[~df.participant.isin(EXCL)]
task_of = df.dropna(subset=["task"]).groupby("participant").task.first()
PIDS = sorted(task_of.index)

ft = df[(df.kind == "finetune") & (df.protocol == "new")].copy()
ft["lr_in4"] = np.isclose(ft.lr.values[:, None], np.array(GRID4)[None, :]).any(1)
KEY = ["participant", "condition", "fraction", "seed"]


def select(frame):
    return frame.loc[frame.groupby(KEY).val_mse.idxmin()]


sel4 = select(ft[ft.lr_in4])
sel6 = select(ft[ft.condition.isin(["sup", "ssl", "placebo"])])
sel6["condition"] = sel6.condition + "_g6"
other = df[(df.kind.isin(["linear", "probe"])) | ((df.kind == "finetune") & (df.protocol == "legacy"))]
SEL = pd.concat([sel4, sel6, other], ignore_index=True)


def wide(metric, cond, seeds=S5):
    x = SEL[(SEL.condition == cond) & SEL.seed.isin(seeds)]
    return x.groupby(["participant", "fraction"])[metric].mean().unstack("fraction").reindex(PIDS)


def boot(x, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    return rng.choice(x, (n, len(x))).mean(1)


def paired(a, b, alternative="two-sided"):
    d = (a - b).dropna()
    w = stats.wilcoxon(d, alternative=alternative)
    nz = d[d != 0]
    rk = stats.rankdata(np.abs(nz))
    bs = boot(d.values)
    return {"mean_a": float(a[d.index].mean()), "mean_b": float(b[d.index].mean()), "delta": float(d.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "lower95_onesided": float(np.percentile(bs, 5)),
            "n_pos": int((d > 0).sum()), "n": int(len(d)), "p": float(w.pvalue),
            "rank_biserial": float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum())}


def holm(ps):
    ps = np.asarray(ps, float)
    order, m, adj, run = np.argsort(ps), len(ps), np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj.tolist()


def fam(a, b, metric, fracs=FR, seeds=S5, pids=None, alternative="two-sided", sign=1):
    A, B = wide(metric, a, seeds), wide(metric, b, seeds)
    if pids is not None:
        A, B = A.loc[pids], B.loc[pids]
    if sign < 0:  # lower is better (MCD): test b - a so that positive = a better
        A, B = -A, -B
    res = {f: paired(A[f], B[f], alternative) for f in fracs}
    for f, q in zip(fracs, holm([res[f]["p"] for f in fracs])):
        res[f]["p_holm"] = q
    return {str(f): v for f, v in res.items()}


def page(conds, metric, frac, seeds, base="sup"):
    """Page trend: the advantage over `base` increases along `conds` (predicted ranks 1..k)."""
    Bm = wide(metric, base, seeds)[frac]
    M = pd.concat([wide(metric, c, seeds)[frac] - Bm for c in conds], axis=1, keys=conds).dropna()
    t = stats.page_trend_test(M.values, predicted_ranks=list(range(1, len(conds) + 1)))
    return {"conds": conds, "means": [float(v) for v in M.mean()], "L": float(t.statistic), "p": float(t.pvalue),
            "n": int(len(M))}


out = {"n_participants": len(PIDS), "excluded": EXCL,
       "provenance": {"code_hash_p2": sorted(d2.code_hash.dropna().unique().tolist()),
                      "status_p2": d2.status.value_counts().to_dict()}}

# ---- label seconds per fraction (median across participants)
ls = d1[(d1.kind == "finetune") & (d1.condition == "sup") & (d1.protocol == "new")]
ls = ls[~ls.participant.isin(EXCL)].groupby(["fraction", "participant"]).label_seconds.first().unstack(0)
out["label_seconds_by_task"] = {t: {str(f): float(ls.loc[[p for p in ls.index if task_of[p] == t], f].median()) for f in FR} for t in TASKS}
out["label_seconds"] = {str(f): {"median": float(ls[f].median()), "min": float(ls[f].min()), "max": float(ls[f].max())}
                        for f in FR}

# ---- group means (participant-level bootstrap CIs)
CONDS = ["sup", "ssl", "placebo", "sup_g6", "ssl_g6", "placebo_g6", "gru", "pooled", "loo_adapt", "loo_noadapt",
         "scale5", "scale15", "ridge", "pca_lab", "pca_all", "probe_random", "probe_placebo", "probe_ssl",
         "probe_pooled", "probe_loo_adapt"]
MASKS = ["mask_%s%d" % (k, r) for k in ("random", "time", "contact", "lagcons") for r in (25, 50, 75)
         if not (k == "random" and r == 50)]
table = {}
for m in MET:
    table[m] = {}
    for c in CONDS + MASKS + ["probe_" + x for x in MASKS]:
        seeds = S3 if (c.startswith("mask") or c.startswith("probe_mask") or c in
                       ("loo_adapt", "loo_noadapt", "scale5", "scale15", "probe_loo_adapt")) else S5
        W = wide(m, c, seeds)
        row = {}
        for f in FR:
            v = W[f].dropna().values if f in W else np.array([])
            if len(v):
                bs = boot(v)
                row[str(f)] = [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        table[m][c] = row
out["table"] = table

R = {}
for metric in ("r_mean", "r_speech"):
    r = {}
    # H5: pooled vs ssl, pooled vs sup at <=25% (Holm across 25/10/5); 5 seeds each
    r["H5_pooled_vs_ssl"] = fam("pooled", "ssl", metric, LOW)
    r["H5_pooled_vs_sup"] = fam("pooled", "sup", metric, LOW)
    r["pooled_vs_ssl_all"] = fam("pooled", "ssl", metric, FR)
    # H6: scaling 1 -> 5 -> 15 -> 30 patients (seeds 1-3 for every arm)
    r["H6_page"] = {str(f): page(["ssl", "scale5", "scale15", "pooled"], metric, f, S3) for f in LOW}
    # post hoc: opposite direction (advantage decreases with more pretraining patients)
    r["H6_page_reverse_posthoc"] = {str(f): page(["pooled", "scale15", "scale5", "ssl"], metric, f, S3) for f in LOW}
    r["H6_scale15_vs_ssl"] = fam("scale15", "ssl", metric, LOW, S3)
    r["H6_scale5_vs_ssl"] = fam("scale5", "ssl", metric, LOW, S3)
    # H7: loo_adapt non-inferior to ssl (margin 0.005) and > sup (seeds 1-3)
    h7 = fam("loo_adapt", "ssl", metric, LOW, S3)
    for f in h7.values():
        f["noninferior_margin"] = 0.005
        f["noninferior"] = bool(f["lower95_onesided"] > -0.005)
    r["H7_looadapt_vs_ssl"] = h7
    r["H7_looadapt_vs_sup"] = fam("loo_adapt", "sup", metric, LOW, S3)
    r["looadapt_vs_ssl_all"] = fam("loo_adapt", "ssl", metric, FR, S3)
    r["looadapt_vs_sup_all"] = fam("loo_adapt", "sup", metric, FR, S3)
    r["loonoadapt_vs_sup_all"] = fam("loo_noadapt", "sup", metric, FR, S3)
    r["looadapt_vs_loonoadapt"] = fam("loo_adapt", "loo_noadapt", metric, FR, S3)
    r["pooled_vs_sup_all"] = fam("pooled", "sup", metric, FR)
    # H8: Phase 1 H1/H2 with six-LR grid; GRU
    r["H8_ssl_vs_sup_g6"] = fam("ssl_g6", "sup_g6", metric)
    r["H8_ssl_vs_placebo_g6"] = fam("ssl_g6", "placebo_g6", metric)
    r["H8_placebo_vs_sup_g6"] = fam("placebo_g6", "sup_g6", metric)
    r["H8_sup_g6_vs_sup_g4"] = fam("sup_g6", "sup", metric)
    r["H8_ssl_g6_vs_ssl_g4"] = fam("ssl_g6", "ssl", metric)
    r["H8_ssl_vs_gru"] = fam("ssl", "gru", metric)
    r["gru_vs_sup"] = fam("gru", "sup", metric)
    r["H1_ssl_vs_sup_g4"] = fam("ssl", "sup", metric)
    for name, (a, b) in {"H8_page_ssl_vs_sup_g6": ("ssl_g6", "sup_g6"), "H1_page_ssl_vs_sup_g4": ("ssl", "sup")}.items():
        D = (wide(metric, a) - wide(metric, b))[FR].dropna()
        t = stats.page_trend_test(D.values, predicted_ranks=[1, 2, 3, 4, 5])
        r[name] = {"L": float(t.statistic), "p": float(t.pvalue)}
    # H9: probes
    r["H9_probe_ssl_vs_placebo"] = fam("probe_ssl", "probe_placebo", metric)
    r["H9_probe_placebo_vs_random"] = fam("probe_placebo", "probe_random", metric)
    r["H9_probe_ssl_vs_random"] = fam("probe_ssl", "probe_random", metric)
    r["probe_pooled_vs_probe_ssl"] = fam("probe_pooled", "probe_ssl", metric)
    r["probe_looadapt_vs_probe_ssl"] = fam("probe_loo_adapt", "probe_ssl", metric, FR, S3)
    # post hoc: frozen representation + linear readout vs raw-feature ridge and vs fine-tuning
    r["probe_ssl_vs_ridge"] = fam("probe_ssl", "ridge", metric)
    r["probe_pooled_vs_ridge"] = fam("probe_pooled", "ridge", metric)
    r["probe_ssl_vs_ssl_ft"] = fam("probe_ssl", "ssl", metric)
    r["ssl_vs_ridge"] = fam("ssl", "ridge", metric)
    r["pooled_vs_ridge"] = fam("pooled", "ridge", metric)
    r["gru_vs_ridge"] = fam("gru", "ridge", metric)
    r["probe_ssl_vs_gru"] = fam("probe_ssl", "gru", metric)
    r["probe_ssl_vs_pca_lab"] = fam("probe_ssl", "pca_lab", metric)
    r["ssl_g6_vs_gru"] = fam("ssl_g6", "gru", metric)
    # masking grid at 10% (and 100%): each vs default random-0.5 (= Phase 1 ssl), seeds 1-3
    mg = {}
    for f in (1.0, 0.1):
        base = wide(metric, "ssl", S3)[f]
        sup = wide(metric, "sup", S3)[f]
        res = {c: paired(wide(metric, c, S3)[f], base) for c in MASKS}
        for c, q in zip(MASKS, holm([res[c]["p"] for c in MASKS])):
            res[c]["p_holm"] = q
        vs_sup = {c: paired(wide(metric, c, S3)[f], sup) for c in MASKS + ["ssl"]}
        for c, q in zip(vs_sup, holm([vs_sup[c]["p"] for c in vs_sup])):
            vs_sup[c]["p_holm"] = q
        allc = ["ssl"] + MASKS
        M = pd.concat([wide(metric, c, S3)[f] for c in allc], axis=1, keys=allc).dropna()
        fr_test = stats.friedmanchisquare(*[M[c] for c in allc])
        # full pairwise matrix (Holm over all 66 pairs)
        pairs, ps = [], []
        for i in range(len(allc)):
            for j in range(i + 1, len(allc)):
                pr = paired(M[allc[i]], M[allc[j]])
                pairs.append([allc[i], allc[j], pr["delta"], pr["p"]])
                ps.append(pr["p"])
        for row, q in zip(pairs, holm(ps)):
            row.append(q)
        mg[str(f)] = {"vs_default": res, "vs_sup": vs_sup, "friedman": [float(fr_test.statistic), float(fr_test.pvalue)],
                      "means": {c: float(M[c].mean()) for c in allc},
                      "pairs_sig_holm": [p for p in pairs if p[4] < 0.05]}
        # geometry main effect (mean over ratios) and ratio main effect
        geo = {}
        for k in ("random", "time", "contact", "lagcons"):
            cs = [c for c in allc if (c == "ssl" and k == "random") or c.startswith("mask_" + k)]
            geo[k] = M[cs].mean(axis=1)
        G = pd.DataFrame(geo)
        mg[str(f)]["geometry_means"] = G.mean().to_dict()
        mg[str(f)]["geometry_friedman"] = float(stats.friedmanchisquare(*[G[c] for c in G]).pvalue)
        gp = {}
        for k in ("time", "contact", "lagcons"):
            gp[k + "_vs_random"] = paired(G[k], G["random"])
        for k, q in zip(gp, holm([v["p"] for v in gp.values()])):
            gp[k]["p_holm"] = q
        mg[str(f)]["geometry_vs_random"] = gp
    r["masking"] = mg
    # probes of masking encoders (seeds 1-3), each vs probe_ssl
    pm = {}
    for f in FR:
        base = wide(metric, "probe_ssl", S3)[f]
        res = {c: paired(wide(metric, "probe_" + c, S3)[f], base) for c in MASKS}
        for c, q in zip(MASKS, holm([res[c]["p"] for c in MASKS])):
            res[c]["p_holm"] = q
        pm[str(f)] = res
    r["probe_masking_vs_probe_ssl"] = pm
    # cohorts for the key comparisons
    coh = {}
    for t in TASKS:
        ps = [p for p in task_of[task_of == t].index]
        coh[t] = {"pooled_vs_ssl": fam("pooled", "ssl", metric, FR, S5, ps),
                  "pooled_vs_sup": fam("pooled", "sup", metric, FR, S5, ps),
                  "looadapt_vs_sup": fam("loo_adapt", "sup", metric, FR, S3, ps),
                  "ssl_g6_vs_sup_g6": fam("ssl_g6", "sup_g6", metric, FR, S5, ps),
                  "probe_ssl_vs_ridge": fam("probe_ssl", "ridge", metric, FR, S5, ps)}
    r["cohorts"] = coh
    # mixed model: pooled - ssl ~ task + fraction
    D = (wide(metric, "pooled") - wide(metric, "ssl")).stack().rename("delta").reset_index()
    D["task"] = D.participant.map(task_of)
    D["fr"] = D.fraction.astype(str)
    mm = smf.mixedlm("delta ~ C(task, Treatment('unique_words')) + C(fr, Treatment('1.0'))", D, groups=D.participant).fit()
    r["mixedlm_pooled_minus_ssl"] = {k: [float(mm.params[k]), float(mm.pvalues[k])] for k in mm.params.index if k != "Group Var"}
    R[metric] = r
out["results"] = R

# MCD (lower is better: positive delta = first arm better)
out["mcd"] = {}
for m in ("mcd_db", "mcd_speech_db"):
    out["mcd"][m] = {"ssl_vs_sup_g6": fam("ssl_g6", "sup_g6", m, sign=-1),
                     "pooled_vs_ssl": fam("pooled", "ssl", m, sign=-1),
                     "looadapt_vs_sup": fam("loo_adapt", "sup", m, FR, S3, sign=-1),
                     "probe_ssl_vs_ridge": fam("probe_ssl", "ridge", m, sign=-1),
                     "ssl_vs_gru": fam("ssl", "gru", m, sign=-1)}

# LR choice with the six-LR grid
out["lr_choice_g6"] = {c: {str(f): sel6[(sel6.condition == c + "_g6") & (sel6.fraction == f)].lr.value_counts().sort_index()
                           .rename(lambda x: f"{x:g}").to_dict() for f in FR} for c in ("sup", "ssl", "placebo")}
out["lr_choice_g4_other"] = {c: {str(f): sel4[(sel4.condition == c) & (sel4.fraction == f)].lr.value_counts().sort_index()
                                 .rename(lambda x: f"{x:g}").to_dict() for f in FR}
                             for c in ("gru", "pooled", "loo_adapt", "loo_noadapt")}
# probe alphas at grid edge
pr = df[df.kind == "probe"]
out["probe_alpha_at_edge"] = pr.groupby("condition").alpha.apply(lambda a: int((a >= 1e10).sum())).to_dict()

# label-equivalence: fraction of labels sup (grid4) would need to match each condition
def equiv(metric, cond, base="sup", seeds=S5):
    s = wide(metric, base, seeds).mean()
    q = wide(metric, cond, seeds).mean()
    xs, ys = np.log(np.array(FR[::-1])), s[FR[::-1]].values
    res = {}
    for f in FR:
        if f not in q or pd.isna(q[f]) or q[f] <= ys[0] or q[f] >= ys[-1] or not np.all(np.diff(ys) > 0):
            res[str(f)] = None
        else:
            res[str(f)] = float(np.exp(np.interp(q[f], ys, xs)))
    return res
out["equiv_vs_sup"] = {m: {c: equiv(m, c, seeds=(S3 if c in ("loo_adapt",) else S5))
                           for c in ("ssl", "pooled", "loo_adapt", "gru", "probe_ssl", "ridge")}
                       for m in ("r_mean", "r_speech")}

# chance
fin = SEL[SEL.kind == "finetune"]
out["above_chance_runs"] = {"n": int(len(fin)), "frac_p_lt_0.05": float((fin.p_value < 0.05).mean()),
                            "min_r_mean_at_5pct_main": float(fin[(fin.fraction == 0.05) &
                                                                  fin.condition.isin(["sup", "ssl", "pooled"])]
                                                             .r_mean.min())}

# contamination summary
C = os.path.join(P2, "contamination")
rows, inj = [], []
for p in PIDS:
    j = json.load(open(os.path.join(C, f"{p}.json")))
    rows.append({k: j[k] for k in ("participant", "n_channels", "max_index", "median_index", "n_flagged",
                                   "flagged_channels", "n_significant_any_size")})
    for db in (30, 20):
        fn = os.path.join(C, f"{p}_inject{db}dB.json")
        if os.path.exists(fn):
            k = json.load(open(fn))
            inj.append({"participant": p, "db": -db, "detected": k["injected_detected"], "rank": k["injected_rank"],
                        "index": next((c["index"] for c in k["per_channel"] if c["channel"] == k["injected_channel"]), None)})
out["contamination"] = {"per_participant": rows, "injection": inj,
                        "n_participants_flagged": int(sum(r["n_flagged"] > 0 for r in rows)),
                        "n_channels_total": int(sum(r["n_channels"] for r in rows)),
                        "n_channels_flagged": int(sum(r["n_flagged"] for r in rows)),
                        "max_index_overall": float(max(r["max_index"] for r in rows)),
                        "n_detected": int(sum(bool(i["detected"]) for i in inj)), "n_inject": len(inj),
                        "median_inject_index": {str(db): float(np.median([i["index"] for i in inj if i["db"] == db]))
                                                for db in (-30, -20)}}
json.dump(out, open(OUT, "w"), indent=1, default=float)


# ---------------- print
def show(metric, name):
    rr = R[metric][name]
    print(f"\n{metric} | {name}")
    for f, x in rr.items():
        extra = f" NI={x['noninferior']} lb={x['lower95_onesided']:+.4f}" if "noninferior" in x else ""
        print(f"  {float(f):4.2f}: {x['mean_a']:.3f} vs {x['mean_b']:.3f} d={x['delta']:+.4f} "
              f"CI[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}] {x['n_pos']:2d}/{x['n']} rb={x['rank_biserial']:+.2f} "
              f"p={x['p']:.2g} holm={x['p_holm']:.2g}{extra}")


print("N =", len(PIDS), "excluded", EXCL)
for metric in ("r_mean", "r_speech"):
    for name, v in R[metric].items():
        if name in ("masking", "probe_masking_vs_probe_ssl", "cohorts", "mixedlm_pooled_minus_ssl", "H6_page") or "page" in name:
            continue
        show(metric, name)
    print(f"\n{metric} H6 Page:", {f: (np.round(v['means'], 4).tolist(), round(v['L'], 1), round(v['p'], 4)) for f, v in R[metric]["H6_page"].items()})
    print(f"{metric} H1/H8 page:", R[metric]["H1_page_ssl_vs_sup_g4"], R[metric]["H8_page_ssl_vs_sup_g6"])
    for f, mg in R[metric]["masking"].items():
        print(f"\n{metric} masking @ {f}: friedman p={mg['friedman'][1]:.3g}; geometry friedman p={mg['geometry_friedman']:.3g}")
        print("   means:", {k: round(v, 4) for k, v in mg["means"].items()})
        print("   geometry means:", {k: round(v, 4) for k, v in mg["geometry_means"].items()})
        print("   geometry vs random:", {k: (round(v['delta'], 4), v['n_pos'], round(v['p_holm'], 3)) for k, v in mg["geometry_vs_random"].items()})
        print("   vs default:", {k: (round(v['delta'], 4), v['n_pos'], round(v['p_holm'], 3)) for k, v in mg["vs_default"].items()})
        print("   sig pairs:", [(a, b, round(d, 4), round(q, 3)) for a, b, d, p, q in mg["pairs_sig_holm"]])
    print(f"\n{metric} probe masks vs probe_ssl @0.05:", {k: (round(v['delta'], 4), v['n_pos'], round(v['p_holm'], 3)) for k, v in R[metric]["probe_masking_vs_probe_ssl"]["0.05"].items()})
    print(f"{metric} mixedlm pooled-ssl:", {k: [round(a, 4), round(b, 4)] for k, (a, b) in R[metric]["mixedlm_pooled_minus_ssl"].items()})
    print(f"\n{metric} COHORTS (delta, n_pos/10, p):")
    for t in TASKS:
        for name, v in R[metric]["cohorts"][t].items():
            print(f"  {t:15s} {name:20s}", " ".join(f"{float(f):.2f}:{x['delta']:+.3f}({x['n_pos']},{x['p']:.2f})" for f, x in v.items()))
print("\nlabel seconds:", out["label_seconds"], out["label_seconds_by_task"])
for metric in ("r_mean", "r_speech"):
    print(metric, "H6 reverse:", {f: (round(v['L'], 1), round(v['p'], 5)) for f, v in R[metric]["H6_page_reverse_posthoc"].items()})
print("equiv:", json.dumps(out["equiv_vs_sup"]))
print("lr g6:", json.dumps(out["lr_choice_g6"]))
print("lr other:", json.dumps(out["lr_choice_g4_other"]))
print("probe alpha edge:", out["probe_alpha_at_edge"])
print("chance:", out["above_chance_runs"])
cc = out["contamination"]
print("contamination:", {k: v for k, v in cc.items() if k not in ("per_participant", "injection")})
print("flagged:", [(r["participant"], r["flagged_channels"], round(r["max_index"], 4)) for r in cc["per_participant"] if r["n_flagged"]])
for m, v in out["mcd"].items():
    for name, x in v.items():
        print(m, name, {f: (round(y["delta"], 3), y["n_pos"], round(y["p"], 3)) for f, y in x.items()})
