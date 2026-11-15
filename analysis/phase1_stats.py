"""Pre-specified Phase 1 analysis (docs/PHASE1_DESIGN.md). Usage: python -I stats.py <bundle_dir> <out_json>"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

B, OUT = sys.argv[1], sys.argv[2]
df = pd.read_csv(os.path.join(B, "phase1_results.csv"))
FR = [1.0, 0.5, 0.25, 0.1, 0.05]
LOW = [0.25, 0.1, 0.05]
TASKS = ["unique_words", "repeated_words", "sentences"]
METRICS = ["r_mean", "r_speech", "mcd_db", "mcd_speech_db"]

new = df[(df.kind == "finetune") & (df.protocol == "new")]
sel_new = new.loc[new.groupby(["participant", "condition", "fraction", "seed"]).val_mse.idxmin()]
other = df[((df.kind == "finetune") & (df.protocol == "legacy")) | (df.kind == "linear")]
sel = pd.concat([sel_new, other])
task_of = df.dropna(subset=["task"]).groupby("participant").task.first()

# participant means over seeds
pm = sel.groupby(["condition", "fraction", "participant"])[METRICS].mean().reset_index()
def wide(metric, cond):
    return pm[pm.condition == cond].pivot(index="participant", columns="fraction", values=metric)[FR]

def boot_ci(x, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

def paired(a, b):
    d = (a - b).dropna()
    w = stats.wilcoxon(d, alternative="two-sided")
    nz = d[d != 0]; rk = stats.rankdata(np.abs(nz))
    rb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
    return {"mean_a": float(a[d.index].mean()), "mean_b": float(b[d.index].mean()), "delta": float(d.mean()),
            "ci95": boot_ci(d.values), "n_pos": int((d > 0).sum()), "n": int(len(d)),
            "p": float(w.pvalue), "rank_biserial": float(rb)}

def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k]); adj[k] = min(1.0, run)
    return adj.tolist()

def family(cond_a, cond_b, metric, participants=None):
    A, Bm = wide(metric, cond_a), wide(metric, cond_b)
    if participants is not None:
        A, Bm = A.loc[participants], Bm.loc[participants]
    res = {f: paired(A[f], Bm[f]) for f in FR}
    adj = holm([res[f]["p"] for f in FR])
    for f, a in zip(FR, adj):
        res[f]["p_holm"] = a
    # Page trend: advantage predicted to increase as labels decrease (1.0 -> 0.05)
    D = (A - Bm)[FR].dropna()
    pt = stats.page_trend_test(D.values, predicted_ranks=[1, 2, 3, 4, 5])
    return {"by_fraction": {str(f): v for f, v in res.items()},
            "page_L": float(pt.statistic), "page_p": float(pt.pvalue)}

out = {"n_participants": int(pm.participant.nunique())}
comparisons = {"H1_ssl_vs_sup": ("ssl", "sup"), "H2_ssl_vs_placebo": ("ssl", "placebo"),
               "placebo_vs_sup": ("placebo", "sup"), "H3_ssllegacy_vs_suplegacy": ("ssl_legacy", "sup_legacy"),
               "H4_ssl_vs_ridge": ("ssl", "ridge"), "H4_ssl_vs_pcaall": ("ssl", "pca_all"),
               "sup_vs_ridge": ("sup", "ridge"), "pcaall_vs_pcalab": ("pca_all", "pca_lab")}
for metric in ("r_mean", "r_speech"):
    out[metric] = {name: family(a, b, metric) for name, (a, b) in comparisons.items()}
    # cohort consistency
    coh = {}
    for t in TASKS:
        ps = task_of[task_of == t].index.tolist()
        coh[t] = {name: family(a, b, metric, ps)["by_fraction"] for name, (a, b) in
                  {"ssl_vs_sup": ("ssl", "sup"), "ssl_vs_placebo": ("ssl", "placebo"), "ssl_vs_ridge": ("ssl", "ridge")}.items()}
    out[metric + "_cohorts"] = coh
# MCD (lower is better)
out["mcd"] = {name: {str(f): paired(wide(m, a)[f], wide(m, b)[f]) for f in FR}
              for m in ("mcd_db", "mcd_speech_db") for name, (a, b) in
              {f"{m}:ssl_vs_sup": ("ssl", "sup"), f"{m}:ssl_vs_ridge": ("ssl", "ridge")}.items()}

# Group means table with participant-level 95% CI
table = {}
for metric in METRICS:
    table[metric] = {}
    for c in ["sup", "ssl", "placebo", "sup_legacy", "ssl_legacy", "ridge", "pca_lab", "pca_all"]:
        W = wide(metric, c)
        table[metric][c] = {str(f): [float(W[f].mean()), *boot_ci(W[f].values)] for f in FR}
        for t in TASKS:
            ps = task_of[task_of == t].index
            table[metric].setdefault(c + "@" + t, {str(f): float(W.loc[ps, f].mean()) for f in FR})
out["table"] = table
ref = df[df.kind == "reference"].set_index("participant")
out["vad_oracle"] = {"all": float(ref.r_mean.mean()), **ref.groupby("task").r_mean.mean().to_dict(),
                     "mcd_speech_db": float(ref.mcd_speech_db.mean())}

# Label-equivalence: what fraction would supervised need to match SSL? (log-linear interpolation of group curves)
def equiv(metric):
    s, q = wide(metric, "sup").mean(), wide(metric, "ssl").mean()
    xs, ys = np.log(np.array(FR[::-1])), s[FR[::-1]].values  # increasing fraction
    res = {}
    for f in FR:
        y = q[f]
        if y <= ys[0] or y >= ys[-1] or not np.all(np.diff(ys) > 0):
            res[str(f)] = None; continue
        res[str(f)] = float(np.exp(np.interp(y, ys, xs)))
    return res
out["sup_fraction_needed_to_match_ssl"] = {m: equiv(m) for m in ("r_mean", "r_speech")}

# Mixed model on the SSL advantage: delta ~ task + fraction, random intercept per participant
rows = []
for metric in ("r_mean", "r_speech"):
    D = (wide(metric, "ssl") - wide(metric, "sup")).stack().rename("delta").reset_index()
    D["task"] = D.participant.map(task_of); D["fr"] = D.fraction.astype(str)
    m = smf.mixedlm("delta ~ C(task, Treatment('unique_words')) + C(fr, Treatment('1.0'))", D, groups=D.participant).fit(reml=True)
    out[f"mixedlm_{metric}"] = {k: [float(m.params[k]), float(m.pvalues[k])] for k in m.params.index if k != "Group Var"}

# Pilot cohort (P01-P10) continuity
p10 = [f"P{i:02d}" for i in range(1, 11)]
out["pilot_cohort_r_mean"] = {c: {str(f): float(wide("r_mean", c).loc[p10, f].mean()) for f in FR}
                              for c in ["sup", "ssl", "sup_legacy", "ssl_legacy", "ridge", "pca_lab"]}
# LR selection summary
out["lr_choice"] = {c: {str(f): sel_new[(sel_new.condition == c) & (sel_new.fraction == f)].lr.value_counts().sort_index().to_dict()
                        for f in FR} for c in ["sup", "ssl", "placebo"]}
json.dump(out, open(OUT, "w"), indent=1, default=float)

# ---- print the essentials
def show(metric, name):
    r = out[metric][name]; print(f"\n{metric} | {name} | Page L={r['page_L']:.0f} p={r['page_p']:.2g}")
    for f in FR:
        x = r["by_fraction"][str(f)]
        print(f"  {f:4.2f}: {x['mean_a']:.3f} vs {x['mean_b']:.3f}  d={x['delta']:+.4f} CI[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}] "
              f"{x['n_pos']:2d}/{x['n']}  rb={x['rank_biserial']:+.2f}  p={x['p']:.2g}  p_holm={x['p_holm']:.2g}")
for metric in ("r_mean", "r_speech"):
    for name in comparisons:
        show(metric, name)
for metric in ("r_mean", "r_speech"):
    print(f"\nCOHORTS {metric}: delta (n_pos/10, p) for ssl_vs_sup | ssl_vs_placebo | ssl_vs_ridge")
    for t in TASKS:
        line = f"  {t:15s}"
        for f in FR:
            a = out[metric + "_cohorts"][t]["ssl_vs_sup"][str(f)]; b = out[metric + "_cohorts"][t]["ssl_vs_placebo"][str(f)]
            line += f" | {f}: {a['delta']:+.3f}({a['n_pos']},{a['p']:.2f}) {b['delta']:+.3f}({b['n_pos']},{b['p']:.2f})"
        print(line)
print("\nsup fraction needed to match SSL:", json.dumps(out["sup_fraction_needed_to_match_ssl"]))
print("mixedlm r_mean:", json.dumps(out["mixedlm_r_mean"]))
print("mixedlm r_speech:", json.dumps(out["mixedlm_r_speech"]))
print("Pilot cohort r_mean:", json.dumps({c: {k: round(v, 3) for k, v in d.items()} for c, d in out["pilot_cohort_r_mean"].items()}))
print("VAD oracle:", out["vad_oracle"])
for k, v in out["mcd"].items():
    print(k, {f: (round(x["delta"], 3), x["n_pos"], round(x["p"], 3)) for f, x in v.items()})
