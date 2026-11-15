"""Pre-specified Phase 5 analysis (docs/PHASE5_DESIGN.md).

Usage: python -I phase5_stats.py <ident5_dir> <out_json> [exclude_pid ...]

Participant score = mean over seeds of the per-prediction score. H18: one-sided Wilcoxon of
(pairwise accuracy - 0.5), Holm across the five budgets. H19 and the reported comparisons:
two-sided paired Wilcoxon, Holm across the budgets of the family.
"""
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats

SRC, OUT = sys.argv[1:3]
EXCL = sys.argv[3:]
FR = [1.0, 0.5, 0.25, 0.1, 0.05]
LOW = [0.25, 0.1, 0.05]
DECODERS = ["sup", "ssl", "gru", "gru_ssl", "lpft_ssl", "lpft_gru_ssl", "ssl_self", "gru_ssl_self", "ridge",
            "probe_ssl", "bb_best_ridge", "bb_best_gru"]

rows, meta = [], {}
for f in sorted(glob(os.path.join(SRC, "P*.json"))):
    j = json.load(open(f))
    if j["participant"] in EXCL:
        continue
    meta[j["participant"]] = {"task": j["task"], "n_segments": j["n_segments"], "segment_seconds": j["segment_seconds"],
                              "test_seconds": j["test_seconds"], "code_hash": j["code_hash"]}
    rows += j["rows"]
E = pd.DataFrame(rows)
E["fr"] = E.fraction.fillna(-1.0)
PIDS = sorted(meta)
task_of = pd.Series({p: m["task"] for p, m in meta.items()})


def wide(col, cond, src=E):
    x = src[src.condition == cond]
    return x.groupby(["participant", "fr"])[col].mean().unstack("fr").reindex(PIDS)


def holm(ps):
    ps = np.asarray(ps, float)
    order, m, adj, run = np.argsort(ps), len(ps), np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj.tolist()


def boot_ci(x, n=10000, seed=1):
    x = np.asarray(x)
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def vs_chance(col, cond, fracs=FR, chance=0.5, pids=None):
    W = wide(col, cond)
    if pids is not None:
        W = W.loc[pids]
    res = {}
    for f in fracs:
        if f not in W or W[f].isna().all():
            continue
        d = (W[f] - chance).dropna()
        w = stats.wilcoxon(d, alternative="greater") if (d != 0).any() else None
        res[f] = {"mean": float(W[f].mean()), "delta": float(d.mean()), "ci95": boot_ci(d.values) if len(d) else None,
                  "n_pos": int((d > 0).sum()), "n": int(len(d)), "p": float(w.pvalue) if w else 1.0}
    for f, q in zip(res, holm([r["p"] for r in res.values()])):
        res[f]["p_holm"] = q
    return {str(f): v for f, v in res.items()}


def paired(col, a, b, fracs=LOW, pids=None):
    A, B = wide(col, a), wide(col, b)
    if pids is not None:
        A, B = A.loc[pids], B.loc[pids]
    res = {}
    for f in fracs:
        if f not in A or f not in B:
            continue
        d = (A[f] - B[f]).dropna()
        if not len(d):
            continue
        w = stats.wilcoxon(d) if (d != 0).any() else None
        res[f] = {"mean_a": float(A[f][d.index].mean()), "mean_b": float(B[f][d.index].mean()),
                  "delta": float(d.mean()), "ci95": boot_ci(d.values), "n_pos": int((d > 0).sum()), "n": int(len(d)),
                  "p": float(w.pvalue) if w else 1.0}
    for f, q in zip(res, holm([r["p"] for r in res.values()])):
        res[f]["p_holm"] = q
    return {str(f): v for f, v in res.items()}


out = {"n_participants": len(PIDS), "excluded": EXCL, "code_hash": sorted({m["code_hash"] for m in meta.values()}),
       "segments": {"n_median_by_task": {t: float(np.median([meta[p]["n_segments"] for p in PIDS if meta[p]["task"] == t]))
                                         for t in sorted(set(task_of))},
                    "n_range": [int(min(m["n_segments"] for m in meta.values())),
                                int(max(m["n_segments"] for m in meta.values()))],
                    "seconds_median_by_task": {t: float(np.median([meta[p]["segment_seconds"]["median"] for p in PIDS
                                                                   if meta[p]["task"] == t])) for t in sorted(set(task_of))}}}
R = {}
# references
R["references"] = {c: {k: float(E[E.condition == c][k].mean()) for k in ("full_pairwise", "full_top1", "loudness_pairwise",
                                                                         "shape_pairwise")}
                   for c in ("true_spectrogram", "shifted_truth")}
R["references"]["shifted_truth_vs_chance"] = vs_chance("full_pairwise", "shifted_truth", [-1.0])
# H18
R["H18_gru_ssl_full"] = vs_chance("full_pairwise", "gru_ssl")
R["all_decoders_full"] = {c: vs_chance("full_pairwise", c) for c in DECODERS}
R["all_decoders_top1"] = {}
for c in DECODERS:
    x = E[E.condition == c].copy()
    x["chance"] = 1.0 / x.n_segments_main
    x["d"] = x.full_top1 - x.chance
    W = x.groupby(["participant", "fr"]).d.mean().unstack("fr").reindex(PIDS)
    T = x.groupby(["participant", "fr"]).full_top1.mean().unstack("fr").reindex(PIDS)
    res = {}
    for f in FR:
        if f in W and W[f].notna().any():
            d = W[f].dropna()
            res[str(f)] = {"mean": float(T[f].mean()), "chance": float((T[f] - W[f]).mean()), "n_pos": int((d > 0).sum()),
                           "p": float(stats.wilcoxon(d, alternative="greater").pvalue)}
    R["all_decoders_top1"][c] = res
R["share_runs_perm_p05"] = {c: {str(f): float((E[(E.condition == c) & (E.fr == f)].full_perm_p < 0.05).mean())
                                for f in FR if ((E.condition == c) & (E.fr == f)).any()} for c in DECODERS}
# H19 and comparisons
R["H19_gru_ssl_vs_gru"] = paired("full_pairwise", "gru_ssl", "gru")
R["H19_ssl_vs_sup"] = paired("full_pairwise", "ssl", "sup")
for a, b in (("gru_ssl", "gru"), ("ssl", "sup"), ("ssl_self", "sup"), ("gru_ssl_self", "gru"), ("lpft_gru_ssl", "gru_ssl"),
             ("lpft_ssl", "ssl"), ("gru_ssl", "bb_best_gru"), ("probe_ssl", "bb_best_ridge"), ("gru_ssl", "ridge"),
             ("lpft_gru_ssl", "ridge"), ("probe_ssl", "ridge"), ("gru_ssl", "ssl")):
    R[f"cmp_{a}_vs_{b}_all"] = paired("full_pairwise", a, b, FR)
# E7 loudness / shape
R["E7"] = {kind: {c: vs_chance(f"{kind}_pairwise", c) for c in DECODERS} for kind in ("loudness", "shape")}
R["E7_cmp"] = {kind: {f"{a}_vs_{b}": paired(f"{kind}_pairwise", a, b, FR) for a, b in
                      (("gru_ssl", "gru"), ("ssl", "sup"), ("lpft_gru_ssl", "gru_ssl"))} for kind in ("loudness", "shape")}
# cohorts
R["cohorts"] = {}
for t in sorted(set(task_of)):
    ps = list(task_of[task_of == t].index)
    R["cohorts"][t] = {"gru_ssl_vs_chance": vs_chance("full_pairwise", "gru_ssl", FR, 0.5, ps),
                       "gru_ssl_vs_gru": paired("full_pairwise", "gru_ssl", "gru", LOW, ps),
                       "ssl_vs_sup": paired("full_pairwise", "ssl", "sup", LOW, ps)}
# sensitivity
R["sensitivity"] = {}
for v in ("k16", "k64", "strict"):
    R["sensitivity"][v] = {"H18": vs_chance(f"{v}_full_pairwise", "gru_ssl"),
                           "H19_gru": paired(f"{v}_full_pairwise", "gru_ssl", "gru"),
                           "H19_ssl": paired(f"{v}_full_pairwise", "ssl", "sup")}
R["table"] = {col: {c: {str(f): float(wide(col, c)[f].mean()) for f in FR if f in wide(col, c) and wide(col, c)[f].notna().any()}
                    for c in DECODERS} for col in ("full_pairwise", "loudness_pairwise", "shape_pairwise", "full_top1")}
out["results"] = R
json.dump(out, open(OUT, "w"), indent=1, default=float)


def show(name, rr, a="mean"):
    print(f"\n{name}")
    for f, x in rr.items():
        if "mean_a" in x:
            print(f"  {float(f):5.2f}: {x['mean_a']:.3f} vs {x['mean_b']:.3f} d={x['delta']:+.4f} "
                  f"CI[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}] {x['n_pos']:2d}/{x['n']} p={x['p']:.3g} holm={x['p_holm']:.3g}")
        else:
            print(f"  {float(f):5.2f}: mean {x['mean']:.3f} (vs chance {x['delta']:+.4f}) {x['n_pos']:2d}/{x['n']} "
                  f"p={x['p']:.3g} holm={x['p_holm']:.3g}")


print("N =", len(PIDS), "excluded", EXCL, out["code_hash"])
print("segments:", json.dumps(out["segments"]))
print("references:", json.dumps({k: v for k, v in R["references"].items() if k != "shifted_truth_vs_chance"}))
show("shifted truth vs chance", R["references"]["shifted_truth_vs_chance"])
show("H18 gru_ssl pairwise vs 0.5", R["H18_gru_ssl_full"])
show("H19 gru_ssl vs gru", R["H19_gru_ssl_vs_gru"])
show("H19 ssl vs sup", R["H19_ssl_vs_sup"])
print("\nPAIRWISE ACCURACY (full) by decoder: mean | Holm p vs 0.5")
for c in DECODERS:
    print(f"  {c:14s}", " ".join(f"{float(f):.2f}:{x['mean']:.3f}|{x['p_holm']:.2g}" for f, x in R["all_decoders_full"][c].items()))
print("\nTOP-1 (mean | chance | n above | p)")
for c in DECODERS:
    print(f"  {c:14s}", " ".join(f"{float(f):.2f}:{x['mean']:.3f}|{x['chance']:.3f}|{x['n_pos']}|{x['p']:.2g}"
                                 for f, x in R["all_decoders_top1"][c].items()))
print("\nshare of runs with permutation p < 0.05:", json.dumps({c: {f: round(v, 2) for f, v in d.items()}
                                                              for c, d in R["share_runs_perm_p05"].items()}))
for k in [k for k in R if k.startswith("cmp_")]:
    show(k, R[k])
for kind in ("loudness", "shape"):
    print(f"\nE7 {kind}: mean | Holm p vs 0.5")
    for c in DECODERS:
        print(f"  {c:14s}", " ".join(f"{float(f):.2f}:{x['mean']:.3f}|{x['p_holm']:.2g}" for f, x in R["E7"][kind][c].items()))
    for name, rr in R["E7_cmp"][kind].items():
        show(f"E7 {kind} {name}", rr)
for t, d in R["cohorts"].items():
    for name, rr in d.items():
        show(f"COHORT {t} {name}", rr)
for v, d in R["sensitivity"].items():
    for name, rr in d.items():
        show(f"SENSITIVITY {v} {name}", rr)
