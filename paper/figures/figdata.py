"""Participant-level data for the Results figures, built exactly as in the pre-specified analyses.

Selections follow analysis/phase4b_stats.py (seven-rate grid, learning rate chosen per
participant, condition, budget and seed on validation MSE; participant score = mean over seeds;
three-seed comparisons use seeds 1-3 of both arms) and analysis/phase2_stats.py (cross-patient
conditions on the four-rate grid, seeds 1-3). Evaluation and identification rows come from the
saved evaluation (eval, eval4b) and identification (ident5) files.

Usage: from figdata import load; D = load(<directory holding the phase bundles>)
"""
from __future__ import annotations

import json
import os
from glob import glob

import numpy as np
import pandas as pd

FR = [1.0, 0.5, 0.25, 0.1, 0.05]
LOW = [0.25, 0.1, 0.05]
G4 = [3e-5, 1e-4, 3e-4, 1e-3]
G6 = [1e-5] + G4 + [3e-3]
G7 = G6 + [1e-2]
S3, S5 = [1, 2, 3], [1, 2, 3, 4, 5]
KEY = ["participant", "condition", "fraction", "seed"]


def _inset(lr, grid):
    return np.isclose(lr.values[:, None], np.array(grid)[None, :]).any(1)


def _select(frame, name):
    s = frame.loc[frame.groupby(KEY).val_mse.idxmin()].copy()
    s["condition"] = name
    return s


def boot(x, n=10000, seed=1, fn=np.mean):
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    return fn(x[idx], axis=1)


def ci95(x, fn=np.mean):
    b = boot(x, fn=fn)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


class Data:
    pass


def load(root: str) -> Data:
    D = Data()
    df = pd.concat([pd.read_csv(os.path.join(root, f"phase{i}_bundle", f"phase{i}_results.csv"))
                    for i in ("1", "2", "3", "4", "4b")], ignore_index=True)
    D.task_of = df.dropna(subset=["task"]).groupby("participant").task.first()
    D.pids = sorted(D.task_of.index)
    ft = df[(df.kind == "finetune") & (df.protocol == "new")].copy()

    specs = [("sup", G7, "sup7"), ("ssl", G7, "ssl7"), ("placebo", G7, "placebo7"), ("gru", G7, "gru7"),
             ("gru_ssl", G7, "gru_ssl7"), ("gru_placebo", G7, "gru_placebo7"), ("lpft_ssl", G7, "lpft_ssl7"),
             ("lpft_gru_ssl", G7, "lpft_gru_ssl7"), ("ssl_self", G7, "ssl_self"), ("gru_ssl_self", G7, "gru_ssl_self")]
    for u in ("125", "250", "500"):
        specs += [(f"ssl_u{u}", G7, f"ssl_u{u}_7"), (f"gru_ssl_u{u}", G7, f"gru_ssl_u{u}_7")]
    # cross-patient comparisons (Phase 2 design): four-rate grid for every arm
    for c in ("sup", "ssl", "pooled", "scale5", "scale15", "loo_adapt", "loo_noadapt"):
        specs.append((c, G4, f"{c}_g4"))
    parts = [_select(ft[(ft.condition == c) & _inset(ft.lr, g)], n) for c, g, n in specs]
    parts.append(_select(ft[ft.condition.isin(["bb_gru", "bb64_gru"]) & _inset(ft.lr, G7)].assign(condition="x"),
                         "bb_best_gru"))
    lin = df[df.kind == "linear"]
    parts.append(_select(lin[lin.condition.isin(["bb_ridge", "bb16_ridge", "bb32_ridge", "bb64_ridge"])]
                         .assign(condition="x"), "bb_best_ridge"))
    D.sel = pd.concat(parts + [df[df.kind.isin(["linear", "probe"])]], ignore_index=True)

    rows = []
    for sub in ("phase3_bundle/eval", "phase4b_bundle/eval4b"):
        for f in sorted(glob(os.path.join(root, sub, "P*.json"))):
            for r in json.load(open(f))["rows"]:
                r["source"] = sub.split("/")[-1]
                rows.append(r)
    D.ev = pd.DataFrame(rows)
    D.ev["fr"] = D.ev.fraction.fillna(-1.0)

    rows = []
    for f in sorted(glob(os.path.join(root, "phase5_bundle/ident5", "P*.json"))):
        rows += json.load(open(f))["rows"]
    D.idn = pd.DataFrame(rows)
    D.idn["fr"] = D.idn.fraction.fillna(-1.0)
    return D


def wide(D, metric, cond, seeds=S5, pids=None):
    x = D.sel[(D.sel.condition == cond) & D.sel.seed.isin(seeds)]
    W = x.groupby(["participant", "fraction"])[metric].mean().unstack("fraction").reindex(D.pids)
    return W if pids is None else W.loc[pids]


def ewide(D, metric, cond, source, seeds=S3):
    x = D.ev[(D.ev.condition == cond) & (D.ev.source == source) & (D.ev.seed.isin(seeds) | D.ev.seed.isna())]
    return x.groupby(["participant", "fr"])[metric].mean().unstack("fr").reindex(D.pids)


def iwide(D, metric, cond, pids=None):
    x = D.idn[D.idn.condition == cond]
    W = x.groupby(["participant", "fr"])[metric].mean().unstack("fr").reindex(D.pids)
    return W if pids is None else W.loc[pids]


def equivalent(sup_curve, score):
    """Supervised budget with the same score (monotone envelope, linear in log2 budget)."""
    fr = np.array(sorted(FR))
    s = np.maximum.accumulate(np.array([sup_curve[f] for f in fr]))
    if score >= s[-1]:
        return 1.0
    if score <= s[0]:
        return float(fr[0])
    return float(2 ** np.interp(score, s, np.log2(fr)))


def label_ratio(D, base, own, f, metric="r_mean"):
    B, O = wide(D, metric, base), wide(D, metric, own)
    r = [equivalent(B.loc[p], O.loc[p, f]) / f for p in D.pids if not (pd.isna(O.loc[p, f]) or B.loc[p].isna().any())]
    return np.array(r)
