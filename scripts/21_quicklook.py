"""Quick summary table (participant-level means; best LR chosen on validation).

    python scripts/21_quicklook.py [--metric r_mean|r_speech|mcd_db] [--task repeated_words]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _common
from ieegssl.analysis import best_lr_rows, flatten, load_runs, participant_means

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--phase", default="phase1", help="one phase or a comma list, e.g. phase1,phase2")
ap.add_argument("--metric", default="r_mean")
ap.add_argument("--task", default=None)
args = ap.parse_args()

root = Path(args.root or _common.default_root())
rows = [flatten(r) for ph in args.phase.split(",") for r in load_runs(root / "results" / ph.strip() / "runs")]
if args.task:
    rows = [r for r in rows if r.get("task") == args.task]
ref = [r for r in rows if r.get("condition") == "vad_oracle" and r.get("status") == "ok"]
sel = best_lr_rows(rows)
pm = participant_means(sel, args.metric)
PREFERRED = ("sup", "ssl", "placebo", "gru", "gru_ssl", "gru_placebo", "pooled", "loo_adapt", "loo_noadapt",
             "scale5", "scale15", "sup_L", "ssl_L", "pooled_L", "lpft_ssl", "lpft_gru_ssl", "ssl_self",
             "gru_ssl_self", "bb_gru", "bb64_gru", "bb_ridge", "bb16_ridge", "bb32_ridge", "bb64_ridge",
             "sup_legacy", "ssl_legacy", "ridge", "pca_lab", "pca_all")
present = {k[0] for k in pm}
conds = [c for c in PREFERRED if c in present] + sorted(present - set(PREFERRED))
fracs = sorted({k[1] for k in pm}, reverse=True)

print(f"\nmetric={args.metric}  task={args.task or 'all'}  (mean over participants of seed-averaged scores)")
if args.metric.startswith("mcd"):
    print("NOTE: lower is better for MCD, so 'SSL>sup' counts participants where SSL is WORSE.")
width = max(len(c) for c in conds) + 2
print(f"{'condition':<{width}}" + "".join(f"{f'{f:.0%} labels':>13s}" for f in fracs) + "   N participants")
for c in conds:
    line = f"{c:<{width}}"
    ns = set()
    for f in fracs:
        v = pm.get((c, f), {})
        ns.add(len(v))
        line += f"{np.mean(list(v.values())):13.3f}" if v else f"{'-':>13s}"
    print(line + f"   {'/'.join(str(n) for n in sorted(ns) if n)}")
line = f"{'SSL - sup':<{width}}"
for f in fracs:
    a, b = pm.get(("ssl", f), {}), pm.get(("sup", f), {})
    common = sorted(set(a) & set(b))
    if common:
        d = np.array([a[p] - b[p] for p in common])
        line += f"{f'{d.mean():+.3f} ({int((d > 0).sum())}/{len(common)})':>13s}"
    else:
        line += f"{'-':>13s}"
print(line)
if ref:
    vals = [r[args.metric] for r in ref if r.get(args.metric) is not None]
    print(f"VAD-oracle reference ({len(vals)} participants): {np.mean(vals):.3f}")
    ch = [r["chance_p95"] for r in sel if r.get("chance_p95") is not None]
    if ch:
        print(f"Chance level (circular shift, 95th pct, mean over runs): {np.mean(ch):.3f}")
