"""Summary of the Phase 3 evaluation suite (results/eval): one table per metric,
conditions x label budgets, mean over participants of seed-averaged values.

    python scripts/22_eval_summary.py [--metric r_shape] [--dir eval]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import _common

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--dir", default="eval")
ap.add_argument("--metric", default=None, help="default: the main metrics, one table each")
args = ap.parse_args()
root = Path(args.root or _common.default_root())
files = sorted((root / "results" / args.dir).glob("P*.json"))
rows = [r for f in files for r in json.loads(f.read_text())["rows"]]
print(f"{len(files)} participants evaluated, {len(rows)} prediction sets")
if not rows:
    raise SystemExit
metrics = [args.metric] if args.metric else ["r_speech", "r_speech_p", "r_loud", "r_shape", "r_shape_p",
                                              "mcd_speech_db", "estoi", "stoi"]
ORDER = ["sup", "ssl", "gru", "gru_ssl", "pooled", "loo_adapt", "ridge", "probe_ssl", "timing_reference", "train_mean",
         "shifted_truth", "true_spectrogram"]
for m in metrics:
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(m)
        if v is None or not np.isfinite(v):
            continue
        if m.endswith("_p"):
            v = float(v < 0.05)  # share of runs above chance
        acc[(r["condition"], r["fraction"])][r["participant"]].append(v)
    if not acc:
        print(f"\n{m}: no values")
        continue
    fracs = sorted({f for _, f in acc if f is not None}, reverse=True)
    conds = [c for c in ORDER if any(k[0] == c for k in acc)] + sorted({k[0] for k in acc} - set(ORDER))
    label = f"{m} (share of runs with p < 0.05)" if m.endswith("_p") else m
    print(f"\n{label}")
    print(f"{'condition':<18}" + "".join(f"{f'{f:.0%}':>9s}" for f in fracs) + f"{'(no budget)':>13s}")
    for c in conds:
        line = f"{c:<18}"
        for f in fracs:
            d = acc.get((c, f))
            line += f"{np.mean([np.mean(v) for v in d.values()]):9.3f}" if d else f"{'-':>9s}"
        d = acc.get((c, None))
        line += f"{np.mean([np.mean(v) for v in d.values()]):13.3f}" if d else f"{'':>13s}"
        print(line)
chk = [(r["r_mean_here"], r["check_r_mean_phase2"]) for r in rows
       if r.get("condition") == "probe_ssl" and r.get("check_r_mean_phase2") is not None]
if chk:
    diff = np.abs(np.array([a - b for a, b in chk]))
    print(f"\nprobe_ssl recomputed vs Phase 2 record: max |difference in r| = {diff.max():.4f} over {len(chk)} runs")
