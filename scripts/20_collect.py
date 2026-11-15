"""Collect all run files of a stage into one CSV and a zip bundle for the statistical analysis.

    python scripts/20_collect.py --phase phase1    # -> $IEEG_ROOT/bundle_phase1_<date>.zip

The bundle contains: results CSV, every run JSON (metrics, curves, provenance),
the resolved config, the data audit and the SLURM logs. Predictions and encoder
weights stay under $IEEG_ROOT.
"""

from __future__ import annotations

import argparse
import csv
import time
import zipfile
from collections import Counter
from pathlib import Path

import _common
from ieegssl.analysis import FLAT_FIELDS, flatten, load_runs

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--phase", default="phase1")
ap.add_argument("--logs", default="logs", help="folder with SLURM .out/.err files")
ap.add_argument("--eval-dir", default="eval", help="results/<dir> of the evaluation suite (Phase 3)")
args = ap.parse_args()

root = Path(args.root or _common.default_root())
base = root / "results" / args.phase
recs = load_runs(base / "runs")
rows = [flatten(r) for r in recs]
csv_path = base / f"{args.phase}_results.csv"
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FLAT_FIELDS)
    w.writeheader()
    w.writerows(rows)

status = Counter(r.get("status") for r in rows)
print(f"{len(rows)} run files: {dict(status)}")
for r in rows:
    if r.get("status") != "ok":
        print(f"  {r['run_id']}: {r.get('error')}")

bundle = root / f"bundle_{args.phase}_{time.strftime('%Y%m%d_%H%M')}.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(csv_path, csv_path.name)
    for p in sorted((base / "runs").glob("*.json")):
        z.write(p, f"runs/{p.name}")
    for p in sorted(base.glob("config_*.json")):
        z.write(p, p.name)
    for p in sorted((root / "audit").glob("*.csv")):
        z.write(p, f"audit/{p.name}")
    for p in sorted((root / "results" / "contamination").glob("*")):
        z.write(p, f"contamination/{p.name}")
    for p in sorted((root / "results" / args.eval_dir).glob("P*.json")):
        z.write(p, f"{args.eval_dir}/{p.name}")
    for p in sorted((root / "results" / "anatomy").glob("P*.csv")):
        z.write(p, f"anatomy/{p.name}")
    for p in sorted((root / "derived_brainbert").glob("P*.json")):
        z.write(p, f"brainbert/{p.name}")
    for p in sorted((root / "derived_brainbert_k64").glob("P*.json")):
        z.write(p, f"brainbert_k64/{p.name}")
    for logdir in {Path(args.logs).resolve(), (_common.REPO / "logs").resolve()}:
        if logdir.exists():
            for p in sorted(logdir.glob("*.out")) + sorted(logdir.glob("*.err")):
                z.write(p, f"logs/{p.name}")
print(f"Bundle written: {bundle}  ({bundle.stat().st_size / 1e6:.1f} MB)")
