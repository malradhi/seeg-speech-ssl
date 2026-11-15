"""Collect the Phase 5 identification results into a zip bundle for the statistical analysis.

    python scripts/24_collect_phase5.py     # -> $IEEG_ROOT/bundle_phase5_<date>.zip
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import _common

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--dir", default="ident5")
ap.add_argument("--logs", default="logs")
args = ap.parse_args()
root = Path(args.root or _common.default_root())
src = root / "results" / args.dir
files = sorted(src.glob("P*.json"))
bad = []
for f in files:
    try:
        j = json.loads(f.read_text())
        if not j.get("rows"):
            bad.append(f.name)
    except Exception:
        bad.append(f.name)
print(f"{len(files)} participant files in results/{args.dir}" + (f"; unreadable or empty: {bad}" if bad else ""))
out = root / f"bundle_phase5_{time.strftime('%Y%m%d_%H%M')}.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f"{args.dir}/{f.name}")
    for f in sorted((root / args.logs).glob(f"{args.dir}_*.out")):
        z.write(f, f"logs/{f.name}")
print(f"Bundle written: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
