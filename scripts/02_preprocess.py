"""Leakage-safe feature extraction (CPU).

    python scripts/02_preprocess.py --participants P15
    python scripts/02_preprocess.py --array-index $SLURM_ARRAY_TASK_ID
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _common
from ieegssl.config import load_config
from ieegssl.preprocessing import preprocess_participant

ap = argparse.ArgumentParser()
ap.add_argument("--participants", default=None)
ap.add_argument("--array-index", default=None)
ap.add_argument("--config", default=None)
ap.add_argument("--root", default=None)
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

root = Path(args.root or _common.default_root())
cfg = load_config(args.config)
failed = []
for pid in _common.resolve_participants(args.participants, args.array_index or os.environ.get("ARRAY_INDEX")):
    try:
        preprocess_participant(pid, root / "raw", root / "derived", cfg, overwrite=args.overwrite)
    except Exception as exc:
        failed.append(pid)
        print(f"{pid}: FAILED {exc!r}", flush=True)
if failed:
    raise SystemExit(f"Preprocessing failed for {failed}")
