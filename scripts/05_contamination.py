"""Acoustic-contamination audit for one or more participants (CPU).

    python scripts/05_contamination.py --participants P15
    python scripts/05_contamination.py --array-index $SLURM_ARRAY_TASK_ID
Output: $IEEG_ROOT/results/contamination/P##.json (+ _matrices.npz)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _common
from ieegssl.config import load_config
from ieegssl.contamination import audit_participant

ap = argparse.ArgumentParser()
ap.add_argument("--participants", default=None)
ap.add_argument("--array-index", default=None)
ap.add_argument("--config", default=None)
ap.add_argument("--root", default=None)
ap.add_argument("--no-inject", action="store_true", help="skip the positive-control injections")
args = ap.parse_args()

root = Path(args.root or _common.default_root())
cfg = load_config(args.config)
failed = []
for pid in _common.resolve_participants(args.participants, args.array_index or os.environ.get("ARRAY_INDEX")):
    try:
        out = root / "results" / "contamination"
        s = audit_participant(pid, root / "raw", out, cfg)
        print(f"{pid}: {s['n_channels']} channels, max index {s['max_index']:.3f}, "
              f"flagged {s['n_flagged']} {s['flagged_channels'][:10]}", flush=True)
        if not args.no_inject:
            for db in cfg["contamination"]["inject_db"]:
                s = audit_participant(pid, root / "raw", out, cfg, inject_db=db,
                                      n_perm=cfg["contamination"]["n_perm_inject"])
                print(f"{pid}: positive control {db:+.0f} dB into {s['injected_channel']}: "
                      f"detected={s['injected_detected']} (rank {s['injected_rank']})", flush=True)
    except Exception as exc:
        failed.append(pid)
        print(f"{pid}: FAILED {exc!r}", flush=True)
if failed:
    raise SystemExit(f"Contamination audit failed for {failed}")
