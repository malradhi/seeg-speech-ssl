"""Phase 4b per-participant runs (self-pretraining, learning-rate grid edge, BrainBERT readout
capacity; docs/PHASE4B_DESIGN.md).

    python scripts/17_run_phase4b.py --participants P15 --workers 8
    python scripts/17_run_phase4b.py --array-index $SLURM_ARRAY_TASK_ID --workers 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _common
from ieegssl.config import load_config
from ieegssl.runner4b import run_participant4b

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=None)
    ap.add_argument("--array-index", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--raw", default=None, help="raw data folder (default: $IEEG_ROOT/raw)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    root = Path(args.root or _common.default_root())
    cfg = load_config(args.config)
    for pid in _common.resolve_participants(args.participants,
                                             args.array_index or os.environ.get("ARRAY_INDEX")):
        run_participant4b(pid, cfg, root, args.workers, args.device, args.raw)
