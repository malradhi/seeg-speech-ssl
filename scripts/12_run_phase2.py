"""Phase 2 per-participant runs (after the multi-participant pretraining has finished).

    python scripts/12_run_phase2.py --participants P15 --workers 8
    python scripts/12_run_phase2.py --array-index $SLURM_ARRAY_TASK_ID --workers 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _common
from ieegssl.config import load_config
from ieegssl.runner2 import run_participant2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=None)
    ap.add_argument("--array-index", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    root = Path(args.root or _common.default_root())
    cfg = load_config(args.config)
    for pid in _common.resolve_participants(args.participants,
                                             args.array_index or os.environ.get("ARRAY_INDEX")):
        run_participant2(pid, cfg, root, args.workers, args.device)
