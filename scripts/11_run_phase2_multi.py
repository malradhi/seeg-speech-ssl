"""Phase 2 multi-participant pretraining (one SLURM array task).

    python scripts/11_run_phase2_multi.py --array-index 1    # leave-one-out + scaling for P01
    python scripts/11_run_phase2_multi.py --array-index 31   # pooled model, seed 1
Array indices 1..30 are participants; 31..35 are the five pooled seeds.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _common
from ieegssl.config import load_config
from ieegssl.multi import run_multi

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--array-index", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    index = args.array_index or int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    cfg = load_config(args.config)
    run_multi(index, cfg, Path(args.root or _common.default_root()),
              args.workers or cfg["phase2"]["multi_workers"], args.device)
