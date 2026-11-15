"""Phase 3 pooled pretraining of the large Transformer (one SLURM array task per seed).

    python scripts/13_run_phase3_pooled.py --array-index 1     # seed 1 of phase3.pooled_large_seeds
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _common
from ieegssl.config import load_config
from ieegssl.runner3 import run_pooled_large

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--array-index", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    index = args.array_index or int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    run_pooled_large(index, load_config(args.config), Path(args.root or _common.default_root()), args.device)
