"""Phase 3 evaluation suite on saved test predictions (CPU; docs/PHASE3_DESIGN.md, part D).

    python scripts/15_evaluate.py --participants P15
    printf "P%02d\\n" $(seq 1 30) | xargs -P 8 -I{} python scripts/15_evaluate.py --participants {}
Finished participants are skipped (results/eval/P##.json); --overwrite recomputes them.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import _common  # noqa: E402
from ieegssl.config import load_config  # noqa: E402
from ieegssl.evaluation import evaluate_participant  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=None)
    ap.add_argument("--array-index", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    root = Path(args.root or _common.default_root())
    cfg = load_config(args.config)
    for pid in _common.resolve_participants(args.participants, args.array_index):
        try:
            evaluate_participant(pid, root, cfg, args.overwrite)
        except Exception as exc:  # report and continue with the next participant
            print(f"{pid}: ERROR {exc!r}", flush=True)
