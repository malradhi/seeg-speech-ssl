"""Shared helpers for the command-line scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.append(str(REPO / "vendor"))  # bundled third-party code (pystoi); an installed copy takes precedence

from ieegssl.config import PARTICIPANTS  # noqa: E402


def default_root() -> str:
    root = os.environ.get("IEEG_ROOT")
    if not root:
        sys.exit("IEEG_ROOT is not set. Run `source slurm/env.sh` first (see KOMONDOR.md).")
    return root


def resolve_participants(spec: str | None, array_index: str | None) -> list[str]:
    """--participants P01,P02 | all ; or a 1-based SLURM array index -> one participant."""
    if array_index is not None:
        return [PARTICIPANTS[int(array_index) - 1]]
    if spec is None or spec.lower() == "all":
        return list(PARTICIPANTS)
    return [s.strip() for s in spec.split(",") if s.strip()]
