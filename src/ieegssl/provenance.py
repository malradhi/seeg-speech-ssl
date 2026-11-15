"""Provenance stamps written into every result file."""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import time
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent.parent


def code_hash() -> str:
    """SHA-256 over the package and script sources (order-independent of the OS)."""
    files = sorted(list(_PKG.glob("*.py")) + list((_REPO / "scripts").glob("*.py")))
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(_REPO).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def environment() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    try:
        import numpy, scipy, torch  # noqa: E401

        info.update(numpy=numpy.__version__, scipy=scipy.__version__, torch=torch.__version__)
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - informational only
        pass
    return info
