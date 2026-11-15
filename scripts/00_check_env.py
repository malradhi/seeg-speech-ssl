"""Check the Python environment (run on the login node and inside the smoke test)."""

from __future__ import annotations

import importlib
import os
import sys

import _common  # noqa: F401  (adds src/ to the path)

ok = True
print(f"python {sys.version.split()[0]}  ({sys.executable})")
for name in ("numpy", "scipy", "torch"):
    try:
        mod = importlib.import_module(name)
        print(f"  {name:6s} {mod.__version__}")
    except Exception as exc:
        ok = False
        print(f"  {name:6s} MISSING ({exc})  -> python -m pip install --user {name}")

try:
    import pystoi  # noqa: F401  (bundled in vendor/ if not installed)

    print(f"  pystoi OK ({os.path.dirname(pystoi.__file__)})")
except Exception as exc:
    ok = False
    print(f"  pystoi MISSING ({exc}) -> the vendor/ folder of the repository is incomplete")

try:
    import torch

    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        print(f"  CUDA OK: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} visible)")
        del x
    else:
        print("  CUDA not available here (expected on the login node)")
except Exception as exc:
    ok = False
    print(f"  torch/CUDA check failed: {exc}")

try:
    import ieegssl
    from ieegssl.provenance import code_hash

    print(f"  ieegssl {ieegssl.__version__}  code_hash={code_hash()}")
except Exception as exc:
    ok = False
    print(f"  cannot import ieegssl: {exc}")

print(f"  IEEG_ROOT={os.environ.get('IEEG_ROOT')}")
print("ENVIRONMENT OK" if ok else "ENVIRONMENT PROBLEM (see above)")
sys.exit(0 if ok else 1)
