"""Download the released BrainBERT checkpoint (login node, ~175 MB, once).

    python scripts/04_download_brainbert.py

Source: huggingface.co/braindecode/brainbert-pretrained (the authors' "large" checkpoint,
re-hosted by braindecode). Writes $IEEG_ROOT/models/brainbert/{config.json, pytorch_model.bin}
and prints the SHA-256 of the weights, which every Phase 4 record stores.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import _common
from ieegssl.config import load_config

cfg = load_config()
root = Path(_common.default_root())
dest = root / cfg["brainbert"]["model_dir"]
dest.mkdir(parents=True, exist_ok=True)
base = f"https://huggingface.co/{cfg['brainbert']['hf_repo']}/resolve/main/"
for name in ("config.json", "pytorch_model.bin"):
    out = dest / name
    if out.exists() and out.stat().st_size > 0:
        print(f"{name}: present ({out.stat().st_size / 1e6:.1f} MB)")
        continue
    tmp = out.with_suffix(out.suffix + ".part")
    req = urllib.request.Request(base + name, headers={"User-Agent": "ieeg-ssl-jne/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(out)
    print(f"{name}: downloaded ({out.stat().st_size / 1e6:.1f} MB)")
conf = json.loads((dest / "config.json").read_text())
expected = {"hidden_dim": 768, "ffn_dim": 3072, "n_layers": 6, "n_heads": 12, "nperseg": 400, "noverlap": 350,
            "idx_freq_cutoff": 40, "stft_clip": 10}
bad = {k: conf.get(k) for k, v in expected.items() if conf.get(k) != v}
digest = hashlib.sha256((dest / "pytorch_model.bin").read_bytes()).hexdigest()
print(f"sha256(pytorch_model.bin) = {digest}")
if bad:
    print(f"UNEXPECTED CONFIG: {bad}")
    sys.exit(1)
print("BRAINBERT OK")
