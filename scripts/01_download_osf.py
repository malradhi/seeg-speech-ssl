"""Download the 30-participant dataset from OSF (project ak3dp) on the login node.

Usage:
    python scripts/01_download_osf.py                    # everything
    python scripts/01_download_osf.py --only P15         # one participant (smoke test)
    python scripts/01_download_osf.py --list             # list files and sizes only

Re-running is safe: verified files are skipped and partial files are resumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common
from ieegssl.config import OSF_NODE
from ieegssl.osf import download, list_files

ap = argparse.ArgumentParser()
ap.add_argument("--dest", default=None, help="default: $IEEG_ROOT/raw")
ap.add_argument("--only", default=None, help="comma-separated participant prefixes, e.g. P15,P01")
ap.add_argument("--list", action="store_true")
args = ap.parse_args()

dest = Path(args.dest or Path(_common.default_root()) / "raw")
dest.mkdir(parents=True, exist_ok=True)

print(f"Listing OSF project {OSF_NODE} ...", flush=True)
files = list_files(OSF_NODE)
if args.only:
    prefixes = tuple(p.strip() + "_" for p in args.only.split(","))
    files = [f for f in files if f["name"].startswith(prefixes)]
total = sum(int(f["size"] or 0) for f in files)
print(f"{len(files)} files, {total / 1e9:.2f} GB", flush=True)
(dest / "osf_manifest.json").write_text(json.dumps(files, indent=1))
if args.list:
    for f in files:
        print(f"  {f['name']:40s} {int(f['size'] or 0) / 1e6:9.1f} MB")
    raise SystemExit(0)

failed = []
for i, meta in enumerate(sorted(files, key=lambda m: m["name"]), 1):
    try:
        download(meta, dest)
        print(f"[{i}/{len(files)}] {meta['name']} OK", flush=True)
    except Exception as exc:
        failed.append(meta["name"])
        print(f"[{i}/{len(files)}] {meta['name']} FAILED: {exc}", flush=True)

if failed:
    print(f"{len(failed)} files failed: {failed}\nRe-run the same command to retry.")
    raise SystemExit(1)
print(f"ALL OK: {len(files)} files in {dest}")
