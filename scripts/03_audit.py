"""Audit all derived files and write the dataset summary table.

Checks: finite features/targets, lag dimensionality, non-overlapping partitions,
boundaries placed in silence, durations, windows, speech fraction, and how many
test stimuli were already seen in training (relevant for repeated words).

Output: $IEEG_ROOT/audit/data_summary.csv  (send this file back)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import _common
from ieegssl.config import load_config
from ieegssl.data import ParticipantData

NON_NEURAL = r"^(MKR|EKG|ECG|PULS|BEAT|SPO2)"

ap = argparse.ArgumentParser()
ap.add_argument("--participants", default=None)
ap.add_argument("--config", default=None)
ap.add_argument("--root", default=None)
args = ap.parse_args()

root = Path(args.root or _common.default_root())
cfg = load_config(args.config)
out_dir = root / "audit"
out_dir.mkdir(parents=True, exist_ok=True)
rows, problems = [], []
for pid in _common.resolve_participants(args.participants, None):
    path = root / "derived" / f"{pid}.npz"
    if not path.exists():
        problems.append(f"{pid}: derived file missing")
        continue
    d = ParticipantData(path, cfg)
    s = d.summary()
    b = d.meta["bounds_samples"]
    s["partitions_disjoint"] = b["train"][1] <= b["val"][0] and b["val"][1] <= b["test"][0]
    s["lag_dim_ok"] = d.Xw["train"].shape[-1] == d.C * d.n_lags
    kept = d.meta.get("channels", [])
    s["no_nonneural_channels"] = not any(re.match(NON_NEURAL, c, re.IGNORECASE) for c in kept)
    rows.append(s)
    for key in ("features_finite", "mel_finite", "partitions_disjoint", "lag_dim_ok", "no_nonneural_channels"):
        if not s[key]:
            problems.append(f"{pid}: {key} failed")
    print(f"{pid} {s['task']:15s} contacts={s['n_contacts']:4d} dur={s['duration_s']:7.1f}s "
          f"windows={s['train_windows']}/{s['val_windows']}/{s['test_windows']} "
          f"speech={s['train_speech_frac']:.2f}", flush=True)

keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "participant", k))
with open(out_dir / "data_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {out_dir / 'data_summary.csv'} ({len(rows)} participants)")
for task in ("unique_words", "repeated_words", "sentences"):
    sel = [r for r in rows if r["task"] == task]
    if sel:
        c = [r["n_contacts"] for r in sel]
        print(f"  {task:15s} {len(sel):2d} participants, contacts used {sum(c)} (range {min(c)}-{max(c)}), "
              f"recorded {sum(r['n_contacts_recorded'] for r in sel)}")
print(f"  TOTAL contacts used: {sum(r['n_contacts'] for r in rows)}")
if problems:
    print("PROBLEMS:\n  " + "\n  ".join(problems))
    raise SystemExit(1)
print("AUDIT PASSED")
