"""Aggregation helpers shared by collect/quicklook (numpy + stdlib only)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FLAT_FIELDS = [
    "run_id", "status", "participant", "task", "kind", "condition", "protocol", "fraction", "seed",
    "lr", "n_label_windows", "label_seconds", "n_train_windows", "n_val_windows", "n_test_windows",
    "n_contacts", "val_mse", "val_r", "r_mean", "r_speech", "mcd_db", "mcd_speech_db", "mse_logmel",
    "chance_mean", "chance_p95", "p_value", "best_step", "steps_run", "epochs_equivalent", "alpha",
    "pretrain_final_loss", "elapsed_s", "code_hash", "config_hash", "error",
    # Phase 4 / 4b additions (empty for earlier phases)
    "lp_steps_run", "lp_best_step", "lp_lr", "unlabelled_fraction", "unlabelled_seconds", "features",
    "bb_components",
]


def load_runs(runs_dir: str | Path) -> list[dict]:
    out = []
    for p in sorted(Path(runs_dir).glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception as exc:
            out.append({"run_id": p.stem, "status": "unreadable", "error": repr(exc)})
    return out


def flatten(rec: dict) -> dict:
    row = {k: rec.get(k) for k in FLAT_FIELDS}
    test = rec.get("test") or rec.get("test_vad_oracle") or {}
    for k in ("r_mean", "r_speech", "mcd_db", "mcd_speech_db", "mse_logmel", "chance_mean",
              "chance_p95", "p_value"):
        row[k] = test.get(k)
    val = rec.get("val") or {}
    row["val_mse"], row["val_r"] = val.get("mse"), val.get("r_mean")
    tr = rec.get("train") or {}
    for k in ("best_step", "steps_run", "epochs_equivalent", "alpha", "lp_steps_run", "lp_best_step", "lp_lr"):
        row[k] = tr.get(k)
    if rec.get("kind") == "reference":
        row["condition"] = "vad_oracle"
    pre = rec.get("pretrain") or {}
    if pre.get("loss_curve"):
        row["pretrain_final_loss"] = pre["loss_curve"][-1]
    if pre.get("unlabelled_seconds") is not None:
        row["unlabelled_seconds"] = pre["unlabelled_seconds"]
    if row.get("error"):
        row["error"] = str(row["error"])[:300]
    return row


def best_lr_rows(rows: list[dict]) -> list[dict]:
    """For conditions run over an LR grid keep, per (participant, condition,
    fraction, seed), the LR with the lowest validation MSE."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok" or r.get("kind") not in ("finetune", "linear", "probe"):
            continue
        groups[(r["participant"], r["condition"], r["fraction"], r["seed"])].append(r)
    return [min(g, key=lambda r: (r["val_mse"] if r["val_mse"] is not None else np.inf))
            for g in groups.values()]


def participant_means(rows: list[dict], metric: str = "r_mean") -> dict:
    """{(condition, fraction): {participant: mean over seeds}}"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(metric)
        if v is not None and np.isfinite(v):
            acc[(r["condition"], r["fraction"])][r["participant"]].append(v)
    return {k: {p: float(np.mean(v)) for p, v in d.items()} for k, d in acc.items()}
