"""Phase 1 orchestration for one participant.

All runs for a participant execute inside one process pool that shares a single
GPU (the models are small, so several workers per A100 keep it busy). Every run
writes one JSON file; finished runs are skipped on restart, so a job can be
resubmitted safely after a time-out or pre-emption.
"""

from __future__ import annotations

import json
import os
import uuid
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import baselines as B
from . import metrics as M
from . import train as T
from .config import config_hash
from .data import ParticipantData, label_subset
from .provenance import code_hash, environment

# condition -> (pretrained encoder kind or None, training protocol)
FT_CONDITIONS = {
    "sup": (None, "new"),
    "ssl": ("ssl", "new"),
    "placebo": ("placebo", "new"),
    "sup_legacy": (None, "legacy"),
    "ssl_legacy": ("ssl_legacy", "legacy"),
}
# encoder kind -> (protocol, placebo data?)
PRETRAIN_KINDS = {"ssl": ("new", False), "placebo": ("new", True), "ssl_legacy": ("legacy", False)}
LINEAR_CONDITIONS = ("ridge", "pca_lab", "pca_all")


def run_id(pid, cond, frac=None, seed=None, lr=None) -> str:
    parts = [pid, cond]
    if frac is not None:
        parts.append(f"f{frac:.2f}")
    if seed is not None:
        parts.append(f"s{seed}")
    if lr is not None:
        parts.append(f"lr{lr:g}")
    return "__".join(parts)


def _atomic_json(path: Path, obj: dict) -> None:
    # unique temporary name: several SLURM tasks may write the same shared file at once
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)


def _done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("status") == "ok"
    except Exception:
        return False


class Paths:
    def __init__(self, root: str | Path, phase: str):
        self.root = Path(root)
        self.derived = self.root / "derived"
        base = self.root / "results" / phase
        self.runs, self.preds, self.encoders = base / "runs", base / "preds", base / "encoders"
        self.base = base

    def ensure(self):
        for p in (self.runs, self.preds, self.encoders):
            p.mkdir(parents=True, exist_ok=True)


def build_jobs(pid: str, cfg: dict):
    ph = cfg["phase1"]
    needed = {FT_CONDITIONS[c][0] for c in ph["deep_conditions"] + ph["legacy_conditions"]} - {None}
    pre = [("pretrain", kind, None, seed, None) for seed in ph["seeds"] for kind in sorted(needed)]
    ft = []
    for frac in ph["fractions"]:
        for seed in ph["seeds"]:
            for cond in ph["deep_conditions"]:
                for lr in cfg["finetune"]["lrs"]:
                    ft.append(("finetune", cond, frac, seed, lr))
            for cond in ph["legacy_conditions"]:
                ft.append(("finetune", cond, frac, seed, None))
            for cond in ph["linear_conditions"]:
                ft.append(("linear", cond, frac, seed, None))
    ft.append(("reference", "reference", None, None, None))
    return pre, ft


def job_id(pid: str, job) -> str:
    kind, cond, frac, seed, lr = job
    if kind == "pretrain":
        return run_id(pid, f"pretrain_{cond}", seed=seed)
    return run_id(pid, cond, frac, seed, lr)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_W: dict = {}


def init_worker(derived_path: str, cfg: dict, root: str, phase: str, device: str, feature_file: str | None = None,
                keep_raw: bool = True):
    """``keep_raw=False`` (Phase 4b) skips the un-normalised copies, which only the legacy
    protocol uses; it saves device memory for wide inputs and changes no result."""
    torch.set_num_threads(1)
    data = ParticipantData(derived_path, cfg, feature_file)
    dev = torch.device(device)
    t = {}
    for s in ("train", "val", "test"):
        t[f"xz_{s}"] = torch.from_numpy(data.z(s)).to(dev)
        if keep_raw:
            t[f"xr_{s}"] = torch.from_numpy(data.Xw[s]).to(dev)
        t[f"y_{s}"] = torch.from_numpy(data.Yw[s]).to(dev)
    t["contact_id"] = torch.as_tensor(data.contact_id, device=dev)
    _W.update(data=data, t=t, cfg=cfg, paths=Paths(root, phase), device=dev,
              stamp={"code_hash": code_hash(), "config_hash": config_hash(cfg), "env": environment()})


def _encoder_path(kind: str, seed: int) -> Path:
    return _W["paths"].encoders / f"{_W['data'].pid}__{kind}__s{seed}.pt"


def _record(job, **fields) -> dict:
    kind, cond, frac, seed, lr = job
    d = _W["data"]
    rec = {"run_id": job_id(d.pid, job), "participant": d.pid, "task": d.task, "kind": kind,
           "condition": cond, "fraction": frac, "seed": seed, "lr": lr,
           "n_train_windows": d.n_windows("train"), "n_val_windows": d.n_windows("val"),
           "n_test_windows": d.n_windows("test"), "n_contacts": d.C}
    rec.update(fields)
    rec.update(_W["stamp"])
    return rec


def _norm_targets(y: torch.Tensor):
    flat = y.reshape(-1, y.shape[-1])
    mean, std = flat.mean(0), flat.std(0).clamp_min(1e-6)
    return mean, std


def _run_pretrain(job):
    _, kind, _, seed, _ = job
    d, t, cfg = _W["data"], _W["t"], _W["cfg"]
    protocol, placebo = PRETRAIN_KINDS[kind]
    if placebo:
        x = torch.from_numpy(T.column_permute(d.z("train"), 777 + seed)).to(_W["device"])
    else:
        x = t["xz_train"] if protocol == "new" else t["xr_train"]
    state, info = T.pretrain_encoder(x, t["contact_id"], cfg, seed, protocol)
    torch.save(state, _encoder_path(kind, seed))
    return {"protocol": protocol, "placebo": placebo, "pretrain": info}


def _run_finetune(job):
    _, cond, frac, seed, lr = job
    d, t, cfg = _W["data"], _W["t"], _W["cfg"]
    enc_kind, protocol = FT_CONDITIONS[cond]
    idx = torch.as_tensor(label_subset(d.n_windows("train"), frac, seed, d.pid), device=_W["device"])
    enc = None
    if enc_kind is not None:
        enc = torch.load(_encoder_path(enc_kind, seed), map_location=_W["device"], weights_only=True)
    if protocol == "new":
        ytr_raw = t["y_train"][idx]
        mean, std = _norm_targets(ytr_raw)
        res = T.finetune(enc, t["xz_train"][idx], (ytr_raw - mean) / std, t["xz_val"],
                         (t["y_val"] - mean) / std, t["xz_test"], cfg, seed, lr, "new")
        m, s = mean.cpu().numpy(), std.cpu().numpy()
        pred_test, pred_val = res["pred_test"] * s + m, res["pred_val"] * s + m
    else:
        res = T.finetune(enc, t["xr_train"][idx], t["y_train"][idx], t["xr_val"], t["y_val"],
                         t["xr_test"], cfg, seed, None, "legacy")
        pred_test, pred_val = res["pred_test"], res["pred_val"]
    info = {k: v for k, v in res.items() if not k.startswith("pred_")}
    return len(idx), protocol, info, pred_val, pred_test


def _run_linear(job):
    _, cond, frac, seed, _ = job
    d, cfg = _W["data"], _W["cfg"]
    idx = label_subset(d.n_windows("train"), frac, seed, d.pid)
    F = d.F
    xz_windows = d.z("train")
    xz_tr_all = xz_windows.reshape(-1, F)
    xtr = xz_windows[idx].reshape(-1, F)
    ytr_raw = d.Yw["train"][idx].reshape(-1, d.Yw["train"].shape[-1])
    m, s = ytr_raw.mean(0), np.maximum(ytr_raw.std(0), 1e-6)
    ytr = (ytr_raw - m) / s
    xva, xte = d.z("val").reshape(-1, F), d.z("test").reshape(-1, F)
    yva = (d.Yw["val"].reshape(-1, ytr.shape[1]) - m) / s
    if cond == "ridge":
        res = B.ridge(xtr, ytr, xva, yva, xte, cfg["linear"]["ridge_alphas"])
    elif cond == "pca_lab":
        res = B.pca_ols(xtr, xtr, ytr, xva, yva, xte, cfg["linear"]["pca_components"])
    elif cond == "pca_all":
        res = B.pca_ols(xz_tr_all, xtr, ytr, xva, yva, xte, cfg["linear"]["pca_components"])
    else:
        raise ValueError(cond)
    shape = d.Yw["test"].shape
    pred_test = (res["pred_test"] * s + m).reshape(shape)
    pred_val = (res["pred_val"] * s + m).reshape(d.Yw["val"].shape)
    info = {k: v for k, v in res.items() if not k.startswith("pred_")}
    return len(idx), "closed_form", info, pred_val, pred_test


def run_job(job) -> str:
    d, cfg, paths = _W["data"], _W["cfg"], _W["paths"]
    rid = job_id(d.pid, job)
    out = paths.runs / f"{rid}.json"
    if _done(out):
        return f"{rid}: already done"
    t0 = time.time()
    try:
        kind = job[0]
        if kind == "pretrain":
            rec = _record(job, status="ok", **_run_pretrain(job))
        elif kind == "reference":
            pred = M.vad_oracle_prediction(d.Yw["train"], d.vad["train"], d.vad["test"])
            test = M.evaluate(d.Yw["test"], pred, d.vad["test"], cfg)
            np.savez_compressed(paths.preds / f"{d.pid}__reference.npz",
                                y_test=d.Yw["test"].astype(np.float16), vad_test=d.vad["test"],
                                vad_oracle=pred.astype(np.float16))
            rec = _record(job, status="ok", test_vad_oracle=test, data_summary=d.summary())
        else:
            runner = _run_finetune if kind == "finetune" else _run_linear
            n_lab, protocol, info, pred_val, pred_test = runner(job)
            test = M.evaluate(d.Yw["test"], pred_test, d.vad["test"], cfg)
            val_r = M.mean_band_r(d.Yw["val"].reshape(-1, pred_val.shape[-1]),
                                  pred_val.reshape(-1, pred_val.shape[-1]))
            if cfg["phase1"].get("save_predictions", True):
                np.savez_compressed(paths.preds / f"{rid}.npz", pred_test=pred_test.astype(np.float16))
            rec = _record(job, status="ok", protocol=protocol, n_label_windows=int(n_lab),
                          label_seconds=d.seconds(n_lab), train=info,
                          val={"mse": info.get("best_val_mse"), "r_mean": val_r}, test=test)
        rec["elapsed_s"] = round(time.time() - t0, 2)
        _atomic_json(out, rec)
        r = rec.get("test", rec.get("test_vad_oracle", {})).get("r_mean")
        return f"{rid}: ok in {rec['elapsed_s']:.1f}s" + (f" | test r={r:.3f}" if r is not None else "")
    except Exception as exc:  # record the failure and keep going
        _atomic_json(out, {"run_id": rid, "status": "error", "error": repr(exc),
                           "traceback": traceback.format_exc(), **_W["stamp"]})
        return f"{rid}: ERROR {exc!r}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_participant(pid: str, cfg: dict, root: str | Path, phase: str = "phase1",
                    workers: int = 8, device: str = "auto") -> None:
    paths = Paths(root, phase)
    paths.ensure()
    derived = paths.derived / f"{pid}.npz"
    if not derived.exists():
        raise FileNotFoundError(f"{derived} not found; run preprocessing first")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _atomic_json(paths.base / f"config_{config_hash(cfg)}.json", cfg)

    pre, ft = build_jobs(pid, cfg)
    todo_pre = [j for j in pre if not _done(paths.runs / f"{job_id(pid, j)}.json")
                or not (paths.encoders / f"{pid}__{j[1]}__s{j[3]}.pt").exists()]
    todo_ft = [j for j in ft if not _done(paths.runs / f"{job_id(pid, j)}.json")]
    # Re-run pretraining whose encoder file is missing.
    for j in todo_pre:
        (paths.runs / f"{job_id(pid, j)}.json").unlink(missing_ok=True)
    print(f"{pid}: {len(todo_pre)} pretraining + {len(todo_ft)} other runs to do "
          f"(device={device}, workers={workers}, code={code_hash()}, config={config_hash(cfg)})",
          flush=True)
    t0 = time.time()
    args = (str(derived), cfg, str(root), phase, device)
    if workers <= 1:
        init_worker(*args)
        for j in todo_pre + todo_ft:
            print(run_job(j), flush=True)
    else:
        ctx = torch.multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=init_worker, initargs=args) as pool:
            for line in pool.imap_unordered(run_job, todo_pre, chunksize=1):
                print(line, flush=True)
            for line in pool.imap_unordered(run_job, todo_ft, chunksize=1):
                print(line, flush=True)
    print(f"{pid}: finished in {(time.time() - t0) / 60:.1f} min", flush=True)
