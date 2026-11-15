"""Phase 4 runs (docs/PHASE4_DESIGN.md). One SLURM array task per participant:

1. BrainBERT features (published pretrained intracranial encoder), extracted once on the GPU
   and frame-aligned with the derived high-gamma features; then two readouts on them with
   the same label subsets, budgets and selection as every other condition: ``bb_ridge``
   (closed-form ridge) and ``bb_gru`` (the bidirectional GRU, six learning rates).
2. Dose-response of own-patient unlabelled data: masked pretraining of the Transformer
   (``ssl_uXX``) and of the GRU (``gru_ssl_uXX``) on 12.5%, 25% and 50% of the training
   partition (nested subsets that contain the labelled windows), fine-tuned at 10% and 5%
   labels; frozen probes of the same encoders.
3. Linear-probe-then-fine-tune of the Phase 1 Transformer and Phase 3 GRU encoders
   (``lpft_ssl``, ``lpft_gru_ssl``) within the same 3,000-step budget.
4. Per-contact anatomy table (location label, grey-matter density, coordinates) for the
   moderator analysis.

Same worker set-up, record format and pairing as Phases 1-3.
"""

from __future__ import annotations

import csv
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import baselines as B
from . import metrics as M
from . import runner as R
from . import train as T
from .config import PARTICIPANTS, config_hash
from .data import label_subset
from .provenance import code_hash


def phase4_participants(cfg: dict) -> list:
    return [str(p) for p in (cfg["phase4"]["participants"] or PARTICIPANTS)]


def dose_name(arch: str, u: float) -> str:
    return f"{'gru_ssl' if arch == 'gru' else 'ssl'}_u{int(round(u * 1000)):03d}"  # 0.125 -> u125


def build_jobs4(pid: str, cfg: dict):
    ph, g4 = cfg["phase4"], cfg["finetune"]["lrs"]
    sm, ss, fr = ph["seeds_main"], ph["seeds_small"], ph["fractions"]
    pre, ft, probes, bb = [], [], [], []
    if ph["run_dose"]:
        for arch in ("transformer", "gru"):
            for u in ph["dose_fractions"]:
                name = dose_name(arch, u)
                pre += [("pretrain4", name, u, s, None) for s in ss]
                grid = g4 if arch == "transformer" else ph["grid6"]
                ft += [("finetune", name, f, s, lr) for f in ph["dose_label_fractions"] for s in ss for lr in grid]
                probes += [("probe", name, None, s, None) for s in ss]
    if ph["run_lpft"]:
        for cond in ("lpft_ssl", "lpft_gru_ssl"):
            ft += [("finetune", cond, f, s, lr) for f in fr for s in sm for lr in ph["grid6"]]
    if ph["run_brainbert"]:
        bb += [("linear", "bb_ridge", f, s, None) for f in fr for s in sm]
        bb += [("finetune", "bb_gru", f, s, lr) for f in fr for s in sm for lr in ph["grid6"]]
    return pre, ft + probes, bb


def _job_id(pid: str, job) -> str:
    kind, cond, frac, seed, lr = job
    if kind == "pretrain4":
        return R.run_id(pid, f"pretrain_{cond}", seed=seed)
    if kind == "probe":
        return R.run_id(pid, f"probe_{cond}", seed=seed)
    return R.run_id(pid, cond, frac, seed, lr)


def _arch(cond: str) -> str:
    return "gru" if ("gru" in cond) else "transformer"


def _enc_path(cond: str, seed: int) -> Path:
    pid, root, ph = R._W["data"].pid, R._W["paths"].root, R._W["cfg"]["phase4"]
    if cond == "lpft_ssl":
        return R.Paths(root, ph["phase1_dir"]).encoders / f"{pid}__ssl__s{seed}.pt"
    if cond == "lpft_gru_ssl":
        return R.Paths(root, ph["phase3_dir"]).encoders / f"{pid}__gru_ssl__s{seed}.pt"
    return R._W["paths"].encoders / f"{pid}__{cond}__s{seed}.pt"


def _load_enc(cond: str, seed: int):
    path = _enc_path(cond, seed)
    if not path.exists():
        raise FileNotFoundError(f"encoder missing: {path}")
    return torch.load(path, map_location=R._W["device"], weights_only=True)


def _run_pretrain4(job) -> dict:
    _, name, u, seed, _ = job
    d, t = R._W["data"], R._W["t"]
    sub = torch.as_tensor(label_subset(d.n_windows("train"), u, seed, d.pid), device=R._W["device"])
    arch = _arch(name)
    state, info = T.pretrain_encoder(t["xz_train"][sub], t["contact_id"], R._W["cfg"], seed, "new", arch=arch)
    torch.save(state, _enc_path(name, seed))
    info["unlabelled_windows"] = int(len(sub))
    info["unlabelled_seconds"] = d.seconds(len(sub))
    return {"protocol": "new", "arch": arch, "unlabelled_fraction": u, "pretrain": info}


def _run_finetune4(job):
    _, cond, frac, seed, lr = job
    d, t, cfg = R._W["data"], R._W["t"], R._W["cfg"]
    idx = torch.as_tensor(label_subset(d.n_windows("train"), frac, seed, d.pid), device=R._W["device"])
    enc = None if cond == "bb_gru" else _load_enc(cond, seed)
    lp = cfg["phase4"]["lp"] if cond.startswith("lpft_") else None
    ytr_raw = t["y_train"][idx]
    mean, std = R._norm_targets(ytr_raw)
    res = T.finetune(enc, t["xz_train"][idx], (ytr_raw - mean) / std, t["xz_val"], (t["y_val"] - mean) / std,
                     t["xz_test"], cfg, seed, lr, "new", arch=_arch(cond), lp=lp)
    m, s = mean.cpu().numpy(), std.cpu().numpy()
    info = {k: v for k, v in res.items() if not k.startswith("pred_")}
    info["arch"] = _arch(cond)
    return len(idx), info, res["pred_val"] * s + m, res["pred_test"] * s + m


def _run_bb_ridge(job):
    _, _, frac, seed, _ = job
    d, cfg = R._W["data"], R._W["cfg"]
    idx = label_subset(d.n_windows("train"), frac, seed, d.pid)
    F = d.F
    xz = d.z("train")
    ytr_raw = d.Yw["train"][idx].reshape(-1, d.Yw["train"].shape[-1])
    m, s = ytr_raw.mean(0), np.maximum(ytr_raw.std(0), 1e-6)
    yva = (d.Yw["val"].reshape(-1, ytr_raw.shape[1]) - m) / s
    res = B.ridge(xz[idx].reshape(-1, F), (ytr_raw - m) / s, d.z("val").reshape(-1, F), yva,
                  d.z("test").reshape(-1, F), cfg["linear"]["ridge_alphas"])
    pred_test = (res["pred_test"] * s + m).reshape(d.Yw["test"].shape)
    pred_val = (res["pred_val"] * s + m).reshape(d.Yw["val"].shape)
    info = {k: v for k, v in res.items() if not k.startswith("pred_")}
    return len(idx), info, pred_val, pred_test


def _run_probe4(job) -> str:
    _, name, _, seed, _ = job
    d, t, cfg, paths = R._W["data"], R._W["t"], R._W["cfg"], R._W["paths"]
    arch = _arch(name)
    state = _load_enc(name, seed)
    feats = {s: T.encode(state, t[f"xz_{s}"], cfg, arch=arch) for s in ("train", "val", "test")}
    dm = feats["train"].shape[-1]
    ftr = feats["train"].reshape(-1, dm)
    mu, sd = ftr.mean(0), np.maximum(ftr.std(0), 1e-6)
    z = {s: (feats[s] - mu) / sd for s in feats}
    lines = []
    for frac in cfg["phase4"]["dose_label_fractions"]:
        sub = ("probe", f"probe_{name}", frac, seed, None)
        out = paths.runs / f"{R.run_id(d.pid, f'probe_{name}', frac, seed)}.json"
        if R._done(out):
            continue
        t0 = time.time()
        idx = label_subset(d.n_windows("train"), frac, seed, d.pid)
        yr = d.Yw["train"][idx].reshape(-1, d.Yw["train"].shape[-1])
        ym, ys = yr.mean(0), np.maximum(yr.std(0), 1e-6)
        yva = (d.Yw["val"].reshape(-1, yr.shape[1]) - ym) / ys
        res = B.ridge(z["train"][idx].reshape(-1, dm), (yr - ym) / ys, z["val"].reshape(-1, dm), yva,
                      z["test"].reshape(-1, dm), cfg["linear"]["ridge_alphas"])
        pred_test = (res["pred_test"] * ys + ym).reshape(d.Yw["test"].shape)
        pred_val = (res["pred_val"] * ys + ym).reshape(d.Yw["val"].shape)
        test = M.evaluate(d.Yw["test"], pred_test, d.vad["test"], cfg)
        val_r = M.mean_band_r(d.Yw["val"].reshape(-1, yr.shape[1]), pred_val.reshape(-1, yr.shape[1]))
        info = {k: v for k, v in res.items() if not k.startswith("pred_")}
        rec = R._record(sub, status="ok", protocol="frozen_probe", encoder=name, n_label_windows=int(len(idx)),
                        label_seconds=d.seconds(len(idx)), train=info,
                        val={"mse": info["best_val_mse"], "r_mean": val_r}, test=test,
                        elapsed_s=round(time.time() - t0, 2))
        rec["kind"] = "probe"
        R._atomic_json(out, rec)
        lines.append(f"f{frac:.2f} r={test['r_mean']:.3f}")
    return ", ".join(lines) or "already done"


def run_job4(job) -> str:
    d, cfg, paths = R._W["data"], R._W["cfg"], R._W["paths"]
    rid = _job_id(d.pid, job)
    out = paths.runs / f"{rid}.json"
    kind = job[0]
    if kind != "probe" and R._done(out):
        return f"{rid}: already done"
    t0 = time.time()
    try:
        if kind == "pretrain4":
            rec = R._record(("pretrain", job[1], None, job[3], None), status="ok", **_run_pretrain4(job))
            rec["run_id"], rec["kind"] = rid, "pretrain"
        elif kind == "probe":
            return f"{rid}: {_run_probe4(job)} ({time.time() - t0:.0f}s)"
        else:
            if kind == "linear":
                n_lab, info, pred_val, pred_test = _run_bb_ridge(job)
                protocol = "closed_form"
            else:
                n_lab, info, pred_val, pred_test = _run_finetune4(job)
                protocol = "new"
            test = M.evaluate(d.Yw["test"], pred_test, d.vad["test"], cfg)
            nm = pred_val.shape[-1]
            val_r = M.mean_band_r(d.Yw["val"].reshape(-1, nm), pred_val.reshape(-1, nm))
            if cfg["phase4"].get("save_predictions", True):
                np.savez_compressed(paths.preds / f"{rid}.npz", pred_test=pred_test.astype(np.float16))
            rec = R._record(job, status="ok", protocol=protocol, n_label_windows=int(n_lab),
                            label_seconds=d.seconds(n_lab), train=info,
                            val={"mse": info.get("best_val_mse"), "r_mean": val_r}, test=test)
            rec["features"] = "brainbert" if job[1].startswith("bb_") else "high_gamma"
        rec["elapsed_s"] = round(time.time() - t0, 2)
        R._atomic_json(out, rec)
        r = rec.get("test", {}).get("r_mean")
        return f"{rid}: ok in {rec['elapsed_s']:.1f}s" + (f" | test r={r:.3f}" if r is not None else "")
    except Exception as exc:
        R._atomic_json(out, {"run_id": rid, "status": "error", "error": repr(exc),
                             "traceback": traceback.format_exc(), **R._W["stamp"]})
        return f"{rid}: ERROR {exc!r}"


def export_anatomy(pid: str, raw_root, derived: Path, out_dir: Path) -> Path | None:
    """Per kept contact: location label, grey-matter density (PTD) and coordinates."""
    import json

    from .preprocessing import find_raw_files

    out = out_dir / f"{pid}.csv"
    if out.exists():
        return out
    files = find_raw_files(raw_root, pid)
    if "locations" not in files:
        return None
    with np.load(derived, allow_pickle=False) as d:
        kept = json.loads(str(d["meta_json"]))["channels"]
    with open(files["locations"], newline="") as fh:
        rows = {r["electrode_name_1"].strip(): r for r in csv.DictReader(fh)}
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["participant", "contact_index", "channel", "location", "PTD", "x", "y", "z"])
        for i, ch in enumerate(kept):
            r = rows.get(ch, {})
            w.writerow([pid, i, ch, r.get("location", ""), r.get("PTD", ""), r.get("coord_x", ""),
                        r.get("coord_y", ""), r.get("coord_z", "")])
    tmp.replace(out)
    return out


def run_participant4(pid: str, cfg: dict, root, workers: int = 8, device: str = "auto", raw_root=None) -> None:
    from .brainbert import extract_participant

    ph = cfg["phase4"]
    phase = ph["results_dir"]
    paths = R.Paths(root, phase)
    paths.ensure()
    derived = paths.derived / f"{pid}.npz"
    if not derived.exists():
        raise FileNotFoundError(f"{derived} not found")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(root)
    raw_root = Path(raw_root) if raw_root else root / "raw"
    R._atomic_json(paths.base / f"config_{config_hash(cfg)}.json", cfg)
    export_anatomy(pid, raw_root, derived, root / "results" / "anatomy")
    bb_file = None
    if ph["run_brainbert"]:
        bcfg = cfg["brainbert"]
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False  # full fp32, as published
        bb_file = extract_participant(pid, raw_root, paths.derived, root / bcfg["features_dir"], cfg,
                                      root / bcfg["model_dir"], device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pre, rest, bb = build_jobs4(pid, cfg)

    def todo(job):
        if job[0] == "probe":
            return not all(R._done(paths.runs / f"{R.run_id(pid, f'probe_{job[1]}', f, job[3])}.json")
                           for f in ph["dose_label_fractions"])
        if job[0] == "pretrain4":
            return not (R._done(paths.runs / f"{_job_id(pid, job)}.json")
                        and (paths.encoders / f"{pid}__{job[1]}__s{job[3]}.pt").exists())
        return not R._done(paths.runs / f"{_job_id(pid, job)}.json")

    pre, rest, bb = [j for j in pre if todo(j)], [j for j in rest if todo(j)], [j for j in bb if todo(j)]
    print(f"{pid}: phase 4 -> {len(pre)} pretraining + {len(rest)} high-gamma + {len(bb)} BrainBERT jobs "
          f"(device={device}, workers={workers}, code={code_hash()}, config={config_hash(cfg)})", flush=True)
    t0 = time.time()
    stages = [(pre + rest, None)]
    if bb:
        stages.append((bb, str(bb_file)))
    for jobs, feature_file in stages:
        if not jobs:
            continue
        args = (str(derived), cfg, str(root), phase, device, feature_file)
        if workers <= 1:
            R.init_worker(*args)
            for j in jobs:
                print(run_job4(j), flush=True)
        else:
            ctx = torch.multiprocessing.get_context("spawn")
            first = [j for j in jobs if j[0] == "pretrain4"]
            later = [j for j in jobs if j[0] != "pretrain4"]
            with ctx.Pool(workers, initializer=R.init_worker, initargs=args) as pool:
                for line in pool.imap_unordered(run_job4, first, chunksize=1):
                    print(line, flush=True)
                for line in pool.imap_unordered(run_job4, later, chunksize=1):
                    print(line, flush=True)
    print(f"{pid}: phase 4 finished in {(time.time() - t0) / 60:.1f} min", flush=True)
