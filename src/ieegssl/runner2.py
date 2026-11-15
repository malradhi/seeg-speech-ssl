"""Phase 2 per-participant runs (docs/PHASE2_DESIGN.md).

Requires the Phase 1 encoders (results/phase1/encoders) and the multi-participant
encoders written by multi.py (results/phase2/encoders). Uses the same worker set-up,
record format and pairing as Phase 1: identical labelled subsets, head
initialisation and minibatch order for every condition of a given seed.
"""

from __future__ import annotations

import time
import traceback

import numpy as np
import torch

from . import baselines as B
from . import metrics as M
from . import runner as R
from . import train as T
from .config import config_hash
from .data import label_subset
from .provenance import code_hash


def mask_name(kind: str, ratio: float) -> str:
    return f"mask_{kind}{int(round(ratio * 100))}"


def build_jobs2(pid: str, cfg: dict):
    ph, lrs = cfg["phase2"], cfg["finetune"]["lrs"]
    seeds_m, seeds_s, fr = ph["seeds_main"], ph["seeds_small"], ph["fractions"]
    grid = [(k, r) for k in ph["mask_kinds"] for r in ph["mask_ratios"] if not (k == "random" and r == 0.5)]
    pre = [("pretrain_mask", mask_name(k, r), r, s, k) for k, r in grid for s in seeds_s]

    ft = []
    for cond in ph["lr_extra_conditions"]:
        ft += [("finetune", cond, f, s, lr) for f in fr for s in seeds_m for lr in ph["lr_extra"]]
    for k, r in grid:
        ft += [("finetune", mask_name(k, r), f, s, lr) for f in ph["mask_fractions"] for s in seeds_s for lr in lrs]
    ft += [("finetune", "pooled", f, s, lr) for f in fr for s in seeds_m for lr in lrs]
    for cond in ("loo_adapt", "loo_noadapt"):
        ft += [("finetune", cond, f, s, lr) for f in fr for s in seeds_s for lr in lrs]
    n_total = len(cfg["phase2"]["participants"] or range(30))
    for k in ph["scaling_k"]:
        if k < n_total:
            ft += [("finetune", f"scale{k}", f, s, lr) for f in ph["scaling_fractions"] for s in seeds_s for lr in lrs]
    if ph["run_gru"]:
        ft += [("finetune", "gru", f, s, lr) for f in fr for s in seeds_m for lr in lrs]

    probes = []
    if ph["run_probes"]:
        probes += [("probe", enc, None, s, None) for enc in ("random", "ssl", "placebo", "pooled") for s in seeds_m]
        probes += [("probe", mask_name(k, r), None, s, None) for k, r in grid for s in seeds_s]
        probes += [("probe", "loo_adapt", None, s, None) for s in seeds_s]
    return pre, ft + probes


def _job_id(pid: str, job) -> str:
    kind, cond, frac, seed, lr = job
    if kind == "pretrain_mask":
        return R.run_id(pid, f"pretrain_{cond}", seed=seed)
    if kind == "probe":
        return R.run_id(pid, f"probe_{cond}", seed=seed)  # one file per fraction is written as well
    return R.run_id(pid, cond, frac, seed, lr)


def _enc_file(cond: str, seed: int):
    pid = R._W["data"].pid
    if cond in ("ssl", "placebo"):
        root = R._W["paths"].root
        return R.Paths(root, R._W["cfg"]["phase2"]["phase1_dir"]).encoders / f"{pid}__{cond}__s{seed}.pt"
    return R._W["paths"].encoders / f"{pid}__{cond}__s{seed}.pt"


def _load_enc(cond: str, seed: int):
    path = _enc_file(cond, seed)
    if not path.exists():
        raise FileNotFoundError(f"encoder missing: {path}")
    return torch.load(path, map_location=R._W["device"], weights_only=True)


def _run_pretrain_mask(job) -> dict:
    _, name, ratio, seed, kind = job
    t = R._W["t"]
    state, info = T.pretrain_encoder(t["xz_train"], t["contact_id"], R._W["cfg"], seed, "new",
                                     mask_kind=kind, mask_ratio=ratio)
    torch.save(state, R._W["paths"].encoders / f"{R._W['data'].pid}__{name}__s{seed}.pt")
    return {"protocol": "new", "pretrain": info}


def _run_finetune2(job):
    _, cond, frac, seed, lr = job
    d, t, cfg = R._W["data"], R._W["t"], R._W["cfg"]
    idx = torch.as_tensor(label_subset(d.n_windows("train"), frac, seed, d.pid), device=R._W["device"])
    arch = "gru" if cond == "gru" else "transformer"
    enc = None if cond in ("sup", "gru") else _load_enc(cond, seed)
    ytr_raw = t["y_train"][idx]
    mean, std = R._norm_targets(ytr_raw)
    res = T.finetune(enc, t["xz_train"][idx], (ytr_raw - mean) / std, t["xz_val"],
                     (t["y_val"] - mean) / std, t["xz_test"], cfg, seed, lr, "new", arch=arch)
    m, s = mean.cpu().numpy(), std.cpu().numpy()
    info = {k: v for k, v in res.items() if not k.startswith("pred_")}
    return len(idx), info, res["pred_val"] * s + m, res["pred_test"] * s + m


def _run_probe(job) -> str:
    _, enc_name, _, seed, _ = job
    d, t, cfg, paths = R._W["data"], R._W["t"], R._W["cfg"], R._W["paths"]
    if enc_name == "random":
        state = T.random_backbone_state(d.F, d.window, cfg, seed)
    else:
        state = _load_enc(enc_name, seed)
    feats = {s: T.encode(state, t[f"xz_{s}"], cfg) for s in ("train", "val", "test")}
    dm = feats["train"].shape[-1]
    ftr = feats["train"].reshape(-1, dm)
    mu, sd = ftr.mean(0), np.maximum(ftr.std(0), 1e-6)
    z = {s: (feats[s] - mu) / sd for s in feats}
    lines = []
    for frac in cfg["phase2"]["fractions"]:
        sub = ("probe", f"probe_{enc_name}", frac, seed, None)
        out = paths.runs / f"{R.run_id(d.pid, f'probe_{enc_name}', frac, seed)}.json"
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
        rec = R._record(sub, status="ok", protocol="frozen_probe", encoder=enc_name, n_label_windows=int(len(idx)),
                        label_seconds=d.seconds(len(idx)), train=info,
                        val={"mse": info["best_val_mse"], "r_mean": val_r}, test=test,
                        elapsed_s=round(time.time() - t0, 2))
        rec["kind"] = "probe"
        R._atomic_json(out, rec)
        lines.append(f"f{frac:.2f} r={test['r_mean']:.3f}")
    return ", ".join(lines) or "already done"


def run_job2(job) -> str:
    d, cfg, paths = R._W["data"], R._W["cfg"], R._W["paths"]
    rid = _job_id(d.pid, job)
    out = paths.runs / f"{rid}.json"
    kind = job[0]
    if kind != "probe" and R._done(out):
        return f"{rid}: already done"
    t0 = time.time()
    try:
        if kind == "pretrain_mask":
            rec = R._record(("pretrain", job[1], None, job[3], None), status="ok", **_run_pretrain_mask(job))
            rec["run_id"], rec["kind"] = rid, "pretrain"
        elif kind == "probe":
            return f"{rid}: {_run_probe(job)} ({time.time() - t0:.0f}s)"
        else:
            n_lab, info, pred_val, pred_test = _run_finetune2(job)
            test = M.evaluate(d.Yw["test"], pred_test, d.vad["test"], cfg)
            val_r = M.mean_band_r(d.Yw["val"].reshape(-1, pred_val.shape[-1]), pred_val.reshape(-1, pred_val.shape[-1]))
            if cfg["phase2"].get("save_predictions", True):
                np.savez_compressed(paths.preds / f"{rid}.npz", pred_test=pred_test.astype(np.float16))
            rec = R._record(job, status="ok", protocol="new", n_label_windows=int(n_lab),
                            label_seconds=d.seconds(n_lab), train=info,
                            val={"mse": info.get("best_val_mse"), "r_mean": val_r}, test=test)
        rec["elapsed_s"] = round(time.time() - t0, 2)
        R._atomic_json(out, rec)
        r = rec.get("test", {}).get("r_mean")
        return f"{rid}: ok in {rec['elapsed_s']:.1f}s" + (f" | test r={r:.3f}" if r is not None else "")
    except Exception as exc:
        R._atomic_json(out, {"run_id": rid, "status": "error", "error": repr(exc),
                             "traceback": traceback.format_exc(), **R._W["stamp"]})
        return f"{rid}: ERROR {exc!r}"


def run_participant2(pid: str, cfg: dict, root, workers: int = 8, device: str = "auto") -> None:
    phase = cfg["phase2"]["results_dir"]
    paths = R.Paths(root, phase)
    paths.ensure()
    derived = paths.derived / f"{pid}.npz"
    if not derived.exists():
        raise FileNotFoundError(f"{derived} not found")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    R._atomic_json(paths.base / f"config_{config_hash(cfg)}.json", cfg)
    pre, rest = build_jobs2(pid, cfg)

    def todo(job):
        if job[0] == "probe":
            return not all(R._done(paths.runs / f"{R.run_id(pid, f'probe_{job[1]}', f, job[3])}.json")
                           for f in cfg["phase2"]["fractions"])
        if job[0] == "pretrain_mask":
            return not (R._done(paths.runs / f"{_job_id(pid, job)}.json")
                        and (paths.encoders / f"{pid}__{job[1]}__s{job[3]}.pt").exists())
        return not R._done(paths.runs / f"{_job_id(pid, job)}.json")

    pre, rest = [j for j in pre if todo(j)], [j for j in rest if todo(j)]
    print(f"{pid}: phase 2 -> {len(pre)} pretraining + {len(rest)} other jobs (device={device}, "
          f"workers={workers}, code={code_hash()}, config={config_hash(cfg)})", flush=True)
    t0 = time.time()
    args = (str(derived), cfg, str(root), phase, device)
    if workers <= 1:
        R.init_worker(*args)
        for j in pre + rest:
            print(run_job2(j), flush=True)
    else:
        ctx = torch.multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=R.init_worker, initargs=args) as pool:
            for line in pool.imap_unordered(run_job2, pre, chunksize=1):
                print(line, flush=True)
            for line in pool.imap_unordered(run_job2, rest, chunksize=1):
                print(line, flush=True)
    print(f"{pid}: phase 2 finished in {(time.time() - t0) / 60:.1f} min", flush=True)
