"""Phase 3 runs (docs/PHASE3_DESIGN.md).

Per participant (one SLURM array task each):
  * GRU pretraining: masked reconstruction of the GRU baseline's projection + bidirectional
    GRU on the participant's own training data (``gru_ssl``) and on column-permuted data
    (``gru_placebo``); fine-tuned over six learning rates; frozen-feature probes.
  * Learning-rate edges: supervised GRU at 1e-5 and 3e-3 (completing its six-rate grid);
    supervised and SSL Transformer at 1e-2.
  * Larger model: own-patient SSL (``ssl_L``), supervised (``sup_L``) and the pooled
    30-patient model (``pooled_L``, written by ``run_pooled_large``) with a 256-unit,
    six-layer Transformer.

Same worker set-up, record format and pairing as Phases 1-2: identical labelled subsets,
head initialisation and minibatch order for every condition of a given seed.
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
from .config import PARTICIPANTS, config_hash
from .data import ParticipantData, label_subset
from .provenance import code_hash, environment

GRU_CONDS = ("gru", "gru_ssl", "gru_placebo")
LARGE_CONDS = ("sup_L", "ssl_L", "pooled_L")
ENCODER_OF = {"sup": None, "gru": None, "sup_L": None, "ssl": "ssl", "gru_ssl": "gru_ssl",
              "gru_placebo": "gru_placebo", "ssl_L": "ssl_L", "pooled_L": "pooled_L"}


def phase3_participants(cfg: dict) -> list:
    return [str(p) for p in (cfg["phase3"]["participants"] or PARTICIPANTS)]


def build_jobs3(pid: str, cfg: dict):
    ph, lrs = cfg["phase3"], cfg["finetune"]["lrs"]
    sm, ss, fr = ph["seeds_main"], ph["seeds_small"], ph["fractions"]
    pre = [("pretrain3", k, None, s, None) for k in ("gru_ssl", "gru_placebo") for s in sm]
    pre += [("pretrain3", "ssl_L", None, s, None) for s in ss]
    ft = []
    for cond in ("gru_ssl", "gru_placebo"):
        ft += [("finetune", cond, f, s, lr) for f in fr for s in sm for lr in ph["gru_lrs"]]
    ft += [("finetune", "gru", f, s, lr) for f in fr for s in sm for lr in ph["gru_lr_extra"]]
    for cond in ph["edge_conditions"]:
        ft += [("finetune", cond, f, s, lr) for f in fr for s in sm for lr in ph["edge_lrs"]]
    for cond in LARGE_CONDS:
        ft += [("finetune", cond, f, s, lr) for f in ph["large_fractions"] for s in ss for lr in lrs]
    probes = []
    if ph["run_probes"]:
        probes = [("probe", enc, None, s, None) for enc in ("gru_random", "gru_ssl", "gru_placebo") for s in sm]
    return pre, ft + probes


def _job_id(pid: str, job) -> str:
    kind, cond, frac, seed, lr = job
    if kind == "pretrain3":
        return R.run_id(pid, f"pretrain_{cond}", seed=seed)
    if kind == "probe":
        return R.run_id(pid, f"probe_{cond}", seed=seed)
    return R.run_id(pid, cond, frac, seed, lr)


def _enc_path(name: str, seed: int):
    pid, root, ph = R._W["data"].pid, R._W["paths"].root, R._W["cfg"]["phase3"]
    if name == "ssl":
        return R.Paths(root, ph["phase1_dir"]).encoders / f"{pid}__ssl__s{seed}.pt"
    return R._W["paths"].encoders / f"{pid}__{name}__s{seed}.pt"


def _load_enc(name: str, seed: int):
    path = _enc_path(name, seed)
    if not path.exists():
        raise FileNotFoundError(f"encoder missing: {path}")
    return torch.load(path, map_location=R._W["device"], weights_only=True)


def _cfg_for(cond: str) -> dict:
    cfg = R._W["cfg"]
    return T.with_model(cfg, "model_large") if cond in LARGE_CONDS else cfg


def _run_pretrain3(job) -> dict:
    _, name, _, seed, _ = job
    d, t = R._W["data"], R._W["t"]
    if name == "gru_placebo":
        x = torch.from_numpy(T.column_permute(d.z("train"), 777 + seed)).to(R._W["device"])
    else:
        x = t["xz_train"]
    arch = "gru" if name.startswith("gru") else "transformer"
    state, info = T.pretrain_encoder(x, t["contact_id"], _cfg_for(name), seed, "new", arch=arch)
    torch.save(state, _enc_path(name, seed))
    return {"protocol": "new", "placebo": name == "gru_placebo", "arch": arch, "pretrain": info}


def _run_finetune3(job):
    _, cond, frac, seed, lr = job
    d, t = R._W["data"], R._W["t"]
    cfg = _cfg_for(cond)
    idx = torch.as_tensor(label_subset(d.n_windows("train"), frac, seed, d.pid), device=R._W["device"])
    enc_name = ENCODER_OF[cond]
    enc = None if enc_name is None else _load_enc(enc_name, seed)
    arch = "gru" if cond in GRU_CONDS else "transformer"
    ytr_raw = t["y_train"][idx]
    mean, std = R._norm_targets(ytr_raw)
    res = T.finetune(enc, t["xz_train"][idx], (ytr_raw - mean) / std, t["xz_val"],
                     (t["y_val"] - mean) / std, t["xz_test"], cfg, seed, lr, "new", arch=arch)
    m, s = mean.cpu().numpy(), std.cpu().numpy()
    info = {k: v for k, v in res.items() if not k.startswith("pred_")}
    info["arch"], info["model_size"] = arch, ("large" if cond in LARGE_CONDS else "base")
    return len(idx), info, res["pred_val"] * s + m, res["pred_test"] * s + m


def _run_probe3(job) -> str:
    _, enc_name, _, seed, _ = job
    d, t, cfg, paths = R._W["data"], R._W["t"], R._W["cfg"], R._W["paths"]
    state = T.random_gru_state(d.F, cfg, seed) if enc_name == "gru_random" else _load_enc(enc_name, seed)
    feats = {s: T.encode(state, t[f"xz_{s}"], cfg, arch="gru") for s in ("train", "val", "test")}
    dm = feats["train"].shape[-1]
    ftr = feats["train"].reshape(-1, dm)
    mu, sd = ftr.mean(0), np.maximum(ftr.std(0), 1e-6)
    z = {s: (feats[s] - mu) / sd for s in feats}
    lines = []
    for frac in cfg["phase3"]["fractions"]:
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


def run_job3(job) -> str:
    d, cfg, paths = R._W["data"], R._W["cfg"], R._W["paths"]
    rid = _job_id(d.pid, job)
    out = paths.runs / f"{rid}.json"
    kind = job[0]
    if kind != "probe" and R._done(out):
        return f"{rid}: already done"
    t0 = time.time()
    try:
        if kind == "pretrain3":
            rec = R._record(("pretrain", job[1], None, job[3], None), status="ok", **_run_pretrain3(job))
            rec["run_id"], rec["kind"] = rid, "pretrain"
        elif kind == "probe":
            return f"{rid}: {_run_probe3(job)} ({time.time() - t0:.0f}s)"
        else:
            n_lab, info, pred_val, pred_test = _run_finetune3(job)
            test = M.evaluate(d.Yw["test"], pred_test, d.vad["test"], cfg)
            val_r = M.mean_band_r(d.Yw["val"].reshape(-1, pred_val.shape[-1]), pred_val.reshape(-1, pred_val.shape[-1]))
            if cfg["phase3"].get("save_predictions", True):
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


def run_participant3(pid: str, cfg: dict, root, workers: int = 8, device: str = "auto") -> None:
    phase = cfg["phase3"]["results_dir"]
    paths = R.Paths(root, phase)
    paths.ensure()
    derived = paths.derived / f"{pid}.npz"
    if not derived.exists():
        raise FileNotFoundError(f"{derived} not found")
    missing = [s for s in cfg["phase3"]["seeds_small"]
               if not (paths.encoders / f"{pid}__pooled_L__s{s}.pt").exists()]
    if missing:
        raise FileNotFoundError(f"{pid}: pooled_L encoders missing for seeds {missing}; run the "
                                "phase3_pooled job first (docs/RUNNING_THE_STUDY.md, step 7)")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    R._atomic_json(paths.base / f"config_{config_hash(cfg)}.json", cfg)
    pre, rest = build_jobs3(pid, cfg)

    def todo(job):
        if job[0] == "probe":
            return not all(R._done(paths.runs / f"{R.run_id(pid, f'probe_{job[1]}', f, job[3])}.json")
                           for f in cfg["phase3"]["fractions"])
        if job[0] == "pretrain3":
            return not (R._done(paths.runs / f"{_job_id(pid, job)}.json")
                        and (paths.encoders / f"{pid}__{job[1]}__s{job[3]}.pt").exists())
        return not R._done(paths.runs / f"{_job_id(pid, job)}.json")

    pre, rest = [j for j in pre if todo(j)], [j for j in rest if todo(j)]
    print(f"{pid}: phase 3 -> {len(pre)} pretraining + {len(rest)} other jobs (device={device}, "
          f"workers={workers}, code={code_hash()}, config={config_hash(cfg)})", flush=True)
    t0 = time.time()
    args = (str(derived), cfg, str(root), phase, device)
    if workers <= 1:
        R.init_worker(*args)
        for j in pre + rest:
            print(run_job3(j), flush=True)
    else:
        ctx = torch.multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=R.init_worker, initargs=args) as pool:
            for line in pool.imap_unordered(run_job3, pre, chunksize=1):
                print(line, flush=True)
            for line in pool.imap_unordered(run_job3, rest, chunksize=1):
                print(line, flush=True)
    print(f"{pid}: phase 3 finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def run_pooled_large(index: int, cfg: dict, root, device: str = "auto") -> None:
    """Pooled pretraining of the large Transformer on all participants (one seed per task)."""
    ph = cfg["phase3"]
    seeds = ph["pooled_large_seeds"]
    if not 1 <= index <= len(seeds):
        raise ValueError(f"array index {index} outside 1..{len(seeds)}")
    seed = seeds[index - 1]
    paths = R.Paths(root, ph["results_dir"])
    paths.ensure()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pids = phase3_participants(cfg)
    name = R.run_id("ALL", "pretrain_pooled_L", seed=seed)
    out = paths.runs / f"{name}.json"
    expected = [paths.encoders / f"{p}__pooled_L__s{seed}.pt" for p in pids]
    stamp = {"code_hash": code_hash(), "config_hash": config_hash(cfg), "env": environment()}
    print(f"pooled_L seed {seed}: {len(pids)} participants (device={device}, code={stamp['code_hash']}, "
          f"config={stamp['config_hash']})", flush=True)
    if R._done(out) and all(p.exists() for p in expected):
        print(f"{name}: already done", flush=True)
        return
    t0 = time.time()
    try:
        dev = torch.device(device)
        data = {}
        for p in pids:
            d = ParticipantData(paths.derived / f"{p}.npz", cfg)
            data[p] = torch.from_numpy(d.z("train")).to(dev)
            del d
        cfg_l = T.with_model(cfg, "model_large")
        model, info = T.pretrain_multi(data, pids, cfg_l, seed, ph["multi_steps_per_participant"] * len(pids))
        for p in pids:
            torch.save(model.export_backbone(p), paths.encoders / f"{p}__pooled_L__s{seed}.pt")
        rec = {"run_id": name, "participant": "ALL", "kind": "pretrain_multi", "condition": "pooled_L",
               "seed": seed, "status": "ok", "pretrain": info, "model": cfg_l["model"],
               "elapsed_s": round(time.time() - t0, 1), **stamp}
        R._atomic_json(out, rec)
        print(f"{name}: ok in {rec['elapsed_s'] / 60:.1f} min (final loss {info['loss_curve'][-1]:.3f})", flush=True)
    except Exception as exc:
        R._atomic_json(out, {"run_id": name, "status": "error", "error": repr(exc),
                             "traceback": traceback.format_exc(), **stamp})
        print(f"{name}: ERROR {exc!r}", flush=True)
        raise
