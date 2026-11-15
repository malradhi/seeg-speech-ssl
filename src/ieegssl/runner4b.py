"""Phase 4b runs (docs/PHASE4B_DESIGN.md). One SLURM array task per participant:

1. Self-pretraining (H17): masked pretraining on only the neural signals of the labelled
   windows of a budget (``ssl_self``, ``gru_ssl_self``; no additional recording), then
   fine-tuning on the same windows with their labels; seven learning rates.
2. Learning-rate grid edge (R2): 1e-2 for every base-size per-patient decoder that was run on
   six rates (and 1e-5 / 3e-3 for the Transformer dose arms), so that all of them share the
   seven-rate grid of ``sup`` / ``ssl``.
3. BrainBERT readout capacity (R1): BrainBERT features re-extracted with 64 principal
   components per contact (variance kept reported at 9, 16, 32 and 64); ridge on the leading
   16, 32 and 64 components (``bb16_ridge`` ...) and the GRU on 64 (``bb64_gru``).

Same worker set-up, record format, labelled subsets and pairing as Phases 1-4. Encoders from
earlier phases are read, never written.
"""

from __future__ import annotations

import copy
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

SELF_CONDS = ("ssl_self", "gru_ssl_self")
NO_ENCODER = ("gru", "bb_gru", "sup")


def phase4b_participants(cfg: dict) -> list:
    return [str(p) for p in (cfg["phase4b"]["participants"] or PARTICIPANTS)]


def bb_ridge_name(k: int) -> str:
    return f"bb{int(k)}_ridge"


def bb_gru_name(k: int) -> str:
    return f"bb{int(k)}_gru"


def _arch(cond: str) -> str:
    return "gru" if "gru" in cond else "transformer"


def build_jobs4b(pid: str, cfg: dict):
    """(high-gamma jobs, BrainBERT k=9 jobs, BrainBERT k=K jobs)."""
    ph = cfg["phase4b"]
    sm, ss, fr, g7 = ph["seeds_main"], ph["seeds_small"], ph["fractions"], ph["grid7"]
    hg, bb9, bbk = [], [], []
    if ph["run_self"]:
        for cond in SELF_CONDS:
            hg += [("pretrain4b", cond, f, s, None) for f in ph["self_fractions"] for s in sm]
            hg += [("finetune", cond, f, s, lr) for f in ph["self_fractions"] for s in sm for lr in g7]
    if ph["run_edge"]:
        for cond in ph["edge_conditions"]:
            jobs = [("finetune", cond, f, s, ph["edge_lr"]) for f in fr for s in sm]
            (bb9 if cond == "bb_gru" else hg).extend(jobs)
        for arch, prefix in (("transformer", "ssl"), ("gru", "gru_ssl")):
            for u in ph["dose_fractions"]:
                name = f"{prefix}_u{int(round(u * 1000)):03d}"
                hg += [("finetune", name, f, s, lr) for f in ph["dose_label_fractions"] for s in ss
                       for lr in ph["dose_extra_lrs"][arch]]
    if ph["run_brainbert"]:
        bbk += [("linear", bb_ridge_name(k), f, s, None) for k in ph["bb_ridge_k"] for f in fr for s in sm]
        bbk += [("finetune", bb_gru_name(ph["bb_gru_k"]), f, s, lr) for f in fr for s in sm for lr in g7]
    return hg, bb9, bbk


def _job_id(pid: str, job) -> str:
    kind, cond, frac, seed, lr = job
    if kind == "pretrain4b":
        return R.run_id(pid, f"pretrain_{cond}", frac, seed)
    return R.run_id(pid, cond, frac, seed, lr)


def _enc_path(cond: str, seed: int, frac: float | None = None) -> Path:
    """Where the pretrained encoder of a condition lives (earlier phases are read-only)."""
    pid, root, ph = R._W["data"].pid, R._W["paths"].root, R._W["cfg"]["phase4b"]
    if cond in SELF_CONDS:
        return R._W["paths"].encoders / f"{pid}__{cond}__f{frac:.2f}__s{seed}.pt"
    if cond in ("placebo", "lpft_ssl"):
        name = "placebo" if cond == "placebo" else "ssl"
        return R.Paths(root, ph["phase1_dir"]).encoders / f"{pid}__{name}__s{seed}.pt"
    if cond in ("gru_ssl", "gru_placebo", "lpft_gru_ssl"):
        name = "gru_placebo" if cond == "gru_placebo" else "gru_ssl"
        return R.Paths(root, ph["phase3_dir"]).encoders / f"{pid}__{name}__s{seed}.pt"
    if cond.startswith(("ssl_u", "gru_ssl_u")):
        return R.Paths(root, ph["phase4_dir"]).encoders / f"{pid}__{cond}__s{seed}.pt"
    raise KeyError(f"no encoder for condition {cond!r}")


def _needs_encoder(cond: str) -> bool:
    return not (cond in NO_ENCODER or cond.startswith("bb"))


def _run_pretrain4b(job) -> dict:
    _, cond, frac, seed, _ = job
    d, t = R._W["data"], R._W["t"]
    sub = torch.as_tensor(label_subset(d.n_windows("train"), frac, seed, d.pid), device=R._W["device"])
    arch = _arch(cond)
    state, info = T.pretrain_encoder(t["xz_train"][sub], t["contact_id"], R._W["cfg"], seed, "new", arch=arch)
    torch.save(state, _enc_path(cond, seed, frac))
    info["unlabelled_windows"] = int(len(sub))
    info["unlabelled_seconds"] = d.seconds(len(sub))
    return {"protocol": "new", "arch": arch, "unlabelled_fraction": frac, "self_pretraining": True,
            "pretrain": info}


def _run_finetune4b(job):
    _, cond, frac, seed, lr = job
    d, t, cfg = R._W["data"], R._W["t"], R._W["cfg"]
    idx = torch.as_tensor(label_subset(d.n_windows("train"), frac, seed, d.pid), device=R._W["device"])
    enc = None
    if _needs_encoder(cond):
        path = _enc_path(cond, seed, frac)
        if not path.exists():
            raise FileNotFoundError(f"encoder missing: {path}")
        enc = torch.load(path, map_location=R._W["device"], weights_only=True)
    lp = cfg["phase4"]["lp"] if cond.startswith("lpft_") else None
    ytr_raw = t["y_train"][idx]
    mean, std = R._norm_targets(ytr_raw)
    res = T.finetune(enc, t["xz_train"][idx], (ytr_raw - mean) / std, t["xz_val"], (t["y_val"] - mean) / std,
                     t["xz_test"], cfg, seed, lr, "new", arch=_arch(cond), lp=lp)
    m, s = mean.cpu().numpy(), std.cpu().numpy()
    info = {k: v for k, v in res.items() if not k.startswith("pred_")}
    info["arch"] = _arch(cond)
    return len(idx), info, res["pred_val"] * s + m, res["pred_test"] * s + m


def leading_components(x: torch.Tensor, n_contacts: int, k_all: int, k: int) -> torch.Tensor:
    """Keep the leading ``k`` of ``k_all`` principal components of every contact (features are
    stored contact-major: column c * k_all + j). Columns are z-scored one by one, so slicing
    z-scored features equals z-scoring sliced ones."""
    if k == k_all:
        return x
    lead = x.shape[:-1]
    return x.reshape(*lead, n_contacts, k_all)[..., :k].reshape(*lead, n_contacts * k)


def _run_bb_ridge_k(job):
    _, cond, frac, seed, _ = job
    d, t, cfg = R._W["data"], R._W["t"], R._W["cfg"]
    k, k_all = int(cond[2:].split("_")[0]), int(cfg["phase4b"]["bb_k"])
    idx = label_subset(d.n_windows("train"), frac, seed, d.pid)
    sel = {s: leading_components(t[f"xz_{s}"], d.C, k_all, k) for s in ("val", "test")}
    xtr = leading_components(t["xz_train"][torch.as_tensor(idx, device=R._W["device"])], d.C, k_all, k)
    F = xtr.shape[-1]
    ytr_raw = d.Yw["train"][idx].reshape(-1, d.Yw["train"].shape[-1])
    m, s = ytr_raw.mean(0), np.maximum(ytr_raw.std(0), 1e-6)
    yva = (d.Yw["val"].reshape(-1, ytr_raw.shape[1]) - m) / s
    res = B.ridge_torch(xtr.reshape(-1, F), (ytr_raw - m) / s, sel["val"].reshape(-1, F), yva,
                        sel["test"].reshape(-1, F), cfg["linear"]["ridge_alphas"])
    del xtr, sel
    pred_test = (res["pred_test"] * s + m).reshape(d.Yw["test"].shape)
    pred_val = (res["pred_val"] * s + m).reshape(d.Yw["val"].shape)
    info = {k_: v for k_, v in res.items() if not k_.startswith("pred_")}
    info["n_components"], info["n_features"] = k, int(F)
    return len(idx), info, pred_val, pred_test


def run_job4b(job) -> str:
    d, cfg, paths = R._W["data"], R._W["cfg"], R._W["paths"]
    rid = _job_id(d.pid, job)
    out = paths.runs / f"{rid}.json"
    kind, cond = job[0], job[1]
    if R._done(out):
        return f"{rid}: already done"
    t0 = time.time()
    try:
        if kind == "pretrain4b":
            rec = R._record(("pretrain", cond, job[2], job[3], None), status="ok", **_run_pretrain4b(job))
            rec["run_id"], rec["kind"] = rid, "pretrain"
        else:
            if kind == "linear":
                n_lab, info, pred_val, pred_test = _run_bb_ridge_k(job)
                protocol = "closed_form"
            else:
                n_lab, info, pred_val, pred_test = _run_finetune4b(job)
                protocol = "new"
            test = M.evaluate(d.Yw["test"], pred_test, d.vad["test"], cfg)
            nm = pred_val.shape[-1]
            val_r = M.mean_band_r(d.Yw["val"].reshape(-1, nm), pred_val.reshape(-1, nm))
            if cfg["phase4b"].get("save_predictions", True):
                np.savez_compressed(paths.preds / f"{rid}.npz", pred_test=pred_test.astype(np.float16))
            rec = R._record(job, status="ok", protocol=protocol, n_label_windows=int(n_lab),
                            label_seconds=d.seconds(n_lab), train=info,
                            val={"mse": info.get("best_val_mse"), "r_mean": val_r}, test=test)
            rec["features"] = "brainbert" if cond.startswith("bb") else "high_gamma"
            if cond == "bb_gru":
                rec["bb_components"] = int(cfg["brainbert"]["pca_components"])
            elif cond.startswith("bb"):
                rec["bb_components"] = int(info.get("n_components", cfg["phase4b"]["bb_gru_k"]))
            if cond in SELF_CONDS:
                rec["self_pretraining"] = True
        rec["elapsed_s"] = round(time.time() - t0, 2)
        R._atomic_json(out, rec)
        r = rec.get("test", {}).get("r_mean")
        return f"{rid}: ok in {rec['elapsed_s']:.1f}s" + (f" | test r={r:.3f}" if r is not None else "")
    except Exception as exc:
        R._atomic_json(out, {"run_id": rid, "status": "error", "error": repr(exc),
                             "traceback": traceback.format_exc(), **R._W["stamp"]})
        return f"{rid}: ERROR {exc!r}"


def bb_config(cfg: dict) -> dict:
    """BrainBERT extraction settings for Phase 4b: K components per contact, own folder."""
    c = copy.deepcopy(cfg)
    ph = cfg["phase4b"]
    c["brainbert"]["pca_components"] = int(ph["bb_k"])
    c["brainbert"]["features_dir"] = ph["bb_features_dir"]
    c["brainbert"]["report_k"] = list(ph["bb_report_k"])
    return c


def _pool(jobs, args, workers):
    if not jobs:
        return
    if workers <= 1:
        R.init_worker(*args)
        for j in jobs:
            print(run_job4b(j), flush=True)
        return
    ctx = torch.multiprocessing.get_context("spawn")
    first = [j for j in jobs if j[0] == "pretrain4b"]
    later = [j for j in jobs if j[0] != "pretrain4b"]
    with ctx.Pool(workers, initializer=R.init_worker, initargs=args) as pool:
        for line in pool.imap_unordered(run_job4b, first, chunksize=1):
            print(line, flush=True)
        for line in pool.imap_unordered(run_job4b, later, chunksize=1):
            print(line, flush=True)


def run_participant4b(pid: str, cfg: dict, root, workers: int = 8, device: str = "auto", raw_root=None) -> None:
    from .brainbert import extract_participant

    ph = cfg["phase4b"]
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
    hg, bb9, bbk = build_jobs4b(pid, cfg)
    todo = lambda job: not R._done(paths.runs / f"{_job_id(pid, job)}.json")  # noqa: E731
    hg, bb9, bbk = [j for j in hg if todo(j)], [j for j in bb9 if todo(j)], [j for j in bbk if todo(j)]
    # a self-pretraining encoder whose record exists but whose file is missing is redone
    for j in build_jobs4b(pid, cfg)[0]:
        if j[0] == "pretrain4b" and j not in hg:
            enc = paths.encoders / f"{pid}__{j[1]}__f{j[2]:.2f}__s{j[3]}.pt"
            if not enc.exists():
                (paths.runs / f"{_job_id(pid, j)}.json").unlink(missing_ok=True)
                hg.insert(0, j)
    bbk_file = None
    if bbk:
        bcfg = bb_config(cfg)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False  # full fp32, as published
        bbk_file = extract_participant(pid, raw_root, paths.derived, root / ph["bb_features_dir"], bcfg,
                                       root / cfg["brainbert"]["model_dir"], device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    bb9_file = root / cfg["brainbert"]["features_dir"] / f"{pid}.npz"
    if bb9 and not bb9_file.exists():
        raise FileNotFoundError(f"{bb9_file} not found (Phase 4 BrainBERT features)")
    print(f"{pid}: phase 4b -> {len(hg)} high-gamma + {len(bb9)} BrainBERT-9 + {len(bbk)} BrainBERT-{ph['bb_k']} "
          f"jobs (device={device}, workers={workers}/{ph['bb_workers']}, code={code_hash()}, "
          f"config={config_hash(cfg)})", flush=True)
    t0 = time.time()
    base = (str(derived), cfg, str(root), phase, device)
    _pool(hg, base + (None, False), workers)
    _pool(bb9, base + (str(bb9_file), False), min(workers, len(bb9)))
    _pool(bbk, base + (str(bbk_file), False), min(int(ph["bb_workers"]), workers))
    print(f"{pid}: phase 4b finished in {(time.time() - t0) / 60:.1f} min", flush=True)
