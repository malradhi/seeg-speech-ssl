"""Phase 2 multi-participant pretraining (one SLURM array task each).

Array index 1..N      target participant: leave-one-out pretraining on the other N-1
                      participants (exported without adaptation and after adapting the
                      target's input layer by SSL on its own unlabelled data), plus the
                      scaling sets (own + k-1 randomly chosen others).
Array index N+1..N+S  pooled pretraining on all N participants, one seed per task;
                      exports an encoder for every participant.

Only training partitions are ever used. Encoders are written to
results/phase2/encoders/<pid>__<kind>__s<seed>.pt in TransformerBackbone format.
"""

from __future__ import annotations

import time
import traceback

import numpy as np
import torch

from . import train as T
from .config import PARTICIPANTS, config_hash
from .data import ParticipantData
from .provenance import code_hash, environment
from .runner import Paths, _atomic_json, _done, run_id


def phase2_participants(cfg: dict) -> list:
    return [str(p) for p in (cfg["phase2"]["participants"] or PARTICIPANTS)]


def multi_tasks(cfg: dict, index: int) -> list:
    pids, ph = phase2_participants(cfg), cfg["phase2"]
    n = len(pids)
    if 1 <= index <= n:
        target = pids[index - 1]
        tasks = [("loo", target, s, None) for s in ph["seeds_small"]]
        tasks += [("scale", target, s, k) for k in ph["scaling_k"] if k < n for s in ph["seeds_small"]]
        return tasks
    s_idx = index - n - 1
    if 0 <= s_idx < len(ph["seeds_main"]):
        return [("pooled", "ALL", ph["seeds_main"][s_idx], None)]
    raise ValueError(f"array index {index} outside 1..{n + len(ph['seeds_main'])}")


def task_name(task) -> str:
    kind, target, seed, k = task
    label = {"loo": "loo", "pooled": "pooled", "scale": f"scale{k}"}[kind]
    return run_id(target, f"pretrain_{label}", seed=seed)


_M: dict = {}


def init_multi_worker(cfg: dict, root: str, device: str) -> None:
    torch.set_num_threads(1)
    paths = Paths(root, cfg["phase2"]["results_dir"])
    dev = torch.device(device)
    data = {}
    for p in phase2_participants(cfg):
        d = ParticipantData(paths.derived / f"{p}.npz", cfg)
        data[p] = torch.from_numpy(d.z("train")).to(dev)
        del d
    _M.update(cfg=cfg, paths=paths, data=data,
              stamp={"code_hash": code_hash(), "config_hash": config_hash(cfg), "env": environment()})


def _enc(pid: str, kind: str, seed: int):
    return _M["paths"].encoders / f"{pid}__{kind}__s{seed}.pt"


def _expected_outputs(task) -> list:
    kind, target, seed, k = task
    if kind == "pooled":
        return [_enc(p, "pooled", seed) for p in _M["data"]]
    if kind == "loo":
        return [_enc(target, "loo_noadapt", seed), _enc(target, "loo_adapt", seed)]
    return [_enc(target, f"scale{k}", seed)]


def run_multi_task(task) -> str:
    cfg, data, paths = _M["cfg"], _M["data"], _M["paths"]
    kind, target, seed, k = task
    name = task_name(task)
    out = paths.runs / f"{name}.json"
    if _done(out) and all(p.exists() for p in _expected_outputs(task)):
        return f"{name}: already done"
    steps_pp = cfg["phase2"]["multi_steps_per_participant"]
    pids = list(data)
    t0 = time.time()
    try:
        rec = {"run_id": name, "participant": target, "kind": "pretrain_multi", "condition": kind,
               "seed": seed, "status": "ok"}
        if kind == "pooled":
            model, info = T.pretrain_multi(data, pids, cfg, seed, steps_pp * len(pids))
            for p in pids:
                torch.save(model.export_backbone(p), _enc(p, "pooled", seed))
            rec["pretrain"] = info
        elif kind == "loo":
            others = [p for p in pids if p != target]
            model, info = T.pretrain_multi({p: data[p] for p in others}, others, cfg, seed,
                                           steps_pp * len(others))
            T.add_new_participant(model, target, data[target].shape[-1], seed)
            torch.save(model.export_backbone(target), _enc(target, "loo_noadapt", seed))
            model, ainfo = T.pretrain_multi({target: data[target]}, [target], cfg, seed,
                                            cfg["phase2"]["adapt_steps"], model=model, train_only=target)
            torch.save(model.export_backbone(target), _enc(target, "loo_adapt", seed))
            rec.update(pretrain=info, adapt=ainfo)
        else:
            others = [p for p in pids if p != target]
            rng = np.random.default_rng([int(seed), int(target[1:]), int(k)])
            chosen = [target] + [str(p) for p in rng.choice(others, size=k - 1, replace=False)]
            model, info = T.pretrain_multi({p: data[p] for p in chosen}, chosen, cfg, seed, steps_pp * k)
            torch.save(model.export_backbone(target), _enc(target, f"scale{k}", seed))
            rec.update(pretrain=info, k=k)
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec.update(_M["stamp"])
        _atomic_json(out, rec)
        return f"{name}: ok in {rec['elapsed_s'] / 60:.1f} min (final loss {rec['pretrain']['loss_curve'][-1]:.3f})"
    except Exception as exc:
        _atomic_json(out, {"run_id": name, "status": "error", "error": repr(exc),
                           "traceback": traceback.format_exc(), **_M["stamp"]})
        return f"{name}: ERROR {exc!r}"


def run_multi(index: int, cfg: dict, root: str, workers: int = 3, device: str = "auto") -> None:
    paths = Paths(root, cfg["phase2"]["results_dir"])
    paths.ensure()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = multi_tasks(cfg, index)
    print(f"multi task {index}: {[task_name(t) for t in tasks]} (device={device}, workers={workers}, "
          f"code={code_hash()}, config={config_hash(cfg)})", flush=True)
    t0 = time.time()
    if workers <= 1 or len(tasks) == 1:
        init_multi_worker(cfg, str(root), device)
        for t in tasks:
            print(run_multi_task(t), flush=True)
    else:
        ctx = torch.multiprocessing.get_context("spawn")
        with ctx.Pool(min(workers, len(tasks)), initializer=init_multi_worker,
                      initargs=(cfg, str(root), device)) as pool:
            for line in pool.imap_unordered(run_multi_task, tasks, chunksize=1):
                print(line, flush=True)
    print(f"multi task {index}: finished in {(time.time() - t0) / 60:.1f} min", flush=True)
