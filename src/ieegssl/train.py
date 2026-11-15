"""Training protocols.

new     Inputs z-scored; targets z-scored with labelled-subset statistics; AdamW;
        a fixed budget of optimisation *steps* (identical for every label
        fraction) with warm-up + cosine decay; early stopping on validation MSE;
        learning rate chosen from a grid on validation MSE.
legacy  Exact pilot-study protocol: raw inputs/targets, Adam, fixed number of *epochs*
        (40 fine-tuning, 30 pretraining), per-epoch cosine schedule.

Pairing: for a given seed, the supervised, SSL and placebo branches use the same
labelled subset, the same Mel-head initialisation and the same minibatch order;
only the encoder initialisation differs.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn

from .masking import make_mask
from .models import (EEGToMelModel, GRUDecoder, GRUSSLModel, MultiParticipantSSL, SSLPretrainModel,
                     TransformerBackbone)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(total: int, warmup: int):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


class CyclicSampler:
    """Reshuffled passes over n items, driven by its own generator."""

    def __init__(self, n: int, batch: int, seed: int):
        self.n, self.batch = n, batch
        self.g = torch.Generator().manual_seed(int(seed))
        self.perm = torch.randperm(n, generator=self.g)
        self.pos = 0

    def next(self) -> torch.Tensor:
        if self.pos >= self.n:
            self.perm = torch.randperm(self.n, generator=self.g)
            self.pos = 0
        idx = self.perm[self.pos : self.pos + self.batch]
        self.pos += self.batch
        return idx


def column_permute(x: np.ndarray, seed: int) -> np.ndarray:
    """Placebo data: permute every feature column independently over time.
    Keeps each feature's marginal distribution (scale) but destroys temporal and
    cross-feature structure."""
    n, w, f = x.shape
    flat = x.reshape(n * w, f)
    rng = np.random.default_rng(int(seed))
    return rng.permuted(flat, axis=0).reshape(n, w, f).astype(np.float32)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch: int = 256) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i : i + batch]) for i in range(0, len(x), batch)])


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------


def pretrain_encoder(x: torch.Tensor, contact_id: torch.Tensor, cfg: dict, seed: int,
                     protocol: str = "new", mask_kind: str | None = None,
                     mask_ratio: float | None = None, arch: str = "transformer"):
    """Masked reconstruction. ``arch="gru"`` pretrains the GRU baseline's projection +
    bidirectional GRU instead of the Transformer (Phase 3); everything else is identical."""
    device = x.device
    n, w, f = x.shape
    set_seed(1000 + seed)
    if arch == "gru":
        model = GRUSSLModel(f, cfg["gru"]).to(device)
    else:
        model = SSLPretrainModel(f, w, cfg["model"]).to(device)
    curve = []
    lag_step = cfg["features"]["step"]

    def step_fn(xb, opt, clip, kind, ratio):
        mask = make_mask(kind, xb, ratio, contact_id, lag_step)
        loss = ((model(xb, mask) - xb) ** 2)[mask].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        return loss

    if protocol == "new":
        pc = cfg["pretrain"]
        kind = mask_kind or pc["mask_kind"]
        ratio = pc["mask_ratio"] if mask_ratio is None else mask_ratio
        opt = torch.optim.AdamW(model.parameters(), lr=pc["lr"], weight_decay=pc["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(pc["steps"], pc["warmup"]))
        sampler = CyclicSampler(n, pc["batch_size"], 30000 + seed)
        model.train()
        running, count = torch.zeros((), device=device), 0
        for step in range(pc["steps"]):
            loss = step_fn(x[sampler.next().to(device)], opt, pc["grad_clip"], kind, ratio)
            sched.step()
            running += loss.detach()
            count += 1
            if (step + 1) % 100 == 0 or step + 1 == pc["steps"]:
                curve.append(round(float(running) / count, 6))  # mean masked MSE over the interval
                running, count = torch.zeros((), device=device), 0
        info = {"steps": pc["steps"], "mask_kind": kind, "mask_ratio": ratio}
    elif protocol == "legacy":
        lc = cfg["legacy"]
        opt = torch.optim.Adam(model.parameters(), lr=lc["pt_lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=lc["pt_epochs"])
        g = torch.Generator().manual_seed(30000 + seed)
        steps = 0
        for _ in range(lc["pt_epochs"]):
            model.train()
            perm = torch.randperm(n, generator=g)
            running, count = torch.zeros((), device=device), 0
            for i in range(0, n, lc["batch_size"]):
                loss = step_fn(x[perm[i : i + lc["batch_size"]].to(device)], opt, lc["grad_clip"],
                               "random", lc["mask_ratio"])
                running += loss.detach()
                count += 1
                steps += 1
            sched.step()
            curve.append(round(float(running) / count, 6))  # epoch-mean masked MSE
        info = {"steps": steps, "mask_kind": "random", "mask_ratio": lc["mask_ratio"]}
    else:
        raise ValueError(protocol)

    info["loss_curve"] = curve
    info["arch"] = arch
    if arch == "gru":
        state = model.backbone_state()
    else:
        state = {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()}
    return state, info


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


HEAD_PREFIXES = ("mel_head.", "head.")


def finetune(encoder_state, xtr, ytr, xva, yva, xte, cfg: dict, seed: int,
             lr: float | None = None, protocol: str = "new", arch: str = "transformer",
             lp: dict | None = None):
    """Train the Mel decoder; return predictions of the best-validation checkpoint.

    ``lp`` (Phase 4) turns on linear-probe-then-fine-tune: the Mel head is first trained
    alone on the frozen pretrained body (``lp["lr"]``, at most ``lp["max_steps"]`` steps,
    early stopping with ``lp["patience"]``), then the whole model is fine-tuned from the
    best probe state at ``lr``. Both stages share the ``finetune.max_steps`` update budget
    and the minibatch order, so the budget equals plain fine-tuning."""
    device = xtr.device
    n, w, f = xtr.shape
    set_seed(50000 + seed)  # identical Mel-head initialisation across conditions
    if arch == "gru":
        model = GRUDecoder(f, cfg["gru"], cfg["model"]["head_hidden"], n_mels=ytr.shape[-1]).to(device)
    else:
        model = EEGToMelModel(f, w, cfg["model"], n_mels=ytr.shape[-1]).to(device)
    if encoder_state is not None:
        if arch == "gru":
            model.load_backbone(encoder_state)
        else:
            model.encoder.load_state_dict(encoder_state, strict=True)
    mse = nn.MSELoss()

    def val_loss() -> float:
        return float(mse(predict(model, xva), yva))

    best, best_state, best_step, steps, curve = float("inf"), None, 0, 0, []

    lp_info = None
    if protocol == "new":
        fc = cfg["finetune"]
        budget = fc["max_steps"]
        if lp is not None:
            sampler = CyclicSampler(n, fc["batch_size"], 20000 + seed)
            head = [p for nm, p in model.named_parameters() if nm.startswith(HEAD_PREFIXES)]
            body = [p for nm, p in model.named_parameters() if not nm.startswith(HEAD_PREFIXES)]
            body_mods = [model.encoder] if arch != "gru" else [model.proj, model.drop, model.gru]
            for prm in body:
                prm.requires_grad_(False)
            lp_steps = min(int(lp["max_steps"]), budget - fc["eval_every"])
            opt = torch.optim.AdamW(head, lr=lp["lr"], weight_decay=fc["weight_decay"])
            sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(lp_steps, fc["warmup"]))
            bad, lp_run = 0, 0
            for step in range(1, lp_steps + 1):
                model.train()
                for mod in body_mods:
                    mod.eval()  # frozen body: no dropout while probing
                idx = sampler.next().to(device)
                loss = mse(model(xtr[idx]), ytr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head, fc["grad_clip"])
                opt.step()
                sched.step()
                lp_run = step
                if step % fc["eval_every"] == 0 or step == lp_steps:
                    v = val_loss()
                    curve.append(round(v, 6))
                    if v < best - 1e-7:
                        best, best_step, bad = v, step, 0
                        best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= lp["patience"]:
                            break
            if best_state is not None:
                model.load_state_dict(best_state)
            for prm in body:
                prm.requires_grad_(True)
            lp_info = {"lp_steps_run": lp_run, "lp_best_val_mse": best, "lp_best_step": best_step,
                       "lp_lr": lp["lr"]}
            budget -= lp_run
            steps = lp_run
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=fc["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(budget, fc["warmup"]))
        if lp is None:
            sampler = CyclicSampler(n, fc["batch_size"], 20000 + seed)
        offset = steps
        bad = 0
        for step in range(offset + 1, offset + budget + 1):
            model.train()
            idx = sampler.next().to(device)
            loss = mse(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), fc["grad_clip"])
            opt.step()
            sched.step()
            steps = step
            if step % fc["eval_every"] == 0 or step == offset + budget:
                v = val_loss()
                curve.append(round(v, 6))
                if v < best - 1e-7:
                    best, best_step, bad = v, step, 0
                    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= fc["patience"]:
                        break
        batch_size = fc["batch_size"]
    elif protocol == "legacy":
        lc = cfg["legacy"]
        lr = lc["ft_lr"]
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=lc["ft_epochs"])
        g = torch.Generator().manual_seed(20000 + seed)
        for _ in range(lc["ft_epochs"]):
            model.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, lc["batch_size"]):
                idx = perm[i : i + lc["batch_size"]].to(device)
                loss = mse(model(xtr[idx]), ytr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), lc["grad_clip"])
                opt.step()
                steps += 1
            sched.step()
            v = val_loss()
            curve.append(round(v, 6))
            if v < best:
                best, best_step = v, steps
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        batch_size = lc["batch_size"]
    else:
        raise ValueError(protocol)

    if best_state is None:  # validation loss never finite: keep the final state
        best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    out = {
        "pred_test": predict(model, xte).cpu().numpy(),
        "pred_val": predict(model, xva).cpu().numpy(),
        "best_val_mse": best,
        "best_step": best_step,
        "steps_run": steps,
        "epochs_equivalent": round(steps * batch_size / n, 2),
        "lr": lr,
        "val_curve": curve,
    }
    if lp_info is not None:
        out.update(lp_info)
    return out


# ---------------------------------------------------------------------------
# Multi-participant pretraining, new-patient adaptation, frozen features (Phase 2)
# ---------------------------------------------------------------------------


def pretrain_multi(data: dict, pids: list, cfg: dict, seed: int, steps: int,
                   model: MultiParticipantSSL | None = None, train_only: str | None = None):
    """Masked reconstruction (random coordinates, ratio from cfg) across participants.

    ``data`` maps pid -> z-scored training windows (n, W, F) on the device. Each step
    draws one participant's minibatch; participants are visited in balanced,
    reshuffled rounds, so every participant contributes the same number of steps.
    With ``train_only=pid`` only that participant's input projection, mask token and
    reconstruction head are trained and the shared body stays frozen.
    """
    pc = cfg["pretrain"]
    device = next(iter(data.values())).device
    w = next(iter(data.values())).shape[1]
    if model is None:
        set_seed(4000 + seed)
        model = MultiParticipantSSL({p: data[p].shape[-1] for p in pids}, w, cfg["model"]).to(device)
    if train_only is None:
        params = list(model.parameters())
    else:
        params = (list(model.proj[train_only].parameters()) + list(model.recon[train_only].parameters())
                  + [model.token[train_only]])
    for p in model.body.parameters():
        p.requires_grad_(train_only is None)
    opt = torch.optim.AdamW(params, lr=pc["lr"], weight_decay=pc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(steps, pc["warmup"]))
    rng = np.random.default_rng([int(seed), 4242, len(pids)])
    samplers = {p: CyclicSampler(data[p].shape[0], pc["batch_size"], 30000 + seed + 97 * i)
                for i, p in enumerate(pids)}
    order: list = []
    curve, running, count = [], torch.zeros((), device=device), 0
    model.train()
    for step in range(steps):
        if not order:
            order = [str(p) for p in rng.permutation(pids)]
        pid = order.pop()
        xb = data[pid][samplers[pid].next().to(device)]
        mask = make_mask("random", xb, pc["mask_ratio"])
        loss = ((model(pid, xb, mask) - xb) ** 2)[mask].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, pc["grad_clip"])
        opt.step()
        sched.step()
        running += loss.detach()
        count += 1
        if (step + 1) % 1000 == 0 or step + 1 == steps:
            curve.append(round(float(running) / count, 6))
            running, count = torch.zeros((), device=device), 0
    for p in model.body.parameters():
        p.requires_grad_(True)
    info = {"steps": steps, "participants": [str(p) for p in pids], "loss_curve": curve,
            "trained": "all" if train_only is None else f"{train_only} only (body frozen)"}
    return model, info


def add_new_participant(model: MultiParticipantSSL, pid: str, n_features: int, seed: int) -> None:
    """Fresh, seeded input projection / mask token / head for an unseen participant."""
    set_seed(6000 + seed)
    model.add_participant(pid, n_features)
    model.to(next(model.body.parameters()).device)


def random_backbone_state(n_features: int, window: int, cfg: dict, seed: int) -> dict:
    """The exact initial encoder that SSL pretraining with this seed starts from."""
    set_seed(1000 + seed)
    enc = TransformerBackbone(n_features, window, cfg["model"])
    return {k: v.detach().cpu() for k, v in enc.state_dict().items()}


@torch.no_grad()
def encode(state: dict, x: torch.Tensor, cfg: dict, batch: int = 256, arch: str = "transformer") -> np.ndarray:
    """Frozen encoder features (n, W, d) for linear probing (d = d_model, or 2 x GRU hidden)."""
    n, w, f = x.shape
    if arch == "gru":
        dec = GRUDecoder(f, cfg["gru"], cfg["model"]["head_hidden"]).to(x.device)
        dec.load_backbone(state)
        dec.eval()
        fn = dec.features
    else:
        enc = TransformerBackbone(f, w, cfg["model"]).to(x.device)
        enc.load_state_dict(state, strict=True)
        enc.eval()
        fn = enc
    return torch.cat([fn(x[i : i + batch]) for i in range(0, n, batch)]).cpu().numpy()


def random_gru_state(n_features: int, cfg: dict, seed: int) -> dict:
    """The exact initial GRU backbone that GRU pretraining with this seed starts from."""
    set_seed(1000 + seed)
    return GRUSSLModel(n_features, cfg["gru"]).backbone_state()


def with_model(cfg: dict, key: str) -> dict:
    """A copy of ``cfg`` whose Transformer size is ``cfg[key]`` (e.g. "model_large")."""
    import copy

    out = copy.deepcopy(cfg)
    out["model"] = copy.deepcopy(cfg[key])
    return out
