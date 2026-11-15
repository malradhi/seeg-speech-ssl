"""Loading derived features: lag stacking, edge trimming, windows, normalisation,
voice-activity labels and nested labelled subsets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .preprocessing import SPLITS


def stack_lags(hg: np.ndarray, centers: np.ndarray, model_order: int, step: int) -> np.ndarray:
    """Lag-major stacking: feature f = lag_index * C + contact (matches the original recipe)."""
    offsets = np.arange(-model_order, model_order + 1) * step
    idx = centers[:, None] + offsets[None, :]
    return hg[idx, :].reshape(len(centers), -1).astype(np.float32)


def kmeans_threshold(x: np.ndarray, iters: int = 100) -> float:
    lo, hi = np.percentile(x, [10, 90])
    thr = 0.5 * (lo + hi)
    for _ in range(iters):
        a, b = x[x < thr], x[x >= thr]
        if len(a) == 0 or len(b) == 0:
            break
        new = 0.5 * (a.mean() + b.mean())
        if abs(new - thr) < 1e-9:
            break
        thr = new
    return float(thr)


def label_subset(n_windows: int, fraction: float, seed: int, pid: str) -> np.ndarray:
    """Nested random subsets: for a given (participant, seed) the 5% subset is
    contained in the 10% subset, which is contained in the 25% subset, etc.
    Identical for every condition, so comparisons are paired."""
    if fraction >= 1.0:
        return np.arange(n_windows)
    rng = np.random.default_rng([int(seed), int(pid[1:]), 20261103])
    perm = rng.permutation(n_windows)
    k = max(2, int(round(fraction * n_windows)))
    return np.sort(perm[:k])


class ParticipantData:
    """``feature_file`` (Phase 4): use frame-aligned features from another file (e.g. BrainBERT
    embeddings, keys ``bb_<split>``) instead of the lag-stacked high-gamma input. Frames, edge
    trimming, windows, targets and voice activity are unchanged, so the windows (and therefore
    the labelled subsets) are identical to those of the high-gamma input."""

    def __init__(self, path: str | Path, cfg: dict, feature_file: str | Path | None = None):
        f = cfg["features"]
        with np.load(path, allow_pickle=False) as d:
            self.meta = json.loads(str(d["meta_json"]))
            raw = {k: d[k] for k in d.files if k != "meta_json"}
        self.pid = self.meta["participant"]
        self.task = self.meta["task"]
        self.C = int(self.meta["n_contacts"])
        self.n_lags = 2 * f["model_order"] + 1
        self.F = self.C * self.n_lags
        self.contact_id = np.tile(np.arange(self.C), self.n_lags)
        self.window = f["window"]
        trim = max(f["edge_trim_frames"], f["model_order"] * f["step"])
        alt = None
        if feature_file is not None:
            with np.load(feature_file, allow_pickle=False) as d:
                alt = {s: d[f"bb_{s}"] for s in SPLITS}
            self.F = int(alt["train"].shape[1])
            per = self.F // self.C
            self.contact_id = np.repeat(np.arange(self.C), per)

        self.Xw, self.Yw, self.Tw = {}, {}, {}
        for split in SPLITS:
            hg, mel, tr = raw[f"hg_{split}"], raw[f"mel_{split}"], raw[f"trial_{split}"]
            centers = np.arange(trim, len(hg) - trim)
            if alt is not None:
                if len(alt[split]) != len(hg):
                    raise ValueError(f"{self.pid} {split}: {len(alt[split])} feature frames vs {len(hg)}")
                X = alt[split][centers].astype(np.float32)
            else:
                X = stack_lags(hg, centers, f["model_order"], f["step"])
            Y = mel[centers].astype(np.float32)
            T = tr[centers]
            starts = np.arange(0, len(centers) - self.window + 1, f["stride"])
            sel = starts[:, None] + np.arange(self.window)[None, :]
            self.Xw[split], self.Yw[split], self.Tw[split] = X[sel], Y[sel], T[sel]
        alt = X = None  # release the full-width copies before the statistics below

        # Neural normalisation from all (unlabelled) training frames.
        xtr = self.Xw["train"].reshape(-1, self.F)
        self.mu = xtr.mean(0).astype(np.float32)
        sd = xtr.std(0)
        sd[sd < 1e-8] = 1.0
        self.sd = sd.astype(np.float32)

        # Voice activity from frame energy; threshold fitted on training audio.
        e_tr = self.Yw["train"].mean(-1).reshape(-1)
        self.vad_threshold = kmeans_threshold(e_tr)
        self.vad = {s: self.Yw[s].mean(-1) >= self.vad_threshold for s in SPLITS}

    def n_windows(self, split: str) -> int:
        return len(self.Xw[split])

    def z(self, split: str) -> np.ndarray:
        return ((self.Xw[split] - self.mu) / self.sd).astype(np.float32)

    def seconds(self, n_windows: int) -> float:
        return n_windows * self.window * 0.01

    def summary(self) -> dict:
        trials = self.meta.get("trial_labels", [])
        out = {
            "participant": self.pid,
            "task": self.task,
            "eeg_sr": self.meta["eeg_sr"],
            "duration_s": round(self.meta["duration_s"], 2),
            "n_contacts": self.C,
            "n_contacts_recorded": self.meta.get("n_contacts_recorded", self.C),
            "channel_policy": self.meta.get("channel_policy", "all"),
            "channels_dropped": " ".join(self.meta.get("channels_dropped", [])),
            "n_features": self.F,
            "n_trials": self.meta.get("n_trials"),
            "n_unique_stimuli": len(set(trials)),
            "boundaries_snapped": self.meta.get("bounds_snapped_to_silence"),
        }
        seen_train = set()
        for s in SPLITS:
            nw = self.n_windows(s)
            out[f"{s}_windows"] = nw
            out[f"{s}_seconds"] = round(self.seconds(nw), 1)
            out[f"{s}_speech_frac"] = round(float(self.vad[s].mean()), 3)
            ids = np.unique(self.Tw[s])
            labels = {trials[i] for i in ids if 0 <= i < len(trials)}
            if s == "train":
                seen_train = labels
            elif labels:
                out[f"{s}_stimuli_seen_in_train"] = round(len(labels & seen_train) / len(labels), 3)
        out["features_finite"] = bool(all(np.isfinite(self.Xw[s]).all() for s in SPLITS))
        out["mel_finite"] = bool(all(np.isfinite(self.Yw[s]).all() for s in SPLITS))
        return out
