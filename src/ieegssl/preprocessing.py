"""Leakage-safe preprocessing: partition first, then extract features per partition.

Pipeline for one participant
----------------------------
1. Load sEEG (samples x contacts), audio (48 kHz) and per-sample stimulus labels.
2. Define chronological 70/15/15 boundaries on the raw sample axis; move each
   boundary to the middle of the nearest inter-trial silence so no trial is split.
3. Within each partition *independently*: detrend, 70-170 Hz band-pass, line-noise
   band-stops, Hilbert envelope, 50-ms/10-ms framing (high-gamma), and the
   23-band log-Mel spectrogram of the simultaneous audio.
4. Save un-stacked high-gamma frames, Mel frames and per-frame trial indices.
   Lag stacking, edge trimming and windowing happen at load time (data.py), so
   no feature can draw on samples from another partition.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.fft import next_fast_len

from .config import task_of

# ---------------------------------------------------------------------------
# Raw files
# ---------------------------------------------------------------------------


def find_raw_files(raw_root: str | Path, pid: str) -> dict[str, Path]:
    root = Path(raw_root)
    out: dict[str, Path] = {}
    for key, suffix in (
        ("eeg", "_sEEG.npy"),
        ("audio", "_audio.npy"),
        ("stimuli", "_stimuli.npy"),
        ("channels", "_channels.npy"),
        ("locations", "_electrode_locations.csv"),
    ):
        hits = sorted(root.rglob(f"{pid}{suffix}"))
        if hits:
            out[key] = hits[0]
    missing = [k for k in ("eeg", "audio") if k not in out]
    if missing:
        raise FileNotFoundError(f"{pid}: missing {missing} under {root}")
    return out


def _load_small_array(path: Path) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except ValueError:
        # Object arrays (e.g. channel names) need pickle; these files come from
        # the public OSF project of the dataset authors.
        return np.load(path, allow_pickle=True)


def infer_eeg_sr(n_eeg: int, n_audio: int, audio_sr: int, candidates) -> float:
    est = audio_sr * n_eeg / n_audio
    best = min(candidates, key=lambda c: abs(c - est))
    if abs(best - est) / best > 0.005:
        raise ValueError(
            f"Cannot infer sEEG rate: estimate {est:.2f} Hz is not within 0.5% of {candidates}. "
            "Check the audio sampling rate in configs."
        )
    return float(best)


def localised_names(path: Path) -> set[str]:
    import csv

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    key = "electrode_name_1" if rows and "electrode_name_1" in rows[0] else next(iter(rows[0]))
    return {str(r[key]).strip() for r in rows}


def select_channels(pid: str, channels: list[str], named: bool, files: dict, p: dict):
    """Channel policy:
    all        keep every recorded channel
    localised  keep only channels listed in P##_electrode_locations.csv
    Channels matching ``exclude_channel_regex`` are always removed."""
    import re

    policy = p.get("channel_policy", "all")
    keep = list(range(len(channels)))
    if policy == "localised":
        if not named or "locations" not in files:
            raise ValueError(f"{pid}: channel_policy='localised' needs P##_channels.npy and the locations CSV")
        loc = localised_names(files["locations"])
        keep = [i for i in keep if channels[i].strip() in loc]
    elif policy != "all":
        raise ValueError(f"Unknown channel_policy {policy!r}")
    pattern = p.get("exclude_channel_regex")
    if pattern:
        rx = re.compile(pattern, re.IGNORECASE)
        keep = [i for i in keep if not rx.search(channels[i])]
    if len(keep) < 10:
        raise ValueError(f"{pid}: only {len(keep)} channels left after selection")
    dropped = [channels[i] for i in range(len(channels)) if i not in set(keep)]
    return np.asarray(keep), dropped


# ---------------------------------------------------------------------------
# Trials and partition boundaries
# ---------------------------------------------------------------------------


def clean_labels(stimuli: np.ndarray) -> np.ndarray:
    return np.char.strip(np.asarray(stimuli).astype(str))


def trial_runs(labels: np.ndarray) -> list[tuple[int, int, str]]:
    """Contiguous runs of non-empty stimulus labels: (start, end, label)."""
    if len(labels) == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    return [(int(s), int(e), str(labels[s])) for s, e in zip(starts, ends) if labels[s] != ""]


def partition_bounds(n: int, runs, train_frac: float, val_frac: float, snap: bool = True):
    nominal = [int(round(n * train_frac)), int(round(n * (train_frac + val_frac)))]
    if not snap or len(runs) < 3:
        return nominal, False
    gaps = np.array([(runs[i][1] + runs[i + 1][0]) // 2 for i in range(len(runs) - 1)])
    snapped = [int(gaps[np.argmin(np.abs(gaps - x))]) for x in nominal]
    if not (0 < snapped[0] < snapped[1] < n):
        return nominal, False
    return snapped, True


# ---------------------------------------------------------------------------
# Feature extraction (applied within one partition)
# ---------------------------------------------------------------------------


def _frame_edges(n_samples: int, sr: float, win_sec: float, shift_sec: float):
    num = int(np.floor((n_samples - win_sec * sr) / (shift_sec * sr)))
    starts = np.floor(np.arange(num) * shift_sec * sr).astype(np.int64)
    stops = np.floor(starts + win_sec * sr).astype(np.int64)
    return starts, stops


def extract_high_gamma(x: np.ndarray, sr: float, cfg: dict) -> np.ndarray:
    """High-gamma Hilbert amplitude averaged in 50-ms windows every 10 ms."""
    p = cfg["preprocess"]
    x = signal.detrend(np.asarray(x, dtype=np.float64), axis=0)
    sos = signal.butter(p["filter_order"], p["hg_band"], btype="bandpass", fs=sr, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=0)
    for lo, hi in p["line_bandstops"]:
        sos = signal.butter(p["filter_order"], [lo, hi], btype="bandstop", fs=sr, output="sos")
        x = signal.sosfiltfilt(sos, x, axis=0)
    amp = np.abs(signal.hilbert(x, N=next_fast_len(len(x)), axis=0)[: len(x)])
    starts, stops = _frame_edges(len(amp), sr, p["win_sec"], p["shift_sec"])
    csum = np.vstack([np.zeros((1, amp.shape[1])), np.cumsum(amp, axis=0)])
    return ((csum[stops] - csum[starts]) / (stops - starts)[:, None]).astype(np.float32)


class MelFilterBank:
    """Mel filterbank identical to the original SingleWordProductionDutch recipe."""

    def __init__(self, spec_size: int, num_bands: int, sample_rate: int):
        max_mel = self._f2m(sample_rate / 2.0)
        mel_step = max_mel / (num_bands + 1)
        edges = np.arange(0, num_bands + 2) * mel_step
        centers = [self._f2b(math.floor(self._m2f(m)), sample_rate, spec_size) for m in edges]
        fm = np.zeros((num_bands, spec_size))
        for i in range(num_bands):
            s, c, e = centers[i : i + 3]
            if c > s:
                fm[i, s:c] = (np.arange(s, c) - s) / float(c - s)
            if e > c:
                fm[i, c:e] = (e - np.arange(c, e)) / float(e - c)
        m = fm.T
        col = m.sum(axis=0)
        col[col == 0] = 1.0
        m = m / col
        m[~np.isfinite(m)] = 0
        self.matrix = m

    @staticmethod
    def _f2m(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    @staticmethod
    def _m2f(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    @staticmethod
    def _f2b(f, sr, size):
        return int(math.floor((f / (sr / 2.0)) * size))

    def log_mel(self, magnitude: np.ndarray) -> np.ndarray:
        out = np.log(magnitude @ self.matrix + 1e-7)
        out[~np.isfinite(out)] = 0
        return out


def downsample_audio(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    q = sr_in / sr_out
    if abs(q - round(q)) > 1e-9:
        return signal.resample_poly(audio, sr_out, sr_in)
    q = int(round(q))
    return signal.decimate(np.asarray(audio, dtype=np.float64), q) if q > 1 else np.asarray(audio, np.float64)


def extract_log_mel(audio16k: np.ndarray, scale: float, cfg: dict) -> np.ndarray:
    p = cfg["preprocess"]
    sr = p["audio_target_sr"]
    x = audio16k * scale
    starts, _ = _frame_edges(len(x), sr, p["win_sec"], p["shift_sec"])
    win_n = int(np.floor(p["win_sec"] * sr))
    window = np.hanning(win_n + 1)[:-1]
    fb = MelFilterBank(win_n // 2 + 1, p["n_mels"], sr)
    out = np.empty((len(starts), p["n_mels"]), dtype=np.float32)
    for a in range(0, len(starts), 4096):
        idx = starts[a : a + 4096]
        frames = np.stack([x[s : s + win_n] for s in idx])
        out[a : a + len(idx)] = fb.log_mel(np.abs(np.fft.rfft(frames * window, axis=1)))
    return out


# ---------------------------------------------------------------------------
# Participant-level driver
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "test")


def preprocess_participant(pid: str, raw_root: str | Path, out_dir: str | Path, cfg: dict,
                           overwrite: bool = False) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}.npz"
    if out_path.exists() and not overwrite:
        print(f"{pid}: derived file exists -> skip", flush=True)
        return out_path

    p = cfg["preprocess"]
    files = find_raw_files(raw_root, pid)
    eeg = np.load(files["eeg"], mmap_mode="r")
    audio = np.load(files["audio"], mmap_mode="r").reshape(-1)
    if eeg.ndim != 2:
        raise ValueError(f"{pid}: expected sEEG [samples, contacts], got {eeg.shape}")
    if eeg.shape[0] < eeg.shape[1]:
        raise ValueError(f"{pid}: sEEG looks transposed {eeg.shape}")
    n = eeg.shape[0]
    eeg_sr = infer_eeg_sr(n, len(audio), p["audio_sr"], p["eeg_sr_candidates"])
    ratio = len(audio) / n

    if "stimuli" in files:
        labels = clean_labels(_load_small_array(files["stimuli"]))[:n]
    else:
        labels = np.array([""] * n)
    runs = trial_runs(labels)
    (b1, b2), snapped = partition_bounds(n, runs, p["train_frac"], p["val_frac"], p["snap_to_silence"])
    bounds = {"train": (0, b1), "val": (b1, b2), "test": (b2, n)}

    channels = None
    if "channels" in files:
        channels = np.asarray(_load_small_array(files["channels"])).astype(str).reshape(-1).tolist()
    named = channels is not None and len(channels) == eeg.shape[1]
    if not named:
        channels = [f"ch{i:03d}" for i in range(eeg.shape[1])]
    keep, dropped = select_channels(pid, channels, named, files, p)

    # Audio scale from the training partition only (int16-equivalent range).
    a_tr = downsample_audio(np.asarray(audio[: int(round(b1 * ratio))]), p["audio_sr"], p["audio_target_sr"])
    peak = float(np.max(np.abs(a_tr)))
    if peak <= 0:
        raise ValueError(f"{pid}: silent training audio")
    scale = 32767.0 / peak

    # Run index for every sample -> per-frame trial index.
    run_index = np.full(n, -1, dtype=np.int32)
    for k, (s, e, _) in enumerate(runs):
        run_index[s:e] = k

    arrays: dict[str, np.ndarray] = {}
    for split, (s, e) in bounds.items():
        hg = extract_high_gamma(np.asarray(eeg[s:e])[:, keep], eeg_sr, cfg)
        a = downsample_audio(np.asarray(audio[int(round(s * ratio)) : int(round(e * ratio))]),
                             p["audio_sr"], p["audio_target_sr"])
        mel = extract_log_mel(a, scale, cfg)
        t = min(len(hg), len(mel))
        hg, mel = hg[:t], mel[:t]
        centers = s + np.floor(np.arange(t) * p["shift_sec"] * eeg_sr + p["win_sec"] * eeg_sr / 2).astype(np.int64)
        arrays[f"hg_{split}"] = hg
        arrays[f"mel_{split}"] = mel
        arrays[f"trial_{split}"] = run_index[np.clip(centers, 0, n - 1)]
        print(f"{pid} {split:5s}: {(e - s) / eeg_sr:7.1f} s -> {t} frames", flush=True)

    meta = {
        "participant": pid,
        "task": task_of(pid),
        "eeg_sr": eeg_sr,
        "audio_sr": p["audio_sr"],
        "n_samples": int(n),
        "duration_s": n / eeg_sr,
        "n_contacts": int(len(keep)),
        "n_contacts_recorded": int(eeg.shape[1]),
        "channel_policy": p.get("channel_policy", "all"),
        "channels": [channels[i] for i in keep],
        "channels_dropped": dropped,
        "bounds_samples": {k: [int(v[0]), int(v[1])] for k, v in bounds.items()},
        "bounds_snapped_to_silence": bool(snapped),
        "n_trials": len(runs),
        "trial_labels": [r[2] for r in runs],
        "audio_scale": scale,
        "source_files": {k: str(v.name) for k, v in files.items()},
    }
    tmp = out_dir / f"{pid}.tmp.npz"
    np.savez_compressed(tmp, meta_json=np.array(json.dumps(meta)), **arrays)
    tmp.replace(out_path)
    print(f"{pid}: saved {out_path} (contacts={meta['n_contacts']}, sr={eeg_sr:g}, "
          f"trials={len(runs)}, snapped={snapped})", flush=True)
    return out_path
