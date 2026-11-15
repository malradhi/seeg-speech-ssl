"""Acoustic-contamination audit (after Roussel et al., 2020, J. Neural Eng. 17:056028).

Principle: genuine speech-related neural activity correlates with the audio *broadly*
(louder speech -> more high-gamma at all frequencies), whereas acoustic contamination
(the microphone signal leaking into the electrodes) correlates with the audio at the
*same* frequency. For each channel we correlate log-power spectrograms of audio and
neural signal over time for every pair of frequencies (70-500 Hz, 200-ms windows) and
take the contamination index

    CI = mean(diagonal correlations) - mean(off-diagonal correlations).

The null distribution comes from circularly shifting the audio spectrogram by >= 10 s.
Each channel's index is converted to a z-score against its own null (mean, SD), and the
one-sided normal p-value is corrected across channels with Benjamini-Hochberg; a
channel is flagged when q < alpha AND the index is at least ``min_index`` (so that a
negligible but very precisely estimated excess is not called contamination).

Positive control: with ``inject_db`` the audio is added to one seeded, randomly chosen
channel at that level relative to the channel's own RMS, to measure sensitivity.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.stats import norm

from .preprocessing import _load_small_array, find_raw_files, infer_eeg_sr, select_channels


def _bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def _logspec(x: np.ndarray, sr: float, win: int, fmin: float, fmax: float):
    f, _, s = signal.spectrogram(x, fs=sr, window="hann", nperseg=win, noverlap=win // 2,
                                 axis=-1, scaling="density")
    keep = (f >= fmin) & (f <= fmax)
    return f[keep], np.log10(s[..., keep, :] + 1e-20)


def _zscore_rows(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True)
    sd = a.std(-1, keepdims=True)
    sd[sd == 0] = 1.0
    return a / sd


def audit_participant(pid: str, raw_root, out_dir, cfg: dict, seed: int = 0,
                      inject_db: float | None = None, n_perm: int | None = None) -> dict:
    c = cfg["contamination"]
    n_perm = n_perm or c["n_perm"]
    out_dir = Path(out_dir)
    tag = "" if inject_db is None else f"_inject{abs(int(inject_db))}dB"
    out_json = out_dir / f"{pid}{tag}.json"
    if out_json.exists():
        with open(out_json) as fh:
            return json.load(fh)
    p = cfg["preprocess"]
    files = find_raw_files(raw_root, pid)
    eeg = np.load(files["eeg"], mmap_mode="r")
    audio = np.load(files["audio"], mmap_mode="r").reshape(-1)
    sr = infer_eeg_sr(eeg.shape[0], len(audio), p["audio_sr"], p["eeg_sr_candidates"])
    channels = np.asarray(_load_small_array(files["channels"])).astype(str).reshape(-1).tolist() \
        if "channels" in files else [f"ch{i:03d}" for i in range(eeg.shape[1])]
    named = len(channels) == eeg.shape[1]
    keep, _ = select_channels(pid, channels, named, files, p)

    frac = Fraction(int(sr), int(p["audio_sr"])).limit_denominator(10000)
    a = signal.resample_poly(np.asarray(audio, np.float64), frac.numerator, frac.denominator)
    n = min(len(a), eeg.shape[0])
    win = int(round(c["win_sec"] * sr))
    x = np.asarray(eeg[:n, keep], np.float64).T                                   # (C, n)
    names = [channels[i] for i in keep]
    injected = None
    if inject_db is not None:
        ch = int(np.random.default_rng([seed, int(pid[1:]), 777]).integers(len(keep)))
        gain = 10 ** (inject_db / 20) * x[ch].std() / a[:n].std()
        x[ch] = x[ch] + gain * a[:n]
        injected = names[ch]
    freqs, A = _logspec(a[:n], sr, win, c["fmin"], c["fmax"])                     # (F, T)
    _, N = _logspec(x, sr, win, c["fmin"], c["fmax"])                               # (C, F, T)
    del x
    A, N = _zscore_rows(A).astype(np.float32), _zscore_rows(N).astype(np.float32)
    n_ch, n_f, n_t = N.shape
    flat = N.reshape(n_ch * n_f, n_t)
    diag_idx = np.arange(n_f)

    def indices(a_spec: np.ndarray):
        r = (a_spec @ flat.T / n_t).reshape(n_f, n_ch, n_f).transpose(1, 0, 2)    # (C, F_audio, F_neural)
        d = r[:, diag_idx, diag_idx].mean(1)
        off = (r.sum((1, 2)) - r[:, diag_idx, diag_idx].sum(1)) / (n_f * n_f - n_f)
        return d - off, r

    ci, r_obs = indices(A)
    hop_sec = (win // 2) / sr
    min_shift = int(np.ceil(c["min_shift_sec"] / hop_sec))
    rng = np.random.default_rng([seed, int(pid[1:])])
    shifts = rng.integers(min_shift, n_t - min_shift, size=n_perm)
    null = np.stack([indices(np.roll(A, int(s), axis=1))[0] for s in shifts])     # (perm, C)
    z = (ci - null.mean(0)) / np.maximum(null.std(0, ddof=1), 1e-12)
    pval = norm.sf(z)
    p_emp = (1 + (null >= ci[None, :]).sum(0)) / (1 + len(shifts))
    q = _bh(pval)
    flagged = np.flatnonzero((q < c["alpha"]) & (ci >= c["min_index"]))
    order = np.argsort(-ci)
    summary = {
        "participant": pid,
        "n_channels": int(n_ch),
        "n_windows": int(n_t),
        "freq_range_hz": [float(freqs[0]), float(freqs[-1])],
        "max_index": float(ci.max()),
        "median_index": float(np.median(ci)),
        "n_flagged": int(len(flagged)),
        "flagged_channels": [names[i] for i in flagged],
        "n_significant_any_size": int((q < c["alpha"]).sum()),
        "n_perm": int(n_perm),
        "min_index": c["min_index"],
        "injected_db": inject_db,
        "injected_channel": injected,
        "injected_detected": None if injected is None else bool(injected in [names[i] for i in flagged]),
        "injected_rank": None if injected is None else int(np.flatnonzero(np.asarray(names)[order] == injected)[0]) + 1,
        "per_channel": [{"channel": names[i], "index": round(float(ci[i]), 5), "z": round(float(z[i]), 3),
                         "p": float(pval[i]), "p_perm": float(p_emp[i]), "q": float(q[i])} for i in range(n_ch)],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=1)
    if inject_db is None:
        top = int(np.argmax(ci))
        np.savez_compressed(out_dir / f"{pid}_matrices.npz", freqs=freqs, mean_matrix=r_obs.mean(0),
                            top_channel_matrix=r_obs[top], top_channel=names[top])
    return summary
