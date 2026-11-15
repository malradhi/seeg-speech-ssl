"""Evaluation metrics on frame sequences (frames x 23 log-Mel bands)."""

from __future__ import annotations

import numpy as np


def band_pearson(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    yc, pc = y - y.mean(0), p - p.mean(0)
    denom = np.sqrt((yc**2).sum(0) * (pc**2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (yc * pc).sum(0) / denom
    r[~np.isfinite(r)] = np.nan
    return r


def mean_band_r(y, p) -> float:
    r = band_pearson(y, p)
    return float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")


def mel_cepstra(logmel: np.ndarray, n: int) -> np.ndarray:
    """Real-cepstrum scaling: c_d = (1/K) * sum_k L_k cos(pi d (k + 0.5) / K),
    i.e. the unnormalised DCT-II divided by 2K (K = number of Mel bands)."""
    from scipy.fft import dct

    x = np.asarray(logmel, np.float64)
    return (dct(x, type=2, norm=None, axis=-1) / (2.0 * x.shape[-1]))[:, 1 : n + 1]


def mcd(y_logmel, p_logmel, n: int = 22, mask=None) -> float:
    """Mel-cepstral distortion (dB), (10/ln10) * sqrt(2 * sum_d (c_d - c'_d)^2) over
    c1..cn of the 23-band natural-log Mel amplitude spectrum (c0, overall level,
    excluded). Values from a 23-band Mel representation are not directly
    comparable with MCD computed from vocoder features of other papers."""
    cy, cp = mel_cepstra(y_logmel, n), mel_cepstra(p_logmel, n)
    d = (10.0 / np.log(10.0)) * np.sqrt(2.0 * ((cy - cp) ** 2).sum(-1))
    if mask is not None:
        d = d[np.asarray(mask, bool)]
    return float(d.mean()) if len(d) else float("nan")


def circular_shift_null(y, p, min_shift: int) -> dict:
    """Chance level by circularly shifting predictions against targets.

    For z-scored series the mean-band Pearson r at every circular lag follows
    from one FFT cross-correlation per band; lags shorter than ``min_shift``
    frames (and their wrap-around) are excluded from the null distribution.
    """
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    n = len(y)
    ys, ps = y.std(0), p.std(0)
    ok = (ys > 0) & (ps > 0)
    if not ok.any() or n <= 2 * min_shift:
        return {"chance_mean": float("nan"), "chance_p95": float("nan"), "p_value": float("nan")}
    zy = (y[:, ok] - y[:, ok].mean(0)) / ys[ok]
    zp = (p[:, ok] - p[:, ok].mean(0)) / ps[ok]
    xc = np.fft.irfft(np.conj(np.fft.rfft(zy, axis=0)) * np.fft.rfft(zp, axis=0), n=n, axis=0) / n
    r_lag = xc.mean(1)
    observed = r_lag[0]
    null = r_lag[min_shift : n - min_shift + 1]
    return {
        "chance_mean": float(null.mean()),
        "chance_p95": float(np.percentile(null, 95)),
        "p_value": float((1 + np.sum(null >= observed)) / (1 + len(null))),
    }


def evaluate(y_logmel, p_logmel, vad, cfg: dict) -> dict:
    """All test metrics for one run. Inputs are frames x bands in log-Mel units."""
    y = np.asarray(y_logmel).reshape(-1, np.asarray(y_logmel).shape[-1])
    p = np.asarray(p_logmel).reshape(-1, y.shape[-1])
    v = np.asarray(vad, bool).reshape(-1)
    mc = cfg["metrics"]
    out = {
        "r_mean": mean_band_r(y, p),
        "r_speech": mean_band_r(y[v], p[v]) if v.sum() > 10 else float("nan"),
        "mcd_db": mcd(y, p, mc["mcd_n_coeffs"]),
        "mcd_speech_db": mcd(y, p, mc["mcd_n_coeffs"], v) if v.any() else float("nan"),
        "mse_logmel": float(np.mean((y - p) ** 2)),
        "r_per_band": [None if not np.isfinite(r) else round(float(r), 5) for r in band_pearson(y, p)],
    }
    out.update(circular_shift_null(y, p, mc["chance_min_shift_frames"]))
    return out


def vad_oracle_prediction(y_train, vad_train, vad_test) -> np.ndarray:
    """Reference that knows only *when* speech occurs: mean speech spectrum during
    speech frames, mean silence spectrum otherwise (means from training data)."""
    yt = np.asarray(y_train).reshape(-1, np.asarray(y_train).shape[-1])
    vt = np.asarray(vad_train, bool).reshape(-1)
    speech, silence = yt[vt].mean(0), yt[~vt].mean(0)
    vte = np.asarray(vad_test, bool)
    return np.where(vte[..., None], speech, silence).astype(np.float32)
