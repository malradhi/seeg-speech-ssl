"""Phase 3 evaluation suite (CPU), computed from saved test predictions.

For each participant and each selected run (best learning rate on validation, as in the
analysis) it reports, on the test partition:

* ``r_speech`` and its own chance level: within-speech r after circularly shifting the
  prediction against the target (>= 1 s shifts), so that a within-speech result can be
  tested against chance like overall r;
* a split of within-speech r into **loudness** (frame mean over the 23 bands) and
  **spectral shape** (each frame minus its band mean), each with its chance level;
* MCD overall and within speech;
* intelligibility of a waveform resynthesis: the predicted and the true log-Mel
  spectrograms are both inverted to audio with the same Griffin-Lim procedure, and
  STOI / ESTOI compare the two (the reference is the resynthesised target, so the score
  isolates decoding error from vocoder loss). The two syntheses start from independent
  random phases: with a shared starting phase even a constant spectrum correlates with
  the target. ESTOI is the primary intelligibility score; classic STOI is reported for
  comparability but scores 0.3-0.5 for predictions with no speech content (its clipping
  step rewards loud stationary output), so it is only read against the floors below.

References evaluated the same way: the speech-timing reference (mean speech spectrum
during speech, mean silence spectrum otherwise), the training-mean spectrum (one
constant prediction), the true spectrogram circularly shifted by half the test
partition (realistic speech at the wrong time: the chance floor of every metric), and
the true spectrogram itself (r = 1, MCD = 0; its STOI/ESTOI is the resynthesis ceiling,
below 1 because of the independent phases). ``probe_ssl`` (ridge on frozen Phase 1 SSL features)
is recomputed here on the CPU because Phase 2 did not store probe predictions; its
r_mean is checked against the Phase 2 record.
"""

from __future__ import annotations

import json
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import baselines as B
from . import metrics as M
from .config import config_hash
from .data import ParticipantData, label_subset
from .preprocessing import MelFilterBank
from .provenance import code_hash, environment

# ---------------------------------------------------------------------------
# Resynthesis
# ---------------------------------------------------------------------------


class Resynthesizer:
    """Log-Mel (the preprocessing's 23-band filterbank) -> waveform by Griffin-Lim."""

    def __init__(self, cfg: dict):
        p = cfg["preprocess"]
        self.sr = p["audio_target_sr"]
        self.win = int(np.floor(p["win_sec"] * self.sr))
        self.hop = int(round(p["shift_sec"] * self.sr))
        self.window = np.hanning(self.win + 1)[:-1]
        if self.win % self.hop:
            raise ValueError("resynthesis needs the window length to be a multiple of the hop")
        fb = MelFilterBank(self.win // 2 + 1, p["n_mels"], self.sr).matrix  # (bins, bands)
        w = fb.T.copy()  # (bands, bins): share each band's level among the bins it covers
        tot = w.sum(0, keepdims=True)
        self.inv = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
        self.iters = cfg["eval"]["griffin_lim_iters"]

    def magnitude(self, logmel: np.ndarray) -> np.ndarray:
        band = np.clip(np.exp(np.asarray(logmel, np.float64)) - 1e-7, 0.0, None)
        return band @ self.inv  # (frames, bins)

    def _stft(self, x: np.ndarray, n_frames: int) -> np.ndarray:
        idx = np.arange(n_frames)[:, None] * self.hop + np.arange(self.win)[None, :]
        return np.fft.rfft(x[idx] * self.window, axis=1)

    def _ola(self, frames: np.ndarray) -> np.ndarray:
        """Overlap-add of (n, win) frames at hop spacing (win is a multiple of hop)."""
        n, k = frames.shape[0], self.win // self.hop
        out = np.zeros((n + k - 1, self.hop))
        blocks = frames.reshape(n, k, self.hop)
        for j in range(k):
            out[j : j + n] += blocks[:, j, :]
        return out.reshape(-1)

    def _istft(self, spec: np.ndarray) -> np.ndarray:
        frames = np.fft.irfft(spec, n=self.win, axis=1) * self.window
        norm = self._ola(np.broadcast_to(self.window ** 2, frames.shape).copy())
        return self._ola(frames) / np.maximum(norm, 1e-8)

    def synth(self, logmel: np.ndarray, seed: int = 0) -> np.ndarray:
        mag = self.magnitude(logmel)
        rng = np.random.default_rng(seed)
        spec = mag * np.exp(2j * np.pi * rng.random(mag.shape))
        for _ in range(self.iters):
            x = self._istft(spec)
            ang = np.angle(self._stft(x, mag.shape[0]))
            spec = mag * np.exp(1j * ang)
        x = self._istft(spec)
        peak = np.max(np.abs(x))
        return (x / peak).astype(np.float64) if peak > 0 else x


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _speech_parts(y: np.ndarray, p: np.ndarray, v: np.ndarray) -> tuple[float, float, float]:
    ys, ps = y[v], p[v]
    r_speech = M.mean_band_r(ys, ps)
    r_loud = _pearson(ys.mean(1), ps.mean(1))
    r_shape = M.mean_band_r(ys - ys.mean(1, keepdims=True), ps - ps.mean(1, keepdims=True))
    return r_speech, r_loud, r_shape


def speech_metrics(y: np.ndarray, p: np.ndarray, v: np.ndarray, n_shifts: int, min_shift: int,
                   seed: int) -> dict:
    """Within-speech r, loudness r and spectral-shape r, each with a circular-shift null."""
    y = np.asarray(y, np.float64).reshape(-1, np.asarray(y).shape[-1])
    p = np.asarray(p, np.float64).reshape(-1, y.shape[1])
    v = np.asarray(v, bool).reshape(-1)
    obs = _speech_parts(y, p, v)
    n = len(y)
    out = dict(zip(("r_speech", "r_loud", "r_shape"), obs))
    if v.sum() < 20 or n <= 2 * min_shift:
        return out
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, n - min_shift + 1, size=n_shifts)
    null = np.array([_speech_parts(y, np.roll(p, int(s), axis=0), v) for s in shifts])
    for i, k in enumerate(("r_speech", "r_loud", "r_shape")):
        col = null[:, i]
        col = col[np.isfinite(col)]
        if len(col) and np.isfinite(obs[i]):
            out[f"{k}_chance_p95"] = float(np.percentile(col, 95))
            out[f"{k}_p"] = float((1 + np.sum(col >= obs[i])) / (1 + len(col)))
    return out


def _stoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> dict:
    try:
        from pystoi import stoi
    except ImportError:
        return {}
    n = min(len(ref), len(deg))
    return {"stoi": float(stoi(ref[:n], deg[:n], sr, extended=False)),
            "estoi": float(stoi(ref[:n], deg[:n], sr, extended=True))}


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------


def _val_mse(rec: dict) -> float:
    v = (rec.get("val") or {}).get("mse")
    return float(v) if v is not None and np.isfinite(v) else np.inf


def _seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def select_runs(pid: str, root: Path, cfg: dict) -> dict:
    """{(condition, fraction, seed): (run record, prediction path)} with the lowest
    validation MSE over every phase's learning rates."""
    ev = cfg["eval"]
    want = set(ev["conditions"])
    groups = defaultdict(list)
    for phase in ev["phases"]:
        runs = root / "results" / phase / "runs"
        preds = root / "results" / phase / "preds"
        for path in sorted(runs.glob(f"{pid}__*.json")):
            try:
                rec = json.loads(path.read_text())
            except Exception:
                continue
            if rec.get("status") != "ok" or rec.get("kind") not in ("finetune", "linear"):
                continue
            if rec.get("protocol") not in ("new", "closed_form"):
                continue
            c, f, s = rec.get("condition"), rec.get("fraction"), rec.get("seed")
            if c not in want or s not in ev["seeds"] or not any(np.isclose(f, x) for x in ev["fractions"]):
                continue
            pred = preds / f"{rec['run_id']}.npz"
            if pred.exists():
                groups[(c, float(f), int(s))].append((rec, pred))
    out = {}
    for key, items in groups.items():
        out[key] = min(items, key=lambda it: _val_mse(it[0]))
    return out


def _probe_predictions(d: ParticipantData, root: Path, cfg: dict) -> dict:
    """Ridge on frozen Phase 1 SSL features, exactly as the Phase 2 probes."""
    import torch

    from . import train as T

    ev, out = cfg["eval"], {}
    enc_dir = root / "results" / ev["probe_encoder_dir"] / "encoders"
    x = {s: torch.from_numpy(d.z(s)) for s in ("train", "val", "test")}
    for seed in ev["seeds"]:
        path = enc_dir / f"{d.pid}__ssl__s{seed}.pt"
        if not path.exists():
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        feats = {s: T.encode(state, x[s], cfg) for s in x}
        dm = feats["train"].shape[-1]
        ftr = feats["train"].reshape(-1, dm)
        mu, sd = ftr.mean(0), np.maximum(ftr.std(0), 1e-6)
        z = {s: (feats[s] - mu) / sd for s in feats}
        for frac in ev["fractions"]:
            idx = label_subset(d.n_windows("train"), frac, seed, d.pid)
            yr = d.Yw["train"][idx].reshape(-1, d.Yw["train"].shape[-1])
            ym, ys = yr.mean(0), np.maximum(yr.std(0), 1e-6)
            yva = (d.Yw["val"].reshape(-1, yr.shape[1]) - ym) / ys
            res = B.ridge(z["train"][idx].reshape(-1, dm), (yr - ym) / ys, z["val"].reshape(-1, dm), yva,
                          z["test"].reshape(-1, dm), cfg["linear"]["ridge_alphas"])
            out[("probe_ssl", float(frac), int(seed))] = (res["pred_test"] * ys + ym).reshape(d.Yw["test"].shape)
    return out


def _phase2_probe_r(pid: str, root: Path, cfg: dict) -> dict:
    out = {}
    runs = root / "results" / cfg["eval"]["probe_check_dir"] / "runs"
    for path in runs.glob(f"{pid}__probe_ssl__*.json"):
        try:
            rec = json.loads(path.read_text())
            out[(float(rec["fraction"]), int(rec["seed"]))] = rec["test"]["r_mean"]
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def evaluate_participant(pid: str, root, cfg: dict, overwrite: bool = False) -> Path:
    root = Path(root)
    ev = cfg["eval"]
    out_dir = root / "results" / ev["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}.json"
    if out_path.exists() and not overwrite:
        print(f"{pid}: evaluation exists -> skip", flush=True)
        return out_path
    t0 = time.time()
    d = ParticipantData(root / "derived" / f"{pid}.npz", cfg)
    y = d.Yw["test"].reshape(-1, d.Yw["test"].shape[-1]).astype(np.float64)
    v = d.vad["test"].reshape(-1)
    rs = Resynthesizer(cfg)
    do_stoi = ev["stoi"]
    if do_stoi:
        try:
            import pystoi  # noqa: F401
        except ImportError:
            print(f"{pid}: pystoi not installed -> STOI/ESTOI skipped (python -m pip install --user pystoi)",
                  flush=True)
            do_stoi = False
    ref_wav = rs.synth(y, seed=0) if do_stoi else None

    preds = {}
    for key, (rec, path) in select_runs(pid, root, cfg).items():
        with np.load(path) as z:
            preds[key] = (z["pred_test"].astype(np.float64), rec)
    if "probe_ssl" in ev["conditions"]:
        check = _phase2_probe_r(pid, root, cfg)
        for key, p in _probe_predictions(d, root, cfg).items():
            r_here = M.mean_band_r(y, p.reshape(-1, y.shape[1]))
            preds[key] = (p, {"run_id": f"{pid}__probe_ssl__f{key[1]:.2f}__s{key[2]}", "lr": None,
                              "check_r_mean_phase2": check.get((key[1], key[2])), "r_mean_here": r_here})
    with np.load(root / "results" / ev["reference_dir"] / "preds" / f"{pid}__reference.npz") as z:
        timing = z["vad_oracle"].astype(np.float64)
    train_mean = np.broadcast_to(d.Yw["train"].reshape(-1, y.shape[1]).mean(0), y.shape)
    refs = {"timing_reference": timing, "train_mean": train_mean,
            "shifted_truth": np.roll(y, len(y) // 2, axis=0), "true_spectrogram": y}
    for name, p in refs.items():
        preds[(name, None, None)] = (p, {"run_id": f"{pid}__{name}", "lr": None})

    rows = []
    for (cond, frac, seed), (p, rec) in sorted(preds.items(), key=lambda kv: str(kv[0])):
        p = np.asarray(p, np.float64).reshape(-1, y.shape[1])
        row = {"participant": pid, "task": d.task, "condition": cond, "fraction": frac, "seed": seed,
               "run_id": rec.get("run_id"), "lr": rec.get("lr"),
               "r_mean": M.mean_band_r(y, p),
               "mcd_db": M.mcd(y, p, cfg["metrics"]["mcd_n_coeffs"]),
               "mcd_speech_db": M.mcd(y, p, cfg["metrics"]["mcd_n_coeffs"], v)}
        for k in ("check_r_mean_phase2", "r_mean_here"):
            if k in rec:
                row[k] = rec[k]
        row.update(speech_metrics(y, p, v, ev["n_shifts"], ev["min_shift_frames"],
                                  seed=_seed_of(pid, cond, frac, seed)))
        if do_stoi and (seed is None or seed in ev["stoi_seeds"]):
            row.update(_stoi(ref_wav, rs.synth(p, seed=1), rs.sr))
        rows.append(row)
    stamp = {"code_hash": code_hash(), "config_hash": config_hash(cfg), "env": environment()}
    res = {"participant": pid, "n_rows": len(rows), "stoi": do_stoi, "test_seconds": len(y) * cfg["preprocess"]["shift_sec"],
           "elapsed_s": round(time.time() - t0, 1), "rows": rows, **stamp}
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=float))
    tmp.replace(out_path)
    print(f"{pid}: evaluated {len(rows)} predictions in {res['elapsed_s'] / 60:.1f} min", flush=True)
    return out_path
