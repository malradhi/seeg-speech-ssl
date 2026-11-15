"""BrainBERT (Wang et al., ICLR 2023) as a published pretrained-encoder baseline (Phase 4).

The encoder reproduces the braindecode 2.0 port (``braindecode.models.BrainBERT``, BSD-3,
which matches the authors' ``MaskedTFModel`` to 1e-5) with identical parameter names, so
the released "large" checkpoint re-hosted at huggingface.co/braindecode/brainbert-pretrained
(``pytorch_model.bin``) loads unchanged. braindecode itself needs Python >= 3.11, so it is
not imported here; ``tests/`` checks this module against it where it is installed.

Front-end as published: Laplacian re-referencing along each shaft, 2048 Hz, magnitude STFT
(400-sample Hann window, 50-sample hop, scipy ``boundary="zeros"`` framing, first 40 bins,
``scaling="spectrum"``), per-bin z-score within the input window, 10 boundary frames
trimmed on each side. Each contact is encoded on its own, as upstream.

For frame-level decoding every encoder output frame is kept (no pooling): windows of
10,250 samples (5.005 s) every 2.5 s, of which only the central 2.5 s are used, are
interpolated onto the 10-ms frame centres of the derived features. Per contact, the
768-dimensional embedding is reduced by PCA fitted on the training partition only to
``k`` components (default 9: the same number of features per contact as the lag-stacked
high-gamma input), so the readouts see inputs of identical size.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import signal

SHAFT = re.compile(r"^([A-Za-z]+'?)(\d+)$")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def sinusoidal_table(n_positions: int, dim: int) -> torch.Tensor:
    dim_even = dim + (dim % 2)
    pos = torch.arange(n_positions).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim_even, 2).float() * (-math.log(10000.0) / dim_even))
    pe = torch.zeros(n_positions, dim_even)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe[:, :dim].contiguous()


class STFTFrontEnd(nn.Module):
    def __init__(self, nperseg=400, noverlap=350, idx_freq_cutoff=40, clip=10):
        super().__init__()
        self.nperseg, self.step, self.cut, self.clip = nperseg, nperseg - noverlap, idx_freq_cutoff, clip
        self.pad = nperseg // 2
        self.register_buffer("window", torch.hann_window(nperseg, periodic=True), persistent=False)

    def frame_centres(self, n_times: int) -> np.ndarray:
        """Sample index (relative to the input start) at the centre of each kept frame."""
        n_seg = (n_times + 2 * self.pad - self.nperseg) // self.step + 1
        if (n_times + 2 * self.pad - self.nperseg) % self.step:
            n_seg += 1
        return np.arange(self.clip, n_seg - self.clip) * self.step

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, frames, cut)
        xp = nn.functional.pad(x, (self.pad, self.pad))
        rem = (xp.shape[-1] - self.nperseg) % self.step
        if rem:
            xp = nn.functional.pad(xp, (0, self.step - rem))
        frames = xp.unfold(-1, self.nperseg, self.step)
        win = self.window.to(frames)
        mag = (torch.fft.rfft(frames * win, dim=-1)[..., : self.cut] / win.sum()).abs()
        mean = mag.mean(dim=-2, keepdim=True)
        std = mag.std(dim=-2, unbiased=False, keepdim=True)
        std = std.masked_fill(std == 0, 1.0)
        mag = (mag - mean) / std
        flat = mag.flatten(start_dim=-2).std(dim=-1) == 0
        mag = mag.masked_fill(flat[..., None, None], 1.0)
        if self.clip:
            mag = mag[..., self.clip : -self.clip, :]
        return torch.nan_to_num(mag, nan=0.0)


class InputEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, drop: float, max_len: int = 5000):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(drop)
        self.register_buffer("pe", sinusoidal_table(max_len, hidden_dim), persistent=False)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(spec)
        return self.dropout(self.layer_norm(h + self.pe[: h.size(1)]))


class BrainBERTEncoder(nn.Module):
    """Frame embeddings ``(B, frames, hidden_dim)`` for single-contact signals ``(B, T)``."""

    def __init__(self, hidden_dim=768, ffn_dim=3072, n_layers=6, n_heads=12, nperseg=400, noverlap=350,
                 idx_freq_cutoff=40, stft_clip=10, drop_prob=0.1):
        super().__init__()
        self.spectrogram = STFTFrontEnd(nperseg, noverlap, idx_freq_cutoff, stft_clip)
        self.input_embedding = InputEmbedding(idx_freq_cutoff, hidden_dim, drop_prob)
        layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, ffn_dim, drop_prob, activation=nn.GELU(),
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.input_embedding(self.spectrogram(x)))


CONFIG_KEYS = ("hidden_dim", "ffn_dim", "n_layers", "n_heads", "nperseg", "noverlap", "idx_freq_cutoff",
               "stft_clip")


def load_pretrained(model_dir, device="cpu") -> tuple[BrainBERTEncoder, dict]:
    """Build the encoder from ``config.json`` and load ``pytorch_model.bin`` (head dropped)."""
    model_dir = Path(model_dir)
    conf = json.loads((model_dir / "config.json").read_text())
    kwargs = {k: conf[k] for k in CONFIG_KEYS if k in conf}
    if conf.get("stft_zscore_before_clip", True) is not True:
        raise ValueError("this port implements the z-score-before-clip recipe of the released checkpoint")
    enc = BrainBERTEncoder(**kwargs)
    weights = model_dir / "pytorch_model.bin"
    state = torch.load(weights, map_location="cpu", weights_only=True)
    # Heads not used for frame embeddings: the downstream classifier and the masked-spectrogram
    # reconstruction head kept from pretraining. Every encoder weight must be present.
    unused = ("final_layer.", "spec_prediction_head.")
    dropped = sorted(k for k in state if k.startswith(unused))
    state = {k: v for k, v in state.items() if not k.startswith(unused)}
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"BrainBERT checkpoint mismatch: missing={missing} unexpected={unexpected}")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    info = {"config": kwargs, "weights_sha256": digest, "n_params": int(sum(p.numel() for p in enc.parameters())),
            "unused_checkpoint_keys": dropped}
    return enc.to(device).eval(), info


# ---------------------------------------------------------------------------
# Signal preparation
# ---------------------------------------------------------------------------


def laplacian(x: np.ndarray, names: list[str], exclude: str | None = None) -> np.ndarray:
    """Shaft Laplacian: each contact minus the mean of its existing neighbours on the same
    shaft (contact numbers +-1). Contacts without a neighbour are left unchanged."""
    rx = re.compile(exclude, re.IGNORECASE) if exclude else None
    index = {}
    for i, nm in enumerate(names):
        m = SHAFT.match(str(nm).strip())
        if m and not (rx and rx.search(nm)):
            index[(m.group(1), int(m.group(2)))] = i
    out = np.array(x, dtype=np.float64, copy=True)
    for (shaft, num), i in index.items():
        nb = [index[(shaft, num + d)] for d in (-1, 1) if (shaft, num + d) in index]
        if nb:
            out[:, i] = x[:, i] - x[:, nb].mean(axis=1)
    return out


def resample(x: np.ndarray, sr_in: float, sr_out: int) -> np.ndarray:
    frac = Fraction(int(round(sr_out)), int(round(sr_in))).limit_denominator(1000)
    if frac == 1:
        return np.asarray(x, dtype=np.float64)
    return signal.resample_poly(x, frac.numerator, frac.denominator, axis=0)


def frame_centres_seconds(n_samples: int, sr: float, win_sec: float, shift_sec: float, n_frames: int):
    """Centres of the derived 50-ms frames (same rule as preprocessing._frame_edges)."""
    starts = np.floor(np.arange(n_frames) * shift_sec * sr).astype(np.int64)
    stops = np.floor(starts + win_sec * sr).astype(np.int64)
    return (starts + stops) / 2.0 / sr


# ---------------------------------------------------------------------------
# Frame-level embeddings
# ---------------------------------------------------------------------------


def window_plan(n: int, centres_s: np.ndarray, sr: int, rel: np.ndarray, bcfg: dict) -> dict:
    """Where each derived frame centre falls: window k (its central hop) and the two encoder
    frames to interpolate between. Shared by every contact of a partition."""
    L, H = int(bcfg["window_samples"]), int(bcfg["hop_samples"])
    left = (L - H) // 2
    k_win = int(math.ceil(n / H))
    target = centres_s * sr
    k = np.clip((target // H).astype(np.int64), 0, k_win - 1)
    pos = target - (k * H - left)
    f = np.interp(pos, rel, np.arange(len(rel)))
    i0 = np.clip(np.floor(f).astype(np.int64), 0, len(rel) - 1)
    i1 = np.clip(i0 + 1, 0, len(rel) - 1)
    return {"L": L, "H": H, "left": left, "k_win": k_win, "k": k, "i0": i0, "i1": i1,
            "w": (f - i0).astype(np.float32)[:, None]}


@torch.no_grad()
def contact_embeddings(enc: BrainBERTEncoder, x: np.ndarray, plan: dict, batch_windows: int, device) -> np.ndarray:
    """Embeddings of one contact's signal at the frame centres described by ``plan``."""
    L, H, left, k_win = plan["L"], plan["H"], plan["left"], plan["k_win"]
    n = len(x)
    right = max(0, (k_win - 1) * H + L - left - n)
    mode = "reflect" if max(left, right) < n else "constant"
    xp = np.pad(np.asarray(x, dtype=np.float32), (left, right), mode=mode)
    wins = np.lib.stride_tricks.sliding_window_view(xp, L)[::H][:k_win]
    z = np.concatenate([enc(torch.from_numpy(np.ascontiguousarray(wins[b:b + batch_windows])).to(device))
                        .float().cpu().numpy() for b in range(0, k_win, batch_windows)])
    k, i0, i1, w = plan["k"], plan["i0"], plan["i1"], plan["w"]
    return (1 - w) * z[k, i0] + w * z[k, i1]


def pca_fit(e: np.ndarray, k: int, report_k=()):
    """Per-contact PCA on training frames. With ``report_k`` a fourth value is returned: the
    share of variance kept by the leading k' components for each k' in ``report_k``."""
    mu = e.mean(0)
    xc = (e - mu).astype(np.float64)
    cov = xc.T @ xc / max(1, len(xc) - 1)
    val, vec = np.linalg.eigh(cov)
    order = np.argsort(val)[::-1]
    val, vec = val[order], vec[:, order]
    comps = vec[:, :k]
    for j in range(comps.shape[1]):  # deterministic sign
        if comps[np.argmax(np.abs(comps[:, j])), j] < 0:
            comps[:, j] *= -1
    total = max(val.sum(), 1e-12)
    out = (mu.astype(np.float32), comps.astype(np.float32), float(val[:k].sum() / total))
    if report_k:
        return out + ({int(r): float(val[: int(r)].sum() / total) for r in report_k},)
    return out


def extract_participant(pid: str, raw_root, derived_dir, out_dir, cfg: dict, model_dir, device="cpu",
                        overwrite=False, encoder=None, enc_info=None) -> Path:
    """Write ``out_dir/<pid>.npz`` with ``bb_train/val/test`` (frames x C*k, float16), frame-aligned
    with ``derived/<pid>.npz``."""
    from .preprocessing import find_raw_files, infer_eeg_sr

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}.npz"
    if out_path.exists() and not overwrite:
        print(f"{pid}: BrainBERT features exist -> skip", flush=True)
        return out_path
    t0 = time.time()
    p, bcfg = cfg["preprocess"], cfg["brainbert"]
    with np.load(Path(derived_dir) / f"{pid}.npz", allow_pickle=False) as d:
        meta = json.loads(str(d["meta_json"]))
        n_frames = {s: int(d[f"hg_{s}"].shape[0]) for s in ("train", "val", "test")}
    files = find_raw_files(raw_root, pid)
    eeg = np.load(files["eeg"], mmap_mode="r")
    audio_len = int(np.load(files["audio"], mmap_mode="r").shape[0])
    sr = infer_eeg_sr(eeg.shape[0], audio_len, p["audio_sr"], p["eeg_sr_candidates"])
    if abs(sr - meta["eeg_sr"]) > 1e-6:
        raise ValueError(f"{pid}: sampling rate {sr} differs from the derived file ({meta['eeg_sr']})")
    all_names = np.asarray(np.load(files["channels"], allow_pickle=True)).astype(str).reshape(-1).tolist()
    kept = meta["channels"]
    idx = [all_names.index(c) for c in kept]
    if encoder is None:
        encoder, enc_info = load_pretrained(model_dir, device)
    sr_bb = int(bcfg["sr"])
    k = int(bcfg["pca_components"])
    encoder.eval()
    rel = encoder.spectrogram.frame_centres(int(bcfg["window_samples"]))
    emb = {}
    for split in ("train", "val", "test"):
        s, e = meta["bounds_samples"][split]
        x = laplacian(np.asarray(eeg[s:e], dtype=np.float64), all_names, p.get("exclude_channel_regex"))[:, idx]
        x = resample(x, sr, sr_bb).astype(np.float32)
        centres = frame_centres_seconds(e - s, sr, p["win_sec"], p["shift_sec"], n_frames[split])
        emb[split] = x, window_plan(len(x), centres, sr_bb, rel, bcfg)
    arrays = {s: np.zeros((n_frames[s], len(kept) * k), dtype=np.float16) for s in emb}
    explained, at = [], []
    report_k = sorted(set(int(r) for r in bcfg.get("report_k", [])) | {k})
    bw = int(bcfg["batch_windows"])
    for c in range(len(kept)):
        per = {s: contact_embeddings(encoder, emb[s][0][:, c], emb[s][1], bw, device) for s in emb}
        mu, comps, ev, ev_at = pca_fit(per["train"], k, report_k)
        explained.append(ev)
        at.append(ev_at)
        for s in per:
            arrays[s][:, c * k : (c + 1) * k] = ((per[s] - mu) @ comps).astype(np.float16)
    info = {"participant": pid, "k": k, "n_contacts": len(kept), "frames": n_frames, "sr": sr_bb,
            "window_samples": bcfg["window_samples"], "hop_samples": bcfg["hop_samples"],
            "pca_explained_mean": float(np.mean(explained)), "pca_explained_min": float(np.min(explained)),
            "pca_explained_at": {str(r): {"mean": float(np.mean([a[r] for a in at])),
                                          "min": float(np.min([a[r] for a in at]))} for r in report_k},
            "encoder": enc_info, "elapsed_s": round(time.time() - t0, 1)}
    tmp = out_dir / f"{pid}.tmp.npz"
    np.savez_compressed(tmp, info_json=np.array(json.dumps(info)), **{f"bb_{s}": a for s, a in arrays.items()})
    tmp.replace(out_path)
    (out_dir / f"{pid}.json").write_text(json.dumps(info, indent=1))
    print(f"{pid}: BrainBERT features {arrays['train'].shape[1]} dims ({len(kept)} contacts x {k}), "
          f"PCA keeps {info['pca_explained_mean']:.0%} of variance, {info['elapsed_s'] / 60:.1f} min", flush=True)
    return out_path
