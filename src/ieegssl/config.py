"""Central configuration.

The configuration is a plain nested dict so it can be written verbatim into every
result file. ``load_config`` merges a JSON override file on top of ``DEFAULTS``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

OSF_NODE = "ak3dp"
PARTICIPANTS = tuple(f"P{i:02d}" for i in range(1, 31))


def task_of(pid: str) -> str:
    """Task cohort of a participant (Verwoert et al., Cell Reports 2025)."""
    n = int(pid[1:])
    if 1 <= n <= 10:
        return "unique_words"
    if 11 <= n <= 20:
        return "repeated_words"
    if 21 <= n <= 30:
        return "sentences"
    return "unknown"


DEFAULTS: dict = {
    "preprocess": {
        # Sampling rate of P##_audio.npy. The sEEG rate is inferred from the
        # length ratio and snapped to the nearest candidate (verified on P15).
        "audio_sr": 48000,
        "eeg_sr_candidates": [512, 1000, 1024, 2000, 2048],
        "win_sec": 0.05,
        "shift_sec": 0.01,
        "hg_band": [70.0, 170.0],
        "line_bandstops": [[98.0, 102.0], [148.0, 152.0]],
        "filter_order": 4,
        "audio_target_sr": 16000,
        "n_mels": 23,
        "train_frac": 0.70,
        "val_frac": 0.15,
        # Move each partition boundary to the middle of the nearest inter-trial
        # silence so that no spoken trial is split across partitions.
        "snap_to_silence": True,
        # Only intracranial contacts with an anatomical location are used. This
        # removes marker (MKR), ECG, pulse-oximetry and unlocalised channels and
        # reproduces the published channel sets (1,103 contacts for P01-P10).
        # The regex is a second, explicit safeguard against non-neural channels.
        "channel_policy": "localised",
        "exclude_channel_regex": "^(MKR|EKG|ECG|PULS|BEAT|SPO2)",
    },
    "features": {
        "model_order": 4,        # 9 lag contexts
        "step": 5,               # 50 ms between lag contexts -> +-200 ms
        # Frames discarded at each partition edge (covers the 20-frame lag
        # margin and filter/Hilbert edge transients). 50 frames = 0.5 s.
        "edge_trim_frames": 50,
        "window": 50,            # 0.5-s model windows
        "stride": 50,            # non-overlapping windows
    },
    "model": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 4,
        "ff_dim": 512,
        "dropout": 0.1,
        "head_hidden": 64,
    },
    # ---- Fair ("new") protocol -------------------------------------------
    "pretrain": {
        "mask_kind": "random",
        "mask_ratio": 0.5,
        "steps": 2000,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "warmup": 100,
        "batch_size": 16,
        "grad_clip": 1.0,
    },
    "finetune": {
        "lrs": [3e-5, 1e-4, 3e-4, 1e-3],
        "weight_decay": 0.01,
        "max_steps": 3000,       # identical update budget for every label fraction
        "warmup": 100,
        "eval_every": 25,
        "patience": 40,          # evaluations without improvement (= 1000 steps) before stopping
        "batch_size": 16,
        "grad_clip": 1.0,
    },
    # ---- Legacy protocol (exact pilot-study settings, for continuity) ----
    "legacy": {
        "ft_epochs": 40,
        "ft_lr": 1e-4,
        "pt_epochs": 30,
        "pt_lr": 1e-3,
        "batch_size": 16,
        "grad_clip": 1.0,
        "mask_ratio": 0.5,
    },
    "linear": {
        "ridge_alphas": [10.0 ** (k / 2.0) for k in range(-2, 21)],  # 0.1 .. 1e10
        "pca_components": 50,
    },
    "metrics": {
        "chance_min_shift_frames": 100,  # circular shifts of >= 1 s
        "mcd_n_coeffs": 22,              # mel-cepstral coefficients c1..c22
    },
    "phase1": {
        "fractions": [1.0, 0.5, 0.25, 0.10, 0.05],
        "seeds": [1, 2, 3, 4, 5],
        "deep_conditions": ["sup", "ssl", "placebo"],
        "legacy_conditions": ["sup_legacy", "ssl_legacy"],
        "linear_conditions": ["ridge", "pca_lab", "pca_all"],
        "save_predictions": True,
    },
    # ---- Phase 2 (docs/PHASE2_DESIGN.md) ---------------------------------
    "phase2": {
        "participants": None,            # None = all 30
        "phase1_dir": "phase1",          # where the Phase 1 encoders live (results/<dir>/encoders)
        "results_dir": "phase2",         # results/<dir>/ for Phase 2 outputs (smoke test: smoke2)
        "fractions": [1.0, 0.5, 0.25, 0.10, 0.05],
        "seeds_main": [1, 2, 3, 4, 5],
        "seeds_small": [1, 2, 3],
        "lr_extra": [1e-5, 3e-3],
        "lr_extra_conditions": ["sup", "ssl", "placebo"],
        "mask_kinds": ["random", "time", "contact", "lagcons"],
        "mask_ratios": [0.25, 0.5, 0.75],
        "mask_fractions": [1.0, 0.10],
        "multi_steps_per_participant": 2000,
        "adapt_steps": 2000,
        "scaling_k": [5, 15],
        "scaling_fractions": [0.25, 0.10, 0.05],
        "multi_workers": 3,
        "run_gru": True,
        "run_probes": True,
        "save_predictions": True,
    },
    "gru": {"proj_dim": 128, "hidden": 128, "layers": 2, "dropout": 0.1},
    # ---- Phase 3 (docs/PHASE3_DESIGN.md) ---------------------------------
    "model_large": {"d_model": 256, "nhead": 8, "num_layers": 6, "ff_dim": 1024, "dropout": 0.1,
                    "head_hidden": 64},
    "phase3": {
        "participants": None,
        "phase1_dir": "phase1",          # Phase 1 SSL encoders (for the 1e-2 SSL runs)
        "results_dir": "phase3",
        "fractions": [1.0, 0.5, 0.25, 0.10, 0.05],
        "seeds_main": [1, 2, 3, 4, 5],
        "seeds_small": [1, 2, 3],
        "gru_lrs": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],  # gru_ssl, gru_placebo
        "gru_lr_extra": [1e-5, 3e-3],                      # supervised GRU: completes the six-rate grid
        "edge_lrs": [1e-2],
        "edge_conditions": ["sup", "ssl"],
        "large_fractions": [1.0, 0.25, 0.10, 0.05],
        "pooled_large_seeds": [1, 2, 3],
        "multi_steps_per_participant": 2000,
        "run_probes": True,
        "save_predictions": True,
    },
    # ---- Phase 4 (docs/PHASE4_DESIGN.md) ---------------------------------
    "brainbert": {
        "model_dir": "models/brainbert",   # under IEEG_ROOT: config.json + pytorch_model.bin
        "hf_repo": "braindecode/brainbert-pretrained",
        "sr": 2048,
        "window_samples": 10250,           # 5.005 s: a whole number of 50-sample STFT hops
        "hop_samples": 5125,               # only the central 2.5 s of each window are used
        "batch_windows": 128,
        "pca_components": 9,               # = features per contact of the lag-stacked high-gamma input
        "features_dir": "derived_brainbert",
    },
    "phase4": {
        "participants": None,
        "results_dir": "phase4",
        "phase1_dir": "phase1",            # Transformer SSL encoders (LP-FT)
        "phase3_dir": "phase3",            # GRU SSL encoders (LP-FT)
        "fractions": [1.0, 0.5, 0.25, 0.10, 0.05],
        "seeds_main": [1, 2, 3, 4, 5],
        "seeds_small": [1, 2, 3],
        "grid6": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
        "lp": {"lr": 1e-3, "max_steps": 1000, "patience": 20},
        "dose_fractions": [0.125, 0.25, 0.5],   # share of the training partition used for pretraining
        "dose_label_fractions": [0.10, 0.05],
        "run_brainbert": True,
        "run_dose": True,
        "run_lpft": True,
        "save_predictions": True,
    },
    # ---- Phase 4b (docs/PHASE4B_DESIGN.md) --------------------------------
    "phase4b": {
        "participants": None,
        "results_dir": "phase4b",
        "phase1_dir": "phase1",            # placebo / ssl encoders (read only)
        "phase3_dir": "phase3",            # gru_ssl / gru_placebo encoders (read only)
        "phase4_dir": "phase4",            # dose-response encoders (read only)
        "fractions": [1.0, 0.5, 0.25, 0.10, 0.05],
        "seeds_main": [1, 2, 3, 4, 5],
        "seeds_small": [1, 2, 3],
        "grid7": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        # H17 self-pretraining: pretrain on the labelled windows only
        "self_fractions": [0.25, 0.10, 0.05],
        # R2 grid edge: conditions run on six rates get the seventh
        "edge_lr": 1e-2,
        "edge_conditions": ["placebo", "lpft_ssl", "gru", "gru_ssl", "gru_placebo", "lpft_gru_ssl", "bb_gru"],
        "dose_fractions": [0.125, 0.25, 0.5],
        "dose_label_fractions": [0.10, 0.05],
        "dose_extra_lrs": {"transformer": [1e-5, 3e-3, 1e-2], "gru": [1e-2]},
        # R1 BrainBERT readout capacity
        "bb_k": 64,                        # principal components per contact kept on disk
        "bb_report_k": [9, 16, 32, 64],    # variance kept is reported at these
        "bb_ridge_k": [16, 32, 64],
        "bb_gru_k": 64,
        "bb_features_dir": "derived_brainbert_k64",
        "bb_workers": 3,                   # wide inputs: fewer workers share the GPU
        "run_self": True,
        "run_edge": True,
        "run_brainbert": True,
        "save_predictions": True,
    },
    # ---- Phase 5 (docs/PHASE5_DESIGN.md): identification of spoken segments, CPU only ----
    "phase5": {
        "phases": ["phase1", "phase2", "phase3", "phase4", "phase4b"],
        # name -> conditions chosen together on validation MSE (all learning rates of all phases)
        "pools": {"sup": ["sup"], "ssl": ["ssl"], "gru": ["gru"], "gru_ssl": ["gru_ssl"],
                  "lpft_ssl": ["lpft_ssl"], "lpft_gru_ssl": ["lpft_gru_ssl"],
                  "ssl_self": ["ssl_self"], "gru_ssl_self": ["gru_ssl_self"], "ridge": ["ridge"],
                  "bb_best_ridge": ["bb_ridge", "bb16_ridge", "bb32_ridge", "bb64_ridge"],
                  "bb_best_gru": ["bb_gru", "bb64_gru"]},
        "probe_ssl": True,                 # ridge on frozen Phase 1 SSL features, recomputed (CPU)
        "seeds": [1, 2, 3, 4, 5],
        "fractions": [1.0, 0.5, 0.25, 0.10, 0.05],
        # frames of 10 ms: gaps < merge_gap closed, segments < min_len dropped, resampled to k frames
        "variants": {"main": {"merge_gap": 15, "min_len": 20, "k": 32},
                     "k16": {"merge_gap": 15, "min_len": 20, "k": 16},
                     "k64": {"merge_gap": 15, "min_len": 20, "k": 64},
                     "strict": {"merge_gap": 10, "min_len": 30, "k": 32}},
        "n_perm": 1000,
        "results_dir": "ident5",
    },
    "eval": {
        "conditions": ["sup", "ssl", "gru", "gru_ssl", "pooled", "loo_adapt", "ridge", "probe_ssl"],
        "phases": ["phase1", "phase2", "phase3"],
        "seeds": [1, 2, 3],
        "fractions": [1.0, 0.5, 0.25, 0.10, 0.05],
        "n_shifts": 200,
        "min_shift_frames": 100,         # circular shifts of >= 1 s
        "griffin_lim_iters": 32,
        "stoi": True,
        "stoi_seeds": [1],               # resynthesis is the slow step: STOI/ESTOI on seed 1 only
        "reference_dir": "phase1",       # where <pid>__reference.npz (timing reference) lives
        "probe_encoder_dir": "phase1",   # Phase 1 SSL encoders for probe_ssl
        "probe_check_dir": "phase2",     # Phase 2 probe records (r_mean cross-check)
        "results_dir": "eval",
    },
    "contamination": {
        "fmin": 70.0,
        "fmax": 500.0,
        "win_sec": 0.2,
        "n_perm": 100,
        "n_perm_inject": 50,
        "inject_db": [-30.0, -20.0],
        "min_shift_sec": 10.0,
        "alpha": 0.05,
        "min_index": 0.01,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key.startswith("_"):
            continue  # comments in JSON files
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, json.load(fh))
    return cfg


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
