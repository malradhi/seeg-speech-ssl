"""Fast synthetic tests of the scientific invariants (no dataset needed).

    python -m pytest -q tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ieegssl import baselines as B  # noqa: E402
from ieegssl import metrics as M  # noqa: E402
from ieegssl import train as T  # noqa: E402
from ieegssl.config import load_config  # noqa: E402
from ieegssl.data import label_subset, stack_lags  # noqa: E402
from ieegssl.masking import contact_mask, time_block_mask  # noqa: E402
from ieegssl.osf import parse_listing  # noqa: E402
from ieegssl.preprocessing import clean_labels, partition_bounds, trial_runs  # noqa: E402

CFG = load_config()


def test_trial_runs_and_snapping():
    lab = np.array(([""] * 10 + ["ja\r"] * 20) * 10 + [""] * 10)
    runs = trial_runs(clean_labels(lab))
    assert len(runs) == 10 and runs[0] == (10, 30, "ja")
    (b1, b2), snapped = partition_bounds(len(lab), runs, 0.7, 0.15)
    assert snapped and clean_labels(lab)[b1] == "" and clean_labels(lab)[b2] == ""
    assert 0 < b1 < b2 < len(lab)


def test_label_subsets_nested_and_deterministic():
    a = label_subset(400, 0.25, 3, "P07")
    b = label_subset(400, 0.10, 3, "P07")
    assert set(b) <= set(a)
    assert np.array_equal(a, label_subset(400, 0.25, 3, "P07"))
    assert not np.array_equal(a, label_subset(400, 0.25, 4, "P07"))
    assert len(label_subset(400, 1.0, 1, "P01")) == 400


def test_lag_stacking_layout():
    hg = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
    x = stack_lags(hg, np.array([50]), model_order=4, step=5)
    assert x.shape == (1, 27)
    assert np.allclose(x[0, :3], hg[30]) and np.allclose(x[0, 12:15], hg[50]) and np.allclose(x[0, -3:], hg[70])


def test_contact_mask_hides_all_lags_of_a_contact():
    cid = torch.as_tensor(np.tile(np.arange(6), 9))
    m = contact_mask(torch.zeros(4, 50, 54), cid, 0.5)
    for b in range(4):
        hidden = {int(c) for c in cid[m[b, 0]]}
        assert len(hidden) == 3
        assert bool(m[b][:, torch.isin(cid, torch.tensor(sorted(hidden)))].all())


def test_time_block_mask_is_contiguous():
    m = time_block_mask(torch.zeros(5, 50, 7), 0.5)
    for b in range(5):
        t = torch.nonzero(m[b, :, 0]).flatten()
        assert len(t) == 25 and int(t[-1] - t[0]) == 24


def test_circular_null_matches_bruteforce():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((300, 4))
    p = 0.5 * y + rng.standard_normal((300, 4))
    res = M.circular_shift_null(y, p, min_shift=20)
    brute = [M.mean_band_r(y, np.roll(p, -k, axis=0)) for k in range(20, 281)]
    assert abs(res["chance_mean"] - np.mean(brute)) < 1e-9
    assert abs(M.mean_band_r(y, p) - np.mean([np.corrcoef(y[:, i], p[:, i])[0, 1] for i in range(4)])) < 1e-12


def test_ridge_matches_closed_form():
    rng = np.random.default_rng(1)
    x, w = rng.standard_normal((200, 10)), rng.standard_normal((10, 3))
    y = x @ w + 0.1 * rng.standard_normal((200, 3))
    res = B.ridge(x, y, x, y, x, [5.0])
    xc, yc = x - x.mean(0), y - y.mean(0)
    wc = np.linalg.solve(xc.T @ xc + 5.0 * np.eye(10), xc.T @ yc)
    assert np.allclose(res["pred_test"], xc @ wc + y.mean(0), atol=1e-4)


def test_placebo_preserves_marginals():
    x = np.random.default_rng(2).standard_normal((20, 50, 6)).astype(np.float32)
    xp = T.column_permute(x, 0)
    assert np.allclose(np.sort(x.reshape(-1, 6), 0), np.sort(xp.reshape(-1, 6), 0))
    assert not np.allclose(x, xp)


def test_matched_head_initialisation_and_short_training():
    cfg = load_config()
    cfg["finetune"].update(max_steps=10, eval_every=5, patience=2)
    cfg["pretrain"].update(steps=5)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(12, 50, 18, generator=g)
    y = torch.randn(12, 50, 23, generator=g)
    cid = torch.as_tensor(np.tile(np.arange(2), 9))
    enc, info = T.pretrain_encoder(x, cid, cfg, seed=1)
    assert len(info["loss_curve"]) == 1
    a = T.finetune(None, x, y, x[:4], y[:4], x[:4], cfg, seed=1, lr=3e-4)
    b = T.finetune(None, x, y, x[:4], y[:4], x[:4], cfg, seed=1, lr=3e-4)
    assert np.allclose(a["pred_test"], b["pred_test"])  # deterministic given the seed
    c = T.finetune(enc, x, y, x[:4], y[:4], x[:4], cfg, seed=1, lr=3e-4)
    assert c["pred_test"].shape == (4, 50, 23)
    # Same Mel-head initialisation regardless of the encoder.
    from ieegssl.models import EEGToMelModel

    T.set_seed(50001)
    h1 = EEGToMelModel(18, 50, cfg["model"]).mel_head[0].weight.clone()
    T.set_seed(50001)
    m2 = EEGToMelModel(18, 50, cfg["model"])
    m2.encoder.load_state_dict(enc)
    assert torch.equal(h1, m2.mel_head[0].weight)


def test_legacy_protocol_runs():
    cfg = load_config()
    cfg["legacy"].update(ft_epochs=2, pt_epochs=1)
    x, y = torch.randn(8, 50, 9), torch.randn(8, 50, 23)
    enc, _ = T.pretrain_encoder(x, torch.as_tensor(np.arange(9) % 1), cfg, seed=2, protocol="legacy")
    out = T.finetune(enc, x, y, x[:2], y[:2], x[:2], cfg, seed=2, protocol="legacy")
    assert out["lr"] == cfg["legacy"]["ft_lr"] and out["steps_run"] == 2


def test_osf_listing_parser():
    payload = {
        "data": [
            {"attributes": {"kind": "file", "name": "P01_audio.npy", "size": 10,
                            "extra": {"hashes": {"md5": "abc", "sha256": None}}},
             "links": {"download": "https://osf.io/download/x/"}},
            {"attributes": {"kind": "folder", "name": "sub"},
             "relationships": {"files": {"links": {"related": {"href": "https://api.osf.io/v2/f/"}}}}},
        ],
        "links": {"next": "https://api.osf.io/v2/next"},
    }
    files, folders, nxt = parse_listing(payload)
    assert files[0]["name"] == "P01_audio.npy" and files[0]["md5"] == "abc"
    assert folders == ["https://api.osf.io/v2/f/"] and nxt == "https://api.osf.io/v2/next"


def test_mcd_zero_for_identical():
    y = np.random.default_rng(3).standard_normal((40, 23))
    assert M.mcd(y, y) == 0.0
    assert M.mcd(y, y + 1.0) < 1e-9  # a constant offset only changes c0, which is excluded


def test_channel_selection(tmp_path):
    from ieegssl.preprocessing import select_channels

    csv_path = tmp_path / "P99_electrode_locations.csv"
    csv_path.write_text("electrode_name_1,electrode_name_2,type\nLA1,LA1,depth\nLA2,LA2,depth\n"
                        + "".join(f"RB{i},RB{i},depth\n" for i in range(1, 11)))
    chans = ["LA1", "LA2"] + [f"RB{i}" for i in range(1, 11)] + ["EKG1", "C99"]
    keep, dropped = select_channels("P99", chans, True, {"locations": csv_path},
                                    {"channel_policy": "localised"})
    assert len(keep) == 12 and dropped == ["EKG1", "C99"]
    keep, dropped = select_channels("P99", chans, True, {}, {"channel_policy": "all",
                                                              "exclude_channel_regex": "^EKG"})
    assert len(keep) == 13 and dropped == ["EKG1"]


def test_lag_consistent_mask_hides_every_copy():
    from ieegssl.masking import lag_consistent_mask

    c, n_lags, step, t = 4, 9, 5, 50
    cid = torch.as_tensor(np.tile(np.arange(c), n_lags))
    torch.manual_seed(0)
    m = lag_consistent_mask(torch.zeros(3, t, c * n_lags), cid, 0.5, step)
    offsets = (np.arange(n_lags) - n_lags // 2) * step
    for b in range(3):
        for ti in range(t):
            for li in range(n_lags):
                frame = ti + offsets[li]
                for lj in range(n_lags):  # every other copy of the same (frame, contact)
                    tj = frame - offsets[lj]
                    if 0 <= tj < t:
                        assert bool(m[b, ti, li * c]) == bool(m[b, tj, lj * c])
    assert 0.4 < m.float().mean().item() < 0.6


def test_multi_participant_export_and_frozen_adaptation():
    from ieegssl.models import TransformerBackbone

    cfg = load_config()
    cfg["pretrain"].update(steps=4)
    data = {"P01": torch.randn(10, 50, 18), "P02": torch.randn(12, 50, 27)}
    model, info = T.pretrain_multi(data, ["P01", "P02"], cfg, seed=1, steps=6)
    assert info["steps"] == 6 and len(info["loss_curve"]) == 1
    enc = TransformerBackbone(27, 50, cfg["model"])
    enc.load_state_dict(model.export_backbone("P02"), strict=True)
    body_before = {k: v.clone() for k, v in model.body.state_dict().items()}
    T.add_new_participant(model, "P03", 36, seed=1)
    model, _ = T.pretrain_multi({"P03": torch.randn(8, 50, 36)}, ["P03"], cfg, seed=1, steps=5,
                                model=model, train_only="P03")
    assert all(torch.equal(body_before[k], v) for k, v in model.body.state_dict().items())
    TransformerBackbone(36, 50, cfg["model"]).load_state_dict(model.export_backbone("P03"), strict=True)


def test_gru_decoder_and_probe_features():
    cfg = load_config()
    cfg["finetune"].update(max_steps=6, eval_every=3, patience=2)
    x, y = torch.randn(6, 50, 18), torch.randn(6, 50, 23)
    out = T.finetune(None, x, y, x[:2], y[:2], x[:2], cfg, seed=1, lr=1e-3, arch="gru")
    assert out["pred_test"].shape == (2, 50, 23)
    feats = T.encode(T.random_backbone_state(18, 50, cfg, 1), x, cfg)
    assert feats.shape == (6, 50, cfg["model"]["d_model"])


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------


def test_gru_backbone_loads_and_head_init_matches():
    import torch

    from ieegssl import train as T
    from ieegssl.config import load_config
    from ieegssl.models import GRUDecoder

    cfg = load_config()
    f = 36
    state = T.random_gru_state(f, cfg, seed=3)
    T.set_seed(50000 + 1)
    a = GRUDecoder(f, cfg["gru"], cfg["model"]["head_hidden"])
    T.set_seed(50000 + 1)
    b = GRUDecoder(f, cfg["gru"], cfg["model"]["head_hidden"])
    b.load_backbone(state)
    for k, v in a.state_dict().items():
        if k.startswith("head."):
            assert torch.equal(v, b.state_dict()[k])  # same seeded head in every GRU condition
    for k, v in state.items():
        assert torch.equal(v, b.state_dict()[k])
    x = torch.randn(2, 50, f)
    feats = T.encode(state, x, cfg, arch="gru")
    assert feats.shape == (2, 50, 2 * cfg["gru"]["hidden"])


def test_gru_pretraining_runs_and_reduces_loss():
    import torch

    from ieegssl import train as T
    from ieegssl.config import load_config

    cfg = load_config()
    cfg["pretrain"]["steps"] = 250
    x = torch.randn(40, 50, 18)
    x[:, :, 9:] = x[:, :, :9]  # redundant copies: learnable structure
    cid = torch.arange(18) % 9
    state, info = T.pretrain_encoder(x, cid, cfg, seed=1, arch="gru")
    assert info["arch"] == "gru" and set(state) == set(T.random_gru_state(18, cfg, 1))
    assert info["loss_curve"][-1] < info["loss_curve"][0]


def test_with_model_large():
    from ieegssl import train as T
    from ieegssl.config import load_config

    cfg = load_config()
    big = T.with_model(cfg, "model_large")
    assert big["model"]["d_model"] == 256 and cfg["model"]["d_model"] == 128


def test_speech_metrics_identity_and_noise():
    from ieegssl.evaluation import speech_metrics

    rng = np.random.default_rng(0)
    y = rng.normal(size=(3000, 23)) + rng.normal(size=(3000, 1))
    v = np.zeros(3000, bool)
    v[::3] = True
    same = speech_metrics(y, y, v, n_shifts=50, min_shift=100, seed=1)
    assert same["r_speech"] > 0.999 and same["r_shape"] > 0.999 and same["r_loud"] > 0.999
    assert same["r_speech_p"] < 0.05
    noise = speech_metrics(y, rng.normal(size=y.shape), v, n_shifts=50, min_shift=100, seed=1)
    assert abs(noise["r_speech"]) < 0.1 and noise["r_speech_p"] > 0.01


def test_resynthesis_round_trip():
    from ieegssl.config import load_config
    from ieegssl.evaluation import Resynthesizer, _stoi
    from ieegssl.preprocessing import extract_log_mel

    cfg = load_config()
    sr = cfg["preprocess"]["audio_target_sr"]
    t = np.arange(sr * 3) / sr
    audio = np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t)) + 0.3 * np.sin(2 * np.pi * 1300 * t)
    logmel = extract_log_mel(audio, 1.0, cfg)
    rs = Resynthesizer(cfg)
    wav = rs.synth(logmel, seed=0)
    back = extract_log_mel(wav * np.abs(audio).max(), 1.0, cfg)
    n = min(len(back), len(logmel))
    r = np.corrcoef(logmel[:n].ravel(), back[:n].ravel())[0, 1]
    assert r > 0.9
    s = _stoi(wav, wav, sr)
    if s:
        assert s["stoi"] > 0.99


# ---------------------------------------------------------------------------
# Phase 4
# ---------------------------------------------------------------------------


def test_brainbert_frame_centres_match_scipy_stft():
    from scipy import signal as sg

    from ieegssl.brainbert import STFTFrontEnd

    fe = STFTFrontEnd()
    n = 10250
    f, t, _ = sg.stft(np.zeros(n), fs=2048, nperseg=400, noverlap=350, boundary="zeros", padded=True)
    assert np.allclose(fe.frame_centres(n) / 2048, t[10:-10])


def test_brainbert_matches_braindecode_if_available():
    import pytest
    import torch

    bd = pytest.importorskip("braindecode.models")
    from ieegssl.brainbert import BrainBERTEncoder

    torch.manual_seed(0)
    kw = dict(hidden_dim=64, ffn_dim=128, n_layers=2, n_heads=4)
    ref = bd.BrainBERT(n_chans=1, n_times=10250, n_outputs=2, sfreq=2048, pool_n_frames=None, **kw).eval()
    mine = BrainBERTEncoder(**kw).eval()
    missing, unexpected = mine.load_state_dict(
        {k: v for k, v in ref.state_dict().items() if not k.startswith("final_layer.")}, strict=False)
    assert not missing and not unexpected
    x = torch.randn(2, 10250) * 30
    with torch.no_grad():
        a = ref.transformer(ref.input_embedding(ref.spectrogram(x[:, None])[:, 0]))
        assert torch.allclose(a, mine(x), atol=1e-6)


def test_brainbert_window_plan_covers_every_frame():
    from ieegssl.brainbert import STFTFrontEnd, window_plan
    from ieegssl.config import load_config

    bc = load_config()["brainbert"]
    rel = STFTFrontEnd().frame_centres(bc["window_samples"])
    n = 2048 * 37 + 123
    centres = np.arange(3700) * 0.01 + 0.025  # frame centres across the whole 37.06-s signal
    plan = window_plan(n, centres, 2048, rel, bc)
    assert plan["k"].min() == 0 and plan["k"].max() == plan["k_win"] - 1
    pos = centres * 2048 - (plan["k"] * plan["H"] - plan["left"])
    assert (pos >= rel[0]).all() and (pos <= rel[-1]).all()  # always inside the window's frames


def test_laplacian_shafts():
    from ieegssl.brainbert import laplacian

    x = np.array([[1.0, 2.0, 4.0, 10.0, 7.0]])
    out = laplacian(x, ["A1", "A2", "A3", "B1", "EKG1"], exclude="^(EKG)")
    assert np.allclose(out[0], [1 - 2, 2 - 2.5, 4 - 2, 10, 7])


def test_lpft_respects_budget_and_reports_probe_stage():
    import torch

    from ieegssl import train as T
    from ieegssl.config import load_config

    cfg = load_config()
    cfg["finetune"].update(max_steps=80, eval_every=10, patience=3)
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(30, 50, 18, generator=g), torch.randn(30, 50, 23, generator=g)
    for arch in ("transformer", "gru"):
        enc = T.random_gru_state(18, cfg, 1) if arch == "gru" else None
        if arch == "transformer":
            from ieegssl.models import TransformerBackbone
            enc = TransformerBackbone(18, 50, cfg["model"]).state_dict()
        r = T.finetune(enc, x[:20], y[:20], x[20:25], y[20:25], x[25:], cfg, 1, 1e-3, arch=arch,
                       lp={"lr": 1e-3, "max_steps": 30, "patience": 2})
        assert 0 < r["lp_steps_run"] <= 30 and r["steps_run"] <= 80
        assert r["best_val_mse"] <= r["lp_best_val_mse"] + 1e-12


def test_brainbert_loader_ignores_pretraining_head(tmp_path):
    import json

    import torch

    from ieegssl.brainbert import BrainBERTEncoder, load_pretrained

    kw = dict(hidden_dim=32, ffn_dim=64, n_layers=1, n_heads=4)
    state = BrainBERTEncoder(**kw).state_dict()
    state["final_layer.weight"] = torch.zeros(2, 32)
    state["spec_prediction_head.output.weight"] = torch.zeros(40, 32)  # as in the released checkpoint
    torch.save(state, tmp_path / "pytorch_model.bin")
    (tmp_path / "config.json").write_text(json.dumps({**kw, "stft_zscore_before_clip": True}))
    enc, info = load_pretrained(tmp_path)
    assert info["unused_checkpoint_keys"] == ["final_layer.weight", "spec_prediction_head.output.weight"]
    del state["transformer.layers.0.linear1.weight"]  # a missing encoder weight must still fail
    torch.save(state, tmp_path / "pytorch_model.bin")
    try:
        load_pretrained(tmp_path)
        raise AssertionError("missing encoder weight not detected")
    except RuntimeError:
        pass


def test_ridge_torch_matches_ridge():
    rng = np.random.default_rng(3)
    for n, f in ((2000, 30), (300, 500)):  # more frames than features, and fewer
        x = rng.normal(size=(n + 600, f)).astype(np.float32)
        y = (x @ rng.normal(size=(f, 4)) + 3 * rng.normal(size=(n + 600, 4))).astype(np.float32)
        a = B.ridge(x[:n], y[:n], x[n:n + 300], y[n:n + 300], x[n + 300:], CFG["linear"]["ridge_alphas"])
        b = B.ridge_torch(torch.from_numpy(x[:n]), y[:n], torch.from_numpy(x[n:n + 300]), y[n:n + 300],
                          torch.from_numpy(x[n + 300:]), CFG["linear"]["ridge_alphas"], chunk=97)
        assert a["alpha"] == b["alpha"]
        assert abs(a["best_val_mse"] - b["best_val_mse"]) < 1e-9
        assert np.allclose(a["pred_test"], b["pred_test"], atol=1e-5)


def test_leading_components_and_nested_pca():
    from ieegssl.brainbert import pca_fit
    from ieegssl.runner4b import leading_components

    x = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4 * 6)  # 4 contacts x 6 comps
    got = leading_components(x, 4, 6, 2)
    cols = [c * 6 + j for c in range(4) for j in range(2)]
    assert torch.equal(got, x[..., cols])
    e = np.random.default_rng(0).normal(size=(500, 20)) @ np.diag(np.linspace(3, 0.1, 20))
    _, c8, ev8, at = pca_fit(e, 8, (2, 8))
    _, c2, ev2 = pca_fit(e, 2)
    assert np.allclose(c8[:, :2], c2) and abs(at[2] - ev2) < 1e-12 and abs(at[8] - ev8) < 1e-12


def test_phase4b_job_list():
    from collections import Counter

    from ieegssl.runner4b import build_jobs4b

    hg, bb9, bbk = build_jobs4b("P01", CFG)
    n = Counter(j[1] for j in hg + bb9 + bbk)
    assert len(hg) + len(bb9) + len(bbk) == 737
    assert n["ssl_self"] == n["gru_ssl_self"] == 15 + 3 * 5 * 7
    assert all(j[4] == 1e-2 for j in bb9) and n["bb_gru"] == 25 and n["bb64_gru"] == 175
    # every pretraining set is the labelled set of the same budget and seed
    pre = [j for j in hg if j[0] == "pretrain4b"]
    assert {(j[2], j[3]) for j in pre} == {(f, s) for f in (0.25, 0.10, 0.05) for s in range(1, 6)}


def test_identification_scores():
    from ieegssl.identification import features, identify, pairwise_accuracy, segments, similarity, top1_accuracy

    v = np.zeros(200, bool)
    v[10:40] = True
    v[45:80] = True      # gap of 5 frames -> merged with the previous run
    v[120:130] = True    # 10 frames -> dropped (min_len 20)
    v[150:190] = True
    assert segments(v, merge_gap=15, min_len=20) == [(10, 80), (150, 190)]
    eye = np.eye(6)
    assert pairwise_accuracy(eye) == 1.0 and top1_accuracy(eye) == 1.0
    assert pairwise_accuracy(np.ones((6, 6))) == 0.5 and abs(top1_accuracy(np.ones((6, 6))) - 1 / 6) < 1e-12
    res = identify(eye, 500, 1)
    assert res["pairwise_perm_p"] < 0.01
    rng = np.random.default_rng(0)
    y = rng.normal(size=(400, 23))
    segs = [(i, i + 30) for i in range(0, 390, 40)]
    t = features(y, segs, 16, "full")
    assert abs(pairwise_accuracy(similarity(t, t)) - 1.0) < 1e-12
    acc = [pairwise_accuracy(similarity(features(rng.normal(size=(400, 23)), segs, 16, "full"), t)) for _ in range(200)]
    assert abs(np.mean(acc) - 0.5) < 0.02  # chance is 0.5
    assert features(y, segs, 16, "loudness").shape == (len(segs), 16)
    assert features(y, segs, 16, "shape").shape == (len(segs), 16 * 23)


def test_word_level_identification():
    from ieegssl.identification import pairwise_word_accuracy, segment_labels

    s = np.eye(4)
    s[0, 1] = s[1, 0] = 1.0           # segments 0 and 1 are the same word and indistinguishable
    labels = ["a", "a", "b", "c"]
    assert pairwise_word_accuracy(s, labels) == 1.0   # same-word pairs are not scored
    from ieegssl.identification import pairwise_accuracy
    assert pairwise_accuracy(s) < 1.0                 # but they are errors/ties at segment level
    t = np.array([-1, 0, 0, 0, -1, 1, 1, 2, 2, 2])
    assert segment_labels(t, ["x", "y", "z"], [(0, 4), (5, 10), (0, 1)]) == ["x", "z", None]
