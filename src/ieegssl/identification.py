"""Phase 5: identification of spoken segments from decoded spectrograms (docs/PHASE5_DESIGN.md).

CPU only, on the saved test predictions. For each participant:

1. **Segments.** Speech segments of the test partition from the frame-level speech labels
   (the same speech/silence split as within-speech r): runs of speech frames, gaps shorter
   than ``merge_gap`` frames closed, segments shorter than ``min_len`` frames dropped. In the
   word tasks a segment is about one spoken word; in the sentence task a sentence or phrase.
   The true segment boundaries are used for both the decoded and the true spectrogram
   (identification, not detection).
2. **Representation.** Each band is standardised with the mean and SD of the *training*
   partition's speech frames (true target), so a segment is described by its deviation from
   the participant's average speech spectrum; each segment is resampled to ``K`` frames
   (linear interpolation), which removes duration as a cue.
3. **Similarity.** Pearson r between the decoded segment i and the true segment j over the
   resampled K x 23 matrix (``full``), over the frame means only (``loudness``), or over each
   frame minus its mean (``shape``).
4. **Scores.** Pairwise (2-alternative) accuracy: the share of pairs (i, j != i) in which
   decoded segment i is more similar to its own true segment than to true segment j (ties
   count one half); chance is exactly 0.5. Top-1 accuracy: the share of decoded segments whose
   most similar true segment is their own; chance 1/N. A per-prediction permutation p-value
   re-pairs decoded and true segments at random.
"""

from __future__ import annotations

import json
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import config_hash
from .data import ParticipantData
from .provenance import code_hash, environment

FEATURES = ("full", "loudness", "shape")


# ---------------------------------------------------------------------------
# Segments and similarity
# ---------------------------------------------------------------------------


def segments(v: np.ndarray, merge_gap: int, min_len: int) -> list[tuple[int, int]]:
    """[start, stop) of speech runs, gaps shorter than ``merge_gap`` closed, runs shorter than
    ``min_len`` dropped."""
    v = np.asarray(v, bool).reshape(-1)
    if not v.any():
        return []
    d = np.diff(np.r_[0, v.astype(np.int8), 0])
    runs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    merged = [list(runs[0])]
    for a, b in runs[1:]:
        if a - merged[-1][1] < merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(int(a), int(b)) for a, b in merged if b - a >= min_len]


def resample(x: np.ndarray, k: int) -> np.ndarray:
    """(L, F) -> (k, F) by linear interpolation over the frame index."""
    x = np.asarray(x, np.float64)
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, k)
    return np.stack([np.interp(dst, src, x[:, f]) for f in range(x.shape[1])], axis=1)


def features(x: np.ndarray, segs, k: int, kind: str) -> np.ndarray:
    """(N, D) segment vectors of a standardised (frames, bands) spectrogram."""
    out = []
    for a, b in segs:
        r = resample(x[a:b], k)
        if kind == "loudness":
            out.append(r.mean(1))
        elif kind == "shape":
            out.append((r - r.mean(1, keepdims=True)).ravel())
        else:
            out.append(r.ravel())
    return np.asarray(out)


def similarity(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """S[i, j] = Pearson r between decoded vector i and true vector j (0 for a constant vector)."""
    def z(m):
        m = m - m.mean(1, keepdims=True)
        n = np.sqrt((m * m).sum(1, keepdims=True))
        return np.divide(m, n, out=np.zeros_like(m), where=n > 0)
    return z(p) @ z(t).T


def pairwise_accuracy(s: np.ndarray) -> float:
    n = len(s)
    if n < 2:
        return float("nan")
    own = np.diag(s)[:, None]
    wins = (own > s).sum(1) + 0.5 * ((own == s).sum(1) - 1)
    return float((wins / (n - 1)).mean())


def pairwise_word_accuracy(s: np.ndarray, labels) -> float:
    """Pairwise accuracy over pairs of *different* spoken words only: a decoded segment is
    compared with its own true segment against true segments of other words (another
    instance of the same word is neither a hit nor an error). Chance 0.5."""
    lab = np.asarray(labels, dtype=object)
    own = np.diag(s)
    accs = []
    for i in range(len(s)):
        other = np.flatnonzero((lab != lab[i]) & np.array([x is not None for x in lab]))
        if lab[i] is None or not len(other):
            continue
        row = s[i, other]
        accs.append(((own[i] > row).sum() + 0.5 * (own[i] == row).sum()) / len(other))
    return float(np.mean(accs)) if accs else float("nan")


def segment_labels(trial_index: np.ndarray, trial_labels: list, segs) -> list:
    """The spoken item of each segment: the stimulus label of the trial that covers most of its
    frames (None if no frame lies in a trial)."""
    t = np.asarray(trial_index).reshape(-1)
    out = []
    for a, b in segs:
        ids = t[a:b]
        ids = ids[(ids >= 0) & (ids < len(trial_labels))]
        out.append(str(trial_labels[int(np.bincount(ids).argmax())]) if len(ids) else None)
    return out


def top1_accuracy(s: np.ndarray) -> float:
    if len(s) < 2:
        return float("nan")
    best = s.max(1, keepdims=True)
    hits = np.diag(s) == best[:, 0]
    ties = (s == best).sum(1)
    return float((hits / ties).mean())  # a tie for first place counts 1/(number tied)


def identify(s: np.ndarray, n_perm: int, seed: int) -> dict:
    """Scores of a similarity matrix and a permutation p-value for the pairwise accuracy."""
    acc2, acc1 = pairwise_accuracy(s), top1_accuracy(s)
    out = {"pairwise": acc2, "top1": acc1, "n": int(len(s))}
    if n_perm and len(s) >= 3 and np.isfinite(acc2):
        rng = np.random.default_rng(seed)
        null = np.array([pairwise_accuracy(s[rng.permutation(len(s))]) for _ in range(n_perm)])
        out["pairwise_perm_p"] = float((1 + np.sum(null >= acc2)) / (1 + n_perm))
    return out


# ---------------------------------------------------------------------------
# Run selection (pools of conditions chosen together on validation MSE)
# ---------------------------------------------------------------------------


def _val_mse(rec: dict) -> float:
    v = (rec.get("val") or {}).get("mse")
    return float(v) if v is not None and np.isfinite(v) else np.inf


def select_pooled(pid: str, root: Path, ph: dict) -> dict:
    """{(name, fraction, seed): (record, prediction path)}: for every name, the run with the lowest
    validation MSE among all runs (all phases, all learning rates) of the conditions it pools."""
    pools = {name: set(conds) for name, conds in ph["pools"].items()}
    member = defaultdict(list)
    for name, conds in pools.items():
        for c in conds:
            member[c].append(name)
    groups = defaultdict(list)
    for phase in ph["phases"]:
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
            if c not in member or s not in ph["seeds"] or not any(np.isclose(f, x) for x in ph["fractions"]):
                continue
            pred = preds / f"{rec['run_id']}.npz"
            if pred.exists():
                for name in member[c]:
                    groups[(name, float(f), int(s))].append((rec, pred))
    return {key: min(items, key=lambda it: _val_mse(it[0])) for key, items in groups.items()}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def identify_participant(pid: str, root, cfg: dict, overwrite: bool = False) -> Path:
    from .evaluation import _probe_predictions

    root = Path(root)
    ph = cfg["phase5"]
    out_dir = root / "results" / ph["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}.json"
    if out_path.exists() and not overwrite:
        print(f"{pid}: identification exists -> skip", flush=True)
        return out_path
    t0 = time.time()
    d = ParticipantData(root / "derived" / f"{pid}.npz", cfg)
    nb = d.Yw["test"].shape[-1]
    y = d.Yw["test"].reshape(-1, nb).astype(np.float64)
    v = d.vad["test"].reshape(-1)
    ytr = d.Yw["train"].reshape(-1, nb)[d.vad["train"].reshape(-1)]
    mu, sd = ytr.mean(0), np.maximum(ytr.std(0), 1e-6)
    std = lambda x: (np.asarray(x, np.float64).reshape(-1, nb) - mu) / sd  # noqa: E731

    variants = {}
    for vname, vc in ph["variants"].items():
        segs = segments(v, vc["merge_gap"], vc["min_len"])
        variants[vname] = (segs, vc["k"])
    seg_main = variants["main"][0]
    words = segment_labels(d.Tw["test"], d.meta.get("trial_labels", []), seg_main)
    known = [w for w in words if w is not None]
    pairs = [(i, j) for i in range(len(words)) for j in range(len(words)) if i != j]
    same = sum(1 for i, j in pairs if words[i] is not None and words[i] == words[j])
    truth = {vname: {kind: features(std(y), segs, k, kind) for kind in FEATURES}
             for vname, (segs, k) in variants.items()}

    preds = {}
    for key, (rec, path) in select_pooled(pid, root, ph).items():
        with np.load(path) as z:
            preds[key] = (z["pred_test"], {"run_id": rec["run_id"], "lr": rec.get("lr"),
                                           "source_condition": rec.get("condition")})
    if ph.get("probe_ssl", True):
        pcfg = dict(cfg)
        pcfg["eval"] = dict(cfg["eval"], seeds=ph["seeds"], fractions=ph["fractions"])
        for key, p in _probe_predictions(d, root, pcfg).items():
            preds[key] = (p, {"run_id": f"{pid}__probe_ssl__f{key[1]:.2f}__s{key[2]}", "lr": None,
                              "source_condition": "probe_ssl"})
    preds[("true_spectrogram", None, None)] = (y, {"run_id": f"{pid}__true_spectrogram"})
    preds[("shifted_truth", None, None)] = (np.roll(y, len(y) // 2, axis=0), {"run_id": f"{pid}__shifted_truth"})

    rows = []
    for (cond, frac, seed), (p, info) in sorted(preds.items(), key=lambda kv: str(kv[0])):
        ps = std(p)
        row = {"participant": pid, "task": d.task, "condition": cond, "fraction": frac, "seed": seed, **info}
        for vname, (segs, k) in variants.items():
            for kind in FEATURES:
                s = similarity(features(ps, segs, k, kind), truth[vname][kind])
                n_perm = ph["n_perm"] if (vname == "main") else 0
                res = identify(s, n_perm, _seed_of(pid, cond, frac, seed, vname, kind))
                tag = kind if vname == "main" else f"{vname}_{kind}"
                row[f"{tag}_pairwise"] = res["pairwise"]
                row[f"{tag}_top1"] = res["top1"]
                if "pairwise_perm_p" in res:
                    row[f"{tag}_perm_p"] = res["pairwise_perm_p"]
                if vname == "main" and ph.get("word_level", True):
                    row[f"{tag}_pairwise_word"] = pairwise_word_accuracy(s, words)
            row[f"n_segments_{vname}"] = len(segs)
        rows.append(row)
    lens = np.array([b - a for a, b in seg_main]) * cfg["preprocess"]["shift_sec"]
    stamp = {"code_hash": code_hash(), "config_hash": config_hash(cfg), "env": environment()}
    res = {"participant": pid, "task": d.task, "n_rows": len(rows), "n_segments": len(seg_main),
           "n_segments_labelled": len(known), "n_distinct_words": len(set(known)),
           "same_word_pair_share": same / len(pairs) if pairs else None, "segment_words": words,
           "segment_seconds": {"median": float(np.median(lens)) if len(lens) else None,
                               "min": float(lens.min()) if len(lens) else None,
                               "max": float(lens.max()) if len(lens) else None},
           "test_seconds": len(y) * cfg["preprocess"]["shift_sec"],
           "elapsed_s": round(time.time() - t0, 1), "rows": rows, **stamp}
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=float))
    tmp.replace(out_path)
    print(f"{pid}: identified {len(rows)} predictions over {len(seg_main)} segments in "
          f"{res['elapsed_s'] / 60:.1f} min", flush=True)
    return out_path
