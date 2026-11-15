"""Linear reference models operating on frames.

ridge     Ridge regression on lag-stacked z-scored features; alpha chosen on
          validation MSE.
pca_lab   Pilot-study reference: PCA(50) fitted on the labelled frames + OLS.
pca_all   PCA(50) fitted on *all* (unlabelled) training frames + OLS on the
          labelled frames: a linear analogue of "unsupervised pretraining".

Both are computed from the F x F Gram matrix accumulated in float64 chunks, so
memory stays small even for the 20-minute recordings (no n x F SVD).
"""

from __future__ import annotations

import numpy as np

CHUNK = 8192


def _mean(x: np.ndarray) -> np.ndarray:
    s = np.zeros(x.shape[1])
    for i in range(0, len(x), CHUNK):
        s += x[i : i + CHUNK].sum(0, dtype=np.float64)
    return s / len(x)


def _gram(x: np.ndarray, xm: np.ndarray, y: np.ndarray | None = None, ym: np.ndarray | None = None):
    f = x.shape[1]
    g = np.zeros((f, f))
    xy = None if y is None else np.zeros((f, y.shape[1]))
    for i in range(0, len(x), CHUNK):
        c = x[i : i + CHUNK].astype(np.float64) - xm
        g += c.T @ c
        if y is not None:
            xy += c.T @ (y[i : i + CHUNK].astype(np.float64) - ym)
    return g, xy


def _project(x: np.ndarray, xm: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.vstack([(x[i : i + CHUNK].astype(np.float64) - xm) @ basis for i in range(0, len(x), CHUNK)])


def ridge(xtr, ytr, xva, yva, xte, alphas):
    xm, ym = _mean(xtr), np.asarray(ytr, np.float64).mean(0)
    g, xy = _gram(xtr, xm, ytr, ym)
    lam, v = np.linalg.eigh(g)
    lam = np.clip(lam, 0.0, None)
    b = v.T @ xy  # equals S U^T Y of the SVD solution
    zva, zte = _project(xva, xm, v), _project(xte, xm, v)
    best = (np.inf, None, None)
    for a in alphas:
        coef = b / (lam + a)[:, None]
        val = float(np.mean((zva @ coef + ym - yva) ** 2))
        if val < best[0]:
            best = (val, a, coef)
    val, a, coef = best
    return {"pred_test": (zte @ coef + ym).astype(np.float32),
            "pred_val": (zva @ coef + ym).astype(np.float32),
            "best_val_mse": val, "alpha": float(a),
            "alpha_at_grid_edge": bool(a in (min(alphas), max(alphas)))}


def ridge_torch(xtr, ytr, xva, yva, xte, alphas, chunk: int = CHUNK):
    """``ridge`` for wide inputs (Phase 4b: BrainBERT with up to 64 components per contact,
    8,128 features): the same estimator, alpha grid and selection, computed in float64 with
    torch on the tensors' device (the GPU in the workers). The n x F projections of ``ridge``
    are never materialised: for each alpha the coefficients are mapped back to feature space
    (V diag(1/(lam+a)) V^T X^T Y), which is algebraically identical."""
    import torch

    dev = xtr.device
    f64 = torch.float64

    def mean(x):
        s = torch.zeros(x.shape[1], dtype=f64, device=dev)
        for i in range(0, len(x), chunk):
            s += x[i : i + chunk].to(f64).sum(0)
        return s / len(x)

    xm = mean(xtr)
    yt = torch.as_tensor(np.asarray(ytr, np.float64), device=dev)
    ym = yt.mean(0)
    g = torch.zeros((xtr.shape[1], xtr.shape[1]), dtype=f64, device=dev)
    xy = torch.zeros((xtr.shape[1], yt.shape[1]), dtype=f64, device=dev)
    for i in range(0, len(xtr), chunk):
        c = xtr[i : i + chunk].to(f64) - xm
        g += c.T @ c
        xy += c.T @ (yt[i : i + chunk] - ym)
    lam, v = torch.linalg.eigh(g)
    del g
    lam = lam.clamp_min(0.0)
    b = v.T @ xy

    def predict(x, w):
        return torch.cat([(x[i : i + chunk].to(f64) - xm) @ w for i in range(0, len(x), chunk)]) + ym

    yva_t = torch.as_tensor(np.asarray(yva, np.float64), device=dev)
    best = (np.inf, None, None)
    for a in alphas:
        w = v @ (b / (lam + a)[:, None])
        val = float(((predict(xva, w) - yva_t) ** 2).mean())
        if val < best[0]:
            best = (val, a, w)
    val, a, w = best
    return {"pred_test": predict(xte, w).cpu().numpy().astype(np.float32),
            "pred_val": predict(xva, w).cpu().numpy().astype(np.float32),
            "best_val_mse": val, "alpha": float(a),
            "alpha_at_grid_edge": bool(a in (min(alphas), max(alphas)))}


def pca_ols(xfit, xtr, ytr, xva, yva, xte, k: int):
    m = _mean(xfit)
    g, _ = _gram(xfit, m)
    _, v = np.linalg.eigh(g)
    k = int(min(k, v.shape[1], len(xtr) - 2))
    p = v[:, ::-1][:, :k]  # leading principal axes

    def design(x):
        z = _project(x, m, p)
        return np.hstack([z, np.ones((len(z), 1))])

    w, *_ = np.linalg.lstsq(design(xtr), np.asarray(ytr, np.float64), rcond=None)
    pv = design(xva) @ w
    return {"pred_test": (design(xte) @ w).astype(np.float32),
            "pred_val": pv.astype(np.float32),
            "best_val_mse": float(np.mean((pv - yva) ** 2)), "n_components": k}
