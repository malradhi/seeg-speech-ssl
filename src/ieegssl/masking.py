"""Masking operators (vectorised).

random    independent Bernoulli(ratio) over (time, feature) coordinates
time      one contiguous block of round(ratio * T) frames, all features
contact   round(ratio * C) physical contacts, all their lag copies, whole window
lagcons   Bernoulli(ratio) over the underlying (frame, contact) cells; a hidden cell
          is hidden in every lag copy that contains it, so it cannot be read off a
          neighbouring lag (lag-consistent random masking)
"""

from __future__ import annotations

import torch


def random_coordinate_mask(x: torch.Tensor, ratio: float) -> torch.Tensor:
    return torch.rand_like(x) < ratio


def time_block_mask(x: torch.Tensor, ratio: float) -> torch.Tensor:
    b, t, f = x.shape
    block = max(1, int(round(t * ratio)))
    start = torch.randint(0, t - block + 1, (b, 1), device=x.device)
    pos = torch.arange(t, device=x.device)[None, :]
    keep = (pos >= start) & (pos < start + block)
    return keep[:, :, None].expand(b, t, f)


def contact_mask(x: torch.Tensor, contact_id: torch.Tensor, ratio: float) -> torch.Tensor:
    """Mask whole physical contacts: all lag-expanded features of a selected
    contact are hidden across the entire window."""
    b, t, f = x.shape
    n_contacts = int(contact_id.max().item()) + 1
    k = max(1, int(round(n_contacts * ratio)))
    scores = torch.rand(b, n_contacts, device=x.device)
    chosen = torch.zeros(b, n_contacts, dtype=torch.bool, device=x.device)
    chosen.scatter_(1, scores.topk(k, dim=1).indices, True)
    return chosen[:, contact_id][:, None, :].expand(b, t, f)


def lag_consistent_mask(x: torch.Tensor, contact_id: torch.Tensor, ratio: float, step: int) -> torch.Tensor:
    """Feature f = lag * C + contact holds frame (t + offset[lag]) of that contact."""
    b, t, f = x.shape
    c = int(contact_id.max().item()) + 1
    n_lags = f // c
    half = (n_lags // 2) * step
    offsets = (torch.arange(n_lags, device=x.device) - n_lags // 2) * step
    base = torch.rand(b, t + 2 * half, c, device=x.device) < ratio          # underlying cells
    frame = torch.arange(t, device=x.device)[:, None] + half + offsets[None, :]  # (t, n_lags)
    return base[:, frame, :].reshape(b, t, n_lags * c)


def make_mask(kind: str, x: torch.Tensor, ratio: float, contact_id: torch.Tensor | None = None,
              step: int = 5) -> torch.Tensor:
    if kind == "random":
        return random_coordinate_mask(x, ratio)
    if kind == "time":
        return time_block_mask(x, ratio)
    if kind == "contact":
        return contact_mask(x, contact_id, ratio)
    if kind == "lagcons":
        return lag_consistent_mask(x, contact_id, ratio, step)
    raise ValueError(f"Unknown mask kind {kind!r}")
