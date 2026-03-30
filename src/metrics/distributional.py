"""Distributional similarity metrics: Wasserstein-p and MMD."""

from __future__ import annotations

import torch


def wasserstein_1d(p: torch.Tensor, q: torch.Tensor) -> float:
    """Wasserstein-1 distance between 1-D empirical distributions.

    Computed via the sorted-difference formula (exact for 1-D).

    Args:
        p: samples from distribution P, shape (N,).
        q: samples from distribution Q, shape (M,).
    """
    assert p.ndim == 1 and q.ndim == 1, (
        f"Expected 1-D tensors, got p={p.shape}, q={q.shape}"
    )
    p_sorted = p.sort().values
    q_sorted = q.sort().values
    # interpolate the shorter one to match length
    if p_sorted.numel() != q_sorted.numel():
        n = max(p_sorted.numel(), q_sorted.numel())
        p_sorted = _resample(p_sorted, n)
        q_sorted = _resample(q_sorted, n)
    return (p_sorted - q_sorted).abs().mean().item()


def _resample(x: torch.Tensor, n: int) -> torch.Tensor:
    """Linearly interpolate 1-D tensor x to length n."""
    idx = torch.linspace(0, x.numel() - 1, n)
    lo = idx.long().clamp(0, x.numel() - 2)
    hi = (lo + 1).clamp(0, x.numel() - 1)
    frac = idx - lo.float()
    return x[lo] * (1 - frac) + x[hi] * frac


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF kernel matrix K(x, y), shape (N, M)."""
    diff = x[:, None] - y[None, :]  # (N, M, D)
    return torch.exp(-diff.pow(2).sum(-1) / (2 * sigma**2))


def mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0,
) -> float:
    """Unbiased MMD² estimate with RBF kernel.

    Args:
        x: samples from distribution P, shape (N, D).
        y: samples from distribution Q, shape (M, D).
        sigma: RBF bandwidth.
    """
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    assert x.shape[1] == y.shape[1], (
        f"Feature dim mismatch: x={x.shape}, y={y.shape}"
    )
    kxx = _rbf_kernel(x, x, sigma)
    kyy = _rbf_kernel(y, y, sigma)
    kxy = _rbf_kernel(x, y, sigma)
    N, M = x.shape[0], y.shape[0]
    # unbiased: zero out diagonal of kxx and kyy
    kxx.fill_diagonal_(0)
    kyy.fill_diagonal_(0)
    return (
        kxx.sum() / (N * (N - 1))
        + kyy.sum() / (M * (M - 1))
        - 2 * kxy.mean()
    ).item()
