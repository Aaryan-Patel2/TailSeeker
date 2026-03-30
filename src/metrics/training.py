"""Training diagnostic metrics: bias-variance decomposition."""

from __future__ import annotations

import torch


def bias_variance_decomposition(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Decompose MSE into bias² + variance.

    Args:
        preds:   repeated model predictions, shape (K, B, ...) where K = num runs.
        targets: ground-truth targets, shape (B, ...) or (K, B, ...).

    Returns:
        Dict with keys: mse, bias_sq, variance, noise (irreducible).
    """
    assert preds.ndim >= 2, f"Expected ≥2-D preds (K, B, ...), got {preds.shape}"
    mean_pred = preds.mean(dim=0)  # (B, ...)

    if targets.ndim == preds.ndim:
        target = targets[0]  # assume all same
    else:
        target = targets
    assert mean_pred.shape == target.shape, (
        f"Shape mismatch: mean_pred={mean_pred.shape}, target={target.shape}"
    )

    bias_sq = (mean_pred - target).pow(2).mean().item()
    variance = preds.var(dim=0).mean().item()
    mse = (preds - target.unsqueeze(0)).pow(2).mean().item()

    return {
        "mse": mse,
        "bias_sq": bias_sq,
        "variance": variance,
        "noise": max(0.0, mse - bias_sq - variance),
    }
