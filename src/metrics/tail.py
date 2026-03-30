"""Right-tail metrics: CVaR (conditional value at risk) and top-k statistics."""

from __future__ import annotations

import torch


def right_cvar(values: torch.Tensor, alpha: float = 0.01) -> float:
    """Right CVaR: mean of the top-(alpha) fraction of values.

    This is the primary tail metric — measures how good the BEST molecules are.
    alpha=0.01 → top 1%, alpha=0.10 → top 10%.

    Args:
        values: 1-D float tensor of per-molecule scores (higher = better).
        alpha:  tail fraction in (0, 1].

    Returns:
        Scalar float — mean score of the top-alpha fraction.
    """
    assert values.ndim == 1, f"Expected 1-D tensor, got shape {values.shape}"
    assert 0 < alpha <= 1, f"alpha must be in (0, 1], got {alpha}"
    k = max(1, int(alpha * values.numel()))
    return values.topk(k).values.mean().item()


def top_k_mean(values: torch.Tensor, k: int = 100) -> float:
    """Mean of the top-k values.

    Args:
        values: 1-D float tensor.
        k:      number of top values. Clipped to len(values).
    """
    assert values.ndim == 1, f"Expected 1-D tensor, got shape {values.shape}"
    k = min(k, values.numel())
    return values.topk(k).values.mean().item()


def tail_improvement_ratio(
    baseline_values: torch.Tensor,
    method_values: torch.Tensor,
    alpha: float = 0.01,
) -> float:
    """Relative improvement in right-CVaR of method vs baseline.

    Returns (CVaR_method - CVaR_baseline) / |CVaR_baseline|.
    Positive means the method produces better tail molecules.
    """
    cvar_base = right_cvar(baseline_values, alpha)
    cvar_meth = right_cvar(method_values, alpha)
    if abs(cvar_base) < 1e-12:
        return float("inf") if cvar_meth > 0 else 0.0
    return (cvar_meth - cvar_base) / abs(cvar_base)
