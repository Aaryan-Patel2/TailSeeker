"""BaseLoss ABC, LossOutput dataclass, and shared per-molecule MSE helper."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _per_molecule_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-molecule MSE, correctly masked for variable-length molecules.

    EDM operates on padded atom tensors (B, N_atoms, n_feat).  A naive
    `.view(B, -1).mean()` divides by *total* (atom × feat) slots, diluting
    the loss for shorter molecules and biasing TERM's log-sum-exp toward
    larger molecules.  This helper divides by the count of *valid* (atom,
    feat) pairs per molecule, making per-molecule losses size-invariant.

    Args:
        pred:      (B, ...) predicted noise.
        target:    (B, ...) ground-truth noise.
        node_mask: (B, N_atoms[, 1]) float mask; 1 = valid atom, 0 = padding.
                   When None, falls back to the original naive mean — no
                   change in behaviour for non-EDM callers.

    Returns:
        per_sample: (B,) scalar per-molecule loss.
    """
    B = pred.shape[0]
    raw = F.mse_loss(pred, target, reduction="none")   # (B, ...)
    if node_mask is None:
        return raw.view(B, -1).mean(dim=1)
    # Broadcast mask over the feature dimension so it covers every (atom, feat) slot.
    mask = node_mask
    while mask.dim() < raw.dim():
        mask = mask.unsqueeze(-1)                       # (B, N_atoms) → (B, N_atoms, 1)
    mask = mask.expand_as(raw)                          # (B, N_atoms, n_feat)
    n_valid = mask.reshape(B, -1).sum(dim=1).clamp(min=1.0)   # (B,)
    return (raw * mask).reshape(B, -1).sum(dim=1) / n_valid


def term_aggregate(per_sample: torch.Tensor, tilt: float) -> torch.Tensor:
    """TERM aggregation of pre-computed per-sample losses.

    This is the **exact drop-in replacement** for ``nll.mean(0)`` in EDM's
    ``qm9/losses.py`` (line 31).  No pred/target recomputation — operates
    directly on the ``(B,)`` per-molecule NLL vector the model already returns.

    EDM injection::

        # Original (ehoogeboom/e3_diffusion_for_molecules, qm9/losses.py:31):
        nll = nll.mean(0)

        # Replaced with V1 tilt:
        nll = term_aggregate(nll, tilt=args.tilt)

    Args:
        per_sample: ``(B,)`` per-molecule losses from model forward pass.
        tilt:       TERM tilt parameter ``t``.  ``0.0`` recovers ``nll.mean(0)``
                    exactly (ERM baseline, sanity-checkable against EDM).

    Returns:
        Scalar loss for ``.backward()``.
    """
    assert per_sample.ndim == 1, (
        f"term_aggregate expects (B,) tensor, got shape {per_sample.shape}. "
        "Did you forget to aggregate atom-level losses to molecule-level first?"
    )
    if tilt == 0.0:
        return per_sample.mean()
    B = per_sample.shape[0]
    return (torch.logsumexp(tilt * per_sample, dim=0) - math.log(B)) / tilt


@dataclass
class LossOutput:
    """Structured output from any loss forward pass."""

    total_loss: torch.Tensor
    """Scalar loss used for the backward pass."""

    per_sample_loss: torch.Tensor
    """Per-sample losses, shape [B]. Used by tilted aggregation."""

    loss_components: dict[str, torch.Tensor] = field(default_factory=dict)
    """Named sub-losses for W&B logging (e.g. {'mse': ..., 'kl': ...})."""

    weights: Optional[torch.Tensor] = None
    """Optional per-sample importance weights, shape [B]."""

    diagnostics: dict[str, float] = field(default_factory=dict)
    """Scalar diagnostic values logged but not backpropagated."""


class BaseLoss(ABC, nn.Module):
    """Abstract base for all loss functions in this project.

    Subclasses implement `forward()` and `log_keys()`.
    Only Loss and Model hierarchies may use ABCs — all other code is concrete.
    """

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> LossOutput:
        """Compute loss.

        Args:
            pred:   model predictions, shape (B, ...).
            target: ground-truth targets, shape (B, ...).

        Returns:
            LossOutput with total_loss, per_sample_loss, etc.
        """

    @abstractmethod
    def log_keys(self) -> list[str]:
        """Return the list of W&B keys this loss will log.

        Called at run init to pre-register metric columns.
        """
