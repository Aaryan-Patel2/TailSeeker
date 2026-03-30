"""BaseLoss ABC and LossOutput dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


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
