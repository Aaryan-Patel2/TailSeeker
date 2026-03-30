"""BaseModel ABC and ModelOutput dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """Structured output from any score-network forward pass."""

    pred_noise: Optional[torch.Tensor] = None
    """Predicted noise ε_θ(x_t, t), shape (B, C, ...)."""

    pred_x0: Optional[torch.Tensor] = None
    """Predicted clean x_0 (if the network predicts x_0 directly)."""

    pred_v: Optional[torch.Tensor] = None
    """Predicted velocity (v-prediction parameterisation)."""

    aux: dict[str, torch.Tensor] = field(default_factory=dict)
    """Hook for TSM intermediates: per-sample norms, attention maps, etc."""


class BaseModel(ABC, nn.Module):
    """Abstract base for all score / diffusion networks.

    Only Loss and Model hierarchies may use ABCs — all other code is concrete.
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> ModelOutput:
        """Denoise *x* at timestep *t*.

        Args:
            x: noisy input tensor, shape (B, ...).
            t: integer timestep indices, shape (B,).

        Returns:
            ModelOutput with at least one of pred_noise / pred_x0 / pred_v set.
        """
