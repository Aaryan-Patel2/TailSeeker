"""ERM baseline loss — standard DDPM denoising MSE (tilt = 0)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseLoss, LossOutput


class DDPMSimpleLoss(BaseLoss):
    """Standard denoising MSE loss — tilt=0 ERM baseline.

    L = (1/B) Σ_i ||ε_θ(x_t^i, t^i) - ε^i||²
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> LossOutput:
        """Compute per-sample MSE and aggregate as a simple mean.

        Args:
            pred:   predicted noise ε_θ, shape (B, ...).
            target: ground-truth noise ε,  shape (B, ...).
        """
        assert pred.shape == target.shape, (
            f"Shape mismatch: pred={pred.shape}, target={target.shape}"
        )
        B = pred.shape[0]
        # per-sample mean over all non-batch dims
        per_sample = F.mse_loss(pred, target, reduction="none").view(B, -1).mean(dim=1)
        assert per_sample.shape == (B,), f"Expected ({B},), got {per_sample.shape}"

        total = per_sample.mean()

        return LossOutput(
            total_loss=total,
            per_sample_loss=per_sample,
            loss_components={"mse": total},
            diagnostics={"tilt": 0.0},
        )

    def log_keys(self) -> list[str]:
        return ["Train/loss", "Train/loss_mse", "Val/loss"]
