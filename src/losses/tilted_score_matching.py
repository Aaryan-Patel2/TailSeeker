"""Tilted Score Matching loss — TERM-style log-sum-exp tilt.

THIS IS THE CORE RESEARCH CONTRIBUTION.
Do not modify the stub interface — only replace the NotImplementedError body.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import BaseLoss, LossOutput, _per_molecule_mse


class TiltedScoreMatchingLoss(BaseLoss):
    """L_tilt = (1/t) * log( (1/B) Σ_i exp(t * l_i) )

    t→0 recovers ERM; t>0 up-weights high-loss (tail-seeking) samples.
    """

    def __init__(self, tilt: float) -> None:
        super().__init__()
        assert tilt != 0.0, "Use DDPMSimpleLoss for tilt=0 (ERM baseline)."
        self.tilt = tilt

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> LossOutput:
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
        B = pred.shape[0]
        per_sample = _per_molecule_mse(pred, target, kwargs.get("node_mask"))
        t = self.tilt
        # Numerically stable via torch.logsumexp's internal max-shift trick.
        total = (torch.logsumexp(t * per_sample, dim=0) - math.log(B)) / t
        weights = torch.softmax(t * per_sample, dim=0)
        entropy = -(weights * (weights + 1e-8).log()).sum()
        return LossOutput(
            total_loss=total,
            per_sample_loss=per_sample,
            loss_components={"tilt": total, "mse_mean": per_sample.mean(), "mse_max": per_sample.max()},
            weights=weights,
            diagnostics={"tilt": float(t), "tilt_effective_weight_entropy": entropy.item()},
        )

    def log_keys(self) -> list[str]:
        return ["Train/loss", "Train/loss_tilt", "Train/loss_mse_mean",
                "Train/loss_mse_max", "Train/tilt_effective_weight_entropy", "Val/loss"]


def get_loss_fn(tilt: float) -> BaseLoss:
    """tilt=0 → DDPMSimpleLoss (ERM baseline); tilt≠0 → TiltedScoreMatchingLoss."""
    if tilt == 0.0:
        from .ddpm_simple import DDPMSimpleLoss
        return DDPMSimpleLoss()
    return TiltedScoreMatchingLoss(tilt=tilt)
