"""Tilted Score Matching loss — TERM-style log-sum-exp tilt.

THIS IS THE CORE RESEARCH CONTRIBUTION.
Do not implement without a corresponding theory section update in the paper.
Do not modify the stub interface — only replace the NotImplementedError body.
"""

from __future__ import annotations

import torch

from .base import BaseLoss, LossOutput


class TiltedScoreMatchingLoss(BaseLoss):
    """L_tilt = (1/t) * log( (1/B) Σ_i exp(t * l_i) )

    where l_i is the per-sample MSE and t is the tilt temperature.
    At t→0 this recovers standard ERM. At t>0 it up-weights high-loss
    (rare / out-of-distribution) samples, biasing the score toward the tail.

    Args:
        tilt: TERM temperature parameter t. Must be non-zero.
    """

    def __init__(self, tilt: float) -> None:
        super().__init__()
        assert tilt != 0.0, "Use DDPMSimpleLoss for tilt=0 (ERM baseline)."
        self.tilt = tilt

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> LossOutput:
        raise NotImplementedError(
            f"TiltedScoreMatchingLoss(tilt={self.tilt}) is not yet implemented. "
            "Edit ONLY src/losses/tilted_score_matching.py to activate."
        )

    def log_keys(self) -> list[str]:
        return [
            "Train/loss",
            "Train/loss_tilt",
            "Train/loss_mse_mean",
            "Train/loss_mse_max",
            "Train/tilt_effective_weight_entropy",
            "Val/loss",
        ]


def get_loss_fn(tilt: float) -> BaseLoss:
    """Registry: return the correct loss for a given tilt value.

    tilt == 0.0  → DDPMSimpleLoss (ERM baseline, fully implemented)
    tilt != 0.0  → TiltedScoreMatchingLoss (stub, raises NotImplementedError)
    """
    if tilt == 0.0:
        from .ddpm_simple import DDPMSimpleLoss
        return DDPMSimpleLoss()
    return TiltedScoreMatchingLoss(tilt=tilt)
