"""Reward-Weighted Tilted Loss — property-driven Gibbs-measure reweighting.

r_i   = QED(i) - λ · (SA(i) - 1) / 9        [composite reward, λ=0.5]
t_eff = t_target · min(1, epoch / warmup)     [linear tilt warmup]
w_i   = softmax(t_eff · r_i)                 [reward Gibbs weights]
L     = Σ_i w_i · l_i                        [reward-weighted regression]

Stationary point: score of p̃(x) ∝ p_data(x) · exp(t · r(x)).
Equivalent to training-time classifier guidance (Dhariwal & Nichol 2021).
t=0 / low reward variance → exact ERM.
"""
from __future__ import annotations

import torch

from .base import BaseLoss, LossOutput, _per_molecule_mse


class RewardWeightedTiltedLoss(BaseLoss):
    """Tilt the DDPM score function toward high-reward (drug-like) molecules."""

    def __init__(
        self,
        tilt: float,
        lambda_: float = 0.5,
        warmup_frac: float = 0.3,
        reward_std_threshold: float = 0.05,
        lambda_logp: float = 0.0,
        lambda_tpsa: float = 0.0,
    ) -> None:
        super().__init__()
        self.tilt = tilt
        self.lambda_ = lambda_
        self.warmup_frac = warmup_frac
        self.reward_std_threshold = reward_std_threshold
        self.lambda_logp = lambda_logp
        self.lambda_tpsa = lambda_tpsa

    def _reward(
        self,
        qed: torch.Tensor,
        sa: torch.Tensor,
        logp: torch.Tensor | None = None,
        tpsa: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert qed.ndim == 1, f"qed must be (B,), got {qed.shape}"
        assert sa.ndim == 1, f"sa must be (B,), got {sa.shape}"
        assert (qed >= 0).all() and (qed <= 1).all(), f"QED out of [0,1]: {qed}"
        assert (sa >= 1).all() and (sa <= 10).all(), f"SA out of [1,10]: {sa}"
        r = qed - self.lambda_ * (sa - 1.0) / 9.0
        if logp is not None and self.lambda_logp > 0.0:
            # Penalise LogP > 3 (too lipophilic → poor solubility); normalised to [0,1]
            r = r - self.lambda_logp * torch.clamp(logp - 3.0, min=0.0) / 2.0
        if tpsa is not None and self.lambda_tpsa > 0.0:
            # Reward low TPSA < 140 Å² (oral bioavailability window); normalised to [0,1]
            r = r + self.lambda_tpsa * (1.0 - (tpsa / 140.0).clamp(0.0, 1.0))
        return r

    def _t_eff(self, epoch: int, max_epochs: int) -> float:
        if self.warmup_frac <= 0.0 or max_epochs <= 0:
            return self.tilt
        return self.tilt * min(1.0, epoch / (self.warmup_frac * max_epochs))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> LossOutput:
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
        qed: torch.Tensor = kwargs["qed"]
        sa: torch.Tensor = kwargs["sa"]
        epoch = int(kwargs.get("epoch", 0))
        max_epochs = int(kwargs.get("max_epochs", 100))

        logp: torch.Tensor | None = kwargs.get("logp")
        tpsa: torch.Tensor | None = kwargs.get("tpsa")
        per_sample = _per_molecule_mse(pred, target, kwargs.get("node_mask"))
        rewards = self._reward(
            qed.float(), sa.float(),
            logp=logp.float() if logp is not None else None,
            tpsa=tpsa.float() if tpsa is not None else None,
        )
        t_eff = self._t_eff(epoch, max_epochs)
        reward_std = float(rewards.std())

        use_erm = (t_eff == 0.0) or (reward_std < self.reward_std_threshold)
        weights = (torch.ones_like(per_sample) / per_sample.shape[0] if use_erm
                   else torch.softmax(t_eff * rewards, dim=0))
        total = (weights * per_sample).sum()

        return LossOutput(
            total_loss=total,
            per_sample_loss=per_sample,
            loss_components={"reward_tilt": total, "mse_mean": per_sample.mean()},
            weights=weights,
            diagnostics={
                "reward_tilt_t_eff": t_eff,
                "reward_std": reward_std,
                "reward_erm_fallback": float(use_erm),
            },
        )

    def log_keys(self) -> list[str]:
        return ["Train/loss", "Train/loss_reward_tilt", "Train/loss_mse_mean",
                "Train/reward_tilt_t_eff", "Train/reward_std", "Train/reward_erm_fallback"]


def get_reward_loss_fn(
    tilt: float,
    lambda_: float = 0.5,
    warmup_frac: float = 0.3,
    lambda_logp: float = 0.0,
    lambda_tpsa: float = 0.0,
) -> BaseLoss:
    """Factory: tilt=0 → DDPMSimpleLoss (ERM); tilt≠0 → RewardWeightedTiltedLoss."""
    if tilt == 0.0:
        from .ddpm_simple import DDPMSimpleLoss
        return DDPMSimpleLoss()
    return RewardWeightedTiltedLoss(
        tilt=tilt, lambda_=lambda_, warmup_frac=warmup_frac,
        lambda_logp=lambda_logp, lambda_tpsa=lambda_tpsa,
    )
