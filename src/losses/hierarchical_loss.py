"""Multi-objective hierarchical TERM loss with Gumbel-Softmax group selection.

Implements math_tilted_objectives.md §3 (J̃) and §4 (L_MO).

§3  Group risk:  R̃_g = (1/τ_g) log[(1/|g|) Σ_{x∈g} exp(τ_g·l_x)]
    Outer loss:  J̃ = (1/t) log[(1/N) Σ_g |g|·exp(t·R̃_g)]

§4  Gumbel-Softmax:  L_MO = Σ_g w_g^GS(λ) · R̃_g
    where logits = R̃_g + log(|g|)/t,  w^GS = gumbel_softmax(logits, τ=λ)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseLoss, LossOutput, _per_molecule_mse


class MultiObjectiveTiltedLoss(BaseLoss):
    """Hierarchical TERM loss with differentiable group selection (§3–4).

    Args:
        outer_tilt:  t — controls emphasis across pharmacological groups.
        group_tilts: [τ_g] — per-group inner tilt; τ>0 tail-seeking, τ<0 robust.
        gumbel_temp: λ — Gumbel-Softmax temperature; anneal downward during training.

    forward() accepts an optional `groups` LongTensor (B,) with values in [0, G).
    If None, all samples are placed in group 0 — equivalent to V1 with group_tilts[0].
    """

    def __init__(
        self,
        outer_tilt: float,
        group_tilts: list[float],
        gumbel_temp: float = 1.0,
    ) -> None:
        super().__init__()
        assert outer_tilt != 0.0, "outer_tilt=0 is undefined; use DDPMSimpleLoss."
        assert len(group_tilts) >= 1, "Provide at least one group tilt."
        assert all(tau != 0.0 for tau in group_tilts), "Inner group tilts must be non-zero."
        assert gumbel_temp > 0.0, f"gumbel_temp must be > 0, got {gumbel_temp}."
        self.outer_tilt = outer_tilt
        self.register_buffer("group_tilts", torch.tensor(group_tilts, dtype=torch.float32))
        self.gumbel_temp = gumbel_temp
        self.G = len(group_tilts)

    def _group_risk(
        self, per_sample: torch.Tensor, mask: torch.BoolTensor, tau: float
    ) -> torch.Tensor:
        """R̃_g = (1/τ) log[(1/|g|) Σ_{x∈g} exp(τ·l_x)]  — numerically stable.

        If the group is absent from this batch, falls back to the batch mean MSE
        (zero-gradient proxy; does not distort other groups' gradients).
        """
        losses_g = per_sample[mask]
        n_g = losses_g.numel()
        if n_g == 0:
            # Detached fallback: missing group contributes no gradient.
            return (
                torch.logsumexp(tau * per_sample, dim=0) - math.log(per_sample.numel())
            ).detach() / tau
        return (torch.logsumexp(tau * losses_g, dim=0) - math.log(n_g)) / tau

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        groups: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LossOutput:
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
        B = pred.shape[0]
        per_sample = _per_molecule_mse(pred, target, kwargs.get("node_mask"))
        assert per_sample.shape == (B,), f"per_sample shape {per_sample.shape} != ({B},)"

        if groups is None:
            groups = torch.zeros(B, dtype=torch.long, device=pred.device)
        assert groups.shape == (B,), f"groups shape {groups.shape} != ({B},)"

        t = self.outer_tilt

        # ── §3: per-group TERM risks ──────────────────────────────────────────
        group_risks = torch.stack([
            self._group_risk(per_sample, groups == g, self.group_tilts[g].item())
            for g in range(self.G)
        ])  # (G,)

        group_sizes = torch.tensor(
            [max((groups == g).sum().item(), 1) for g in range(self.G)],
            dtype=torch.float32, device=pred.device,
        )  # (G,)

        # J̃ = (1/t) log[(1/N) Σ_g |g|·exp(t·R̃_g)]   (diagnostic, not backpropagated)
        N = group_sizes.sum()
        j_tilt = (
            torch.logsumexp(t * group_risks + group_sizes.log(), dim=0) - N.log()
        ) / t

        # ── §4: Gumbel-Softmax differentiable group selection ─────────────────
        # logits: π_g = R̃_g + (1/t)·log|g|
        logits = group_risks + group_sizes.log() / t  # (G,)
        weights_gs = F.gumbel_softmax(logits, tau=self.gumbel_temp, hard=False)
        loss_mo = (weights_gs * group_risks).sum()

        return LossOutput(
            total_loss=loss_mo,
            per_sample_loss=per_sample,
            loss_components={
                "mo_loss": loss_mo,
                "j_tilt": j_tilt,
                "mse_mean": per_sample.mean(),
                "mse_max": per_sample.max(),
            },
            weights=weights_gs,
            diagnostics={
                "outer_tilt": float(t),
                "gumbel_temp": float(self.gumbel_temp),
                "n_groups": float(self.G),
            },
        )

    def log_keys(self) -> list[str]:
        return [
            "Train/loss", "Train/loss_mo_loss", "Train/loss_j_tilt",
            "Train/loss_mse_mean", "Train/loss_mse_max", "Val/loss",
        ]


def get_hierarchical_loss_fn(
    outer_tilt: float,
    group_tilts: list[float],
    gumbel_temp: float = 1.0,
) -> MultiObjectiveTiltedLoss:
    """Factory: returns a configured MultiObjectiveTiltedLoss."""
    return MultiObjectiveTiltedLoss(
        outer_tilt=outer_tilt,
        group_tilts=group_tilts,
        gumbel_temp=gumbel_temp,
    )
