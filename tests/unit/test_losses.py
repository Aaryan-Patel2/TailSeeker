"""Unit tests for loss functions."""

from __future__ import annotations

import pytest
import torch


# ── DDPMSimpleLoss ────────────────────────────────────────────────────

def test_erm_zero_loss_on_perfect_prediction():
    """DDPMSimpleLoss returns 0 when pred == target."""
    from src.losses.ddpm_simple import DDPMSimpleLoss

    loss_fn = DDPMSimpleLoss()
    x = torch.randn(4, 4, 8, 8)
    out = loss_fn(x, x)
    assert out.total_loss.item() < 1e-6, (
        f"Expected ~0 loss on perfect prediction, got {out.total_loss.item()}"
    )


def test_erm_returns_loss_output_fields():
    """DDPMSimpleLoss returns a LossOutput with all required fields."""
    from src.losses.ddpm_simple import DDPMSimpleLoss
    from src.losses.base import LossOutput

    loss_fn = DDPMSimpleLoss()
    pred = torch.randn(4, 4, 8, 8)
    target = torch.randn(4, 4, 8, 8)
    out = loss_fn(pred, target)

    assert isinstance(out, LossOutput)
    assert out.total_loss.ndim == 0, "total_loss must be a scalar"
    assert out.per_sample_loss.shape == (4,), (
        f"per_sample_loss must be shape (B,)==(4,), got {out.per_sample_loss.shape}"
    )
    assert "mse" in out.loss_components


def test_erm_shape_mismatch_raises():
    """DDPMSimpleLoss raises AssertionError on shape mismatch."""
    from src.losses.ddpm_simple import DDPMSimpleLoss

    loss_fn = DDPMSimpleLoss()
    with pytest.raises(AssertionError, match="Shape mismatch"):
        loss_fn(torch.randn(4, 4), torch.randn(4, 8))


# ── TiltedScoreMatchingLoss implementation ────────────────────────────

def test_tsm_zero_loss_on_perfect_prediction():
    """L_tilt = 0 when pred == target (all per-sample MSE = 0)."""
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    for tilt in [1.0, -1.0, 5.0, -5.0]:
        loss_fn = TiltedScoreMatchingLoss(tilt=tilt)
        x = torch.randn(4, 4, 8, 8)
        out = loss_fn(x, x)
        assert out.total_loss.item() < 1e-5, (
            f"tilt={tilt}: expected ~0 on perfect pred, got {out.total_loss.item()}"
        )


def test_tsm_returns_loss_output_fields():
    """TiltedScoreMatchingLoss returns a LossOutput with all required fields."""
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss
    from src.losses.base import LossOutput

    loss_fn = TiltedScoreMatchingLoss(tilt=1.0)
    pred, target = torch.randn(4, 4, 8, 8), torch.randn(4, 4, 8, 8)
    out = loss_fn(pred, target)

    assert isinstance(out, LossOutput)
    assert out.total_loss.ndim == 0, "total_loss must be scalar"
    assert out.per_sample_loss.shape == (4,), f"per_sample_loss shape {out.per_sample_loss.shape}"
    assert "tilt" in out.loss_components
    assert "mse_mean" in out.loss_components
    assert "mse_max" in out.loss_components
    assert out.weights is not None and out.weights.shape == (4,)


@pytest.mark.parametrize("tilt", [0.5, 1.0, 2.0, 5.0])
def test_tsm_positive_tilt_geq_erm(tilt: float):
    """t>0 → L_tilt ≥ mean_mse (Jensen's inequality on convex exp)."""
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    torch.manual_seed(0)
    pred, target = torch.randn(16, 4), torch.randn(16, 4)
    loss_fn = TiltedScoreMatchingLoss(tilt=tilt)
    out = loss_fn(pred, target)
    mean_mse = out.per_sample_loss.mean().item()
    assert out.total_loss.item() >= mean_mse - 1e-5, (
        f"tilt={tilt}: L_tilt={out.total_loss.item():.6f} < mean_mse={mean_mse:.6f}"
    )


@pytest.mark.parametrize("tilt", [-0.5, -1.0, -2.0, -5.0])
def test_tsm_negative_tilt_leq_erm(tilt: float):
    """t<0 → L_tilt ≤ mean_mse (Jensen + negative denominator flips inequality)."""
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    torch.manual_seed(0)
    pred, target = torch.randn(16, 4), torch.randn(16, 4)
    loss_fn = TiltedScoreMatchingLoss(tilt=tilt)
    out = loss_fn(pred, target)
    mean_mse = out.per_sample_loss.mean().item()
    assert out.total_loss.item() <= mean_mse + 1e-5, (
        f"tilt={tilt}: L_tilt={out.total_loss.item():.6f} > mean_mse={mean_mse:.6f}"
    )


def test_tsm_stable_at_extreme_tilt():
    """t=10 and t=-5 produce finite, non-NaN loss."""
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    torch.manual_seed(1)
    pred, target = torch.randn(8, 4), torch.randn(8, 4)
    for tilt in [10.0, -5.0]:
        out = TiltedScoreMatchingLoss(tilt=tilt)(pred, target)
        assert torch.isfinite(out.total_loss), f"tilt={tilt}: loss is not finite"


def test_tsm_erm_limit_at_tiny_tilt():
    """t→0 limit: L_tilt ≈ mean_mse (Taylor: O(t) error)."""
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    torch.manual_seed(2)
    pred, target = torch.randn(32, 4), torch.randn(32, 4)
    import torch.nn.functional as F
    per_sample = F.mse_loss(pred, target, reduction="none").view(32, -1).mean(dim=1)
    mean_mse = per_sample.mean().item()

    out = TiltedScoreMatchingLoss(tilt=1e-3)(pred, target)
    assert abs(out.total_loss.item() - mean_mse) < 1e-3, (
        f"ERM limit failed: L_tilt={out.total_loss.item():.6f}, mean_mse={mean_mse:.6f}"
    )


def test_tsm_zero_tilt_raises_assertion():
    """TiltedScoreMatchingLoss disallows tilt=0 (use DDPMSimpleLoss instead)."""
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    with pytest.raises(AssertionError):
        TiltedScoreMatchingLoss(tilt=0.0)


def test_get_loss_fn_returns_erm_for_zero():
    """get_loss_fn(0.0) returns DDPMSimpleLoss."""
    from src.losses.tilted_score_matching import get_loss_fn
    from src.losses.ddpm_simple import DDPMSimpleLoss

    loss_fn = get_loss_fn(0.0)
    assert isinstance(loss_fn, DDPMSimpleLoss)


def test_get_loss_fn_returns_tsm_for_nonzero():
    """get_loss_fn(1.0) returns TiltedScoreMatchingLoss."""
    from src.losses.tilted_score_matching import get_loss_fn, TiltedScoreMatchingLoss

    loss_fn = get_loss_fn(1.0)
    assert isinstance(loss_fn, TiltedScoreMatchingLoss)
    assert loss_fn.tilt == 1.0


# ── MultiObjectiveTiltedLoss (§3–4) ──────────────────────────────────

def test_mo_zero_loss_on_perfect_prediction():
    """L_MO = 0 when pred == target (all per-sample MSE = 0)."""
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, -1.0])
    x = torch.randn(8, 4, 8, 8)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    out = loss_fn(x, x, groups=groups)
    assert out.total_loss.item() < 1e-5, (
        f"Expected ~0 on perfect pred, got {out.total_loss.item()}"
    )


def test_mo_returns_loss_output_fields():
    """MultiObjectiveTiltedLoss returns a LossOutput with all required fields."""
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss
    from src.losses.base import LossOutput

    torch.manual_seed(0)
    loss_fn = MultiObjectiveTiltedLoss(outer_tilt=2.0, group_tilts=[1.0, -1.0], gumbel_temp=0.5)
    pred, target = torch.randn(8, 4), torch.randn(8, 4)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    out = loss_fn(pred, target, groups=groups)

    assert isinstance(out, LossOutput)
    assert out.total_loss.ndim == 0, "total_loss must be scalar"
    assert out.per_sample_loss.shape == (8,), f"per_sample_loss shape {out.per_sample_loss.shape}"
    assert out.weights is not None and out.weights.shape == (2,), (
        f"weights must be shape (G=2,), got {out.weights.shape}"
    )
    for key in ("mo_loss", "j_tilt", "mse_mean", "mse_max"):
        assert key in out.loss_components, f"Missing key '{key}' in loss_components"


def test_mo_gumbel_weights_are_valid_distribution():
    """Gumbel-Softmax weights are positive and sum to 1."""
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    torch.manual_seed(1)
    loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 2.0, -1.0])
    pred, target = torch.randn(12, 4), torch.randn(12, 4)
    groups = torch.arange(12) % 3  # 4 samples per group
    out = loss_fn(pred, target, groups=groups)

    w = out.weights
    assert (w > 0).all(), f"All Gumbel weights must be positive, got {w}"
    assert abs(w.sum().item() - 1.0) < 1e-5, f"Weights must sum to 1, got {w.sum().item()}"


def test_mo_single_group_matches_v1():
    """G=1 MultiObjectiveTiltedLoss equals V1 TiltedScoreMatchingLoss with same tilt."""
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    torch.manual_seed(2)
    pred, target = torch.randn(16, 4), torch.randn(16, 4)

    # G=1: gumbel_softmax([single_logit]) = [1.0] always → L_MO = R̃_0 = V1(tau)
    tau = 1.5
    mo_fn = MultiObjectiveTiltedLoss(outer_tilt=2.0, group_tilts=[tau], gumbel_temp=1.0)
    v1_fn = TiltedScoreMatchingLoss(tilt=tau)

    mo_out = mo_fn(pred, target)          # groups=None → single group
    v1_out = v1_fn(pred, target)

    assert abs(mo_out.total_loss.item() - v1_out.total_loss.item()) < 1e-5, (
        f"G=1 MO loss {mo_out.total_loss.item():.6f} != V1 {v1_out.total_loss.item():.6f}"
    )


def test_mo_stable_at_extreme_outer_tilt():
    """Extreme outer_tilt values produce finite, non-NaN L_MO and J̃."""
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    torch.manual_seed(3)
    pred, target = torch.randn(8, 4), torch.randn(8, 4)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    for t in [10.0, -5.0]:
        loss_fn = MultiObjectiveTiltedLoss(outer_tilt=t, group_tilts=[1.0, -1.0])
        out = loss_fn(pred, target, groups=groups)
        assert torch.isfinite(out.total_loss), f"t={t}: L_MO is not finite"
        assert torch.isfinite(out.loss_components["j_tilt"]), f"t={t}: J̃ is not finite"


def test_mo_missing_group_does_not_crash():
    """A batch missing one group entirely falls back gracefully (no crash, finite loss,
    gradients still flow through the present group)."""
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    torch.manual_seed(4)
    pred = torch.randn(8, 4, requires_grad=True)
    target = torch.randn(8, 4)
    # All samples in group 0; group 1 is absent from this batch
    groups = torch.zeros(8, dtype=torch.long)
    loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 2.0])
    out = loss_fn(pred, target, groups=groups)
    assert torch.isfinite(out.total_loss), "Loss must be finite even with missing group"
    # Gradient must flow back through the present group (group 0)
    out.total_loss.backward()
    assert pred.grad is not None, "grad must flow to pred even with missing group"
    assert torch.isfinite(pred.grad).all(), "pred.grad must be finite"


def test_mo_zero_tilt_raises_for_inner_group():
    """Inner group tilt of 0.0 raises AssertionError."""
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    with pytest.raises(AssertionError):
        MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[0.0])


def test_mo_outer_zero_tilt_raises():
    """outer_tilt=0 raises AssertionError."""
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    with pytest.raises(AssertionError):
        MultiObjectiveTiltedLoss(outer_tilt=0.0, group_tilts=[1.0])


def test_get_hierarchical_loss_fn_factory():
    """get_hierarchical_loss_fn returns a correctly configured instance."""
    from src.losses.hierarchical_loss import get_hierarchical_loss_fn, MultiObjectiveTiltedLoss

    fn = get_hierarchical_loss_fn(outer_tilt=2.0, group_tilts=[1.0, -1.0], gumbel_temp=0.5)
    assert isinstance(fn, MultiObjectiveTiltedLoss)
    assert fn.outer_tilt == 2.0
    assert fn.G == 2
    assert fn.gumbel_temp == 0.5


# ── log_sum_exp ───────────────────────────────────────────────────────

def test_log_sum_exp_correctness():
    """log_sum_exp matches torch.logsumexp."""
    from src.utils import log_sum_exp

    x = torch.randn(10)
    expected = torch.logsumexp(x, dim=0)
    actual = log_sum_exp(x, dim=0)
    assert torch.allclose(actual, expected, atol=1e-5), (
        f"log_sum_exp mismatch: expected {expected.item():.6f}, got {actual.item():.6f}"
    )
