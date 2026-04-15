"""Mathematical limit convergence tests for tilted losses.

Verifies three theoretical properties:
1. ERM limit:            tilt → 0     ⟹  L_tilt → arithmetic mean of per-sample losses
2. Minimax limit:        tilt → +∞    ⟹  L_tilt → max of per-sample losses
3. Hierarchical collapse: τ_outer = τ_group, groups=None  ⟹  L_MO ≈ L_tilt(τ)

All tests use random tensors with planted outliers to stress the tail behavior.
"""

import pytest
import torch

from src.losses.ddpm_simple import DDPMSimpleLoss
from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss
from src.losses.tilted_score_matching import TiltedScoreMatchingLoss
from src.losses.base import _per_molecule_mse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_with_outlier():
    """8-sample batch (float32) where sample 7 has large error (planted outlier)."""
    torch.manual_seed(42)
    B, C, H, W = 8, 4, 8, 8
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)
    # Plant a large outlier in the last sample
    target[-1] = pred[-1] + 3.0
    return pred, target


@pytest.fixture
def batch_with_outlier_f64():
    """Float64 version of the batch — used for ERM limit tests to avoid float32
    catastrophic cancellation in (logsumexp - log(B)) / t at tiny tilt values."""
    torch.manual_seed(42)
    B, C, H, W = 8, 4, 8, 8
    pred = torch.randn(B, C, H, W, dtype=torch.float64)
    target = torch.randn(B, C, H, W, dtype=torch.float64)
    target[-1] = pred[-1] + 3.0
    return pred, target


@pytest.fixture
def per_sample_losses(batch_with_outlier):
    """Pre-computed per-sample MSE losses for the fixture batch."""
    pred, target = batch_with_outlier
    return _per_molecule_mse(pred, target, node_mask=None)


@pytest.fixture
def per_sample_losses_f64(batch_with_outlier_f64):
    """Float64 per-sample losses."""
    pred, target = batch_with_outlier_f64
    return _per_molecule_mse(pred, target, node_mask=None)


# ---------------------------------------------------------------------------
# Limit 1: ERM (tilt → 0)
# ---------------------------------------------------------------------------

class TestERMLimit:
    """Use float64 fixtures to avoid float32 catastrophic cancellation.

    At tilt=1e-6, the formula (logsumexp(t*l) - log(B)) / t involves
    subtracting two nearly equal numbers.  In float32 (~7 sig figs) this
    loses ~5 digits; in float64 (~15 sig figs) it retains ~9 digits — enough
    to verify the limit to atol=1e-4.
    """

    def test_tiny_tilt_matches_mean(self, batch_with_outlier_f64, per_sample_losses_f64):
        """L_tilt(tilt=1e-6) ≈ mean of per-sample losses (float64)."""
        pred, target = batch_with_outlier_f64
        loss_fn = TiltedScoreMatchingLoss(tilt=1e-6)
        out = loss_fn(pred, target)
        expected = per_sample_losses_f64.mean()
        assert torch.allclose(out.total_loss, expected, atol=1e-4), (
            f"ERM limit failed: L_tilt={out.total_loss.item():.8f}, "
            f"mean={expected.item():.8f}"
        )

    def test_tiny_tilt_matches_ddpm_simple(self, batch_with_outlier_f64):
        """L_tilt(tilt=1e-6) ≈ DDPMSimpleLoss (float64)."""
        pred, target = batch_with_outlier_f64
        tilt_fn = TiltedScoreMatchingLoss(tilt=1e-6)
        erm_fn = DDPMSimpleLoss()
        tilt_out = tilt_fn(pred, target)
        erm_out = erm_fn(pred, target)
        assert torch.allclose(tilt_out.total_loss, erm_out.total_loss, atol=1e-4), (
            f"L_tilt(1e-6)={tilt_out.total_loss.item():.8f} != "
            f"DDPMSimple={erm_out.total_loss.item():.8f}"
        )

    def test_negative_tiny_tilt_matches_mean(self, batch_with_outlier_f64, per_sample_losses_f64):
        """L_tilt(tilt=-1e-6) also recovers the mean (float64)."""
        pred, target = batch_with_outlier_f64
        loss_fn = TiltedScoreMatchingLoss(tilt=-1e-6)
        out = loss_fn(pred, target)
        expected = per_sample_losses_f64.mean()
        assert torch.allclose(out.total_loss, expected, atol=1e-4), (
            f"Negative ERM limit failed: L_tilt={out.total_loss.item():.8f}, "
            f"mean={expected.item():.8f}"
        )


# ---------------------------------------------------------------------------
# Limit 2: Minimax (tilt → +∞)
# ---------------------------------------------------------------------------

class TestMinimaxLimit:

    def test_large_tilt_approaches_max(self, batch_with_outlier, per_sample_losses):
        """L_tilt(tilt=1e3) ≈ max of per-sample losses."""
        pred, target = batch_with_outlier
        loss_fn = TiltedScoreMatchingLoss(tilt=1e3)
        out = loss_fn(pred, target)
        expected = per_sample_losses.max()
        # Tolerance is looser (1e-2) because 1e3 is not infinite
        assert torch.allclose(out.total_loss, expected, atol=1e-2), (
            f"Minimax limit failed: L_tilt={out.total_loss.item():.6f}, "
            f"max={expected.item():.6f}"
        )

    def test_large_tilt_exceeds_mean(self, batch_with_outlier):
        """L_tilt(tilt=1e3) > DDPMSimpleLoss — tail-seeking raises loss."""
        pred, target = batch_with_outlier
        large_fn = TiltedScoreMatchingLoss(tilt=1e3)
        erm_fn = DDPMSimpleLoss()
        assert large_fn(pred, target).total_loss > erm_fn(pred, target).total_loss, (
            "Large-tilt loss should exceed ERM loss on a batch with outliers"
        )

    def test_monotone_in_tilt(self, batch_with_outlier):
        """L_tilt is monotonically non-decreasing in tilt for tilt > 0."""
        pred, target = batch_with_outlier
        tilts = [1e-6, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        losses = [TiltedScoreMatchingLoss(t)(pred, target).total_loss.item() for t in tilts]
        for i in range(len(losses) - 1):
            assert losses[i] <= losses[i + 1] + 1e-6, (
                f"Monotonicity violated: L({tilts[i]})={losses[i]:.6f} > "
                f"L({tilts[i+1]})={losses[i+1]:.6f}"
            )

    def test_large_negative_tilt_approaches_min(self, batch_with_outlier, per_sample_losses):
        """L_tilt(tilt=-1e3) ≈ min of per-sample losses (reverse minimax)."""
        pred, target = batch_with_outlier
        loss_fn = TiltedScoreMatchingLoss(tilt=-1e3)
        out = loss_fn(pred, target)
        expected = per_sample_losses.min()
        assert torch.allclose(out.total_loss, expected, atol=1e-2), (
            f"Negative minimax limit failed: L_tilt={out.total_loss.item():.6f}, "
            f"min={expected.item():.6f}"
        )


# ---------------------------------------------------------------------------
# Limit 3: Hierarchical collapse (τ_outer = τ_group, groups=None → L_tilt)
# ---------------------------------------------------------------------------

class TestHierarchicalCollapse:

    @pytest.mark.parametrize("tau", [0.5, 1.0, 2.0, 5.0])
    def test_collapse_to_single_objective(self, batch_with_outlier, tau):
        """MultiObjectiveTiltedLoss(outer_tilt=τ, group_tilts=[τ], groups=None) ≈ TiltedScoreMatchingLoss(τ)."""
        pred, target = batch_with_outlier
        # Single-group multi-objective (groups=None → all in group 0)
        mo_fn = MultiObjectiveTiltedLoss(
            outer_tilt=tau,
            group_tilts=[tau],
            gumbel_temp=1e-6,  # near-deterministic Gumbel = no stochasticity
        )
        so_fn = TiltedScoreMatchingLoss(tilt=tau)

        mo_out = mo_fn(pred, target, groups=None)
        so_out = so_fn(pred, target)

        # With a single group and τ_outer = τ_group, J̃ = R̃_0 = L_tilt(τ)
        # Gumbel weight is 1.0 (only one group), so L_MO = 1.0 * R̃_0 = L_tilt(τ)
        assert torch.allclose(mo_out.total_loss, so_out.total_loss, atol=1e-5), (
            f"Hierarchical collapse failed at τ={tau}: "
            f"L_MO={mo_out.total_loss.item():.6f}, "
            f"L_tilt={so_out.total_loss.item():.6f}"
        )

    def test_multi_group_splits_correctly(self, batch_with_outlier):
        """With two groups and equal tilts, L_MO is a weighted combination of group risks."""
        pred, target = batch_with_outlier  # B=8
        B = pred.shape[0]
        groups = torch.zeros(B, dtype=torch.long)
        groups[B // 2:] = 1  # split evenly

        tau = 1.0
        mo_fn = MultiObjectiveTiltedLoss(
            outer_tilt=tau,
            group_tilts=[tau, tau],
            gumbel_temp=1e-6,
        )
        out = mo_fn(pred, target, groups)
        # The loss must be a positive scalar
        assert out.total_loss.ndim == 0
        assert out.total_loss.item() > 0

    def test_j_tilt_diagnostic_is_scalar(self, batch_with_outlier):
        """j_tilt diagnostic in loss_components must be a scalar tensor."""
        pred, target = batch_with_outlier
        groups = torch.zeros(pred.shape[0], dtype=torch.long)
        mo_fn = MultiObjectiveTiltedLoss(
            outer_tilt=1.0, group_tilts=[1.0], gumbel_temp=0.5
        )
        out = mo_fn(pred, target, groups)
        j_tilt = out.loss_components["j_tilt"]
        assert j_tilt.ndim == 0, f"j_tilt should be scalar, got shape {j_tilt.shape}"


# ---------------------------------------------------------------------------
# Limit 4: Jensen's inequality (sanity — matches ablation finding)
# ---------------------------------------------------------------------------

class TestJensensInequality:

    def test_positive_tilt_above_erm(self, batch_with_outlier):
        """L_tilt(τ > 0) ≥ ERM loss — Jensen's inequality for convex exp."""
        pred, target = batch_with_outlier
        erm_loss = DDPMSimpleLoss()(pred, target).total_loss
        for tau in [0.5, 1.0, 2.0, 5.0]:
            tilt_loss = TiltedScoreMatchingLoss(tilt=tau)(pred, target).total_loss
            assert tilt_loss >= erm_loss - 1e-6, (
                f"Jensen violated at τ={tau}: L_tilt={tilt_loss:.6f} < ERM={erm_loss:.6f}"
            )

    def test_negative_tilt_below_erm(self, batch_with_outlier):
        """L_tilt(τ < 0) ≤ ERM loss — negative tilt is the robust direction.

        Negative tilt de-weights outliers via softmin weighting, so the
        aggregated loss sits below the arithmetic mean.  This is the reverse
        of Jensen's inequality (concave direction of logsumexp for t<0).
        """
        pred, target = batch_with_outlier
        erm_loss = DDPMSimpleLoss()(pred, target).total_loss
        for tau in [-0.5, -1.0, -2.0, -5.0]:
            tilt_loss = TiltedScoreMatchingLoss(tilt=tau)(pred, target).total_loss
            assert tilt_loss <= erm_loss + 1e-6, (
                f"Negative tilt should be ≤ ERM at τ={tau}: "
                f"L_tilt={tilt_loss:.6f} > ERM={erm_loss:.6f}"
            )
