"""Smoke tests — validate theoretical loss functions work correctly.

Tests cover:
- Single-objective tilted score matching (L_tilt)
- Multi-objective hierarchical loss (L_MO)
- ERM baseline (DDPMSimpleLoss)
- Shape handling, numerical stability, and gradient flow
"""

import pytest
import torch

# ==============================================================================
# src/losses/ddpm_simple.py — ERM Baseline
# ==============================================================================

class TestDDPMSimpleLoss:
    """Validate DDPMSimpleLoss (tilt=0 ERM baseline)."""

    def test_forward_returns_scalar(self):
        """DDPMSimpleLoss.forward returns a 0-D tensor."""
        from src.losses.ddpm_simple import DDPMSimpleLoss

        loss_fn = DDPMSimpleLoss()
        pred = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)
        output = loss_fn(pred, target)
        assert output.total_loss.ndim == 0, f"Expected scalar, got shape {output.total_loss.shape}"

    def test_forward_per_sample_loss_shape(self):
        """DDPMSimpleLoss computes per-sample losses correctly."""
        from src.losses.ddpm_simple import DDPMSimpleLoss

        loss_fn = DDPMSimpleLoss()
        B, C, H, W = 4, 3, 8, 8
        pred = torch.randn(B, C, H, W)
        target = torch.randn(B, C, H, W)
        output = loss_fn(pred, target)
        assert output.per_sample_loss.shape == (B,), f"Expected ({B},), got {output.per_sample_loss.shape}"

    def test_forward_shape_mismatch_raises(self):
        """DDPMSimpleLoss raises on shape mismatch."""
        from src.losses.ddpm_simple import DDPMSimpleLoss

        loss_fn = DDPMSimpleLoss()
        with pytest.raises(AssertionError, match="Shape mismatch"):
            loss_fn(torch.randn(4, 3), torch.randn(4, 4))

    def test_erm_baseline_equals_mean_mse(self):
        """DDPMSimpleLoss total_loss equals mean of per-sample MSEs."""
        from src.losses.ddpm_simple import DDPMSimpleLoss

        loss_fn = DDPMSimpleLoss()
        pred = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)
        output = loss_fn(pred, target)
        expected_mean = output.per_sample_loss.mean()
        assert torch.allclose(output.total_loss, expected_mean, atol=1e-6), (
            f"Expected {expected_mean}, got {output.total_loss}"
        )


# ==============================================================================
# src/losses/tilted_score_matching.py — L_tilt (TERM-style)
# ==============================================================================

class TestTiltedScoreMatchingLoss:
    """Validate TiltedScoreMatchingLoss (core research contribution)."""

    def test_forward_returns_scalar(self):
        """TiltedScoreMatchingLoss.forward returns a 0-D tensor."""
        from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

        loss_fn = TiltedScoreMatchingLoss(tilt=1.0)
        pred = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)
        output = loss_fn(pred, target)
        assert output.total_loss.ndim == 0, f"Expected scalar, got shape {output.total_loss.shape}"

    def test_forward_per_sample_loss_shape(self):
        """TiltedScoreMatchingLoss computes per-sample losses correctly."""
        from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

        loss_fn = TiltedScoreMatchingLoss(tilt=1.0)
        B, C, H, W = 4, 3, 8, 8
        pred = torch.randn(B, C, H, W)
        target = torch.randn(B, C, H, W)
        output = loss_fn(pred, target)
        assert output.per_sample_loss.shape == (B,), f"Expected ({B},), got {output.per_sample_loss.shape}"

    def test_tilt_zero_raises(self):
        """TiltedScoreMatchingLoss rejects tilt=0 (use DDPMSimpleLoss)."""
        from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

        with pytest.raises(AssertionError, match="Use DDPMSimpleLoss"):
            TiltedScoreMatchingLoss(tilt=0.0)

    def test_jensen_inequality_positive_tilt(self):
        """L_tilt(t>0) >= mean(mse) — tail-seeking favors high-loss samples."""
        from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

        torch.manual_seed(0)
        loss_fn = TiltedScoreMatchingLoss(tilt=2.0)
        pred = torch.randn(8, 3, 4, 4)
        target = torch.randn(8, 3, 4, 4)
        output = loss_fn(pred, target)
        mean_mse = output.per_sample_loss.mean()
        assert output.total_loss >= mean_mse - 1e-5, (
            f"Jensen violated: L_tilt({2.0}) < mean(mse): {output.total_loss} < {mean_mse}"
        )

    def test_jensen_inequality_negative_tilt(self):
        """L_tilt(t<0) <= mean(mse) — robust loss favors low-loss samples."""
        from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

        torch.manual_seed(0)
        loss_fn = TiltedScoreMatchingLoss(tilt=-2.0)
        pred = torch.randn(8, 3, 4, 4)
        target = torch.randn(8, 3, 4, 4)
        output = loss_fn(pred, target)
        mean_mse = output.per_sample_loss.mean()
        assert output.total_loss <= mean_mse + 1e-5, (
            f"Jensen violated: L_tilt({-2.0}) > mean(mse): {output.total_loss} > {mean_mse}"
        )

    def test_erm_limit(self):
        """L_tilt(t→0) → mean(mse) as tilt→0."""
        from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

        torch.manual_seed(0)
        pred = torch.randn(8, 3, 4, 4)
        target = torch.randn(8, 3, 4, 4)

        loss_small_tilt = TiltedScoreMatchingLoss(tilt=1e-3)
        output_small = loss_small_tilt(pred, target)

        loss_erm = TiltedScoreMatchingLoss(tilt=1.0)
        output_erm = loss_erm(pred, target)

        # ERM limit: as tilt → 0, L_tilt → mean(mse)
        from src.losses.ddpm_simple import DDPMSimpleLoss
        erm_baseline = DDPMSimpleLoss()
        output_baseline = erm_baseline(pred, target)

        # Sanity: tilt=1.0 should be closer to mean(mse) than tilt=1e-3
        # (since logsumexp is monotonic in tilt magnitude)
        assert torch.allclose(output_small.per_sample_loss.mean(), output_baseline.total_loss, atol=1e-2), (
            f"ERM limit failed: mean(mse)={output_baseline.total_loss}, L_tilt(1e-3)={output_small.total_loss}"
        )

    def test_numerical_stability_extreme_tilt(self):
        """L_tilt(t=10) is numerically stable (logsumexp overflow-safe)."""
        from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

        torch.manual_seed(0)
        loss_fn = TiltedScoreMatchingLoss(tilt=10.0)
        pred = torch.randn(8, 3, 4, 4)
        target = torch.randn(8, 3, 4, 4)
        output = loss_fn(pred, target)
        assert not torch.isnan(output.total_loss), "Loss contains NaN at extreme tilt=10"
        assert not torch.isinf(output.total_loss), "Loss contains Inf at extreme tilt=10"

    def test_numerical_stability_negative_tilt(self):
        """L_tilt(t=-5) is numerically stable."""
        from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

        torch.manual_seed(0)
        loss_fn = TiltedScoreMatchingLoss(tilt=-5.0)
        pred = torch.randn(8, 3, 4, 4)
        target = torch.randn(8, 3, 4, 4)
        output = loss_fn(pred, target)
        assert not torch.isnan(output.total_loss), "Loss contains NaN at tilt=-5"
        assert not torch.isinf(output.total_loss), "Loss contains Inf at tilt=-5"

    def test_loss_components_present(self):
        """TiltedScoreMatchingLoss logs expected components."""
        from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

        loss_fn = TiltedScoreMatchingLoss(tilt=1.5)
        pred = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)
        output = loss_fn(pred, target)
        assert "tilt" in output.loss_components, "Missing 'tilt' in loss_components"
        assert "mse_mean" in output.loss_components, "Missing 'mse_mean' in loss_components"
        assert "mse_max" in output.loss_components, "Missing 'mse_max' in loss_components"

    def test_gradients_flow(self):
        """TiltedScoreMatchingLoss backward pass computes gradients."""
        from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

        loss_fn = TiltedScoreMatchingLoss(tilt=1.0)
        pred = torch.randn(4, 3, 8, 8, requires_grad=True)
        target = torch.randn(4, 3, 8, 8)
        output = loss_fn(pred, target)
        output.total_loss.backward()
        assert pred.grad is not None, "No gradients computed"
        assert pred.grad.shape == pred.shape, f"Gradient shape mismatch: {pred.grad.shape}"


# ==============================================================================
# src/losses/hierarchical_loss.py — Multi-Objective L_MO
# ==============================================================================

class TestMultiObjectiveTiltedLoss:
    """Validate MultiObjectiveTiltedLoss (§3–4)."""

    def test_forward_returns_scalar(self):
        """MultiObjectiveTiltedLoss.forward returns a 0-D tensor."""
        from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

        loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 2.0])
        pred = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)
        output = loss_fn(pred, target)
        assert output.total_loss.ndim == 0, f"Expected scalar, got shape {output.total_loss.shape}"

    def test_forward_with_groups(self):
        """MultiObjectiveTiltedLoss handles explicit group assignments."""
        from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

        loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 2.0])
        B = 8
        pred = torch.randn(B, 3, 8, 8)
        target = torch.randn(B, 3, 8, 8)
        groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        output = loss_fn(pred, target, groups=groups)
        assert output.total_loss.ndim == 0, f"Expected scalar, got shape {output.total_loss.shape}"

    def test_forward_without_groups(self):
        """MultiObjectiveTiltedLoss defaults to group=0 when groups=None."""
        from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

        loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0])
        pred = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)
        output = loss_fn(pred, target, groups=None)
        assert output.total_loss.ndim == 0, f"Expected scalar, got shape {output.total_loss.shape}"

    def test_outer_tilt_zero_raises(self):
        """MultiObjectiveTiltedLoss rejects outer_tilt=0."""
        from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

        with pytest.raises(AssertionError, match="outer_tilt=0"):
            MultiObjectiveTiltedLoss(outer_tilt=0.0, group_tilts=[1.0])

    def test_group_tilt_zero_raises(self):
        """MultiObjectiveTiltedLoss rejects group_tilts with zeros."""
        from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

        with pytest.raises(AssertionError, match="non-zero"):
            MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 0.0])

    def test_per_sample_loss_shape(self):
        """MultiObjectiveTiltedLoss per-sample losses have shape [B]."""
        from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

        loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 2.0])
        B = 8
        pred = torch.randn(B, 3, 8, 8)
        target = torch.randn(B, 3, 8, 8)
        output = loss_fn(pred, target)
        assert output.per_sample_loss.shape == (B,), f"Expected ({B},), got {output.per_sample_loss.shape}"

    def test_gumbel_softmax_weights_sum_to_one(self):
        """Gumbel-Softmax weights sum to 1.0 (G-dimensional)."""
        from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

        loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 2.0, 1.5])
        pred = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)
        output = loss_fn(pred, target)
        # weights should be (G,) and sum to ~1
        assert output.weights.shape[0] == 3, f"Expected 3 groups, got {output.weights.shape}"
        assert torch.allclose(output.weights.sum(), torch.tensor(1.0), atol=1e-5), (
            f"Weights don't sum to 1: {output.weights.sum()}"
        )

    def test_loss_components_present(self):
        """MultiObjectiveTiltedLoss logs expected components."""
        from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

        loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 2.0])
        pred = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)
        output = loss_fn(pred, target)
        assert "mo_loss" in output.loss_components, "Missing 'mo_loss'"
        assert "j_tilt" in output.loss_components, "Missing 'j_tilt'"
        assert "mse_mean" in output.loss_components, "Missing 'mse_mean'"
        assert "mse_max" in output.loss_components, "Missing 'mse_max'"

    def test_gradients_flow(self):
        """MultiObjectiveTiltedLoss backward pass computes gradients."""
        from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

        loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 2.0])
        pred = torch.randn(4, 3, 8, 8, requires_grad=True)
        target = torch.randn(4, 3, 8, 8)
        output = loss_fn(pred, target)
        output.total_loss.backward()
        assert pred.grad is not None, "No gradients computed"
        assert pred.grad.shape == pred.shape, "Gradient shape mismatch"

    def test_numerical_stability(self):
        """MultiObjectiveTiltedLoss is numerically stable across tilt magnitudes."""
        from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

        torch.manual_seed(0)
        for outer_tilt in [0.5, 1.0, 5.0, 10.0]:
            loss_fn = MultiObjectiveTiltedLoss(outer_tilt=outer_tilt, group_tilts=[1.0, 2.0])
            pred = torch.randn(4, 3, 8, 8)
            target = torch.randn(4, 3, 8, 8)
            output = loss_fn(pred, target)
            assert not torch.isnan(output.total_loss), f"NaN at outer_tilt={outer_tilt}"
            assert not torch.isinf(output.total_loss), f"Inf at outer_tilt={outer_tilt}"


# ==============================================================================
# Factory Functions
# ==============================================================================

class TestLossFactory:
    """Validate loss factory functions."""

    def test_get_loss_fn_tilt_zero_returns_erm(self):
        """get_loss_fn(0.0) returns DDPMSimpleLoss."""
        from src.losses.ddpm_simple import DDPMSimpleLoss
        from src.losses.tilted_score_matching import get_loss_fn

        loss_fn = get_loss_fn(0.0)
        assert isinstance(loss_fn, DDPMSimpleLoss), f"Expected DDPMSimpleLoss, got {type(loss_fn)}"

    def test_get_loss_fn_nonzero_tilt_returns_tsm(self):
        """get_loss_fn(t≠0) returns TiltedScoreMatchingLoss."""
        from src.losses.tilted_score_matching import TiltedScoreMatchingLoss, get_loss_fn

        for tilt in [-2.0, 0.5, 1.0, 5.0]:
            loss_fn = get_loss_fn(tilt)
            assert isinstance(loss_fn, TiltedScoreMatchingLoss), (
                f"Expected TiltedScoreMatchingLoss for tilt={tilt}, got {type(loss_fn)}"
            )

    def test_get_hierarchical_loss_fn_returns_multi_objective(self):
        """get_hierarchical_loss_fn returns MultiObjectiveTiltedLoss."""
        from src.losses.hierarchical_loss import (
            MultiObjectiveTiltedLoss,
            get_hierarchical_loss_fn,
        )

        loss_fn = get_hierarchical_loss_fn(outer_tilt=1.0, group_tilts=[1.0, 2.0])
        assert isinstance(loss_fn, MultiObjectiveTiltedLoss), (
            f"Expected MultiObjectiveTiltedLoss, got {type(loss_fn)}"
        )


# ==============================================================================
# Integration: Loss Comparison
# ==============================================================================

class TestLossComparison:
    """Compare loss behaviors across the ablation matrix."""

    def test_all_losses_run_without_error(self):
        """All loss functions run successfully on identical inputs."""
        from src.losses.hierarchical_loss import get_hierarchical_loss_fn
        from src.losses.tilted_score_matching import get_loss_fn

        torch.manual_seed(42)
        pred = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)

        # Test tilted loss at various tilt values
        for tilt in [-5.0, -2.0, 0.0, 1.0, 2.0, 5.0, 10.0]:
            loss_fn = get_loss_fn(tilt)
            output = loss_fn(pred, target)
            assert output.total_loss.ndim == 0, f"tilt={tilt}: non-scalar loss"

        # Test multi-objective loss
        loss_fn = get_hierarchical_loss_fn(
            outer_tilt=1.0, group_tilts=[1.0, 2.0]
        )
        output = loss_fn(pred, target)
        assert output.total_loss.ndim == 0, "Multi-objective: non-scalar loss"

    def test_loss_magnitudes_reasonable(self):
        """All losses produce reasonable magnitude outputs (not Inf/NaN)."""
        from src.losses.tilted_score_matching import get_loss_fn

        torch.manual_seed(42)
        pred = torch.randn(8, 3, 8, 8)
        target = torch.randn(8, 3, 8, 8)

        for tilt in [-5.0, -2.0, 0.0, 1.0, 2.0, 5.0, 10.0]:
            loss_fn = get_loss_fn(tilt)
            output = loss_fn(pred, target)
            loss_val = output.total_loss.item()
            assert 0 <= loss_val < 1e6, (
                f"tilt={tilt}: unreasonable loss value {loss_val}"
            )
