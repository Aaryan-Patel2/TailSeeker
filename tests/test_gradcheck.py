"""Gradient connectivity tests for MultiObjectiveTiltedLoss.

Verifies that:
1. total_loss has requires_grad=True and a grad_fn (computation graph intact).
2. torch.autograd.gradcheck passes — Gumbel-Softmax weights are differentiable
   w.r.t. the prediction tensor.
3. Gradients flow back through all groups, not just the selected one.
"""

import pytest
import torch
from torch.autograd import gradcheck

from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loss_fn(**kwargs) -> MultiObjectiveTiltedLoss:
    defaults = dict(outer_tilt=1.0, group_tilts=[1.0, 2.0], gumbel_temp=0.5)
    defaults.update(kwargs)
    return MultiObjectiveTiltedLoss(**defaults)


def _double_inputs(B: int = 6, C: int = 4, H: int = 4, W: int = 4):
    """Return float64 pred (requires_grad) + target + groups for gradcheck."""
    torch.manual_seed(0)
    pred = torch.randn(B, C, H, W, dtype=torch.float64, requires_grad=True)
    target = torch.randn(B, C, H, W, dtype=torch.float64)
    groups = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    return pred, target, groups


# ---------------------------------------------------------------------------
# Test 1: Computation graph intact after forward pass
# ---------------------------------------------------------------------------

class TestComputationGraph:

    def test_total_loss_requires_grad(self):
        """total_loss.requires_grad must be True."""
        loss_fn = _make_loss_fn()
        pred = torch.randn(6, 4, 4, 4, requires_grad=True)
        target = torch.randn(6, 4, 4, 4)
        groups = torch.tensor([0, 0, 0, 1, 1, 1])
        out = loss_fn(pred, target, groups)
        assert out.total_loss.requires_grad, "total_loss.requires_grad is False — graph broken"

    def test_total_loss_has_grad_fn(self):
        """total_loss must have a grad_fn (is not a leaf)."""
        loss_fn = _make_loss_fn()
        pred = torch.randn(6, 4, 4, 4, requires_grad=True)
        target = torch.randn(6, 4, 4, 4)
        groups = torch.tensor([0, 0, 0, 1, 1, 1])
        out = loss_fn(pred, target, groups)
        assert out.total_loss.grad_fn is not None, "total_loss.grad_fn is None — leaf tensor"

    def test_backward_no_error(self):
        """loss.backward() must not raise."""
        loss_fn = _make_loss_fn()
        pred = torch.randn(6, 4, 4, 4, requires_grad=True)
        target = torch.randn(6, 4, 4, 4)
        groups = torch.tensor([0, 0, 0, 1, 1, 1])
        out = loss_fn(pred, target, groups)
        out.total_loss.backward()  # must not raise

    def test_pred_grad_nonzero_both_groups(self):
        """pred.grad must be nonzero for samples in both groups after backward."""
        loss_fn = _make_loss_fn()
        pred = torch.randn(6, 4, 4, 4, requires_grad=True)
        target = torch.zeros(6, 4, 4, 4)
        groups = torch.tensor([0, 0, 0, 1, 1, 1])
        out = loss_fn(pred, target, groups)
        out.total_loss.backward()
        assert pred.grad is not None, "pred.grad is None after backward"
        grad_g0 = pred.grad[:3].abs().sum()
        grad_g1 = pred.grad[3:].abs().sum()
        assert grad_g0 > 0, "No gradient for group 0 samples"
        assert grad_g1 > 0, "No gradient for group 1 samples"

    def test_groups_none_still_differentiable(self):
        """groups=None (single-group fallback) must also be differentiable."""
        loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0], gumbel_temp=0.5)
        pred = torch.randn(4, 4, 4, 4, requires_grad=True)
        target = torch.randn(4, 4, 4, 4)
        out = loss_fn(pred, target, groups=None)
        assert out.total_loss.requires_grad
        out.total_loss.backward()
        assert pred.grad is not None and pred.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Test 2: Numerical gradcheck
# ---------------------------------------------------------------------------

class TestNumericalGradcheck:

    def test_gradcheck_two_groups(self):
        """gradcheck must pass for the standard two-group configuration.

        gumbel_softmax adds random Gumbel noise each call, so we seed inside
        the closure to make it deterministic for numerical Jacobian estimation.
        """
        loss_fn = _make_loss_fn(outer_tilt=1.0, group_tilts=[1.0, 2.0], gumbel_temp=0.5)
        pred, target, groups = _double_inputs(B=6, C=2, H=3, W=3)
        target = target.detach()
        groups = groups.detach()

        def fn(p):
            torch.manual_seed(0)  # fix Gumbel noise for numerical Jacobian
            return loss_fn(p, target, groups).total_loss

        assert gradcheck(fn, (pred,), eps=1e-5, atol=1e-4, rtol=1e-3, raise_exception=True)

    def test_gradcheck_single_group_fallback(self):
        """gradcheck must pass when groups=None (all samples in group 0)."""
        loss_fn = MultiObjectiveTiltedLoss(
            outer_tilt=1.0, group_tilts=[1.0], gumbel_temp=0.5
        )
        torch.manual_seed(1)
        pred = torch.randn(4, 2, 3, 3, dtype=torch.float64, requires_grad=True)
        target = torch.randn(4, 2, 3, 3, dtype=torch.float64)

        def fn(p):
            torch.manual_seed(0)
            return loss_fn(p, target, groups=None).total_loss

        assert gradcheck(fn, (pred,), eps=1e-5, atol=1e-4, rtol=1e-3, raise_exception=True)

    def test_gradcheck_high_tilt(self):
        """gradcheck must pass even at high tilt values (potential overflow risk)."""
        loss_fn = _make_loss_fn(outer_tilt=5.0, group_tilts=[5.0, 5.0], gumbel_temp=0.1)
        pred, target, groups = _double_inputs(B=4, C=2, H=3, W=3)

        def fn(p):
            torch.manual_seed(0)
            return loss_fn(p, target, groups[:4]).total_loss

        assert gradcheck(fn, (pred,), eps=1e-5, atol=1e-3, rtol=1e-2, raise_exception=True)


# ---------------------------------------------------------------------------
# Test 3: Gumbel weights properties
# ---------------------------------------------------------------------------

class TestGumbelWeights:

    def test_weights_sum_to_one(self):
        """Gumbel-Softmax weights must sum to 1 (soft simplex)."""
        loss_fn = _make_loss_fn()
        pred = torch.randn(6, 4, 4, 4, requires_grad=True)
        target = torch.randn(6, 4, 4, 4)
        groups = torch.tensor([0, 0, 0, 1, 1, 1])
        out = loss_fn(pred, target, groups)
        assert out.weights is not None
        assert torch.allclose(out.weights.sum(), torch.tensor(1.0), atol=1e-5), (
            f"Weights sum to {out.weights.sum().item():.6f}, expected 1.0"
        )

    def test_weights_nonnegative(self):
        """Gumbel-Softmax weights must be non-negative."""
        loss_fn = _make_loss_fn()
        pred = torch.randn(6, 4, 4, 4)
        target = torch.randn(6, 4, 4, 4)
        groups = torch.tensor([0, 0, 0, 1, 1, 1])
        out = loss_fn(pred, target, groups)
        assert (out.weights >= 0).all(), "Negative weights found in Gumbel-Softmax output"

    def test_weights_shape(self):
        """Weights shape must equal number of groups G."""
        G = 3
        loss_fn = MultiObjectiveTiltedLoss(
            outer_tilt=1.0, group_tilts=[1.0, 2.0, 3.0], gumbel_temp=1.0
        )
        pred = torch.randn(9, 4, 4, 4)
        target = torch.randn(9, 4, 4, 4)
        groups = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        out = loss_fn(pred, target, groups)
        assert out.weights.shape == (G,), f"Expected weights shape ({G},), got {out.weights.shape}"

    def test_low_temperature_concentrates_weights(self):
        """At very low Gumbel temperature, weights should concentrate on one group."""
        loss_fn = MultiObjectiveTiltedLoss(
            outer_tilt=2.0, group_tilts=[2.0, 2.0], gumbel_temp=0.01
        )
        # Make group 1 have much higher loss by setting target far from pred
        pred = torch.zeros(6, 4, 4, 4)
        target = torch.zeros(6, 4, 4, 4)
        target[3:] = 10.0  # group 1 has large error
        groups = torch.tensor([0, 0, 0, 1, 1, 1])
        out = loss_fn(pred, target, groups)
        max_weight = out.weights.max().item()
        # With low temp and very unequal risks, the dominant group should get > 0.9 weight
        assert max_weight > 0.9, (
            f"Expected concentrated weight > 0.9 at low temperature, got max={max_weight:.4f}"
        )
