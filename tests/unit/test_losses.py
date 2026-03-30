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


# ── TiltedScoreMatchingLoss stub ──────────────────────────────────────

@pytest.mark.parametrize("tilt", [0.5, 1.0, 2.0, 5.0, -1.0])
def test_tsm_stub_raises_not_implemented(tilt: float):
    """TiltedScoreMatchingLoss raises NotImplementedError with tilt in message."""
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    loss_fn = TiltedScoreMatchingLoss(tilt=tilt)
    pred = torch.randn(4, 4, 8, 8)
    target = torch.randn(4, 4, 8, 8)
    with pytest.raises(NotImplementedError, match=f"tilt={tilt}"):
        loss_fn(pred, target)


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
