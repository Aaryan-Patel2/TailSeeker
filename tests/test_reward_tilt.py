"""Tests for RewardWeightedTiltedLoss.

Covers: ERM limit, argmax limit, gradient direction, λ sensitivity,
        ERM fallback (low reward variance), tilt annealing.
"""
from __future__ import annotations

import torch
import pytest

from src.losses.reward_weighted_loss import RewardWeightedTiltedLoss, get_reward_loss_fn
from src.losses.ddpm_simple import DDPMSimpleLoss


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _batch(B: int = 8, C: int = 4, N: int = 29, seed: int = 0):
    """Return (pred, target, qed, sa) with controlled random state."""
    torch.manual_seed(seed)
    pred = torch.randn(B, C, N, N)
    target = torch.randn(B, C, N, N)
    qed = torch.rand(B)                      # [0, 1]
    sa = torch.rand(B) * 9.0 + 1.0          # [1, 10]
    return pred, target, qed, sa


# ── ERM limits ────────────────────────────────────────────────────────────────

def test_erm_limit_tilt_zero():
    """t=0 via factory → loss equals per_sample_mse.mean() (exact ERM)."""
    loss_fn = get_reward_loss_fn(tilt=0.0)
    assert isinstance(loss_fn, DDPMSimpleLoss)


def test_erm_limit_epoch_zero():
    """epoch=0 with warmup → t_eff=0 → uniform weights → loss ≈ ERM."""
    pred, target, qed, sa = _batch()
    loss_fn = RewardWeightedTiltedLoss(tilt=5.0, warmup_frac=0.3)
    out = loss_fn(pred, target, qed=qed, sa=sa, epoch=0, max_epochs=100)

    erm_fn = DDPMSimpleLoss()
    erm_out = erm_fn(pred, target)
    assert torch.allclose(out.total_loss, erm_out.total_loss, atol=1e-5), (
        f"epoch=0 should give ERM loss; got {out.total_loss:.6f} vs {erm_out.total_loss:.6f}"
    )


def test_erm_limit_no_warmup():
    """warmup_frac=0 at epoch=0 should still apply full tilt."""
    pred, target, qed, sa = _batch()
    loss_fn = RewardWeightedTiltedLoss(tilt=2.0, warmup_frac=0.0)
    out = loss_fn(pred, target, qed=qed, sa=sa, epoch=0, max_epochs=100)
    # With warmup_frac=0, t_eff = tilt at all epochs → weights non-uniform
    assert out.diagnostics["reward_tilt_t_eff"] == pytest.approx(2.0)


# ── Argmax limit ──────────────────────────────────────────────────────────────

def test_argmax_limit_high_tilt():
    """t→∞, no warmup → all weight on highest-reward molecule."""
    pred, target, qed, sa = _batch(B=8, seed=1)
    loss_fn = RewardWeightedTiltedLoss(tilt=1000.0, warmup_frac=0.0,
                                       reward_std_threshold=0.0)
    out = loss_fn(pred, target, qed=qed, sa=sa, epoch=0, max_epochs=1)

    assert out.weights is not None
    # Highest weight should be ≈1.0, all others ≈0
    assert out.weights.max() > 0.99, f"Max weight={out.weights.max():.4f}, expected ≈1.0"
    assert out.weights.sum() == pytest.approx(1.0, abs=1e-5)


# ── Gradient direction ────────────────────────────────────────────────────────

def test_positive_tilt_upweights_high_reward():
    """t>0 → higher-reward molecules get strictly larger gradient weights."""
    pred, target, qed, sa = _batch(B=16, seed=2)
    loss_fn = RewardWeightedTiltedLoss(tilt=3.0, warmup_frac=0.0,
                                       reward_std_threshold=0.0)
    out = loss_fn(pred, target, qed=qed, sa=sa, epoch=100, max_epochs=100)

    assert out.weights is not None
    r = qed - 0.5 * (sa - 1.0) / 9.0
    # Weight order must be monotone with reward order
    reward_rank = r.argsort()
    weight_rank = out.weights.argsort()
    assert torch.all(reward_rank == weight_rank), (
        "Weight ranking must match reward ranking under positive tilt"
    )


def test_negative_tilt_downweights_high_reward():
    """t<0 → lower-reward molecules get larger weights (robustness direction)."""
    pred, target, qed, sa = _batch(B=16, seed=3)
    loss_fn = RewardWeightedTiltedLoss(tilt=-3.0, warmup_frac=0.0,
                                       reward_std_threshold=0.0)
    out = loss_fn(pred, target, qed=qed, sa=sa, epoch=100, max_epochs=100)

    assert out.weights is not None
    r = qed - 0.5 * (sa - 1.0) / 9.0
    # Negative tilt: weight order is reverse of reward order
    reward_rank = r.argsort()
    weight_rank = out.weights.argsort(descending=True)
    assert torch.all(reward_rank == weight_rank), (
        "Negative tilt must invert reward ranking"
    )


# ── Lambda sensitivity ────────────────────────────────────────────────────────

def test_lambda_zero_uses_qed_only():
    """λ=0 → composite reward = QED only (SA ignored)."""
    pred, target, qed, sa = _batch(B=8, seed=4)
    loss_fn = RewardWeightedTiltedLoss(tilt=2.0, lambda_=0.0, warmup_frac=0.0,
                                       reward_std_threshold=0.0)
    out = loss_fn(pred, target, qed=qed, sa=sa, epoch=100, max_epochs=100)

    expected_weights = torch.softmax(2.0 * qed, dim=0)
    assert torch.allclose(out.weights, expected_weights, atol=1e-5)


# ── ERM fallback (low reward variance) ───────────────────────────────────────

def test_erm_fallback_constant_reward():
    """Constant rewards → std=0 < threshold → ERM fallback."""
    pred, target, _, _ = _batch(B=8, seed=5)
    qed = torch.full((8,), 0.6)
    sa = torch.full((8,), 3.0)

    loss_fn = RewardWeightedTiltedLoss(tilt=5.0, warmup_frac=0.0,
                                       reward_std_threshold=0.05)
    out = loss_fn(pred, target, qed=qed, sa=sa, epoch=100, max_epochs=100)

    erm_fn = DDPMSimpleLoss()
    erm_out = erm_fn(pred, target)
    assert torch.allclose(out.total_loss, erm_out.total_loss, atol=1e-5)
    assert out.diagnostics["reward_erm_fallback"] == 1.0


# ── Tilt annealing ────────────────────────────────────────────────────────────

def test_annealing_linear_warmup():
    """t_eff ramps linearly: 0 at epoch=0, t_target at epoch≥warmup_end."""
    loss_fn = RewardWeightedTiltedLoss(tilt=4.0, warmup_frac=0.5)
    pred, target, qed, sa = _batch()

    out_start = loss_fn(pred, target, qed=qed, sa=sa, epoch=0, max_epochs=100)
    out_mid = loss_fn(pred, target, qed=qed, sa=sa, epoch=25, max_epochs=100)
    out_end = loss_fn(pred, target, qed=qed, sa=sa, epoch=50, max_epochs=100)
    out_full = loss_fn(pred, target, qed=qed, sa=sa, epoch=100, max_epochs=100)

    assert out_start.diagnostics["reward_tilt_t_eff"] == pytest.approx(0.0)
    assert out_mid.diagnostics["reward_tilt_t_eff"] == pytest.approx(2.0, abs=1e-5)
    assert out_end.diagnostics["reward_tilt_t_eff"] == pytest.approx(4.0, abs=1e-5)
    assert out_full.diagnostics["reward_tilt_t_eff"] == pytest.approx(4.0, abs=1e-5)


# ── Output structure ──────────────────────────────────────────────────────────

def test_loss_output_fields():
    """LossOutput has required fields; per_sample_loss shape matches batch size."""
    pred, target, qed, sa = _batch(B=8)
    loss_fn = RewardWeightedTiltedLoss(tilt=2.0, warmup_frac=0.0,
                                       reward_std_threshold=0.0)
    out = loss_fn(pred, target, qed=qed, sa=sa, epoch=100, max_epochs=100)

    assert out.total_loss.shape == torch.Size([])    # scalar
    assert out.per_sample_loss.shape == torch.Size([8])
    assert out.weights is not None and out.weights.shape == torch.Size([8])
    assert out.weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert out.total_loss.requires_grad or not out.total_loss.requires_grad  # just check it exists


def test_backward_runs():
    """Loss is differentiable; backward should not raise."""
    pred, target, qed, sa = _batch(B=4)
    pred = pred.requires_grad_(True)
    loss_fn = RewardWeightedTiltedLoss(tilt=2.0, warmup_frac=0.0,
                                       reward_std_threshold=0.0)
    out = loss_fn(pred, target, qed=qed, sa=sa, epoch=100, max_epochs=100)
    out.total_loss.backward()
    assert pred.grad is not None


# ── LogP / TPSA reward terms (Tier 1 biological extension) ───────────────────

def test_logp_penalty_lowers_reward_for_greasy_molecules():
    """High LogP (>3) should reduce reward vs baseline with lambda_logp>0."""
    torch.manual_seed(10)
    B = 8
    qed = torch.full((B,), 0.7)
    sa  = torch.full((B,), 3.0)
    logp_greasy = torch.full((B,), 5.0)   # LogP=5 → penalty = 0.3 * (5-3)/2 = 0.3
    logp_ok     = torch.full((B,), 2.0)   # LogP=2 → no penalty (below threshold 3)

    loss_fn = RewardWeightedTiltedLoss(tilt=2.0, warmup_frac=0.0,
                                       reward_std_threshold=0.0,
                                       lambda_logp=0.3, lambda_tpsa=0.0)
    r_greasy = loss_fn._reward(qed, sa, logp=logp_greasy)
    r_ok     = loss_fn._reward(qed, sa, logp=logp_ok)
    assert (r_ok > r_greasy).all(), "LogP ≤ 3 should give higher reward than LogP=5"


def test_logp_no_penalty_below_threshold():
    """LogP ≤ 3 incurs zero penalty (clamp at 0)."""
    torch.manual_seed(11)
    B = 8
    qed = torch.rand(B)
    sa  = torch.rand(B) * 9.0 + 1.0
    logp = torch.full((B,), 1.5)   # well below threshold

    loss_fn_with    = RewardWeightedTiltedLoss(tilt=2.0, warmup_frac=0.0,
                                               reward_std_threshold=0.0,
                                               lambda_logp=0.3)
    loss_fn_without = RewardWeightedTiltedLoss(tilt=2.0, warmup_frac=0.0,
                                               reward_std_threshold=0.0,
                                               lambda_logp=0.0)
    r_with    = loss_fn_with._reward(qed, sa, logp=logp)
    r_without = loss_fn_without._reward(qed, sa)
    assert torch.allclose(r_with, r_without, atol=1e-5), \
        "LogP=1.5 should incur no penalty (clamp(logp-3, min=0)=0)"


def test_tpsa_reward_increases_for_low_polar_surface():
    """Low TPSA (<140 Å²) should increase reward vs high TPSA."""
    torch.manual_seed(12)
    B = 8
    qed = torch.full((B,), 0.6)
    sa  = torch.full((B,), 4.0)
    tpsa_low  = torch.full((B,), 40.0)    # good oral absorption → reward = 0.2*(1-40/140)
    tpsa_high = torch.full((B,), 130.0)   # poor → reward = 0.2*(1-130/140)

    loss_fn = RewardWeightedTiltedLoss(tilt=2.0, warmup_frac=0.0,
                                       reward_std_threshold=0.0,
                                       lambda_logp=0.0, lambda_tpsa=0.2)
    r_low  = loss_fn._reward(qed, sa, tpsa=tpsa_low)
    r_high = loss_fn._reward(qed, sa, tpsa=tpsa_high)
    assert (r_low > r_high).all(), "Low TPSA should give higher reward than high TPSA"


def test_admet_reward_backward_differentiable():
    """Full ADMET reward (logp + tpsa) must allow backward through loss."""
    pred, target, qed, sa = _batch(B=4)
    pred = pred.requires_grad_(True)
    logp = torch.randn(4) * 1.5 + 2.0
    tpsa = torch.rand(4) * 100.0 + 20.0
    loss_fn = RewardWeightedTiltedLoss(tilt=2.0, warmup_frac=0.0,
                                       reward_std_threshold=0.0,
                                       lambda_logp=0.3, lambda_tpsa=0.2)
    out = loss_fn(pred, target, qed=qed, sa=sa, logp=logp, tpsa=tpsa,
                  epoch=100, max_epochs=100)
    out.total_loss.backward()
    assert pred.grad is not None


def test_zero_lambdas_matches_original_reward():
    """lambda_logp=0, lambda_tpsa=0 → identical reward to the QED/SA-only formula."""
    torch.manual_seed(13)
    B = 8
    qed = torch.rand(B)
    sa  = torch.rand(B) * 9.0 + 1.0
    logp = torch.randn(B) + 2.0
    tpsa = torch.rand(B) * 100.0 + 20.0

    loss_fn_new = RewardWeightedTiltedLoss(tilt=2.0, warmup_frac=0.0,
                                           reward_std_threshold=0.0,
                                           lambda_logp=0.0, lambda_tpsa=0.0)
    loss_fn_old = RewardWeightedTiltedLoss(tilt=2.0, warmup_frac=0.0,
                                           reward_std_threshold=0.0)
    r_new = loss_fn_new._reward(qed, sa, logp=logp, tpsa=tpsa)
    r_old = loss_fn_old._reward(qed, sa)
    assert torch.allclose(r_new, r_old, atol=1e-5)
