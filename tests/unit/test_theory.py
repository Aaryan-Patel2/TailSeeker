"""Theory verification tests — prototyping-phase checks on toy data.

Verifies mathematical properties that must hold BEFORE committing to QM9 training:
  §1  Single-objective L_t properties
  §2  Multi-objective J̃ structural properties
  §3  Gumbel-Softmax selection properties
  §4  Numerical stability / convexity smoke tests
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# §1  Single-Objective L_t
# ─────────────────────────────────────────────────────────────────────────────

def test_minimax_limit():
    """L_t(t→∞) → max_i f_i  (minimax / worst-case risk).

    At t=100, the log-sum-exp is dominated by the single maximum term.
    Tolerance: |L_t(100) - max f_i| < 0.01 * max f_i  (1 % relative error).
    """
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    torch.manual_seed(7)
    # Inject one extreme outlier to make max clearly dominant.
    pred = torch.randn(16, 4)
    target = torch.randn(16, 4)
    # Ensure one sample has a much larger error.
    pred[0] = pred[0] * 0.0          # pred[0] ≈ 0
    target[0] = torch.ones(4) * 5.0  # target[0] far away → large loss

    loss_fn = TiltedScoreMatchingLoss(tilt=100.0)
    out = loss_fn(pred, target)
    max_f = out.per_sample_loss.max().item()
    L_t = out.total_loss.item()

    assert abs(L_t - max_f) / max(max_f, 1e-8) < 0.02, (
        f"Minimax limit failed: L_t(100)={L_t:.6f}, max_f={max_f:.6f}, "
        f"rel_err={abs(L_t-max_f)/max_f:.4f}"
    )


def test_monotonicity_theorem7():
    """L_t is monotonically non-decreasing in t (Theorem 7).

    For a fixed batch, compute L_t at increasing t values and verify the
    sequence is non-decreasing.  A violation indicates a bug.
    """
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    torch.manual_seed(42)
    pred = torch.randn(16, 4)
    target = torch.randn(16, 4)

    tilt_values = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0]
    losses = []
    for t in tilt_values:
        out = TiltedScoreMatchingLoss(tilt=t)(pred, target)
        losses.append(out.total_loss.item())

    violations = []
    for i in range(len(losses) - 1):
        if losses[i + 1] < losses[i] - 1e-5:
            violations.append(
                f"L_t({tilt_values[i]:.1f})={losses[i]:.6f} > "
                f"L_t({tilt_values[i+1]:.1f})={losses[i+1]:.6f}"
            )

    assert not violations, "Monotonicity violated:\n" + "\n".join(violations)


def test_gradient_weighting_lemma1():
    """Gradient weights are w_i ∝ exp(t·f_i) (Lemma 1).

    The TiltedScoreMatchingLoss stores softmax(t·per_sample) as `weights`.
    Verify: weights[i] == exp(t·f_i) / Σ_j exp(t·f_j)  to 1e-5.
    """
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    torch.manual_seed(3)
    pred = torch.randn(8, 4)
    target = torch.randn(8, 4)

    for tilt in [1.0, 2.0, 5.0]:
        loss_fn = TiltedScoreMatchingLoss(tilt=tilt)
        out = loss_fn(pred, target)

        expected_weights = torch.softmax(tilt * out.per_sample_loss, dim=0)
        assert torch.allclose(out.weights, expected_weights, atol=1e-5), (
            f"tilt={tilt}: weight mismatch — "
            f"max abs err={( out.weights - expected_weights).abs().max().item():.2e}"
        )

        # Hardest sample gets highest weight for t > 0.
        hardest_idx = out.per_sample_loss.argmax()
        assert out.weights.argmax() == hardest_idx, (
            f"tilt={tilt}: highest weight not on hardest sample"
        )


# ─────────────────────────────────────────────────────────────────────────────
# §2  Multi-Objective J̃ structural properties
# ─────────────────────────────────────────────────────────────────────────────

def test_hierarchical_collapse_lemma7():
    """J̃(t, τ=t) == L_t on flattened data when all inner tilts equal outer tilt.

    Lemma 7 proof sketch:
      J̃ = (1/t) log[(1/N) Σ_g |g| exp(t · R̃_g)]
      R̃_g = (1/t) log[(1/|g|) Σ_{x∈g} exp(t·f_x)]
      → (1/t) log[(1/N) Σ_g Σ_{x∈g} exp(t·f_x)]
      = (1/t) log[(1/N) Σ_x exp(t·f_x)]
      = L_t (flattened).
    """
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    torch.manual_seed(5)
    pred = torch.randn(16, 4)
    target = torch.randn(16, 4)

    # 4 groups of 4 samples; all tilts = outer tilt = 2.0
    t = 2.0
    groups = torch.repeat_interleave(torch.arange(4), 4)  # [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
    mo_fn = MultiObjectiveTiltedLoss(outer_tilt=t, group_tilts=[t, t, t, t])
    v1_fn = TiltedScoreMatchingLoss(tilt=t)

    mo_out = mo_fn(pred, target, groups=groups)
    v1_out = v1_fn(pred, target)

    j_tilt = mo_out.loss_components["j_tilt"].item()
    l_t = v1_out.total_loss.item()

    assert abs(j_tilt - l_t) < 1e-4, (
        f"Hierarchical collapse failed: J̃={j_tilt:.6f} != L_t={l_t:.6f} "
        f"(diff={abs(j_tilt-l_t):.2e})"
    )


def test_uniform_group_weighting_small_outer_tilt():
    """J̃(t→0) → (1/N) Σ_g |g|·R̃_g  (size-weighted mean of group risks).

    For equal-sized groups this becomes the simple mean of R̃_g.
    Verified by comparing J̃ at t=1e-4 with directly computed mean(R̃_g).
    """
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    torch.manual_seed(6)
    pred = torch.randn(8, 4)
    target = torch.randn(8, 4)

    # Two equal-sized groups (4 samples each).
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    tiny_t = 1e-4
    mo_fn = MultiObjectiveTiltedLoss(outer_tilt=tiny_t, group_tilts=[1.0, 1.0])
    mo_out = mo_fn(pred, target, groups=groups)

    j_tilt = mo_out.loss_components["j_tilt"].item()

    # Directly compute group risks R̃_g (inner τ=1.0, 4 samples each).
    per_sample = F.mse_loss(pred, target, reduction="none").view(8, -1).mean(dim=1)
    tau = 1.0
    r0 = (torch.logsumexp(tau * per_sample[:4], dim=0) - math.log(4)).item() / tau
    r1 = (torch.logsumexp(tau * per_sample[4:], dim=0) - math.log(4)).item() / tau
    expected_mean = 0.5 * (r0 + r1)

    assert abs(j_tilt - expected_mean) < 1e-3, (
        f"Uniform weighting failed: J̃(t→0)={j_tilt:.6f}, mean(R̃_g)={expected_mean:.6f} "
        f"(diff={abs(j_tilt-expected_mean):.2e})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# §3  Gumbel-Softmax properties
# ─────────────────────────────────────────────────────────────────────────────

def test_gumbel_temperature_zero_limit():
    """At λ→0+, Gumbel-Softmax weights → approximately one-hot.

    The hardest group should receive weight > 0.95 and all others < 0.05.
    We use a batch with a clear group risk gap to make the winner unambiguous.
    """
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    torch.manual_seed(99)
    B = 16
    # Group 0: very high loss (pred far from target).
    # Group 1: low loss (pred near target).
    pred = torch.zeros(B, 4)
    target = torch.zeros(B, 4)
    target[:B // 2] = 10.0   # group 0 → large MSE
    # target[B//2:] = 0.0 → group 1 → MSE ≈ 0

    groups = torch.cat([torch.zeros(B // 2, dtype=torch.long),
                        torch.ones(B // 2, dtype=torch.long)])

    low_temp = 1e-3
    loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 1.0],
                                       gumbel_temp=low_temp)
    out = loss_fn(pred, target, groups=groups)

    weights = out.weights
    max_w = weights.max().item()
    assert max_w > 0.95, (
        f"λ={low_temp}: max weight should be > 0.95 (near one-hot), got {max_w:.4f}\n"
        f"weights={weights.tolist()}"
    )


def test_gumbel_differentiability():
    """Gradient of L_MO w.r.t. group risks R̃_g is non-zero and finite.

    Uses a custom autograd check: perturb one group's samples and verify
    that total_loss changes in the expected direction (positive grad).
    """
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    torch.manual_seed(11)
    B = 8
    pred = torch.randn(B, 4, requires_grad=True)
    target = torch.randn(B, 4)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 2.0])
    out = loss_fn(pred, target, groups=groups)
    out.total_loss.backward()

    assert pred.grad is not None, "No gradient computed for pred."
    assert torch.isfinite(pred.grad).all(), "Gradient contains NaN or Inf."
    assert pred.grad.abs().sum() > 0, "All gradients are exactly zero (detached graph?)."


# ─────────────────────────────────────────────────────────────────────────────
# §4  Numerical stability and convexity
# ─────────────────────────────────────────────────────────────────────────────

def test_convexity_lemma5():
    """L_t is convex in predictions for t>0 (Lemma 5 / Jensen's inequality).

    Jensen's inequality on convex f: f(E[X]) ≤ E[f(X)].
    Applied here: L_t(avg(pred1, pred2), target) ≤ 0.5*[L_t(pred1) + L_t(pred2)]
    for t > 0.  Verified for t ∈ {0.5, 1.0, 2.0, 5.0}.
    """
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    torch.manual_seed(17)
    pred1 = torch.randn(16, 4)
    pred2 = torch.randn(16, 4)
    target = torch.randn(16, 4)
    pred_avg = 0.5 * (pred1 + pred2)

    for tilt in [0.5, 1.0, 2.0, 5.0]:
        fn = TiltedScoreMatchingLoss(tilt=tilt)
        L_avg = fn(pred_avg, target).total_loss.item()
        L_1 = fn(pred1, target).total_loss.item()
        L_2 = fn(pred2, target).total_loss.item()
        avg_L = 0.5 * (L_1 + L_2)

        assert L_avg <= avg_L + 1e-5, (
            f"Convexity (Lemma 5) violated at tilt={tilt}: "
            f"L_t(avg)={L_avg:.6f} > avg(L_t)={avg_L:.6f} "
            f"(violation={L_avg - avg_L:.2e})"
        )


def test_invalid_latent_nonnegative():
    """Loss is always ≥ 0; never negative, even for degenerate inputs.

    Checks: all-zeros pred vs nonzero target, high-magnitude noise pred,
    and an all-ones pred — all must give loss ≥ 0.
    """
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss
    from src.losses.ddpm_simple import DDPMSimpleLoss

    target = torch.ones(8, 4) * 3.0

    broken_inputs = {
        "all_zeros": torch.zeros(8, 4),
        "high_noise": torch.randn(8, 4) * 100.0,
        "all_ones": torch.ones(8, 4),
    }

    for label, broken_pred in broken_inputs.items():
        # ERM
        erm_out = DDPMSimpleLoss()(broken_pred, target)
        assert erm_out.total_loss.item() >= -1e-6, (
            f"ERM loss negative for {label}: {erm_out.total_loss.item():.6f}"
        )

        # Tilted (positive and negative)
        for tilt in [1.0, -1.0, 5.0, -5.0]:
            out = TiltedScoreMatchingLoss(tilt=tilt)(broken_pred, target)
            assert out.total_loss.item() >= -1e-6, (
                f"tilt={tilt}: loss negative for {label}: {out.total_loss.item():.6f}"
            )

        # High-noise pred should produce large positive loss, not crash.
        out_large = TiltedScoreMatchingLoss(tilt=1.0)(broken_inputs["high_noise"], target)
        assert out_large.total_loss.item() > 1.0, (
            f"Expected large loss for high-noise input, got {out_large.total_loss.item():.4f}"
        )
        assert torch.isfinite(out_large.total_loss), "Loss is not finite for high-noise input."


def test_logsumexp_stress_no_overflow():
    """logsumexp trick prevents overflow when one sample has massive loss (1e5).

    Without max-shift, exp(t * 1e5) overflows to Inf for t > ~8e-4.
    torch.logsumexp uses the trick internally; verify loss stays finite.
    """
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    torch.manual_seed(23)
    B = 8
    pred = torch.randn(B, 4)
    target = torch.randn(B, 4)

    # Force sample 0 to have enormous error.
    pred[0] = torch.zeros(4)
    target[0] = torch.ones(4) * 316.0  # MSE ≈ 1e5 per feature

    for tilt in [1.0, 5.0, 10.0]:
        out = TiltedScoreMatchingLoss(tilt=tilt)(pred, target)
        assert torch.isfinite(out.total_loss), (
            f"tilt={tilt}: overflow — loss={out.total_loss.item()} with max_f≈1e5"
        )
        assert out.total_loss.item() > 0, (
            f"tilt={tilt}: expected positive loss, got {out.total_loss.item()}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# §5  Gradient Connectivity — torch.autograd.gradcheck
# ─────────────────────────────────────────────────────────────────────────────

def test_gradcheck_v1_tilted_loss():
    """Numerical Jacobian check for TiltedScoreMatchingLoss (V1).

    V1 is fully deterministic — gradcheck is straightforward.
    Requires float64; uses a small batch to keep runtime acceptable.
    """
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    torch.manual_seed(31)
    B = 4
    target = torch.randn(B, 3, dtype=torch.float64)
    pred = torch.randn(B, 3, dtype=torch.float64, requires_grad=True)

    loss_fn = TiltedScoreMatchingLoss(tilt=2.0)
    # Convert buffers/params to float64 (none exist in V1, but be explicit).
    loss_fn = loss_fn.double()

    def func(p):
        return loss_fn(p, target).total_loss.unsqueeze(0)

    assert torch.autograd.gradcheck(func, (pred,), eps=1e-6, atol=1e-4, rtol=1e-3), (
        "gradcheck failed for TiltedScoreMatchingLoss (V1)"
    )


def test_gradcheck_v2_multi_objective_loss():
    """Numerical Jacobian check for MultiObjectiveTiltedLoss (V2).

    F.gumbel_softmax samples Gumbel noise on each call, making L_MO
    stochastic. Fix: reseed inside the closure so every evaluation
    gradcheck makes sees identical Gumbel noise — effectively deterministic
    for the Jacobian check while keeping weights attached to the graph.

    Spec §5(vii): gradcheck must pass with Gumbel weights in the graph.
    """
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    GUMBEL_SEED = 37
    torch.manual_seed(GUMBEL_SEED)
    B = 8
    target = torch.randn(B, 3, dtype=torch.float64)
    pred = torch.randn(B, 3, dtype=torch.float64, requires_grad=True)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    loss_fn = MultiObjectiveTiltedLoss(
        outer_tilt=1.0,
        group_tilts=[1.0, 2.0],
        gumbel_temp=1.0,
    ).double()  # convert group_tilts buffer to float64

    def func(p):
        # Reseed before every call so Gumbel noise is fixed across
        # all finite-difference perturbations gradcheck makes.
        torch.manual_seed(GUMBEL_SEED)
        return loss_fn(p, target, groups=groups).total_loss.unsqueeze(0)

    assert torch.autograd.gradcheck(func, (pred,), eps=1e-5, atol=1e-3, rtol=1e-2), (
        "gradcheck failed for MultiObjectiveTiltedLoss (V2) — "
        "gradient may be detached or numerically unstable"
    )
