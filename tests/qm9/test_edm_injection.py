"""EDM loss-injection tests — validates the single-line swap at EDM's reduction point.

EDM training loop (ehoogeboom/e3_diffusion_for_molecules, qm9/losses.py:31):

    nll = generative_model(x, h, node_mask, edge_mask, context)  # (B,) per-molecule NLL
    nll = nll.mean(0)   ← ORIGINAL EDM REDUCTION — replace with term_aggregate

    # V1 injection:
    from src.losses import term_aggregate
    nll = term_aggregate(nll, tilt=args.tilt)

    # V2 injection (multi-objective with property groups):
    from src.losses import MultiObjectiveTiltedLoss
    loss_fn = MultiObjectiveTiltedLoss(outer_tilt=t, group_tilts=[tau_qed, tau_sa])
    # (per-molecule NLL already computed; use as surrogate per_sample input)

NOTE: EDM's generative_model returns (B,) shaped per-molecule NLL from
sum_except_batch (x.reshape(B,-1).sum(-1)), so atom-level aggregation is
already done inside the model — term_aggregate operates on these scalars directly.

These tests verify:
  1. tilt=0 injection is numerically identical to EDM's baseline mean.
  2. Jensen inequality holds after masked aggregation (V1).
  3. All 8 ablation tilt values run without NaN/Inf on EDM-shaped inputs.
  4. V2 hierarchical loss integrates with property groups + node_mask.
  5. Gradients flow from loss back through masked per-molecule MSE to pred.
"""

from __future__ import annotations

import pytest
import torch

# Ablation matrix from CLAUDE.md §Ablation Matrix
ABLATION_TILTS = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]


# ─────────────────────────────────────────────────────────────────────────────
# Sanity: tilt=0 must reproduce EDM's baseline exactly
# ─────────────────────────────────────────────────────────────────────────────

def test_tilt_zero_matches_edm_baseline(edm_batch):
    """get_loss_fn(0.0) with node_mask reproduces EDM's naive mean over valid atoms."""
    from src.losses.tilted_score_matching import get_loss_fn
    from src.losses.base import _per_molecule_mse

    pred, target, node_mask = edm_batch["pred"], edm_batch["target"], edm_batch["node_mask"]

    # Our ERM baseline
    out = get_loss_fn(0.0)(pred, target, node_mask=node_mask)

    # EDM-equivalent: mean of per-molecule masked losses
    per_mol = _per_molecule_mse(pred, target, node_mask)
    edm_equivalent = per_mol.mean()

    assert torch.allclose(out.total_loss, edm_equivalent, atol=1e-5), (
        f"tilt=0 injection diverges from EDM baseline: "
        f"ours={out.total_loss.item():.6f}, edm={edm_equivalent.item():.6f}"
    )


def test_tilt_zero_with_mask_differs_from_naive_mean(edm_batch):
    """Masked ERM != naive mean when molecules have different sizes.

    This confirms the normalization fix is load-bearing — before the fix,
    these two values would differ silently (shorter molecules penalized less).
    """
    from src.losses.tilted_score_matching import get_loss_fn

    pred, target, node_mask = edm_batch["pred"], edm_batch["target"], edm_batch["node_mask"]
    B = pred.shape[0]

    masked_loss = get_loss_fn(0.0)(pred, target, node_mask=node_mask).total_loss
    naive_loss  = get_loss_fn(0.0)(pred, target).total_loss  # no mask

    # For variable-length molecules, these must be different.
    assert not torch.allclose(masked_loss, naive_loss, atol=1e-4), (
        "Masked and unmasked ERM are identical despite variable molecule sizes — "
        "the node_mask has no effect (bug)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# V1: Jensen + ablation sweep
# ─────────────────────────────────────────────────────────────────────────────

def test_v1_jensen_holds_after_masking(edm_batch):
    """L_tilt(t>0) >= masked mean MSE — Jensen must still hold after masking."""
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss
    from src.losses.base import _per_molecule_mse

    pred, target, node_mask = edm_batch["pred"], edm_batch["target"], edm_batch["node_mask"]

    for tilt in [1.0, 2.0, 5.0]:
        out = TiltedScoreMatchingLoss(tilt=tilt)(pred, target, node_mask=node_mask)
        mean_mse = _per_molecule_mse(pred, target, node_mask).mean()

        assert out.total_loss.item() >= mean_mse.item() - 1e-5, (
            f"Jensen violated at tilt={tilt} with node_mask: "
            f"L_tilt={out.total_loss.item():.6f} < mean_mse={mean_mse.item():.6f}"
        )


def test_v1_ablation_sweep_no_nan(edm_batch):
    """All 8 ablation-matrix tilt values produce finite loss on EDM-shaped inputs."""
    from src.losses.tilted_score_matching import get_loss_fn

    pred, target, node_mask = edm_batch["pred"], edm_batch["target"], edm_batch["node_mask"]

    for tilt in ABLATION_TILTS:
        out = get_loss_fn(tilt)(pred, target, node_mask=node_mask)
        assert torch.isfinite(out.total_loss), (
            f"tilt={tilt}: loss is not finite: {out.total_loss.item()}"
        )
        assert out.total_loss.item() >= -1e-6, (
            f"tilt={tilt}: loss is negative: {out.total_loss.item():.6f}"
        )


def test_v1_monotonicity_with_masking(edm_batch):
    """L_tilt is non-decreasing in t even after masked per-molecule aggregation."""
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    pred, target, node_mask = edm_batch["pred"], edm_batch["target"], edm_batch["node_mask"]
    tilt_values = [-5.0, -2.0, -1.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    losses = [
        TiltedScoreMatchingLoss(tilt=t)(pred, target, node_mask=node_mask).total_loss.item()
        for t in tilt_values
    ]

    violations = [
        f"L({tilt_values[i]})={losses[i]:.4f} > L({tilt_values[i+1]})={losses[i+1]:.4f}"
        for i in range(len(losses) - 1)
        if losses[i + 1] < losses[i] - 1e-5
    ]
    assert not violations, "Monotonicity violated after masking:\n" + "\n".join(violations)


# ─────────────────────────────────────────────────────────────────────────────
# V2: Multi-objective with property groups + node_mask
# ─────────────────────────────────────────────────────────────────────────────

def test_v2_with_node_mask_and_groups(edm_batch_with_props):
    """MultiObjectiveTiltedLoss accepts node_mask + groups; loss is finite."""
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    pred      = edm_batch_with_props["pred"]
    target    = edm_batch_with_props["target"]
    node_mask = edm_batch_with_props["node_mask"]
    groups    = edm_batch_with_props["groups"]

    # QED group: τ=+2 (tail-seek); SA group: τ=-1 (robustness)
    loss_fn = MultiObjectiveTiltedLoss(
        outer_tilt=2.0,
        group_tilts=[2.0, -1.0],
        gumbel_temp=1.0,
    )
    out = loss_fn(pred, target, groups=groups, node_mask=node_mask)

    assert torch.isfinite(out.total_loss), (
        f"V2 loss not finite with node_mask: {out.total_loss.item()}"
    )
    assert out.total_loss.ndim == 0, "V2 loss must be a scalar."
    assert out.weights.shape == (2,), f"Expected 2 group weights, got {out.weights.shape}"
    assert torch.allclose(out.weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_v2_gumbel_curriculum_with_masking(edm_batch_with_props):
    """At λ→0, V2 selects the higher-risk group even with masking."""
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    pred      = edm_batch_with_props["pred"]
    target    = edm_batch_with_props["target"]
    node_mask = edm_batch_with_props["node_mask"]
    groups    = edm_batch_with_props["groups"]

    # Make group 0 have very high loss: pred far from target.
    pred_biased          = pred.clone()
    pred_biased[:2, :5] = 100.0   # group 0 molecules, first 5 valid atoms

    loss_fn = MultiObjectiveTiltedLoss(
        outer_tilt=1.0,
        group_tilts=[1.0, 1.0],
        gumbel_temp=1e-3,   # near-zero temp → near one-hot
    )
    out = loss_fn(pred_biased, target, groups=groups, node_mask=node_mask)

    # Group 0 has much higher risk → should dominate the Gumbel weights.
    assert out.weights[0].item() > 0.9, (
        f"Expected group-0 weight > 0.9 at λ=1e-3, got {out.weights.tolist()}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradient flow through masked path
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_flows_through_masked_v1(edm_batch):
    """Gradients reach pred through the masked per-molecule MSE path (V1)."""
    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    pred      = edm_batch["pred"].requires_grad_(True)
    target    = edm_batch["target"]
    node_mask = edm_batch["node_mask"]

    out = TiltedScoreMatchingLoss(tilt=2.0)(pred, target, node_mask=node_mask)
    out.total_loss.backward()

    assert pred.grad is not None, "No gradient reached pred."
    # Padding positions must have zero gradient (mask zeroed them out).
    # Skip molecules that use all atoms (no padding to check).
    for i, n in enumerate(edm_batch["n_atoms"].tolist()):
        pad_slice = pred.grad[i, n:, :]
        if pad_slice.numel() == 0:
            continue  # no padding for this molecule
        pad_grad = pad_slice.abs().max().item()
        assert pad_grad < 1e-6, (
            f"Molecule {i}: padding atom {n}+ has non-zero gradient={pad_grad:.2e} "
            "(gradient leaking through masked positions)"
        )


def test_gradient_flows_through_masked_v2(edm_batch_with_props):
    """Gradients reach pred through the masked per-molecule MSE path (V2)."""
    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    pred      = edm_batch_with_props["pred"].requires_grad_(True)
    target    = edm_batch_with_props["target"]
    node_mask = edm_batch_with_props["node_mask"]
    groups    = edm_batch_with_props["groups"]

    loss_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[2.0, -1.0])
    out = loss_fn(pred, target, groups=groups, node_mask=node_mask)
    out.total_loss.backward()

    assert pred.grad is not None, "No gradient reached pred."
    assert torch.isfinite(pred.grad).all(), "Gradient contains NaN or Inf."
    assert pred.grad.abs().sum() > 0, "All gradients are zero (detached graph?)."


# ─────────────────────────────────────────────────────────────────────────────
# EDM NLL-level injection — term_aggregate operating on (B,) per-molecule NLL
# (ehoogeboom/e3_diffusion_for_molecules, qm9/losses.py:31 drop-in)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_edm_nll(B: int, seed: int = 42) -> torch.Tensor:
    """Simulate (B,) per-molecule NLL output from EDM's generative_model forward.

    EDM's generative_model returns sum_except_batch(error), so per-molecule
    NLL values are positive scalars.  We plant one large outlier to make
    the tail-seeking behaviour visible.
    """
    torch.manual_seed(seed)
    nll = torch.rand(B) * 2.0 + 0.5   # typical range [0.5, 2.5]
    nll[0] = 8.0                       # one hard molecule — TERM should up-weight it
    return nll


def test_term_aggregate_tilt_zero_matches_edm_mean():
    """term_aggregate(nll, 0.0) == nll.mean(0) — bit-exact sanity check against EDM."""
    from src.losses import term_aggregate

    nll = _mock_edm_nll(8)
    edm_result = nll.mean(0)
    our_result  = term_aggregate(nll, tilt=0.0)

    assert torch.allclose(our_result, edm_result, atol=1e-7), (
        f"term_aggregate(tilt=0) diverges from EDM mean: "
        f"ours={our_result.item():.8f}, edm={edm_result.item():.8f}"
    )


def test_term_aggregate_all_ablation_tilts():
    """All 8 ablation tilt values run without NaN/Inf on EDM-style NLL."""
    from src.losses import term_aggregate

    nll = _mock_edm_nll(16)

    for tilt in ABLATION_TILTS:
        result = term_aggregate(nll, tilt=tilt)
        assert torch.isfinite(result), f"tilt={tilt}: term_aggregate returned {result.item()}"
        assert result.ndim == 0, f"tilt={tilt}: expected scalar, got shape {result.shape}"


def test_term_aggregate_monotonicity():
    """term_aggregate is non-decreasing in tilt (Theorem 1 / monotonicity)."""
    from src.losses import term_aggregate

    nll = _mock_edm_nll(16)
    tilt_values = [-5.0, -2.0, -1.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    losses = [term_aggregate(nll, t).item() for t in tilt_values]

    violations = [
        f"L({tilt_values[i]:.1f})={losses[i]:.4f} > L({tilt_values[i+1]:.1f})={losses[i+1]:.4f}"
        for i in range(len(losses) - 1)
        if losses[i + 1] < losses[i] - 1e-5
    ]
    assert not violations, "Monotonicity violated on EDM NLL:\n" + "\n".join(violations)


def test_term_aggregate_up_weights_hard_molecule():
    """At t>0, term_aggregate > nll.mean() — the outlier drives the loss up."""
    from src.losses import term_aggregate

    nll = _mock_edm_nll(16)   # nll[0]=8.0 is the hard molecule

    erm  = nll.mean().item()
    tilt = term_aggregate(nll, tilt=2.0).item()

    assert tilt > erm, (
        f"t=2 tilted loss ({tilt:.4f}) should exceed ERM ({erm:.4f}) "
        "because the outlier molecule dominates log-sum-exp."
    )


def test_term_aggregate_jensen_positive_tilt():
    """term_aggregate(t>0) >= nll.mean() — Jensen's inequality on NLL."""
    from src.losses import term_aggregate

    nll = _mock_edm_nll(16)
    mean_nll = nll.mean().item()

    for tilt in [0.5, 1.0, 2.0, 5.0, 10.0]:
        agg = term_aggregate(nll, tilt).item()
        assert agg >= mean_nll - 1e-5, (
            f"Jensen violated at tilt={tilt}: term_aggregate={agg:.6f} < mean={mean_nll:.6f}"
        )


def test_term_aggregate_gradient_flows():
    """Gradient flows back through term_aggregate to the nll tensor."""
    from src.losses import term_aggregate

    nll = _mock_edm_nll(8).requires_grad_(True)
    loss = term_aggregate(nll, tilt=2.0)
    loss.backward()

    assert nll.grad is not None, "No gradient reached nll."
    assert torch.isfinite(nll.grad).all(), "Gradient contains NaN or Inf."
    # Hard molecule (index 0, nll=8.0) must receive the largest gradient weight.
    assert nll.grad.argmax().item() == 0, (
        f"Hardest molecule (index 0) should have largest gradient weight; "
        f"grad={nll.grad.tolist()}"
    )


def test_term_aggregate_rejects_non_1d():
    """term_aggregate asserts (B,) shape — catches forgetting to aggregate atoms first."""
    from src.losses import term_aggregate

    bad_input = torch.randn(4, 9, 8)   # atom-level, not molecule-level
    with pytest.raises(AssertionError, match="term_aggregate expects"):
        term_aggregate(bad_input, tilt=1.0)
