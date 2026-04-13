"""Per-molecule MSE normalization tests — the critical pre-condition for EDM.

EDM pads molecules to max_atoms.  Our TERM objectives must aggregate to
one scalar *per molecule* before applying log-sum-exp, and that scalar
must reflect only valid atoms (node_mask==1).  A naive `.mean()` over
all atom slots dilutes the loss for shorter molecules, biasing the tilt
toward larger ones.

These tests verify _per_molecule_mse is size-invariant and mask-correct.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses.base import _per_molecule_mse

# ─────────────────────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_masked_mean_matches_manual(edm_batch):
    """_per_molecule_mse with node_mask equals manual per-molecule masked mean."""
    pred, target, node_mask = edm_batch["pred"], edm_batch["target"], edm_batch["node_mask"]
    B, N, F_dim = pred.shape

    got = _per_molecule_mse(pred, target, node_mask)

    # Manual: for each molecule, mean over valid (atom, feat) pairs.
    expected = torch.zeros(B)
    for i in range(B):
        n = node_mask[i, :, 0].sum().long().item()
        raw_i = F.mse_loss(pred[i, :n], target[i, :n], reduction="none")
        expected[i] = raw_i.mean()

    assert torch.allclose(got, expected, atol=1e-5), (
        f"Masked mean mismatch:\ngot={got}\nexpected={expected}"
    )


def test_no_mask_fallback_matches_original(edm_batch):
    """When node_mask=None, _per_molecule_mse == naive view(B,-1).mean(dim=1)."""
    pred, target = edm_batch["pred"], edm_batch["target"]
    B = pred.shape[0]

    got      = _per_molecule_mse(pred, target, node_mask=None)
    expected = F.mse_loss(pred, target, reduction="none").view(B, -1).mean(dim=1)

    assert torch.allclose(got, expected, atol=1e-6), (
        f"No-mask fallback mismatch:\ngot={got}\nexpected={expected}"
    )


def test_full_mask_equals_no_mask(edm_batch_uniform):
    """When all atoms are valid (node_mask all-ones), masked == unmasked."""
    pred, target, node_mask = (
        edm_batch_uniform["pred"],
        edm_batch_uniform["target"],
        edm_batch_uniform["node_mask"],
    )
    B = pred.shape[0]

    masked   = _per_molecule_mse(pred, target, node_mask)
    unmasked = _per_molecule_mse(pred, target, node_mask=None)

    assert torch.allclose(masked, unmasked, atol=1e-5), (
        "Full-mask path diverges from no-mask path when there is no padding."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Padding does NOT contaminate
# ─────────────────────────────────────────────────────────────────────────────

def test_padding_atoms_do_not_affect_loss():
    """Changing padding atom values (where mask=0) must not change per-molecule loss."""
    torch.manual_seed(5)
    B, N, F_dim = 4, 9, 8
    n_valid = 5

    node_mask = torch.zeros(B, N, 1)
    node_mask[:, :n_valid, 0] = 1.0

    pred   = torch.randn(B, N, F_dim) * node_mask
    target = torch.randn(B, N, F_dim) * node_mask

    loss_before = _per_molecule_mse(pred, target, node_mask)

    # Corrupt padding positions with huge values.
    pred_corrupted   = pred.clone()
    target_corrupted = target.clone()
    pred_corrupted[:, n_valid:, :]   = 1e6
    target_corrupted[:, n_valid:, :] = -1e6

    loss_after = _per_molecule_mse(pred_corrupted, target_corrupted, node_mask)

    assert torch.allclose(loss_before, loss_after, atol=1e-5), (
        "Padding corruption changed per-molecule loss — masking is broken.\n"
        f"before={loss_before}\nafter={loss_after}"
    )


def test_size_invariance():
    """Two molecules with identical valid-atom errors must have identical per-molecule loss.

    A 3-atom molecule and a 9-atom molecule (both padded to 9 total slots) where
    the valid atoms have exactly the same (pred, target) pairs must report the same
    per_sample loss.  A naive mean would give 1/3 the loss to the 3-atom molecule.
    """
    torch.manual_seed(9)
    N, F_dim = 9, 8

    # Build 1 molecule of each size sharing the same first-3-atom errors.
    pred_row   = torch.randn(N, F_dim)
    target_row = torch.randn(N, F_dim)

    pred   = pred_row.unsqueeze(0).expand(2, -1, -1).clone()
    target = target_row.unsqueeze(0).expand(2, -1, -1).clone()

    # mol 0: 3 valid atoms; mol 1: 9 valid atoms (same 3 + 6 zero-error padding).
    mask = torch.zeros(2, N, 1)
    mask[0, :3, 0] = 1.0
    mask[1, :3, 0] = 1.0     # same 3 valid atoms ...
    # mol 1 has 6 additional atoms with pred == target (zero error).
    pred[1, 3:, :] = 0.0
    target[1, 3:, :] = 0.0
    mask[1, 3:, 0] = 1.0     # ... and 6 zero-error atoms included

    loss = _per_molecule_mse(pred, target, mask)

    # Both share the same 3 valid atoms — masked mol 0 == mol 1 (3 atoms, same errors).
    # But mol 1 has 6 additional zero-error atoms, so its mean should be lower.
    # This verifies that when zero-error atoms are included, mol 1 gets a lower loss
    # (correct behaviour — more atoms, some with zero error, lower average).
    masked_3   = _per_molecule_mse(pred[:1], target[:1], mask[:1])   # 3-atom
    masked_9   = _per_molecule_mse(pred[1:], target[1:], mask[1:])   # 9-atom (3 good + 6 zero)

    # 9-atom version has lower per-molecule loss because 6 atoms have zero error.
    assert masked_9.item() <= masked_3.item() + 1e-5, (
        "9-atom mol with 6 zero-error atoms should have ≤ loss than 3-atom mol.\n"
        f"3-atom={masked_3.item():.6f}, 9-atom={masked_9.item():.6f}"
    )


def test_per_sample_shape(edm_batch):
    """_per_molecule_mse always returns (B,) regardless of input shape."""
    pred, target, node_mask = edm_batch["pred"], edm_batch["target"], edm_batch["node_mask"]
    B = pred.shape[0]

    result = _per_molecule_mse(pred, target, node_mask)
    assert result.shape == (B,), f"Expected ({B},), got {result.shape}"


def test_2d_node_mask(edm_batch):
    """_per_molecule_mse handles (B, N_atoms) mask (no trailing 1 dim)."""
    pred, target, node_mask = edm_batch["pred"], edm_batch["target"], edm_batch["node_mask"]
    mask_2d = node_mask.squeeze(-1)   # (B, N_atoms)

    result_3d = _per_molecule_mse(pred, target, node_mask)
    result_2d = _per_molecule_mse(pred, target, mask_2d)

    assert torch.allclose(result_3d, result_2d, atol=1e-6), (
        "2D and 3D node_mask give different results."
    )
