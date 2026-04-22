"""Coordinate decode: (4, max_atoms, max_atoms) tensor → 3-D positions.

Inverse of QM9Dataset._encode_molecule.

Row-mean reconstruction: given delta[i,j] = x_i - x_j (centred, scaled by 5Å),
    x_i = mean_j delta[i,j]   (exact when molecule is zero-centred)

Atom-type decoding (SMILES generation) requires the Direction-B 9-channel
representation and is NOT implemented here; that is `src/generation/smiles.py`.
"""
from __future__ import annotations

import torch


def decode_coords(feat: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Recover 3-D positions from a (4+, max_atoms, max_atoms) encoding.

    Uses channels 0-2 (Δx, Δy, Δz), ignores adjacency and any extra channels.
    Returns (n_atoms, 3) in Ångströms, zero-centred.

    Args:
        feat:    Tensor of shape (C, max_atoms, max_atoms), C ≥ 4.
        n_atoms: Number of real (non-padded) atoms.

    Returns:
        Tensor of shape (n_atoms, 3), dtype float32, zero-centred.
    """
    assert feat.ndim == 3, f"Expected (C, N, N), got {feat.shape}"
    assert feat.shape[0] >= 4, f"Need ≥4 channels, got {feat.shape[0]}"
    assert 1 <= n_atoms <= feat.shape[1], (
        f"n_atoms={n_atoms} out of range [1, {feat.shape[1]}]"
    )
    # delta[k, i, j] = coord_k[i] - coord_k[j]  for k in {x,y,z}
    delta = feat[:3, :n_atoms, :n_atoms]          # (3, n, n)
    # _encode_molecule stores diff[a,b,k] = pos[b,k] - pos[a,k]  (j − i, not i − j)
    # so row-mean over j gives:  mean_j(pos[j,k] - pos[i,k]) = 0 - pos[i,k] = -pos[i,k]
    # → negate to recover original positions, then undo the /5Å normalisation
    pos = -delta.mean(dim=2).T * 5.0             # (n, 3)
    pos = pos - pos.mean(dim=0)                  # re-centre (numerical clean-up)
    return pos.float()


def pairwise_distances(pos: torch.Tensor) -> torch.Tensor:
    """Return (n, n) Euclidean distance matrix for positions (n, 3)."""
    assert pos.ndim == 2 and pos.shape[1] == 3, (
        f"Expected (n, 3), got {pos.shape}"
    )
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # (n, n, 3)
    return diff.pow(2).sum(-1).sqrt()            # (n, n)


def triangle_inequality_check(
    pos: torch.Tensor,
    tol: float = 1e-4,
) -> bool:
    """Return True iff all triples satisfy the triangle inequality.

    Used as a validity gate: structures that violate this are discarded before
    passing to RDKit, since they cannot represent physical 3-D geometries.

    Args:
        pos: (n, 3) atomic positions.
        tol: numerical tolerance for the inequality d_ij ≤ d_ik + d_kj.
    """
    assert pos.ndim == 2 and pos.shape[1] == 3, f"Expected (n, 3), got {pos.shape}"
    d = pairwise_distances(pos)                  # (n, n)
    n = d.shape[0]
    for k in range(n):
        # d[i,j] <= d[i,k] + d[k,j] for all i,j
        upper = d[:, k:k+1] + d[k:k+1, :]      # broadcast (n,n)
        if (d > upper + tol).any():
            return False
    return True
