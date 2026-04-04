"""Unit tests for src/data/qm9.py encoding logic."""

from __future__ import annotations

import torch
import pytest

from src.data.qm9 import _encode_molecule


def test_encode_molecule_output_shape():
    """_encode_molecule always returns (4, max_atoms, max_atoms)."""
    pos = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    out = _encode_molecule(pos, edge_index, max_atoms=29)
    assert out.shape == (4, 29, 29), f"Expected (4, 29, 29), got {out.shape}"
    assert out.dtype == torch.float32


def test_encode_molecule_adjacency_channel():
    """Channel 3 reflects edge_index bonds exactly."""
    pos = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1], [1, 0]])  # single bond 0↔1
    out = _encode_molecule(pos, edge_index, max_atoms=10)
    assert out[3, 0, 1] == 1.0, "Expected bond 0→1"
    assert out[3, 1, 0] == 1.0, "Expected bond 1→0"
    assert out[3, 0, 2] == 0.0, "Expected no bond 0→2"


def test_encode_molecule_padded_region_is_zero():
    """Padding region (beyond N atoms) is all zeros in all channels."""
    N = 5
    pos = torch.randn(N, 3)
    edge_index = torch.zeros(2, 0, dtype=torch.long)  # no bonds
    out = _encode_molecule(pos, edge_index, max_atoms=29)
    padded = out[:, N:, :]
    assert padded.abs().max() == 0.0, "Padded rows should be zero"
    padded_cols = out[:, :, N:]
    assert padded_cols.abs().max() == 0.0, "Padded cols should be zero"


def test_encode_molecule_coord_channels_antisymmetric():
    """Δ channels 0-2 satisfy diff[k, i, j] == -diff[k, j, i] (antisymmetric)."""
    pos = torch.randn(4, 3)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    out = _encode_molecule(pos, edge_index, max_atoms=4)
    for ch in range(3):
        mat = out[ch, :4, :4]
        assert torch.allclose(mat, -mat.T, atol=1e-5), f"Channel {ch} is not antisymmetric"


@pytest.mark.parametrize("n_atoms", [1, 5, 29])
def test_encode_molecule_various_sizes(n_atoms: int):
    """Encoding works for molecules of any size ≤ max_atoms."""
    pos = torch.randn(n_atoms, 3)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    out = _encode_molecule(pos, edge_index, max_atoms=29)
    assert out.shape == (4, 29, 29)
    assert torch.isfinite(out).all(), f"Non-finite values for n_atoms={n_atoms}"
