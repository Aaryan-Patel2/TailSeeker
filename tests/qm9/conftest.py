"""Shared fixtures for QM9/EDM integration tests.

EDM represents molecules as padded tensors (B, N_atoms, n_feat) with a
node_mask (B, N_atoms, 1) indicating valid atoms.  These fixtures reproduce
that structure without requiring EDM or torch-geometric to be installed.

QM9 stats: max 9 heavy atoms, 5 one-hot atom types (H,C,N,O,F), 3 xyz coords
→ n_feat = 3 (pos) + 5 (types) = 8 per atom in EDM's equivariant graph.
"""

from __future__ import annotations

import pytest
import torch

MAX_ATOMS = 9
N_FEAT = 8   # 3 xyz + 5 atom-type one-hot (EDM default)


@pytest.fixture
def edm_batch():
    """4-molecule batch with variable atom counts, padded to MAX_ATOMS=9.

    Molecule sizes: [5, 7, 9, 3] — deliberately varied to expose any
    normalization bug that would treat short and long molecules differently.
    """
    B = 4
    n_atoms = [5, 7, 9, 3]
    torch.manual_seed(0)

    node_mask = torch.zeros(B, MAX_ATOMS, 1)
    for i, n in enumerate(n_atoms):
        node_mask[i, :n, 0] = 1.0

    # EDM zeroes out padding positions in both pred and target.
    pred   = torch.randn(B, MAX_ATOMS, N_FEAT) * node_mask
    target = torch.randn(B, MAX_ATOMS, N_FEAT) * node_mask

    return {
        "pred":      pred,
        "target":    target,
        "node_mask": node_mask,
        "n_atoms":   torch.tensor(n_atoms, dtype=torch.long),
    }


@pytest.fixture
def edm_batch_uniform():
    """8-molecule batch where all molecules have the same size (9 atoms).

    Used to verify that masked and unmasked paths agree when there is no
    padding — the two implementations must be numerically identical here.
    """
    B = 8
    torch.manual_seed(1)
    node_mask = torch.ones(B, MAX_ATOMS, 1)
    pred   = torch.randn(B, MAX_ATOMS, N_FEAT)
    target = torch.randn(B, MAX_ATOMS, N_FEAT)
    return {"pred": pred, "target": target, "node_mask": node_mask,
            "n_atoms": torch.full((B,), MAX_ATOMS, dtype=torch.long)}


@pytest.fixture
def edm_batch_with_props(edm_batch):
    """Extends edm_batch with QED/SA property group labels for V2 tests.

    Group 0 = QED-like (inner tilt τ=+2, tail-seek)
    Group 1 = SA-like  (inner tilt τ=-1, robustness)
    """
    batch = dict(edm_batch)
    batch["groups"] = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    return batch
