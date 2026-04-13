"""QM9 dataset — returns per-molecule tensors for DDPM training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

# Standard QM9 split boundaries (deterministic by molecule index, no random seed).
# Uses first 130 000 of 130 831 molecules to give round numbers.
# train [0, 110 000), val [110 000, 120 000), test [120 000, 130 000).
_SPLIT_RANGES: dict[str, tuple[int, int]] = {
    "train": (0, 110_000),
    "val":   (110_000, 120_000),
    "test":  (120_000, 130_000),
}


def _encode_molecule(
    pos: torch.Tensor,        # (N, 3) atomic coords in Angstroms
    edge_index: torch.Tensor, # (2, E) bond edges
    max_atoms: int,
) -> torch.Tensor:
    """Encode a molecule as a (4, max_atoms, max_atoms) pairwise feature tensor.

    Channels:
        0: Δx = pos_i[x] - pos_j[x]  (centred, scaled by 5 Å)
        1: Δy = pos_i[y] - pos_j[y]
        2: Δz = pos_i[z] - pos_j[z]
        3: adjacency A_ij (binary 0/1 from bond edge_index)

    Zero-pads to (4, max_atoms, max_atoms). Molecules with N > max_atoms
    are truncated (QM9 never exceeds max_atoms=29 for heavy atoms).
    """
    N = min(pos.shape[0], max_atoms)
    pos_n = pos[:N]
    pos_n = (pos_n - pos_n.mean(dim=0)) / 5.0  # centre + scale

    # Pairwise diffs: (N, N, 3)
    diff = pos_n.unsqueeze(0) - pos_n.unsqueeze(1)  # (N, N, 3)

    # Adjacency (only edges where both endpoints are within N)
    adj = torch.zeros(N, N, dtype=torch.float32)
    mask = (edge_index[0] < N) & (edge_index[1] < N)
    ei = edge_index[:, mask]
    if ei.numel() > 0:
        adj[ei[0], ei[1]] = 1.0

    feat = torch.stack([diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], adj], dim=0)  # (4, N, N)

    out = torch.zeros(4, max_atoms, max_atoms, dtype=torch.float32)
    out[:, :N, :N] = feat
    return out


class QM9Dataset(Dataset):
    """QM9 molecular dataset for diffusion-based generation.

    Each item returns a dict with:
        coords     (4, max_atoms, max_atoms)  float32 — pairwise coord + adj encoding
        n_atoms    int                         — number of heavy atoms (before padding)

    Args:
        root:      path to QM9 raw data (torch_geometric cache dir)
        split:     "train" | "val" | "test"
        max_atoms: pad/truncate to this size (default 29 = QM9 max heavy atoms)
        download:  if True, download QM9 via torch_geometric (default False)
    """

    MAX_ATOMS_QM9 = 29

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        max_atoms: int = MAX_ATOMS_QM9,
        download: bool = False,
    ) -> None:
        self.root = Path(root)
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.max_atoms = max_atoms

        try:
            from torch_geometric.datasets import QM9 as _PyGQM9
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required: uv pip install torch-geometric"
            ) from exc

        # Note: torch-geometric 2.7.0 QM9 auto-downloads if needed; download param not supported
        self._pyg = _PyGQM9(root=str(self.root))
        lo, hi = _SPLIT_RANGES[split]
        total = len(self._pyg)
        assert total >= hi, (
            f"QM9 has only {total} molecules but split '{split}' requires up to {hi}."
        )
        self._indices = list(range(lo, hi))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mol = self._pyg[self._indices[idx]]
        coords = _encode_molecule(mol.pos, mol.edge_index, self.max_atoms)
        return {"coords": coords, "n_atoms": mol.pos.shape[0]}

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate variable-molecule dicts into a batched dict."""
        return {
            "coords": torch.stack([b["coords"] for b in batch]),
            "n_atoms": [b["n_atoms"] for b in batch],
        }
