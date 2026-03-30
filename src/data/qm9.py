"""QM9 dataset — returns per-molecule tensors for DDPM training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class QM9Dataset(Dataset):
    """QM9 molecular dataset for diffusion-based generation.

    Each item returns a dict with:
        coords     (N, 3)  float32 — 3-D atomic coordinates
        atom_types (N,)    int64   — atomic numbers
        adj        (N, N)  float32 — adjacency matrix (0/1)
        smiles     str             — canonical SMILES
        qed        float           — QED drug-likeness score [0, 1]
        sa         float           — SA synthetic-accessibility score [1, 10]
        n_atoms    int             — number of heavy atoms N

    Args:
        root:     path to QM9 raw data (torch_geometric cache dir)
        split:    "train" | "val" | "test"
        max_atoms: pad/truncate to this size (default 29 = QM9 max)
        download: if True, download QM9 via torch_geometric (default False)
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

        if not self.root.exists():
            raise FileNotFoundError(
                f"Data root does not exist: {self.root}. "
                "Pass download=True or run tailseeker-setup first."
            )

        # TODO: load torch_geometric QM9 and split into train/val/test
        # from torch_geometric.datasets import QM9
        # dataset = QM9(root=str(self.root), transform=None)
        # Placeholder — replace with real split indices
        self._data: list[dict] = []

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Pad variable-length molecules to a fixed atom budget."""
        return {
            k: torch.stack([b[k] for b in batch])
            if isinstance(batch[0][k], torch.Tensor)
            else [b[k] for b in batch]
            for k in batch[0]
        }
