"""QM9 dataset — returns per-molecule tensors for DDPM training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

_SPLIT_RANGES: dict[str, tuple[int, int]] = {
    "train": (0, 110_000),
    "val":   (110_000, 120_000),
    "test":  (120_000, 130_000),
}


def _encode_molecule(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    max_atoms: int,
) -> torch.Tensor:
    """Encode molecule as (4, max_atoms, max_atoms) pairwise feature tensor.

    Channels: Δx, Δy, Δz (centred/scaled by 5 Å), adjacency A_ij.
    Zero-pads to (4, max_atoms, max_atoms).
    """
    N = min(pos.shape[0], max_atoms)
    pos_n = (pos[:N] - pos[:N].mean(dim=0)) / 5.0
    diff = pos_n.unsqueeze(0) - pos_n.unsqueeze(1)           # (N, N, 3)
    adj = torch.zeros(N, N, dtype=torch.float32)
    mask = (edge_index[0] < N) & (edge_index[1] < N)
    ei = edge_index[:, mask]
    if ei.numel() > 0:
        adj[ei[0], ei[1]] = 1.0
    feat = torch.stack([diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], adj], dim=0)
    out = torch.zeros(4, max_atoms, max_atoms, dtype=torch.float32)
    out[:, :N, :N] = feat
    return out


def _compute_properties(smiles: str) -> tuple[float, float, float, float]:
    """Return (QED, SA, LogP, TPSA) for a SMILES string via RDKit.

    Falls back to neutral defaults if RDKit is unavailable or SMILES unparseable.
    SA: [1, 10] lower = more accessible. LogP: Crippen. TPSA: Å² polar surface area.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import QED, Crippen, Descriptors
        try:
            from rdkit.Contrib.SA_Score import sascorer
        except ImportError:
            sascorer = None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.5, 5.0, 2.0, 60.0
        qed = float(QED.qed(mol))
        sa = float(sascorer.calculateScore(mol)) if sascorer is not None else 5.0
        logp = float(Crippen.MolLogP(mol))
        tpsa = float(Descriptors.TPSA(mol))
        return qed, sa, logp, tpsa
    except Exception:
        return 0.5, 5.0, 2.0, 60.0


class QM9Dataset(Dataset):
    """QM9 molecular dataset for diffusion-based generation.

    Each item returns a dict with:
        coords  (4, max_atoms, max_atoms) float32 — pairwise coord + adj encoding
        n_atoms int                        — number of heavy atoms (before padding)
        qed     float32 scalar            — QED drug-likeness score [0, 1]
        sa      float32 scalar            — SA score [1, 10] (lower = more accessible)
        group   int64 scalar              — QED tertile group: 0=low, 1=mid, 2=high

    Args:
        root:      path to QM9 raw data (torch_geometric cache dir)
        split:     "train" | "val" | "test"
        max_atoms: pad/truncate to this size (default 29 = QM9 max heavy atoms)
        download:  unused (torch_geometric 2.7+ auto-downloads)
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

        self._pyg = _PyGQM9(root=str(self.root))
        lo, hi = _SPLIT_RANGES[split]
        total = len(self._pyg)
        assert total >= hi, (
            f"QM9 has only {total} molecules but split '{split}' requires up to {hi}."
        )
        self._indices = list(range(lo, hi))

        # Pre-compute QED/SA scores and QED-tertile group labels.
        # Results are cached to <root>/qm9_properties_<split>.pt so the
        # expensive RDKit pass runs only once (~2–5 min for train split).
        # v2 cache adds logp + tpsa; v1 cache (qed/sa only) is ignored.
        cache_path = self.root / f"qm9_properties_v2_{split}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            self._qed  = cached["qed"]
            self._sa   = cached["sa"]
            self._logp = cached["logp"]
            self._tpsa = cached["tpsa"]
        else:
            print(f"[QM9Dataset] Computing QED/SA/LogP/TPSA for {len(self._indices)} "
                  f"molecules (split={split}). Cached to {cache_path}.")
            qed_vals, sa_vals, logp_vals, tpsa_vals = [], [], [], []
            for idx in self._indices:
                mol = self._pyg[idx]
                smiles = getattr(mol, "smiles", None) or ""
                q, s, lp, tp = _compute_properties(smiles)
                qed_vals.append(q); sa_vals.append(s)
                logp_vals.append(lp); tpsa_vals.append(tp)
            self._qed  = torch.tensor(qed_vals,  dtype=torch.float32)
            self._sa   = torch.tensor(sa_vals,   dtype=torch.float32)
            self._logp = torch.tensor(logp_vals, dtype=torch.float32)
            self._tpsa = torch.tensor(tpsa_vals, dtype=torch.float32)
            torch.save(
                {"qed": self._qed, "sa": self._sa,
                 "logp": self._logp, "tpsa": self._tpsa},
                cache_path,
            )
        q33, q66 = self._qed.quantile(torch.tensor([1/3, 2/3]))
        self._groups = torch.zeros(len(self._indices), dtype=torch.long)
        self._groups[self._qed >= q33] = 1
        self._groups[self._qed >= q66] = 2

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mol = self._pyg[self._indices[idx]]
        coords = _encode_molecule(mol.pos, mol.edge_index, self.max_atoms)
        return {
            "coords": coords,
            "n_atoms": mol.pos.shape[0],
            "qed":   self._qed[idx],
            "sa":    self._sa[idx],
            "logp":  self._logp[idx],
            "tpsa":  self._tpsa[idx],
            "group": self._groups[idx],
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate variable-molecule dicts into a batched dict."""
        return {
            "coords": torch.stack([b["coords"] for b in batch]),
            "n_atoms": [b["n_atoms"] for b in batch],
            "qed":   torch.stack([b["qed"]   for b in batch]),
            "sa":    torch.stack([b["sa"]    for b in batch]),
            "logp":  torch.stack([b["logp"]  for b in batch]),
            "tpsa":  torch.stack([b["tpsa"]  for b in batch]),
            "group": torch.stack([b["group"] for b in batch]),
        }
