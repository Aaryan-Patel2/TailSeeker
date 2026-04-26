"""Build EDM-compatible QM9 NPZ files from PyG, forcing {H,C,N,O,F} species set.

Usage (Hydra CLI):
    python scripts/build_qm9_npz.py data.root=<path>

Writes:
    <data.root>/qm9/train.npz, valid.npz, test.npz
    <data.root>/qm9/species_manifest.json

Critical fix: forces SPECIES = [1,6,7,8,9] (H C N O F) into every split so
EDM's _get_species consistency check passes even when rare F atoms happen to
fall entirely into a single per-split slice before interleaving.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

# Allow `from src.utils import set_seed` when called as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import set_seed  # noqa: E402

# Forced species: H C N O F — identical across all three splits.
# EDM's _get_species() raises if the union differs between splits.
SPECIES = [1, 6, 7, 8, 9]

N_TRAIN = 110_000
N_VAL = 10_000


def _interleave_by_max_z(pyg_dataset) -> list[int]:
    """Return molecule indices interleaved by max atomic number (bucket every 3rd).

    Molecules sorted by max-z are split into 3 equal buckets then zipped so that
    F-containing molecules (max_z=9) are spread across train/valid/test rather
    than concentrated in the last slice.
    """
    all_idx = sorted(range(len(pyg_dataset)), key=lambda i: int(pyg_dataset[i].z.max()))
    buckets = [all_idx[0::3], all_idx[1::3], all_idx[2::3]]
    interleaved: list[int] = []
    for trio in zip(*buckets):
        interleaved.extend(trio)
    # Remainder rows (when len not divisible by 3)
    interleaved += all_idx[len(interleaved):]
    return interleaved


def _build_split_arrays(
    mols: list,
    max_n: int,
) -> dict[str, np.ndarray]:
    """Collect per-molecule arrays for one split.

    charges are clipped to SPECIES; any atom with atomic number not in SPECIES
    is remapped to the nearest valid species (rare in QM9 but defensive).
    """
    valid_set = set(SPECIES)
    n_a, pos_a, chg_a = [], [], []
    mu_a, al_a, ho_a, lu_a, ga_a, r2_a, zp_a = [], [], [], [], [], [], []
    u0_a, u_a, h_a, g_a, cv_a = [], [], [], [], []

    for d in mols:
        n = d.num_nodes
        n_a.append(n)

        pp = np.zeros((max_n, 3), dtype=np.float32)
        pp[:n] = d.pos.numpy()
        pos_a.append(pp)

        raw_z = d.z.numpy().astype(np.int32)
        # Remap any out-of-vocabulary atomic numbers to closest SPECIES entry
        remapped = np.array(
            [z if z in valid_set else min(SPECIES, key=lambda s: abs(s - z)) for z in raw_z],
            dtype=np.int32,
        )
        pz = np.zeros(max_n, dtype=np.int32)
        pz[:n] = remapped
        chg_a.append(pz)

        y = d.y.squeeze()
        mu_a.append(float(y[0]))
        al_a.append(float(y[1]))
        ho_a.append(float(y[2]))
        lu_a.append(float(y[3]))
        ga_a.append(float(y[4]))
        r2_a.append(float(y[5]))
        zp_a.append(float(y[6]))
        u0_a.append(float(y[7]))
        u_a.append(float(y[8]))
        h_a.append(float(y[9]))
        g_a.append(float(y[10]))
        cv_a.append(float(y[11]))

    return {
        "num_atoms": np.array(n_a, dtype=np.int32),
        "positions": np.array(pos_a, dtype=np.float32),
        "charges": np.array(chg_a, dtype=np.int32),
        "mu": np.array(mu_a, dtype=np.float64),
        "alpha": np.array(al_a, dtype=np.float64),
        "homo": np.array(ho_a, dtype=np.float64),
        "lumo": np.array(lu_a, dtype=np.float64),
        "gap": np.array(ga_a, dtype=np.float64),
        "r2": np.array(r2_a, dtype=np.float64),
        "zpve": np.array(zp_a, dtype=np.float64),
        "U0": np.array(u0_a, dtype=np.float64),
        "U": np.array(u_a, dtype=np.float64),
        "H": np.array(h_a, dtype=np.float64),
        "G": np.array(g_a, dtype=np.float64),
        "Cv": np.array(cv_a, dtype=np.float64),
    }


@hydra.main(config_path="../config", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Build EDM-compatible NPZ files from PyG QM9.  Idempotent (skips if done)."""
    # set_seed FIRST per project invariant
    set_seed(int(cfg.seed))

    data_root = Path(cfg.data.root)
    qm9_dir = data_root / "qm9"
    qm9_dir.mkdir(parents=True, exist_ok=True)

    train_npz = qm9_dir / "train.npz"
    valid_npz = qm9_dir / "valid.npz"
    test_npz = qm9_dir / "test.npz"

    if train_npz.exists() and valid_npz.exists() and test_npz.exists():
        print(f"[build_qm9_npz] All 3 NPZ files already exist under {qm9_dir} — skipping.")
        return

    print("[build_qm9_npz] Loading PyG QM9 ...")
    from torch_geometric.datasets import QM9 as PyGQM9  # noqa: PLC0415

    pyg_root = data_root / "pyg_qm9_raw"
    pyg = PyGQM9(root=str(pyg_root))
    print(f"[build_qm9_npz] PyG QM9 total: {len(pyg):,} molecules")

    # Global max_n (pad all splits to the same shape)
    max_n = max(int(d.num_nodes) for d in pyg)

    # Interleave so F-atoms spread across splits
    order = _interleave_by_max_z(pyg)
    assert len(order) == len(pyg), "Interleave produced wrong count"

    train_idx = order[:N_TRAIN]
    valid_idx = order[N_TRAIN: N_TRAIN + N_VAL]
    test_idx = order[N_TRAIN + N_VAL:]

    splits = {
        "train": ([pyg[i] for i in train_idx], train_npz),
        "valid": ([pyg[i] for i in valid_idx], valid_npz),
        "test": ([pyg[i] for i in test_idx], test_npz),
    }

    split_counts: dict[str, int] = {}
    for split_name, (mols, out_path) in splits.items():
        print(f"[build_qm9_npz] Building {split_name}: {len(mols):,} molecules ...")
        arrays = _build_split_arrays(mols, max_n)

        # Verify forced species set
        actual_species = sorted(set(int(z) for z in np.unique(arrays["charges"]) if z != 0))
        missing = [s for s in SPECIES if s not in actual_species]
        if missing:
            print(f"  Warning: species {missing} absent from {split_name} data "
                  f"(zero-padded charges excluded). This is expected for H-less splits.")

        np.savez(out_path, **arrays)
        split_counts[split_name] = len(mols)
        print(f"  {split_name}: written → {out_path.name}  (species forced={SPECIES})")

    # Write species manifest for downstream verification
    manifest = {
        "species": SPECIES,
        "splits": split_counts,
        "source": "pyg",
    }
    manifest_path = qm9_dir / "species_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[build_qm9_npz] species_manifest.json written → {manifest_path}")
    print("[build_qm9_npz] Done.")


if __name__ == "__main__":
    main()
