"""Molecular validity, QED, SA score, and uniqueness metrics."""

from __future__ import annotations

from typing import Sequence


def validity(smiles_list: Sequence[str]) -> float:
    """Fraction of SMILES that parse to a valid RDKit molecule.

    Args:
        smiles_list: list of generated SMILES strings.

    Returns:
        Float in [0, 1]. Returns 0.0 for empty input.
    """
    if not smiles_list:
        return 0.0
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("rdkit is required for molecular metrics. pip install rdkit-pypi")
    valid = sum(1 for s in smiles_list if Chem.MolFromSmiles(s) is not None)
    return valid / len(smiles_list)


def mean_qed(smiles_list: Sequence[str]) -> float:
    """Mean QED drug-likeness score over valid molecules in [0, 1].

    Filters invalid SMILES before computing. Returns 0.0 if none are valid.
    """
    from rdkit import Chem
    from rdkit.Chem import QED

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    scores = [QED.qed(m) for m in mols if m is not None]
    return float(sum(scores) / len(scores)) if scores else 0.0


def mean_sa(smiles_list: Sequence[str]) -> float:
    """Mean SA (synthetic accessibility) score over valid molecules in [1, 10].

    Lower = more accessible. Returns 10.0 if none are valid.
    """
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError:
        # fallback: try the standalone package
        try:
            import sascorer  # type: ignore[no-redef]
        except ImportError:
            raise ImportError(
                "SA score requires rdkit with SA_Score contrib or sascorer package."
            )
    from rdkit import Chem

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    scores = [sascorer.calculateScore(m) for m in mols if m is not None]
    return float(sum(scores) / len(scores)) if scores else 10.0


def uniqueness(smiles_list: Sequence[str]) -> float:
    """Fraction of valid SMILES that are unique (deduplicated canonical forms).

    Returns 0.0 for empty input or when no SMILES are valid.
    """
    from rdkit import Chem

    canonical = [
        Chem.MolToSmiles(m)
        for s in smiles_list
        if (m := Chem.MolFromSmiles(s)) is not None
    ]
    if not canonical:
        return 0.0
    return len(set(canonical)) / len(canonical)
