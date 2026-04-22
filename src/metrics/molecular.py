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


def diversity(smiles_list: Sequence[str]) -> float:
    """Mean pairwise Tanimoto distance over valid molecules (higher = more diverse).

    Uses Morgan fingerprints (radius=2). Returns 0.0 for < 2 valid molecules.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        for s in smiles_list
        if (m := Chem.MolFromSmiles(s)) is not None
    ]
    if len(fps) < 2:
        return 0.0
    dists = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            dists.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return float(sum(dists) / len(dists))


def novelty(smiles_list: Sequence[str], train_smiles: Sequence[str]) -> float:
    """Fraction of valid generated molecules not seen in training set.

    Compares canonical SMILES. Returns 0.0 for empty/invalid input.
    """
    from rdkit import Chem

    train_canonical = {
        Chem.MolToSmiles(m)
        for s in train_smiles
        if (m := Chem.MolFromSmiles(s)) is not None
    }
    gen_canonical = [
        Chem.MolToSmiles(m)
        for s in smiles_list
        if (m := Chem.MolFromSmiles(s)) is not None
    ]
    if not gen_canonical:
        return 0.0
    novel = sum(1 for s in gen_canonical if s not in train_canonical)
    return novel / len(gen_canonical)


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
