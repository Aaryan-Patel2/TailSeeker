"""Metric computation stubs — validity, QED, SA score, top-k tail."""

from typing import Sequence

import torch


def validity(molecules: Sequence) -> float:
    """Fraction of generated molecules that are chemically valid.

    TODO: implement using rdkit.Chem.MolFromSmiles (returns None for invalid).
    """
    # TODO: iterate molecules, count valid RDKit Mol objects
    raise NotImplementedError("validity() not yet implemented")


def qed_score(molecules: Sequence) -> float:
    """Mean QED (drug-likeness) score over *molecules* in [0, 1].

    TODO: implement using rdkit.Chem.QED.qed().
    """
    # TODO: filter valid molecules, compute QED for each, return mean
    raise NotImplementedError("qed_score() not yet implemented")


def sa_score(molecules: Sequence) -> float:
    """Mean SA (synthetic accessibility) score over *molecules* in [1, 10].

    Lower is more synthetically accessible.

    TODO: implement using the SA score from rdkit contrib.
    """
    # TODO: filter valid molecules, compute SA for each, return mean
    raise NotImplementedError("sa_score() not yet implemented")


def top_k_tail(values: torch.Tensor, k: int = 100) -> float:
    """Mean of the top-*k* values (tail quality metric).

    Args:
        values: 1-D float tensor of per-molecule scores.
        k:      number of top values to average.

    Returns:
        Scalar float.
    """
    assert values.ndim == 1, (
        f"Expected 1-D tensor, got shape {values.shape}"
    )
    k = min(k, values.numel())
    return values.topk(k).values.mean().item()
