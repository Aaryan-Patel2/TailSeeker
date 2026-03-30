"""Evaluator: runs the full metric suite and returns a flat W&B-ready dict."""

from __future__ import annotations

from typing import Sequence

import torch


class Evaluator:
    """Run all metrics on a batch of generated molecules.

    Usage:
        ev = Evaluator(cfg)
        metrics = ev.evaluate(smiles_list, qed_scores, sa_scores)
        wandb.log(metrics)
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        smiles: Sequence[str],
        qed_scores: torch.Tensor | None = None,
        sa_scores: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute all metrics and return a flat dict.

        Args:
            smiles:     generated SMILES strings.
            qed_scores: precomputed QED scores (1-D) — computed here if None.
            sa_scores:  precomputed SA scores (1-D) — computed here if None.

        Returns:
            Flat dict suitable for wandb.log().
        """
        from src.metrics.molecular import mean_qed, mean_sa, uniqueness, validity
        from src.metrics.tail import right_cvar, top_k_mean

        out: dict[str, float] = {}

        out["Eval/validity"] = validity(smiles)
        out["Eval/uniqueness"] = uniqueness(smiles)

        if smiles and out["Eval/validity"] > 0:
            out["Eval/qed_mean"] = mean_qed(smiles)
            out["Eval/sa_mean"] = mean_sa(smiles)

        if qed_scores is not None and qed_scores.numel() > 0:
            out["Eval/qed_top1pct"] = right_cvar(qed_scores, alpha=0.01)
            out["Eval/qed_top10pct"] = right_cvar(qed_scores, alpha=0.10)
            out["Eval/qed_top100"] = top_k_mean(qed_scores, k=100)

        if sa_scores is not None and sa_scores.numel() > 0:
            # SA: lower is better, so negate for "top" semantics
            neg_sa = -sa_scores
            out["Eval/sa_best1pct"] = -right_cvar(neg_sa, alpha=0.01)
            out["Eval/sa_best10pct"] = -right_cvar(neg_sa, alpha=0.10)

        return out
