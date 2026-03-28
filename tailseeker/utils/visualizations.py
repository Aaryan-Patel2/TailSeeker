"""Visualization helpers for loss curves and molecule grids."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def plot_loss_curve(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training and validation loss curves over epochs.

    Args:
        train_losses: per-epoch training losses.
        val_losses:   per-epoch validation losses.
        save_path:    if provided, save the figure here at 150 dpi.

    TODO: add optional exponential-moving-average smoothing parameter.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_molecule_grid(
    molecules: Sequence,
    save_path: str | Path | None = None,
    n_cols: int = 4,
) -> plt.Figure:
    """Render a grid of molecules using RDKit.

    Args:
        molecules: sequence of RDKit Mol objects (or SMILES strings).
        save_path: if provided, save the figure here.
        n_cols:    number of columns in the grid.

    TODO: implement using rdkit.Chem.Draw.MolsToGridImage.
    """
    # TODO: convert SMILES strings to Mol objects if needed
    # TODO: call MolsToGridImage and embed the PIL image in matplotlib
    raise NotImplementedError("plot_molecule_grid() not yet implemented")
