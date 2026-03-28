"""Raw data loading helpers."""

from pathlib import Path
from typing import Any


def load_raw_data(root: str | Path) -> Any:
    """Load raw dataset splits from *root* directory.

    Returns a dict with keys "train", "val", (optionally "test") mapping to
    raw data structures suitable for wrapping in a Dataset.

    TODO: implement QM9 loading via torch_geometric.datasets.QM9.
    """
    root = Path(root)
    assert root.exists(), f"Data root does not exist: {root}"
    # TODO: load QM9 or other molecular dataset here
    raise NotImplementedError(f"load_raw_data not implemented for root={root}")
