"""LightningDataModule for TailSeeker."""

from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from dotmap import DotMap
from torch.utils.data import DataLoader, Dataset

from tailseeker.data.utils import load_raw_data


class TailSeekerDataModule(pl.LightningDataModule):
    """DataModule wrapping train/val/test splits for TailSeeker."""

    def __init__(self, config: DotMap) -> None:
        super().__init__()
        self.config = config
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and split raw data into train/val datasets.

        TODO: wrap the raw splits in a proper Dataset subclass.
        """
        raw = load_raw_data(self.config.get("data_root", "data/"))
        # TODO: convert raw["train"] / raw["val"] into Dataset objects
        self._train_dataset = raw["train"]
        self._val_dataset = raw["val"]

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        assert self._train_dataset is not None, (
            "setup() must be called before train_dataloader()"
        )
        return DataLoader(
            self._train_dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        assert self._val_dataset is not None, (
            "setup() must be called before val_dataloader()"
        )
        return DataLoader(
            self._val_dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
        )
