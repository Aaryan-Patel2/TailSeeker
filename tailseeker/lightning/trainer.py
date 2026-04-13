"""LightningModule for TailSeeker — training, validation, optimisation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from dotmap import DotMap

from tailseeker.lightning.loss import TailSeekerLoss
from tailseeker.utils.utils import format_batch


class TailSeekerModule(pl.LightningModule):
    """LightningModule wrapping model training, validation, and optimisation."""

    def __init__(self, model: nn.Module, config: DotMap) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = TailSeekerLoss(config)

        if config.get("use_ema", False):
            self.ema_model = deepcopy(model)
            self.ema_model.eval()
            self._ema_decay: float = config.get("ema_decay", 0.9999)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Single gradient-update step."""
        x, y = format_batch(batch)
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log(
            "Train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if hasattr(self, "ema_model"):
            self._update_ema()
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Single validation step."""
        # sanity-check fast path — skip expensive evaluation during sanity run
        if self.trainer.sanity_checking:
            return torch.tensor(0.0, device=self.device)

        x, y = format_batch(batch)
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log(
            "Val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        """Post-epoch hook — compute full metrics using EMA model if available."""
        if hasattr(self, "ema_model"):
            # TODO: run validity / QED / SA / top-k-tail evaluation with self.ema_model
            pass

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict:
        """Return AdamW optimiser (and optional LR scheduler)."""
        lr: float = self.config.get("learning_rate", 1e-4)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # TODO: add a CosineAnnealingLR or other scheduler if needed
        return {"optimizer": optimizer}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_ema(self) -> None:
        """Polyak-average model weights into ema_model."""
        decay = self._ema_decay
        for ema_p, p in zip(
            self.ema_model.parameters(),
            self.model.parameters(),
        ):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
