"""DDPM training loop with stub-safe loss handling.

Catches NotImplementedError from TiltedScoreMatchingLoss, logs
loss_stub_active=true, skips the backward pass, and continues.
Never crashes in stub mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.diffusion.forward_process import q_sample
from src.diffusion.noise_schedule import get_schedule, precompute_schedule
from src.losses.base import LossOutput
from src.losses.base import BaseLoss
from src.losses.tilted_score_matching import get_loss_fn
from src.models.base import ModelOutput


class Trainer:
    """Minimal training loop for DDPM score matching.

    Args:
        model:    score network (BaseModel subclass).
        cfg:      Hydra DictConfig or dict-like config.
        output_dir: directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        output_dir: Path,
        loss_fn: BaseLoss | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Caller may supply a pre-built loss (e.g. MultiObjectiveTiltedLoss).
        # Falls back to single-objective get_loss_fn(tilt) for backwards compat.
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            tilt: float = float(cfg.get("tilt", 1.0))
            self.loss_fn = get_loss_fn(tilt)
        self._stub_active = False  # set True when NotImplementedError is caught

        # build noise schedule and move to target device later
        schedule_name: str = cfg.get("schedule", "linear")
        self.num_timesteps: int = cfg.get("num_timesteps", 1000)
        betas = get_schedule(schedule_name, self.num_timesteps)
        self.schedule = precompute_schedule(betas)

        lr: float = float(cfg.get("learning_rate", 1e-4))
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        self._step = 0
        self._epoch = 0

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """One gradient update (or stub no-op).

        Returns a flat dict of loggable scalars.
        """
        device = next(self.model.parameters()).device
        x0 = batch["coords"].to(device)  # (B, N, 3) or (B, C, H, W)
        B = x0.shape[0]

        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # move schedule tensors to device on first call
        for k in self.schedule:
            if self.schedule[k].device != device:
                self.schedule[k] = self.schedule[k].to(device)

        x_t, noise = q_sample(
            x0, t,
            self.schedule["sqrt_alphas_cumprod"],
            self.schedule["sqrt_one_minus_alphas_cumprod"],
        )

        output: ModelOutput = self.model(x_t, t)
        pred = output.pred_noise
        assert pred is not None, "model must return pred_noise"

        log: dict[str, Any] = {}

        # Pass group labels to loss_fn if available (multi-objective mode).
        loss_kwargs = {}
        if "group" in batch:
            loss_kwargs["groups"] = batch["group"].to(device)

        try:
            loss_out: LossOutput = self.loss_fn(pred, noise, **loss_kwargs)
            self._stub_active = False

            self.optimizer.zero_grad()
            loss_out.total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            log["Train/loss"] = loss_out.total_loss.item()
            for k, v in loss_out.loss_components.items():
                log[f"Train/loss_{k}"] = v.item() if hasattr(v, "item") else float(v)
            log["loss_stub_active"] = False

        except NotImplementedError as exc:
            self._stub_active = True
            log["loss_stub_active"] = True
            log["stub_message"] = str(exc)
            # skip backward — continue to next step

        self._step += 1
        log["step"] = self._step
        return log

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader) -> dict[str, float]:
        """Run one full epoch. Returns aggregated log dict."""
        self.model.train()
        agg: dict[str, list] = {}
        for batch in dataloader:
            step_log = self.train_step(batch)
            for k, v in step_log.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(v)

        self._epoch += 1
        return {k: sum(vs) / len(vs) for k, vs in agg.items() if vs}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self) -> Path:
        """Save model + optimizer + config to output_dir/checkpoints/."""
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        path = ckpt_dir / f"epoch_{self._epoch:04d}.pt"
        torch.save(
            {
                "epoch": self._epoch,
                "step": self._step,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": dict(self.cfg),
                "stub_active": self._stub_active,
            },
            path,
        )
        return path

    @classmethod
    def load_checkpoint(cls, path: Path, model: nn.Module, cfg: DictConfig) -> "Trainer":
        """Restore a trainer from a checkpoint file."""
        ckpt = torch.load(path, map_location="cpu")
        trainer = cls(model, cfg, path.parent.parent)
        model.load_state_dict(ckpt["state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer._epoch = ckpt["epoch"]
        trainer._step = ckpt["step"]
        return trainer
