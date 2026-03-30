"""Integration: one training step with each loss — must not crash."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.ddpm_unet import DDPMUNet
from src.training.trainer import Trainer
from src.utils import set_seed


_STUB_CFG = {
    "tilt": 0.0,        # overridden per test
    "schedule": "linear",
    "num_timesteps": 10,  # tiny T for fast tests
    "learning_rate": 1e-4,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "grad_clip": 1.0,
}

_STUB_BATCH = {"coords": torch.randn(2, 4, 8, 8)}


def _make_trainer(tilt: float, tmp_path: Path) -> Trainer:
    set_seed(0)
    model = DDPMUNet(in_channels=4, out_channels=4, model_channels=16, num_res_blocks=1)
    cfg = dict(_STUB_CFG)
    cfg["tilt"] = tilt
    return Trainer(model, cfg, tmp_path)  # type: ignore[arg-type]


# ── ERM baseline (DDPMSimpleLoss, tilt=0) ────────────────────────────

def test_erm_one_step_no_crash(tmp_path: Path):
    """One training step with DDPMSimpleLoss completes without error."""
    trainer = _make_trainer(tilt=0.0, tmp_path=tmp_path)
    log = trainer.train_step(_STUB_BATCH)
    assert "Train/loss" in log, f"Expected Train/loss in log, got keys: {list(log.keys())}"
    assert log.get("loss_stub_active") is False
    assert isinstance(log["Train/loss"], float)


def test_erm_epoch_one_checkpoint_saved(tmp_path: Path):
    """train_epoch saves a checkpoint and returns Train/loss."""
    trainer = _make_trainer(tilt=0.0, tmp_path=tmp_path)
    log = trainer.train_epoch([_STUB_BATCH])
    ckpt = trainer.save_checkpoint()
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    assert "Train/loss" in log


# ── TSM stub (TiltedScoreMatchingLoss, tilt=1) ───────────────────────

def test_tsm_stub_one_step_no_crash(tmp_path: Path):
    """One training step with TSM stub does not raise — backward is skipped."""
    trainer = _make_trainer(tilt=1.0, tmp_path=tmp_path)
    log = trainer.train_step(_STUB_BATCH)
    assert log.get("loss_stub_active") is True, (
        f"Expected loss_stub_active=True, got: {log}"
    )


def test_tsm_stub_epoch_no_crash(tmp_path: Path):
    """Full epoch with TSM stub completes and logs loss_stub_active=True."""
    trainer = _make_trainer(tilt=1.0, tmp_path=tmp_path)
    log = trainer.train_epoch([_STUB_BATCH])
    assert log.get("loss_stub_active") == 1.0, (  # aggregated as float mean of True=1
        f"Expected loss_stub_active=1.0 in epoch log, got: {log}"
    )


def test_tsm_stub_checkpoint_saved(tmp_path: Path):
    """Checkpoint embeds stub_active=True when TSM is unimplemented."""
    trainer = _make_trainer(tilt=1.0, tmp_path=tmp_path)
    trainer.train_epoch([_STUB_BATCH])
    ckpt_path = trainer.save_checkpoint()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt["stub_active"] is True


# ── Multiple tilt values ──────────────────────────────────────────────

@pytest.mark.parametrize("tilt", [-2.0, -1.0, 2.0, 5.0, 10.0])
def test_tsm_stub_various_tilts_no_crash(tilt: float, tmp_path: Path):
    """TSM stub handles all tilt values without crashing."""
    trainer = _make_trainer(tilt=tilt, tmp_path=tmp_path)
    log = trainer.train_step(_STUB_BATCH)
    assert log.get("loss_stub_active") is True
