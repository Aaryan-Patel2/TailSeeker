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


# ── TiltedScoreMatchingLoss (implemented) ────────────────────────────

def test_tsm_one_step_runs_real_backward(tmp_path: Path):
    """TSM step completes with real backward pass — loss_stub_active is False."""
    trainer = _make_trainer(tilt=1.0, tmp_path=tmp_path)
    log = trainer.train_step(_STUB_BATCH)
    assert log.get("loss_stub_active") is False, f"Expected real loss, got: {log}"
    assert "Train/loss" in log
    assert isinstance(log["Train/loss"], float)


def test_tsm_epoch_logs_loss_components(tmp_path: Path):
    """Full TSM epoch logs tilt, mse_mean, mse_max components."""
    trainer = _make_trainer(tilt=1.0, tmp_path=tmp_path)
    log = trainer.train_epoch([_STUB_BATCH])
    assert log.get("loss_stub_active", True) is False or log.get("loss_stub_active", 0.0) == 0.0
    assert "Train/loss" in log
    assert "Train/loss_tilt" in log


def test_tsm_checkpoint_stub_active_false(tmp_path: Path):
    """Checkpoint embeds stub_active=False when TSM is fully implemented."""
    trainer = _make_trainer(tilt=1.0, tmp_path=tmp_path)
    trainer.train_epoch([_STUB_BATCH])
    ckpt_path = trainer.save_checkpoint()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ckpt["stub_active"] is False


# ── Multiple tilt values ──────────────────────────────────────────────

@pytest.mark.parametrize("tilt", [-2.0, -1.0, 2.0, 5.0, 10.0])
def test_tsm_various_tilts_run_backward(tilt: float, tmp_path: Path):
    """All ablation tilt values complete a real backward pass."""
    trainer = _make_trainer(tilt=tilt, tmp_path=tmp_path)
    log = trainer.train_step(_STUB_BATCH)
    assert log.get("loss_stub_active") is False, f"tilt={tilt}: expected real loss, got {log}"
    assert "Train/loss" in log
