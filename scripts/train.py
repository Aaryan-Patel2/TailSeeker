"""Hydra training entrypoint.

Usage:
    python scripts/train.py                       # default config
    python scripts/train.py loss.tilt=2.0 seed=1  # override
    python scripts/train.py --multirun +experiment=ablation_tilt
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# make src/ importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.ddpm_unet import get_model
from src.training.trainer import Trainer
from src.utils import set_seed


@hydra.main(config_path="../config", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Full training run. set_seed() is the first call."""
    try:
        _run(cfg)
    except Exception:
        tb = traceback.format_exc()
        hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        crash_path = hydra_dir / "crash_report.txt"
        crash_path.write_text(tb)
        # also patch CLAUDE.md with a dated crash note
        _patch_claude_md_crash(tb)
        raise


def _run(cfg: DictConfig) -> None:
    # 1. Seed — must be first
    set_seed(int(cfg.seed))

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Log full config
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # 3. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Model
    model = get_model(cfg.model).to(device)

    # 5. Trainer
    trainer_cfg = {
        **OmegaConf.to_container(cfg.loss, resolve=True),
        **OmegaConf.to_container(cfg.diffusion, resolve=True),
        **OmegaConf.to_container(cfg.training, resolve=True),
    }
    trainer = Trainer(model, trainer_cfg, output_dir)

    # 6. Stub-mode dataloader (synthetic until QM9 is wired up)
    dataloader = _make_stub_dataloader(cfg, device)

    # 7. W&B (optional)
    wandb_run = _init_wandb(cfg, output_dir)

    # 8. Train
    tilt = float(cfg.loss.tilt)
    print(f"[train.py] tilt={tilt}  seed={cfg.seed}  device={device}  output={output_dir}")

    for epoch in range(int(cfg.training.max_epochs)):
        epoch_log = trainer.train_epoch(dataloader)

        stub = epoch_log.get("loss_stub_active", False)
        if stub:
            print(f"  epoch {epoch:04d}  loss_stub_active=True  (backward skipped)")
        else:
            loss_val = epoch_log.get("Train/loss", float("nan"))
            print(f"  epoch {epoch:04d}  loss={loss_val:.6f}")

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, **epoch_log})

        # save checkpoint at end of epoch 0 and every 10 epochs
        if epoch == 0 or (epoch + 1) % 10 == 0:
            ckpt_path = trainer.save_checkpoint()
            print(f"  checkpoint saved → {ckpt_path}")

        # one epoch is enough to verify the harness
        break  # TODO: remove once QM9 dataloader is wired up

    if wandb_run is not None:
        wandb_run.finish()

    print("[train.py] done.")


def _make_stub_dataloader(cfg: DictConfig, device: torch.device):
    """Return a single-batch iterable of synthetic data for stub mode."""
    B = int(cfg.training.batch_size)
    C = int(cfg.model.in_channels)
    # Use a small spatial size for fast stub runs
    stub_batch = {"coords": torch.randn(B, C, 8, 8)}
    return [stub_batch]  # one-element list = one batch per epoch


def _init_wandb(cfg: DictConfig, output_dir: Path):
    """Init W&B if entity is set and mode != disabled. Returns run or None."""
    wc = cfg.wandb
    if wc.mode == "disabled" or wc.entity is None:
        return None
    try:
        import wandb
        run = wandb.init(
            project=wc.project,
            entity=wc.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(output_dir),
            mode=wc.mode,
        )
        return run
    except Exception as exc:
        print(f"[train.py] W&B init failed (continuing without logging): {exc}")
        return None


def _patch_claude_md_crash(tb: str) -> None:
    """Append a crash note to .claude/CLAUDE.md."""
    claude_md = Path(__file__).resolve().parents[1] / ".claude" / "CLAUDE.md"
    if not claude_md.exists():
        return
    from datetime import date
    note = (
        f"\n### ## Crash {date.today()} — scripts/train.py\n"
        f"```\n{tb[-2000:]}\n```\n"
    )
    with open(claude_md, "a") as f:
        f.write(note)


if __name__ == "__main__":
    main()
