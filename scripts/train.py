"""Hydra training entrypoint.

Usage:
    python scripts/train.py                       # default config
    python scripts/train.py loss.tilt=2.0 seed=1  # override
    python scripts/train.py --multirun +experiment=ablation_tilt
"""

from __future__ import annotations

import csv
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

    # 5. Loss function — single-objective, multi-objective, or reward-tilt
    loss_mode: str = str(cfg.loss.get("mode", "single"))
    if loss_mode == "multi":
        from src.losses.hierarchical_loss import get_hierarchical_loss_fn
        loss_fn = get_hierarchical_loss_fn(
            outer_tilt=float(cfg.loss.outer_tilt),
            group_tilts=list(cfg.loss.group_tilts),
            gumbel_temp=float(cfg.loss.gumbel_temp),
        )
        print(f"[train.py] loss=multi  outer_tilt={cfg.loss.outer_tilt}  "
              f"group_tilts={list(cfg.loss.group_tilts)}  gumbel_temp={cfg.loss.gumbel_temp}")
    elif loss_mode == "reward":
        from src.losses.reward_weighted_loss import get_reward_loss_fn
        rt = cfg.loss.reward_tilt
        loss_fn = get_reward_loss_fn(
            tilt=float(rt.tilt),
            lambda_=float(rt.get("lambda", 0.5)),
            warmup_frac=float(rt.get("warmup_frac", 0.3)),
            lambda_logp=float(rt.get("lambda_logp", 0.0)),
            lambda_tpsa=float(rt.get("lambda_tpsa", 0.0)),
        )
        print(f"[train.py] loss=reward  tilt={rt.tilt}  lambda={rt.get('lambda', 0.5)}  "
              f"lambda_logp={rt.get('lambda_logp', 0.0)}  "
              f"lambda_tpsa={rt.get('lambda_tpsa', 0.0)}")
    else:
        loss_fn = None  # Trainer builds get_loss_fn(tilt) internally

    # 6. Trainer
    trainer_cfg = {
        **OmegaConf.to_container(cfg.loss, resolve=True),
        **OmegaConf.to_container(cfg.diffusion, resolve=True),
        **OmegaConf.to_container(cfg.training, resolve=True),
    }
    trainer = Trainer(model, trainer_cfg, output_dir, loss_fn=loss_fn)

    # 7. Dataloader — real QM9 if available, synthetic stub otherwise
    dataloader = _make_dataloader(cfg, device)

    # 8. W&B (optional)
    wandb_run = _init_wandb(cfg, output_dir)

    # 9. Train
    tilt = (float(cfg.loss.reward_tilt.tilt) if loss_mode == "reward"
            else float(cfg.loss.tilt))
    print(f"[train.py] mode={loss_mode}  tilt={tilt}  seed={cfg.seed}  "
          f"device={device}  output={output_dir}")

    # CSV loss log — one row per epoch, written incrementally
    csv_path = output_dir / "losses.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_fields = ["epoch", "loss", "tilt", "mode", "seed",
                  "mean_reward_weight", "reward_std", "reward_erm_fallback"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields, extrasaction="ignore")
    csv_writer.writeheader()

    for epoch in range(int(cfg.training.max_epochs)):
        epoch_log = trainer.train_epoch(dataloader)

        stub = epoch_log.get("loss_stub_active", False)
        if stub:
            print(f"  epoch {epoch:04d}  loss_stub_active=True  (backward skipped)")
        else:
            loss_val = epoch_log.get("Train/loss", float("nan"))
            reward_info = ""
            if loss_mode == "reward":
                t_eff = epoch_log.get("Train/reward_tilt_t_eff", float("nan"))
                r_std = epoch_log.get("Train/reward_std", float("nan"))
                reward_info = f"  t_eff={t_eff:.3f}  reward_std={r_std:.3f}"
            print(f"  epoch {epoch:04d}  loss={loss_val:.6f}{reward_info}")
            tilt_val = (float(cfg.loss.reward_tilt.tilt) if loss_mode == "reward"
                        else float(cfg.loss.tilt))
            csv_writer.writerow({
                "epoch": epoch, "loss": loss_val,
                "tilt": tilt_val, "mode": loss_mode, "seed": int(cfg.seed),
                "mean_reward_weight": epoch_log.get("Train/reward_tilt_t_eff", ""),
                "reward_std": epoch_log.get("Train/reward_std", ""),
                "reward_erm_fallback": epoch_log.get("Train/reward_erm_fallback", ""),
            })
            csv_file.flush()

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, **epoch_log})

        # save checkpoint at end of epoch 0 and every 10 epochs
        if epoch == 0 or (epoch + 1) % 10 == 0:
            ckpt_path = trainer.save_checkpoint()
            print(f"  checkpoint saved → {ckpt_path}")

    csv_file.close()

    if wandb_run is not None:
        wandb_run.finish()

    print("[train.py] done.")


def _make_dataloader(cfg: DictConfig, device: torch.device):
    """Real QM9 DataLoader; falls back to synthetic stub if data unavailable."""
    from torch.utils.data import DataLoader

    from src.data.qm9 import QM9Dataset

    data_root = Path(cfg.data.root)
    download = bool(cfg.data.get("download", False))
    try:
        dataset = QM9Dataset(
            root=data_root,
            split="train",
            max_atoms=int(cfg.data.max_atoms),
            download=download,
        )
        print(f"[train.py] QM9 train split: {len(dataset)} molecules")
        return DataLoader(
            dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=True,
            num_workers=int(cfg.data.num_workers),
            collate_fn=QM9Dataset.collate_fn,
        )
    except (FileNotFoundError, RuntimeError, ImportError, AssertionError, AttributeError) as exc:
        print(f"[train.py] QM9 unavailable ({type(exc).__name__}); using synthetic stub data")
        return _make_stub_dataloader(cfg, device)


def _make_stub_dataloader(cfg: DictConfig, device: torch.device):
    """Synthetic stub: shape matches real QM9 output (B, C, max_atoms, max_atoms).

    Includes qed/sa/group so reward-tilt and multi-objective modes work without data.
    QED drawn from Beta(2,2) ≈ realistic drug-likeness distribution.
    SA drawn uniformly from [1, 10].
    """
    B = int(cfg.training.batch_size)
    C = int(cfg.model.in_channels)
    N = int(cfg.data.max_atoms)
    qed = torch.distributions.Beta(2.0, 2.0).sample((B,))          # (B,) in [0,1]
    sa = torch.rand(B) * 9.0 + 1.0                                  # (B,) in [1,10]
    stub_batch = {
        "coords": torch.randn(B, C, N, N),
        "qed": qed,
        "sa": sa,
        "logp": torch.randn(B) * 1.5 + 2.0,                         # LogP ~ N(2, 1.5)
        "tpsa": torch.rand(B) * 100.0 + 20.0,                       # TPSA ~ U[20, 120] Å²
        "group": (qed * 3).long().clamp(0, 2),
    }
    return [stub_batch]


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
