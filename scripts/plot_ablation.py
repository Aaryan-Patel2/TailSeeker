#!/usr/bin/env python3
"""
Ablation results plotter.

Reconstructs per-checkpoint loss curves from saved model weights by
evaluating each checkpoint on a fixed synthetic batch. Produces:
  1. Loss curves per tilt value (mean ± std over 3 seeds)
  2. Final loss vs. tilt bar chart
  3. Monotonicity verification: loss-vs-tilt at fixed epoch

Usage:
    python scripts/plot_ablation.py
    python scripts/plot_ablation.py --multirun-dir multirun/2026-04-12/20-01-33
    python scripts/plot_ablation.py --epochs-only  # skip reconstruction, show final epoch only
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F

from src.losses.tilted_score_matching import get_loss_fn
from src.models.ddpm_unet import get_model
from dotmap import DotMap


# ── Fixed evaluation batch (same seed → reproducible reconstruction) ──────────
EVAL_SEED = 0
EVAL_B = 64
EVAL_C = 4
EVAL_N = 29
DEVICE = torch.device("cpu")

TILT_VALUES = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]
SEEDS = [0, 1, 2]
CHECKPOINT_EPOCHS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Colour map: blue (negative tilt) → grey (zero) → red (positive tilt)
import matplotlib
_CMAP_POS = matplotlib.colormaps["Reds"]
_CMAP_NEG = matplotlib.colormaps["Blues"]


def tilt_color(t: float):
    vals = TILT_VALUES
    if t < 0:
        idx = vals.index(t)
        neg_vals = [v for v in vals if v < 0]
        frac = (neg_vals.index(t) + 1) / len(neg_vals)
        return _CMAP_NEG(0.4 + 0.5 * frac)
    elif t == 0:
        return "dimgray"
    else:
        pos_vals = [v for v in vals if v > 0]
        frac = (pos_vals.index(t) + 1) / len(pos_vals)
        return _CMAP_POS(0.4 + 0.5 * frac)


def make_eval_batch():
    torch.manual_seed(EVAL_SEED)
    coords = torch.randn(EVAL_B, EVAL_C, EVAL_N, EVAL_N)
    return {"coords": coords}


def default_model_cfg():
    return DotMap({
        "in_channels": EVAL_C,
        "out_channels": EVAL_C,
        "model_channels": 64,
        "num_res_blocks": 2,
        "dropout": 0.0,
    })


def eval_checkpoint(ckpt_path: Path, tilt: float, eval_batch: dict) -> float:
    """Load checkpoint and compute loss on fixed eval batch. Returns scalar loss."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = get_model(default_model_cfg()).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loss_fn = get_loss_fn(tilt)
    x0 = eval_batch["coords"].to(DEVICE)
    B = x0.shape[0]

    # Simple single-timestep evaluation (t=500, midpoint of schedule)
    with torch.no_grad():
        noise = torch.randn_like(x0)
        # approximate noisy input at t=500 using closed-form alpha ≈ 0.5
        alpha = 0.5
        x_t = alpha ** 0.5 * x0 + (1 - alpha) ** 0.5 * noise

        t_tensor = torch.full((B,), 500, dtype=torch.long, device=DEVICE)
        from src.models.base import ModelOutput
        pred_noise = model(x_t, t_tensor).pred_noise
        loss_out = loss_fn(pred_noise, noise)
        return loss_out.total_loss.item()


def collect_results(multirun_dir: Path) -> dict:
    """
    Returns results[tilt][seed] = list of (epoch, loss) pairs.
    Reads losses.csv if present (fast); falls back to checkpoint reconstruction.
    """
    results: dict[float, dict[int, list]] = {t: {} for t in TILT_VALUES}
    eval_batch = make_eval_batch()

    total_jobs = len(TILT_VALUES) * len(SEEDS)
    done = 0

    for job_dir in sorted(multirun_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 999):
        if not job_dir.is_dir() or not job_dir.name.isdigit():
            continue
        override = job_dir / ".hydra" / "overrides.yaml"
        if not override.exists():
            continue
        tilt = seed = None
        for line in override.read_text().splitlines():
            if "loss.tilt=" in line:
                tilt = float(line.split("=")[1])
            if "seed=" in line:
                seed = int(line.split("=")[1])
        if tilt is None or seed is None or tilt not in TILT_VALUES:
            continue

        # ── Fast path: read losses.csv written during training ────────────
        csv_path = job_dir / "losses.csv"
        if csv_path.exists():
            epoch_losses = []
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    epoch_losses.append((int(row["epoch"]), float(row["loss"])))
            if epoch_losses:
                results[tilt][seed] = epoch_losses
                done += 1
                print(f"  [{done}/{total_jobs}] tilt={tilt:>5.1f}  seed={seed}  "
                      f"final_loss={epoch_losses[-1][1]:.4f}  (from CSV)")
                continue

        # ── Fallback: reconstruct from checkpoints ────────────────────────
        ckpts = sorted(
            job_dir.glob("checkpoints/*.pt"),
            key=lambda p: int(re.search(r"epoch_(\d+)", p.name).group(1)),
        )
        if not ckpts:
            continue

        epoch_losses = []
        for ckpt in ckpts:
            epoch_num = int(re.search(r"epoch_(\d+)", ckpt.name).group(1))
            loss_val = eval_checkpoint(ckpt, tilt, eval_batch)
            epoch_losses.append((epoch_num, loss_val))

        results[tilt][seed] = epoch_losses
        done += 1
        print(f"  [{done}/{total_jobs}] tilt={tilt:>5.1f}  seed={seed}  "
              f"final_loss={epoch_losses[-1][1]:.4f}  (reconstructed)")

    return results


def plot_loss_curves(results: dict, out_dir: Path):
    """Plot 1: loss curves per tilt, mean ± std over seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for tilt in TILT_VALUES:
        seed_data = results.get(tilt, {})
        if not seed_data:
            continue
        # Align on common epochs
        all_epochs = sorted({ep for sd in seed_data.values() for ep, _ in sd})
        per_seed_losses = []
        for seed, curve in seed_data.items():
            epoch_map = dict(curve)
            per_seed_losses.append([epoch_map[e] for e in all_epochs if e in epoch_map])

        if not per_seed_losses:
            continue
        arr = np.array(per_seed_losses)  # (n_seeds, n_epochs)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        epochs = [e for e in all_epochs if e in dict(list(seed_data.values())[0])]

        color = tilt_color(tilt)
        label = f"tilt={tilt:+g}" + (" (ERM)" if tilt == 0 else "")
        ax.plot(epochs, mean, color=color, linewidth=2, label=label)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (eval on fixed batch)", fontsize=12)
    ax.set_title("TailSeeker Ablation: Loss Curves per Tilt Value\n(mean ± std over 3 seeds)", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = out_dir / "loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {path}")


def plot_final_loss_vs_tilt(results: dict, out_dir: Path):
    """Plot 2: final-epoch loss vs. tilt bar chart."""
    tilts, means, stds = [], [], []
    for tilt in TILT_VALUES:
        seed_data = results.get(tilt, {})
        finals = [curve[-1][1] for curve in seed_data.values() if curve]
        if not finals:
            continue
        tilts.append(tilt)
        means.append(np.mean(finals))
        stds.append(np.std(finals))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [tilt_color(t) for t in tilts]
    x = np.arange(len(tilts))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, width=0.6, edgecolor="black", linewidth=0.8)

    # Highlight ERM baseline
    if 0.0 in tilts:
        erm_idx = tilts.index(0.0)
        bars[erm_idx].set_edgecolor("black")
        bars[erm_idx].set_linewidth(2.5)
        ax.axhline(means[erm_idx], color="dimgray", linestyle="--", linewidth=1.2, alpha=0.6, label=f"ERM baseline = {means[erm_idx]:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:+g}" for t in tilts], fontsize=11)
    ax.set_xlabel("Tilt (τ)", fontsize=12)
    ax.set_ylabel("Final Loss (epoch 100)", fontsize=12)
    ax.set_title("Final Loss vs. Tilt Value\n(mean ± std over 3 seeds)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = out_dir / "final_loss_vs_tilt.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_monotonicity(results: dict, out_dir: Path):
    """Plot 3: loss-vs-tilt at epochs 1, 50, 100."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_ep = {1: "#b0c4de", 50: "#4682b4", 100: "#00008b"}

    for check_epoch in [1, 50, 100]:
        tilts_x, loss_y = [], []
        for tilt in TILT_VALUES:
            seed_data = results.get(tilt, {})
            vals = []
            for curve in seed_data.values():
                epoch_map = dict(curve)
                if check_epoch in epoch_map:
                    vals.append(epoch_map[check_epoch])
            if vals:
                tilts_x.append(tilt)
                loss_y.append(np.mean(vals))
        if tilts_x:
            ax.plot(tilts_x, loss_y, marker="o", linewidth=2,
                    color=colors_ep[check_epoch], label=f"Epoch {check_epoch}")

    ax.set_xlabel("Tilt (τ)", fontsize=12)
    ax.set_ylabel("Mean Loss across Seeds", fontsize=12)
    ax.set_title("Loss vs. Tilt at Selected Epochs\n(Jensen's inequality: t>0 raises loss above ERM)", fontsize=13)
    ax.legend(fontsize=11)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = out_dir / "monotonicity_vs_tilt.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multirun-dir",
        default="multirun/2026-04-12/20-01-33",
        help="Path to Hydra multirun directory containing job subdirectories",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/ablation_plots",
        help="Directory to save generated plots",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    multirun_dir = repo_root / args.multirun_dir
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not multirun_dir.exists():
        print(f"ERROR: multirun directory not found: {multirun_dir}")
        sys.exit(1)

    print(f"Loading checkpoints from: {multirun_dir}")
    print(f"Evaluating on fixed synthetic batch (seed={EVAL_SEED}, B={EVAL_B})...\n")

    results = collect_results(multirun_dir)

    n_runs = sum(len(v) for v in results.values())
    if n_runs == 0:
        print("No completed runs found. Run the ablation first:")
        print("  python scripts/train.py --multirun loss.tilt=-5,-2,-1,0,1,2,5,10 seed=0,1,2")
        sys.exit(1)

    print(f"\nGenerating plots for {n_runs} runs...")
    plot_loss_curves(results, out_dir)
    plot_final_loss_vs_tilt(results, out_dir)
    plot_monotonicity(results, out_dir)

    print(f"\nAll plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
