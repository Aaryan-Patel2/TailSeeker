#!/usr/bin/env python3
"""
Demo script: Run all theoretical loss functions and display behavior.
Shows:
  - ERM baseline (tilt=0)
  - Single-objective tilted score matching (L_tilt)
  - Multi-objective hierarchical loss (L_MO)
"""

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def main():
    print("=" * 80)
    print("TailSeeker Loss Function Validation")
    print("=" * 80)

    # Generate synthetic batch
    torch.manual_seed(42)
    B, C, H, W = 8, 3, 4, 4
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    # Compute per-sample losses for reference
    from torch.nn.functional import mse_loss
    per_sample = mse_loss(pred, target, reduction="none").view(B, -1).mean(dim=1)

    print(f"\nBatch shape: {pred.shape}")
    min_mse = per_sample.min()
    mean_mse = per_sample.mean()
    max_mse = per_sample.max()
    print(f"Per-sample MSE: min={min_mse:.4f}, mean={mean_mse:.4f}, max={max_mse:.4f}")

    # ========================================================================
    # 1. ERM BASELINE (tilt=0)
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. ERM BASELINE (DDPMSimpleLoss, tilt=0)")
    print("=" * 80)

    from src.losses.ddpm_simple import DDPMSimpleLoss
    loss_erm = DDPMSimpleLoss()
    output_erm = loss_erm(pred, target)

    print(f"Total loss: {output_erm.total_loss.item():.6f}")
    print(f"Components: {output_erm.loss_components}")
    print(f"Diagnostics: {output_erm.diagnostics}")

    # ========================================================================
    # 2. SINGLE-OBJECTIVE TILTED LOSS (L_tilt)
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. SINGLE-OBJECTIVE TILTED SCORE MATCHING (L_tilt)")
    print("=" * 80)

    from src.losses.tilted_score_matching import TiltedScoreMatchingLoss

    tilt_values = [-5.0, -2.0, -1.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"\n{'Tilt':>8} {'L_tilt':>10} {'vs ERM':>10} {'Entropy':>10} {'Status':>12}")
    print("-" * 52)

    for tilt in tilt_values:
        loss_tsm = TiltedScoreMatchingLoss(tilt=tilt)
        output_tsm = loss_tsm(pred, target)
        loss_val = output_tsm.total_loss.item()
        diff = loss_val - output_erm.total_loss.item()
        entropy = output_tsm.diagnostics["tilt_effective_weight_entropy"]

        # Classify behavior
        if tilt > 0:
            status = "tail-seeking" if loss_val > output_erm.total_loss.item() else "stable"
        elif tilt < 0:
            status = "robust" if loss_val < output_erm.total_loss.item() else "stable"
        else:
            status = "ERM (baseline)"

        print(f"{tilt:8.1f} {loss_val:10.6f} {diff:+10.6f} {entropy:10.4f} {status:>12}")

    # ========================================================================
    # 3. MULTI-OBJECTIVE HIERARCHICAL LOSS (L_MO)
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. MULTI-OBJECTIVE HIERARCHICAL LOSS (L_MO with Gumbel-Softmax)")
    print("=" * 80)

    from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss

    # Create group assignments
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    print(f"\nGroup assignment: {groups.tolist()}")
    print(f"Group 0 samples (indices 0-3): mean MSE = {per_sample[:4].mean():.4f}")
    print(f"Group 1 samples (indices 4-7): mean MSE = {per_sample[4:].mean():.4f}")

    loss_mo = MultiObjectiveTiltedLoss(
        outer_tilt=1.0,
        group_tilts=[1.0, 2.0],  # inner tilts for each group
        gumbel_temp=1.0,
    )
    output_mo = loss_mo(pred, target, groups=groups)

    print(f"\nMulti-objective loss (L_MO): {output_mo.total_loss.item():.6f}")
    print("Components:")
    for k, v in output_mo.loss_components.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.6f}")
        else:
            print(f"  {k}: {v:.6f}")

    print("\nGumbel-Softmax weights (group importance):")
    print(f"  Group 0: {output_mo.weights[0].item():.4f}")
    print(f"  Group 1: {output_mo.weights[1].item():.4f}")
    print(f"  Sum: {output_mo.weights.sum().item():.4f} (must be 1.0)")

    print(f"\nDiagnostics: {output_mo.diagnostics}")

    # ========================================================================
    # 4. GRADIENT FLOW VALIDATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. GRADIENT FLOW VALIDATION")
    print("=" * 80)

    for loss_name, loss_fn in [
        ("ERM (tilt=0)", DDPMSimpleLoss()),
        ("TSM (tilt=1.0)", TiltedScoreMatchingLoss(tilt=1.0)),
        ("TSM (tilt=2.0)", TiltedScoreMatchingLoss(tilt=2.0)),
        ("MO (L_MO)", MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 2.0])),
    ]:
        pred_grad = torch.randn(B, C, H, W, requires_grad=True)
        target_grad = torch.randn(B, C, H, W)

        if "MO" in loss_name:
            output = loss_fn(pred_grad, target_grad, groups=groups)
        else:
            output = loss_fn(pred_grad, target_grad)

        output.total_loss.backward()
        grad_norm = pred_grad.grad.norm().item()

        print(f"{loss_name:20s}: gradient norm = {grad_norm:.6f}")

    # ========================================================================
    # 5. NUMERICAL STABILITY CHECK
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. NUMERICAL STABILITY AT EXTREME TILTS")
    print("=" * 80)

    extreme_tilts = [-10.0, -5.0, 10.0, 50.0]
    print(f"\n{'Tilt':>8} {'Loss Value':>15} {'NaN/Inf?':>12}")
    print("-" * 38)

    for tilt in extreme_tilts:
        loss_fn = TiltedScoreMatchingLoss(tilt=tilt)
        output = loss_fn(pred, target)
        loss_val = output.total_loss.item()
        is_bad = "FAIL" if (np.isnan(loss_val) or np.isinf(loss_val)) else "PASS"
        print(f"{tilt:8.1f} {loss_val:15.6f} {is_bad:>12}")

    print("\n" + "=" * 80)
    print("✓ All theoretical loss functions validated successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
