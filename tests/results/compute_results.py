"""Compute numerical results for theory_verification.tex.

Run from repo root:
    python tests/results/compute_results.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F

from src.losses.ddpm_simple import DDPMSimpleLoss
from src.losses.hierarchical_loss import MultiObjectiveTiltedLoss
from src.losses.tilted_score_matching import TiltedScoreMatchingLoss


def fmt(x, d=6):
    if isinstance(x, torch.Tensor):
        x = x.item()
    return f"{x:.{d}f}"


results = {}

# ── §1a ERM limit ─────────────────────────────────────────────────────────────
torch.manual_seed(2)
pred, target = torch.randn(32, 4), torch.randn(32, 4)
per_sample = F.mse_loss(pred, target, reduction="none").view(32, -1).mean(dim=1)
mean_mse = per_sample.mean().item()
L_tiny = TiltedScoreMatchingLoss(tilt=1e-3)(pred, target).total_loss.item()
results["erm_limit_mean_mse"] = mean_mse
results["erm_limit_L_tiny"]   = L_tiny
results["erm_limit_diff"]     = abs(L_tiny - mean_mse)

# ── §1b Minimax limit ─────────────────────────────────────────────────────────
torch.manual_seed(7)
pred2, target2 = torch.randn(16, 4), torch.randn(16, 4)
pred2[0] = pred2[0] * 0.0
target2[0] = torch.ones(4) * 5.0
out_mm = TiltedScoreMatchingLoss(tilt=100.0)(pred2, target2)
max_f = out_mm.per_sample_loss.max().item()
L_t100 = out_mm.total_loss.item()
results["minimax_max_f"]   = max_f
results["minimax_L_t100"]  = L_t100
results["minimax_rel_err"] = abs(L_t100 - max_f) / max(max_f, 1e-8)

# ── §1c Monotonicity ─────────────────────────────────────────────────────────
torch.manual_seed(42)
pred3, target3 = torch.randn(16, 4), torch.randn(16, 4)
tilt_values = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0]
mono_losses = []
for t in tilt_values:
    out = TiltedScoreMatchingLoss(tilt=t)(pred3, target3)
    mono_losses.append(out.total_loss.item())
results["monotonicity_tilts"]  = tilt_values
results["monotonicity_losses"] = mono_losses

# ── §1d Gradient weighting ────────────────────────────────────────────────────
torch.manual_seed(3)
pred4, target4 = torch.randn(8, 4), torch.randn(8, 4)
tilt_gw = 2.0
out_gw = TiltedScoreMatchingLoss(tilt=tilt_gw)(pred4, target4)
expected_w = torch.softmax(tilt_gw * out_gw.per_sample_loss, dim=0)
results["grad_weight_tilt"]       = tilt_gw
results["grad_weight_max_err"]    = (out_gw.weights - expected_w).abs().max().item()
results["grad_weight_hardest_idx"] = out_gw.per_sample_loss.argmax().item()
results["grad_weight_hardest_w"]  = out_gw.weights.max().item()
results["grad_weight_per_sample"] = out_gw.per_sample_loss.tolist()
results["grad_weight_weights"]    = out_gw.weights.tolist()

# ── §2a Hierarchical collapse ─────────────────────────────────────────────────
torch.manual_seed(5)
pred5, target5 = torch.randn(16, 4), torch.randn(16, 4)
t_collapse = 2.0
groups5 = torch.repeat_interleave(torch.arange(4), 4)
mo_fn5 = MultiObjectiveTiltedLoss(outer_tilt=t_collapse, group_tilts=[t_collapse]*4)
v1_fn5 = TiltedScoreMatchingLoss(tilt=t_collapse)
mo_out5 = mo_fn5(pred5, target5, groups=groups5)
v1_out5 = v1_fn5(pred5, target5)
j_tilt5 = mo_out5.loss_components["j_tilt"].item()
l_t5    = v1_out5.total_loss.item()
results["hier_collapse_j_tilt"] = j_tilt5
results["hier_collapse_l_t"]    = l_t5
results["hier_collapse_diff"]   = abs(j_tilt5 - l_t5)

# ── §2b Uniform group weighting ───────────────────────────────────────────────
torch.manual_seed(6)
pred6, target6 = torch.randn(8, 4), torch.randn(8, 4)
groups6 = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
tiny_t = 1e-4
mo_fn6 = MultiObjectiveTiltedLoss(outer_tilt=tiny_t, group_tilts=[1.0, 1.0])
mo_out6 = mo_fn6(pred6, target6, groups=groups6)
j6 = mo_out6.loss_components["j_tilt"].item()
per6 = F.mse_loss(pred6, target6, reduction="none").view(8, -1).mean(dim=1)
tau6 = 1.0
r6_0 = (torch.logsumexp(tau6 * per6[:4], dim=0) - math.log(4)).item() / tau6
r6_1 = (torch.logsumexp(tau6 * per6[4:], dim=0) - math.log(4)).item() / tau6
expected_mean6 = 0.5 * (r6_0 + r6_1)
results["uniform_j_tilt"]      = j6
results["uniform_mean_risks"]  = expected_mean6
results["uniform_r0"]          = r6_0
results["uniform_r1"]          = r6_1
results["uniform_diff"]        = abs(j6 - expected_mean6)

# ── §3a Gumbel temperature limit ─────────────────────────────────────────────
torch.manual_seed(99)
B7 = 16
pred7 = torch.zeros(B7, 4)
target7 = torch.zeros(B7, 4)
target7[:B7 // 2] = 10.0
groups7 = torch.cat([torch.zeros(B7//2, dtype=torch.long),
                     torch.ones(B7//2, dtype=torch.long)])
lo_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 1.0], gumbel_temp=1e-3)
lo_out = lo_fn(pred7, target7, groups=groups7)
results["gumbel_low_temp_weights"] = lo_out.weights.tolist()
results["gumbel_low_temp_max_w"]   = lo_out.weights.max().item()

# Also show at high temperature for contrast
hi_fn = MultiObjectiveTiltedLoss(outer_tilt=1.0, group_tilts=[1.0, 1.0], gumbel_temp=10.0)
hi_out = hi_fn(pred7, target7, groups=groups7)
results["gumbel_high_temp_weights"] = hi_out.weights.tolist()

# ── §4a Convexity ─────────────────────────────────────────────────────────────
torch.manual_seed(17)
pred_a = torch.randn(16, 4)
pred_b = torch.randn(16, 4)
target8 = torch.randn(16, 4)
pred_avg = 0.5 * (pred_a + pred_b)
convex_results = {}
for tilt in [0.5, 1.0, 2.0, 5.0]:
    fn = TiltedScoreMatchingLoss(tilt=tilt)
    L_avg = fn(pred_avg, target8).total_loss.item()
    L_1   = fn(pred_a, target8).total_loss.item()
    L_2   = fn(pred_b, target8).total_loss.item()
    avg_L = 0.5 * (L_1 + L_2)
    convex_results[tilt] = {"L_avg": L_avg, "avg_L": avg_L, "gap": avg_L - L_avg}
results["convexity"] = convex_results

# ── §4b Invalid latent ────────────────────────────────────────────────────────
target9 = torch.ones(8, 4) * 3.0
invalid_results = {}
for label, bp in {
    "all_zeros": torch.zeros(8, 4),
    "high_noise": torch.randn(8, 4) * 100.0,
    "all_ones": torch.ones(8, 4),
}.items():
    erm_loss = DDPMSimpleLoss()(bp, target9).total_loss.item()
    tsm_loss = TiltedScoreMatchingLoss(tilt=1.0)(bp, target9).total_loss.item()
    invalid_results[label] = {"erm": erm_loss, "tsm_t1": tsm_loss}
results["invalid_latent"] = invalid_results

# ── §4c logsumexp stress ──────────────────────────────────────────────────────
torch.manual_seed(23)
pred10 = torch.randn(8, 4)
target10 = torch.randn(8, 4)
pred10[0] = torch.zeros(4)
target10[0] = torch.ones(4) * 316.0
stress_results = {}
for tilt in [1.0, 5.0, 10.0]:
    out = TiltedScoreMatchingLoss(tilt=tilt)(pred10, target10)
    stress_results[tilt] = {
        "loss": out.total_loss.item(),
        "max_f": out.per_sample_loss.max().item(),
        "is_finite": torch.isfinite(out.total_loss).item(),
    }
results["logsumexp_stress"] = stress_results

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n=== NUMERICAL RESULTS FOR TEX ===\n")
print(f"ERM limit: mean_mse={fmt(results['erm_limit_mean_mse'])}, "
      f"L_t(1e-3)={fmt(results['erm_limit_L_tiny'])}, "
      f"diff={fmt(results['erm_limit_diff'])}")

print(f"\nMinimax limit (t=100): max_f={fmt(results['minimax_max_f'])}, "
      f"L_t={fmt(results['minimax_L_t100'])}, "
      f"rel_err={fmt(results['minimax_rel_err'])}")

print(f"\nMonotonicity: tilts={results['monotonicity_tilts']}")
print(f"              losses={[fmt(x,4) for x in results['monotonicity_losses']]}")

print(f"\nGrad weights (t={results['grad_weight_tilt']}): max_abs_err={fmt(results['grad_weight_max_err'])}")
print(f"  per_sample_loss={[fmt(x,4) for x in results['grad_weight_per_sample']]}")
print(f"  weights        ={[fmt(x,4) for x in results['grad_weight_weights']]}")

print(f"\nHierarchical collapse: J̃={fmt(results['hier_collapse_j_tilt'])}, "
      f"L_t={fmt(results['hier_collapse_l_t'])}, diff={fmt(results['hier_collapse_diff'])}")

print(f"\nUniform weighting: J̃(t→0)={fmt(results['uniform_j_tilt'])}, "
      f"mean(R̃_g)={fmt(results['uniform_mean_risks'])}, diff={fmt(results['uniform_diff'])}")
print(f"  R̃_0={fmt(results['uniform_r0'])}, R̃_1={fmt(results['uniform_r1'])}")

print(f"\nGumbel λ=1e-3 weights: {[fmt(x,4) for x in results['gumbel_low_temp_weights']]} "
      f"(max={fmt(results['gumbel_low_temp_max_w'])})")
print(f"Gumbel λ=10  weights: {[fmt(x,4) for x in results['gumbel_high_temp_weights']]}")

print("\nConvexity (Lemma 5):")
for t, v in results["convexity"].items():
    print(f"  t={t}: L_t(avg)={fmt(v['L_avg'])}, avg(L_t)={fmt(v['avg_L'])}, "
          f"gap={fmt(v['gap'])} ({'OK' if v['gap']>=-1e-5 else 'FAIL'})")

print("\nInvalid latent:")
for label, v in results["invalid_latent"].items():
    print(f"  {label}: ERM={fmt(v['erm'])}, TSM(t=1)={fmt(v['tsm_t1'])}")

print("\nlogsumexp stress test (max_f ≈ 1e5):")
for t, v in results["logsumexp_stress"].items():
    print(f"  t={t}: loss={fmt(v['loss'])}, max_f={fmt(v['max_f'])}, finite={v['is_finite']}")
