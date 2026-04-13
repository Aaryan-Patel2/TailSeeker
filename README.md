# TailSeeker

**Tilted Score Matching for Drug Discovery on QM9**

TailSeeker replaces the standard DDPM denoising MSE loss with a TERM-style log-sum-exp tilt that biases diffusion model scores toward rare, high-value molecular regions. The goal is to improve tail performance — generating the *best* molecules, not just average ones.

Target venue: **NeurIPS 2026 Workshop on ML for Drug Discovery**.

---

## Core Idea

Standard diffusion training minimizes average loss (ERM), which treats rare, high-property molecules the same as common ones. TailSeeker introduces a tilt parameter `t` that up-weights high-loss samples during training:

```
L_tilt = (1/t) * log( (1/B) sum_i exp(t * l_i) )
```

- `t = 0` — standard ERM (DDPMSimpleLoss baseline)
- `t > 0` — up-weights hard/rare samples (tail-seeking)
- `t < 0` — down-weights hard samples (reverse tilt, ablation control)

At `t -> 0`, the loss recovers standard ERM exactly. The ablation axis sweeps `t in {-5, -2, -1, 0, 1, 2, 5, 10}` over 3 seeds (24 total runs).

---

## Installation

Requires Python 3.11, CUDA 12.1 (CPU fallback works for dev/testing).

```bash
# 1. Create and activate virtualenv
uv venv --python 3.11 && source .venv/bin/activate

# 2. Install PyTorch (CUDA 12.1)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
uv pip install numpy pandas matplotlib seaborn tqdm wandb rdkit torch-geometric hydra-core omegaconf scipy imageio

# 4. Install this package in editable mode
uv pip install -e ".[dev]"
```

> **Note:** No GPU is required to run the harness in stub mode. All dev and test runs work on CPU.

---

## Running Experiments

```bash
# Single run with default config (tilt=1.0, seed=42)
python scripts/train.py

# Override any config key via Hydra
python scripts/train.py loss.tilt=2.0 seed=1

# Full ablation sweep (24 runs: 8 tilt values x 3 seeds)
python scripts/train.py --multirun loss.tilt=-5,-2,-1,0,1,2,5,10 seed=0,1,2
```

Outputs land in `outputs/<timestamp>/` (Hydra managed). Each run saves `config.yaml` and checkpoints to that directory.

### Testing & Validation

```bash
# Plot ablation results (requires completed multirun)
python scripts/plot_ablation.py
# outputs to outputs/ablation_plots/{loss_curves,final_loss_vs_tilt,monotonicity_vs_tilt}.png

# Run full test suite (109 tests: QM9 + smoke + unit + integration + theory)
pytest

# Smoke tests only (loss functions + shapes + gradients)
pytest tests/test_smoke.py -v

# Specific loss function tests
pytest tests/unit/test_losses.py -v

# Demo: show all loss functions in action (L_tilt ablation + L_MO)
python scripts/demo_losses.py
```

The demo script validates:
- ERM baseline (tilt=0)
- Single-objective L_tilt at 8 ablation points (-5 to 10)
- Multi-objective L_MO with Gumbel-Softmax group weighting
- Numerical stability and gradient flow across all tilt values

### W&B Logging

Set your entity in `config/default.yaml`:

```yaml
wandb:
  entity: your-username   # or null to disable
  mode: online            # "online" | "offline" | "disabled"
```

---

## File Structure

```
TailSeeker/
├── scripts/
│   └── train.py                      # Hydra entrypoint; set_seed() is first call
│
├── src/
│   ├── losses/
│   │   ├── base.py                   # BaseLoss ABC + LossOutput dataclass
│   │   ├── ddpm_simple.py            # ERM baseline (tilt=0), fully implemented
│   │   ├── tilted_score_matching.py  # V1: single-objective TERM loss (L_tilt) — implemented
│   │   └── hierarchical_loss.py      # V2: multi-objective J̃ + L_MO (Gumbel-Softmax) — implemented
│   │
│   ├── models/
│   │   ├── base.py                   # BaseModel ABC + ModelOutput dataclass
│   │   └── ddpm_unet.py              # U-Net score network; get_model() factory
│   │
│   ├── diffusion/
│   │   ├── noise_schedule.py         # Linear/cosine beta schedules + precomputation
│   │   ├── forward_process.py        # q_sample: add noise at timestep t
│   │   └── reverse_process.py        # Denoising reverse pass
│   │
│   ├── data/
│   │   └── qm9.py                    # QM9Dataset (torch-geometric wrapper)
│   │
│   ├── metrics/
│   │   ├── molecular.py              # validity, QED, SA score, uniqueness
│   │   ├── tail.py                   # right_cvar, top_k_mean, tail_improvement_ratio
│   │   ├── distributional.py         # FCD, MMD, coverage
│   │   └── training.py               # per-step and per-epoch training diagnostics
│   │
│   ├── eval/
│   │   └── evaluator.py              # Full metric suite -> flat W&B dict
│   │
│   ├── training/
│   │   └── trainer.py                # Training loop; catches NotImplementedError (stub-safe)
│   │
│   └── utils.py                      # set_seed(), misc helpers
│
├── config/
│   ├── default.yaml                  # All hyperparameters with inline comments
│   └── experiment/
│       └── ablation_tilt.yaml        # Multirun sweep: tilt x seed = 24 runs
│
├── tests/
│   ├── unit/
│   │   ├── test_losses.py            # Loss shape/value unit tests
│   │   ├── test_theory.py            # Theory validation + gradcheck
│   │   └── test_data.py              # Data encoding tests
│   ├── integration/
│   │   └── test_train_one_step.py    # End-to-end single-step smoke test
│   ├── qm9/
│   │   ├── conftest.py               # EDM batch fixtures (padded, uniform, with groups)
│   │   ├── test_normalization.py     # Per-molecule MSE masking validation (7 tests)
│   │   └── test_edm_injection.py     # EDM integration + term_aggregate tests (17 tests)
│   └── test_smoke.py                 # Smoke tests for all loss functions (29 tests)
│
└── _archive/
    └── tailseeker_v1/                # Original package -- reference only, not imported
```

### Key File Purposes

| File | Purpose |
|---|---|
| `src/losses/tilted_score_matching.py` | **V1 contribution.** Single-objective TERM loss (L_tilt) — fully implemented. |
| `src/losses/hierarchical_loss.py` | **V2 contribution.** Multi-objective J̃ + L_MO with Gumbel-Softmax — fully implemented. |
| `src/losses/ddpm_simple.py` | ERM baseline (tilt=0). Used as the control in all ablations. |
| `src/training/trainer.py` | Training loop. Full backward pass for all tilt values (loss is implemented). |
| `config/default.yaml` | Single source of truth for all hyperparameters. Override at runtime via Hydra. |
| `src/metrics/tail.py` | Primary evaluation: right-CVaR and tail improvement ratio vs. baseline. |

---

## Ablation Matrix (Paper Table 1)

| `loss.tilt` | Type | Status |
|---|---|---|
| `-5.0` | reverse tilt | **implemented** |
| `-2.0` | reverse tilt | **implemented** |
| `-1.0` | reverse tilt | **implemented** |
| `0.0` | standard ERM baseline | **implemented** |
| `1.0` | mild tilt | **implemented** |
| `2.0` | moderate tilt | **implemented** |
| `5.0` | strong tilt | **implemented** |
| `10.0` | extreme tilt | **implemented** |

---

## Loss Implementation

`TiltedScoreMatchingLoss` is fully implemented. Formula:

```
L_tilt = (1/t) * [logsumexp(t·l, dim=0) - log(B)]
```

where `l_i` is per-sample MSE and `t` is `loss.tilt`. Uses `torch.logsumexp` for numerical stability. Jensen's inequality guarantees `t>0 → L_tilt ≥ mean_mse` and `t<0 → L_tilt ≤ mean_mse`.

When `tilt=0`, `get_loss_fn()` routes to `DDPMSimpleLoss` (standard ERM).

## QM9 Dataloader

`src/data/qm9.py` loads real QM9 via `torch_geometric`. Install: `uv pip install torch-geometric`. Set `data.download: true` in config for first-time download.

Molecules are encoded as `(4, 29, 29)` pairwise feature tensors:
- Channels 0–2: Δx, Δy, Δz pairwise coordinate differences (centred, scaled by 5 Å)
- Channel 3: adjacency matrix (0/1 from bond edges)

Split: train [0, 110 000) / val [110 000, 120 000) / test [120 000, 130 000), deterministic by index.

If `torch-geometric` is not installed or `data/` is missing, training falls back to synthetic stub data with a warning.

---

## Metrics

**Primary (tail quality):**
- `right_cvar(alpha=0.01)` — mean score of the top-1% of generated molecules
- `tail_improvement_ratio` — CVaR improvement over ERM baseline

**Molecular validity:**
- `validity` — fraction of generated SMILES that parse as valid RDKit molecules
- `mean_qed` — mean drug-likeness score [0, 1]
- `mean_sa` — mean synthetic accessibility [1, 10] (lower = more accessible)
- `uniqueness` — fraction of valid molecules that are structurally unique

**Distributional:**
- FCD, MMD, coverage (see `src/metrics/distributional.py`)

---

## Reproducibility

Every run is fully determined by its Hydra config + `seed=` override. `set_seed()` is the first call in every entrypoint and covers `random`, `numpy`, `torch`, `torch.cuda`, and cuDNN determinism flags. The full config is written to `outputs/<run>/config.yaml` and embedded in every checkpoint alongside the model state dict and optimizer state.

---

## Development Constraints

- No argparse — Hydra overrides only (`python scripts/train.py key=value`)
- No distributed training — single GPU is sufficient for QM9
- All tests must pass without a GPU
- ABCs are permitted only for `BaseLoss` and `BaseModel` hierarchies; all other code is concrete
- File length budgets are enforced (see `.claude/CLAUDE.md` for limits per file)
- `src/losses/tilted_score_matching.py` must not be modified without a corresponding paper update
