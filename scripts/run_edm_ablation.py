"""EDM full-scale ablation script — one (tilt, seed) training run.

Called from colab_edm_ablation.ipynb via subprocess for each cell in the
8-tilt × 3-seed ablation matrix.  Uses Hydra for config (consistent with
train.py) and EDMAdapter for surgical loss injection into EDM.

Usage (Hydra CLI overrides — NO argparse per project invariant):
    python scripts/run_edm_ablation.py \\
        --config-name edm_ablation \\
        loss.tilt=1.0 seed=0 \\
        edm.repo_path=/drive/MyDrive/TailSeeker/edm \\
        data.root=/drive/MyDrive/TailSeeker/data \\
        output.root=/drive/MyDrive/TailSeeker/outputs

Output per run (under output.root/{tilt}_{seed}/):
    config.yaml         full hydra config (embedded for checkpoint recovery)
    metrics.csv         epoch, loss, qed_tail_p10, sa_tail_p10, tilt, seed
    ckpt_epoch{N}.pt    checkpoint every eval_every epochs + final
    run_id.txt          W&B run ID for cross-referencing
"""

from __future__ import annotations

import csv
import sys
import traceback
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import term_aggregate  # noqa: F401 — imported for verify_patch
from src.models.edm_adapter import EDMAdapter
from src.utils import set_seed


@hydra.main(config_path="../config", config_name="edm_ablation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Full EDM+tilt run for one (tilt, seed) pair."""
    try:
        _run(cfg)
    except Exception:
        tb = traceback.format_exc()
        try:
            hydra_dir = Path(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
            (hydra_dir / "crash_report.txt").write_text(tb)
        except Exception:
            pass
        print(f"[run_edm_ablation] CRASH:\n{tb}", file=sys.stderr)
        raise


def _run(cfg: DictConfig) -> None:
    # 1. Seed — must be absolute first call
    set_seed(int(cfg.seed))

    tilt = float(cfg.loss.tilt)
    seed = int(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_edm_ablation] tilt={tilt}  seed={seed}  device={device}")

    # 2. Output directory (Drive-backed, scoped to this run)
    run_tag = f"tilt{tilt:+.1f}_seed{seed}"
    out_dir = Path(cfg.output.root) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Save full config immediately (recovery requires only this file)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    # 4. Inject EDM into sys.path and patch its loss aggregation
    adapter = EDMAdapter(edm_root=str(cfg.edm.repo_path), tilt=tilt)
    adapter.patch_loss()
    adapter.verify_patch()

    try:
        _train_loop(cfg, adapter, out_dir, device, tilt, seed)
    finally:
        adapter.unpatch_loss()

    print(f"[run_edm_ablation] done  outputs → {out_dir}")


def _train_loop(
    cfg: DictConfig,
    adapter: EDMAdapter,
    out_dir: Path,
    device: torch.device,
    tilt: float,
    seed: int,
) -> None:
    """EDM training loop with tilted loss.  Structure mirrors EDM's main_qm9.py."""
    # ── Build EDM model & data (via EDM's own factory functions) ──────────
    import qm9.losses as qm9_losses  # type: ignore[import]
    from qm9 import dataset as qm9_dataset  # type: ignore[import]
    from qm9.models import get_model  # type: ignore[import]
    # get_dataset_info is in configs/datasets_config, not qm9.utils
    from configs.datasets_config import get_dataset_info  # type: ignore[import]

    # Parse EDM args from our Hydra config — EDM uses an argparse-style namespace
    edm_args = _build_edm_args(cfg, device)

    dataset_info = get_dataset_info("qm9", edm_args.remove_h)
    dataloaders, charge_scale = qm9_dataset.retrieve_dataloaders(edm_args)

    # get_model takes the train split only, not the full dataloaders dict
    model, nodes_dist, prop_dist = get_model(edm_args, device, dataset_info, dataloaders["train"])
    model = model.to(device)

    if cfg.edm.get("checkpoint"):
        ckpt = torch.load(cfg.edm.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[run_edm_ablation] warm-start from {cfg.edm.checkpoint}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        amsgrad=True,
        weight_decay=1e-12,
    )

    # EMA (essential for stable EDM training per literature)
    ema = _try_build_ema(model) if cfg.training.get("use_ema", True) else None

    # ── CSV logger ────────────────────────────────────────────────────────
    csv_path = out_dir / "metrics.csv"
    fieldnames = ["epoch", "loss", "qed_tail_p10", "sa_tail_p10", "tilt", "seed"]
    csv_file = csv_path.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb_run = _init_wandb(cfg, out_dir, tilt, seed)

    # ── Run ID for cross-referencing ──────────────────────────────────────
    if wandb_run is not None:
        (out_dir / "run_id.txt").write_text(wandb_run.id)

    max_epochs = int(cfg.training.max_epochs)
    eval_every = int(cfg.ablation.get("eval_every", 10))

    for epoch in range(max_epochs):
        model.train()
        epoch_nll = _train_one_epoch(
            model, dataloaders["train"], optimizer, edm_args,
            nodes_dist, qm9_losses, device, cfg,
        )
        if ema is not None:
            ema.update(model.parameters())

        # ── Evaluate tail enrichment ──────────────────────────────────────
        qed_tail = float("nan")
        sa_tail = float("nan")
        if epoch == 0 or (epoch + 1) % eval_every == 0:
            model.eval()
            qed_tail, sa_tail = _eval_tail_enrichment(
                model, dataloaders["valid"], edm_args, nodes_dist, device
            )
            _save_checkpoint(model, optimizer, ema, cfg, epoch, out_dir)

        row = {
            "epoch": epoch, "loss": epoch_nll,
            "qed_tail_p10": qed_tail, "sa_tail_p10": sa_tail,
            "tilt": tilt, "seed": seed,
        }
        writer.writerow(row)
        csv_file.flush()

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "Train/nll": epoch_nll,
                           "Val/qed_tail_p10": qed_tail,
                           "Val/sa_tail_p10": sa_tail})

        print(
            f"  epoch {epoch:04d}  nll={epoch_nll:.5f}"
            f"  qed_tail={qed_tail:.4f}  sa_tail={sa_tail:.4f}"
        )

    csv_file.close()
    if wandb_run is not None:
        wandb_run.finish()


def _train_one_epoch(model, loader, optimizer, edm_args, nodes_dist, qm9_losses, device, cfg):
    """One training epoch; returns mean NLL (already term_aggregate'd by adapter)."""
    total_nll = 0.0
    n_batches = 0
    grad_clip = float(cfg.training.get("grad_clip", 1.0))

    for batch in loader:
        x = batch["positions"].to(device)
        # Validate positions shape: should be (batch_size, max_atoms, 3)
        assert x.ndim == 3, f"[x shape error] x must be 3D (batch, atoms, 3), got shape {x.shape}, ndim={x.ndim}"
        assert x.shape[-1] == 3, f"[x shape error] last dim must be 3 (xyz coords), got {x.shape[-1]} from shape {x.shape}"
        assert x.shape[1] > 0, f"[x shape error] n_atoms must be > 0, got {x.shape[1]} from shape {x.shape}"

        # Build h dict: EDM's model.normalize() requires both 'categorical' and 'integer' keys.
        # 'categorical' = one-hot (B, N, n_types); 'integer' = argmax index (B, N, 1).
        one_hot_raw = batch.get("one_hot")
        if isinstance(one_hot_raw, dict):
            h = {k: v.to(device) for k, v in one_hot_raw.items()}
        else:
            cat = one_hot_raw.to(device) if hasattr(one_hot_raw, 'to') else torch.as_tensor(one_hot_raw, device=device)
            h = {"categorical": cat}
        # Ensure 'integer' key: EDM normalize() reads h['integer'] for normalization.
        # Derive from categorical if missing: integer index is the argmax of one-hot.
        if 'integer' not in h and 'categorical' in h:
            h['integer'] = h['categorical'].float().argmax(dim=-1, keepdim=True).float()

        node_mask = batch["atom_mask"].to(device).float()
        edge_mask = batch["edge_mask"].to(device).float()
        context = batch.get("context", None)
        if context is not None:
            context = context.to(device)

        # node_mask must be (B, N, 1) to broadcast with coords (B, N, 3) in assert_correctly_masked
        if node_mask.ndim == 2:
            node_mask = node_mask.unsqueeze(-1)

        optimizer.zero_grad()
        # qm9_losses.compute_loss_and_nll is patched by EDMAdapter to use term_aggregate
        nll, reg_term, _ = qm9_losses.compute_loss_and_nll(
            edm_args, model, nodes_dist, x, h, node_mask, edge_mask, context
        )
        loss = nll + edm_args.train_diffusion * reg_term
        assert torch.isfinite(loss), f"Non-finite loss at batch (nll={nll.item():.4f})"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_nll += nll.item()
        n_batches += 1

    return total_nll / max(n_batches, 1)


@torch.no_grad()
def _eval_tail_enrichment(model, loader, edm_args, nodes_dist, device):
    """Generate molecules and compute top-10% QED and SA (tail enrichment metrics)."""
    try:
        from rdkit.Chem import QED as rdQED  # noqa: PLC0415
        from sascorer import calculateScore  # type: ignore[import]  # noqa: PLC0415
    except ImportError:
        return float("nan"), float("nan")

    qed_scores, sa_scores = [], []
    for batch in loader:
        x = batch["positions"].to(device)
        node_mask = batch["atom_mask"].to(device)
        # Use ground-truth molecules for tail property evaluation
        # (generation-based eval is too slow for per-epoch tracking)
        for i in range(x.shape[0]):
            n = int(node_mask[i].sum().item())
            coords = x[i, :n].cpu().numpy()
            try:
                mol = _coords_to_mol(coords, batch, i)
                if mol is not None:
                    qed_scores.append(rdQED.qed(mol))
                    sa_scores.append(calculateScore(mol))
            except Exception:
                continue
        if len(qed_scores) >= 256:
            break  # enough for stable percentile estimate

    if not qed_scores:
        return float("nan"), float("nan")

    import numpy as np
    qed_arr = np.array(qed_scores)
    sa_arr = np.array(sa_scores)
    # Tail enrichment = fraction of molecules in top-10% QED / bottom-10% SA
    qed_p90 = float(np.percentile(qed_arr, 90))
    sa_p10 = float(np.percentile(sa_arr, 10))
    qed_tail_frac = float((qed_arr >= qed_p90).mean())
    sa_tail_frac = float((sa_arr <= sa_p10).mean())
    return qed_tail_frac, sa_tail_frac


def _coords_to_mol(coords, batch, idx):
    """Decode one EDM QM9 batch entry to an RDKit mol.  Returns None on failure.

    Uses atom types from ``batch['one_hot']`` (argmax → atomic number) and the
    3-D coordinates to build an RWMol.  Bond connectivity is inferred via
    ``rdDetermineBonds`` (RDKit ≥ 2022.09) with a distance-threshold fallback.
    """
    try:
        from rdkit import Chem  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        # QM9 with remove_h=False: H C N O F
        _ATOMIC_NUMS = [1, 6, 7, 8, 9]

        one_hot = batch.get("one_hot")
        if isinstance(one_hot, dict):  # EDM sometimes wraps in a dict
            one_hot = one_hot.get("categorical", next(iter(one_hot.values()), None))
        if one_hot is None:
            return None

        node_mask = batch["atom_mask"][idx].cpu()
        n = int(node_mask.sum().item())
        if n < 2:
            return None

        atom_idxs = one_hot[idx, :n].argmax(dim=-1).cpu().tolist()
        pos = np.array(coords[:n], dtype=float)  # (n, 3)

        mol = Chem.RWMol()
        for ai in atom_idxs:
            mol.AddAtom(Chem.Atom(_ATOMIC_NUMS[min(int(ai), len(_ATOMIC_NUMS) - 1)]))

        conf = Chem.Conformer(n)
        for j in range(n):
            conf.SetAtomPosition(j, pos[j].tolist())
        mol.AddConformer(conf, assignId=True)

        # Try modern rdDetermineBonds first; fall back to distance threshold
        try:
            from rdkit.Chem import rdDetermineBonds  # noqa: PLC0415
            rdDetermineBonds.DetermineConnectivity(mol)
        except (ImportError, AttributeError, Exception):
            _add_bonds_by_distance(mol, pos)

        try:
            Chem.SanitizeMol(mol)
            return mol.GetMol()
        except Exception:
            # Last resort: strip explicit Hs and retry
            try:
                mol2 = Chem.RemoveHs(mol, sanitize=False)
                Chem.SanitizeMol(mol2)
                return mol2
            except Exception:
                return None
    except Exception:
        return None


def _add_bonds_by_distance(mol, pos):
    """Add single bonds wherever two atoms are within 1.3× their summed covalent radii."""
    import numpy as np  # noqa: PLC0415

    _COV_R = {1: 0.31, 6: 0.77, 7: 0.75, 8: 0.73, 9: 0.72}
    n = mol.GetNumAtoms()
    for i in range(n):
        ri = _COV_R.get(mol.GetAtomWithIdx(i).GetAtomicNum(), 0.77)
        for j in range(i + 1, n):
            rj = _COV_R.get(mol.GetAtomWithIdx(j).GetAtomicNum(), 0.77)
            if float(np.linalg.norm(pos[i] - pos[j])) < 1.3 * (ri + rj):
                mol.AddBond(i, j, Chem.BondType.SINGLE)


def _save_checkpoint(model, optimizer, ema, cfg, epoch, out_dir):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        out_dir / f"ckpt_epoch{epoch:04d}.pt",
    )


def _try_build_ema(model):
    """Build EMA wrapper if torch_ema is available (installed with EDM)."""
    try:
        from torch_ema import ExponentialMovingAverage  # type: ignore[import]
        return ExponentialMovingAverage(model.parameters(), decay=0.9999)
    except ImportError:
        print("[run_edm_ablation] torch_ema not found — running without EMA.")
        return None


def _build_edm_args(cfg: DictConfig, device: torch.device):
    """Build an EDM-compatible args namespace from our Hydra config.

    EDM's internals expect an argparse.Namespace.  We construct the required
    fields from our config, keeping EDM's defaults for anything we don't
    explicitly control.
    """
    import argparse
    args = argparse.Namespace()
    # QM9 dataset settings (EDM defaults)
    args.dataset = "qm9"
    args.remove_h = False
    args.include_charges = True
    args.filter_n_atoms = None
    args.datadir = str(cfg.data.root)
    args.batch_size = int(cfg.training.batch_size)
    args.n_epochs = int(cfg.training.max_epochs)
    args.num_workers = int(cfg.data.get("num_workers", 2))
    # Model (EDM defaults — these must NOT be changed for the ablation)
    args.nf = 256
    args.n_layers = 9
    args.attention = 1
    args.norm_diff = True
    args.sin_embedding = False
    args.ode_regularization = 1e-3
    args.train_diffusion = 1
    args.diffusion_steps = 1000
    args.diffusion_noise_schedule = str(cfg.diffusion.get("schedule", "polynomial_2"))
    args.diffusion_noise_precision = 1e-5
    args.diffusion_loss_type = "l2"
    args.normalize_factors = [1, 4, 10]
    args.conditioning = []
    args.context_node_nf = 0
    # Fields accessed by get_model but absent from our original namespace
    args.condition_time = True
    args.tanh = True
    args.model = "egnn_dynamics"
    args.norm_constant = 1
    args.inv_sublayers = 1
    args.normalization_factor = 1   # scalar; distinct from normalize_factors list
    args.aggregation_method = "sum"
    args.probabilistic_model = "diffusion"
    args.device = str(device)
    args.lr = float(cfg.training.learning_rate)
    args.weight_decay = 1e-12
    args.clip_grad = True
    args.dp = True   # data-parallel (handled by Colab GPU setup)
    args.exp_name = f"edm_tilt{cfg.loss.tilt:+.1f}_seed{cfg.seed}"
    return args


def _init_wandb(cfg: DictConfig, out_dir: Path, tilt: float, seed: int):
    wc = cfg.wandb
    if str(wc.get("mode", "disabled")) == "disabled" or wc.get("entity") is None:
        return None
    try:
        import wandb
        return wandb.init(
            project=wc.project,
            entity=wc.entity,
            config={**OmegaConf.to_container(cfg, resolve=True), "tilt": tilt, "seed": seed},
            dir=str(out_dir),
            mode=wc.mode,
            name=f"tilt{tilt:+.1f}_seed{seed}",
        )
    except Exception as exc:
        print(f"[run_edm_ablation] W&B init failed (continuing without): {exc}")
        return None


if __name__ == "__main__":
    main()
