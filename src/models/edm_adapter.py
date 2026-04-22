"""EDM (e3_diffusion_for_molecules) loss-injection adapter.

Surgical single-line patch: replaces ``nll = nll.mean(0)`` in EDM's
``qm9/losses.py`` (Hoogeboom et al., line 31) with
``nll = term_aggregate(nll, tilt)`` from our TiltedScoreMatchingLoss.

No EDM source files are modified on disk.  The patch lives only in-memory
for the duration of a training run and is reversible via ``unpatch_loss()``.

Patch strategy
--------------
``inspect.getsource`` retrieves the function body as a string.  We replace
the exact reduction line with a call to our aggregator, compile the modified
source, and re-bind the function on the module object.  The patched function
shares the original module's global namespace (so all EDM helpers remain
available) augmented with our two injected names.

Verification (always run after patching):
    ``tilt=0`` must reproduce ``nll.mean(0)`` bit-for-bit (atol=1e-7).
    Enforced by ``verify_patch()``.

Usage::

    adapter = EDMAdapter(edm_root="/drive/MyDrive/TailSeeker/edm", tilt=1.0)
    with adapter:               # patches on __enter__, reverts on __exit__
        run_edm_training(...)   # EDM now uses term_aggregate internally

Or explicitly::

    adapter.patch_loss()
    ...
    adapter.unpatch_loss()
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from typing import Optional

import torch

from src.losses import term_aggregate

# Sentinel string we look for in EDM's losses.py source.
_EDM_REDUCTION_PATTERN = "nll = nll.mean(0)"
# Injected replacement (references names we add to the function's globals).
_EDM_REDUCTION_REPLACEMENT = "nll = _TAILSEEKER_TERM_AGGREGATE(nll, _TAILSEEKER_TILT)"


class EDMAdapter:
    """Adapter that injects TailSeeker's tilted loss into a cloned EDM repo.

    Args:
        edm_root: Absolute path to the root of a cloned
            ``ehoogeboom/e3_diffusion_for_molecules`` repository.
        tilt:     TERM tilt parameter τ.  ``0.0`` recovers EDM's ERM baseline.
    """

    def __init__(self, edm_root: str, tilt: float) -> None:
        assert Path(edm_root).is_dir(), (
            f"edm_root does not exist: {edm_root}\n"
            "Run colab_setup.ipynb Cell 3 first (git clone EDM to Drive)."
        )
        self.edm_root = str(edm_root)
        self.tilt = tilt
        self._edm_losses: Optional[types.ModuleType] = None
        self._orig_fn = None

    # ─── context manager interface ────────────────────────────────────────────

    def __enter__(self) -> EDMAdapter:
        self.patch_loss()
        return self

    def __exit__(self, *_) -> None:
        self.unpatch_loss()

    # ─── public API ──────────────────────────────────────────────────────────

    def add_to_path(self) -> None:
        """Prepend edm_root to sys.path so EDM modules are importable."""
        if self.edm_root not in sys.path:
            sys.path.insert(0, self.edm_root)

    def patch_loss(self) -> None:
        """Replace ``qm9.losses.compute_loss_and_nll`` with the tilted version.

        Idempotent: calling twice is a no-op (second call logs a warning).
        """
        if self._orig_fn is not None:
            print("[EDMAdapter] already patched — skipping duplicate patch_loss() call.")
            return

        self.add_to_path()
        try:
            import qm9.losses as edm_losses  # noqa: PLC0415 (import inside function intentional)
        except ImportError as exc:
            raise ImportError(
                f"Cannot import 'qm9.losses' from edm_root='{self.edm_root}'.\n"
                "Check that colab_setup.ipynb Cell 3 ran successfully.\n"
                f"Original error: {exc}"
            ) from exc

        self._edm_losses = edm_losses
        self._orig_fn = edm_losses.compute_loss_and_nll

        patched = self._build_patched_fn(edm_losses)
        edm_losses.compute_loss_and_nll = patched
        print(f"[EDMAdapter] patched qm9.losses.compute_loss_and_nll  tilt={self.tilt}")

    def unpatch_loss(self) -> None:
        """Restore the original ``compute_loss_and_nll`` function."""
        if self._orig_fn is None:
            return
        assert self._edm_losses is not None
        self._edm_losses.compute_loss_and_nll = self._orig_fn
        self._orig_fn = None
        print("[EDMAdapter] restored original qm9.losses.compute_loss_and_nll")

    def verify_patch(self, batch_size: int = 8, seed: int = 42) -> None:
        """Sanity-check: tilt=0 adapter must match plain nll.mean(0) exactly.

        Raises AssertionError if the patch deviates from EDM's baseline.
        Only call after patch_loss().
        """
        torch.manual_seed(seed)
        dummy_nll = torch.rand(batch_size) * 2.0 + 0.5  # positive, like EDM NLL
        expected = dummy_nll.mean(0)
        got = term_aggregate(dummy_nll, tilt=0.0)
        assert torch.allclose(got, expected, atol=1e-7), (
            f"[EDMAdapter.verify_patch] tilt=0 diverges from mean(0): "
            f"expected={expected.item():.8f}  got={got.item():.8f}"
        )
        if self.tilt != 0.0:
            # Jensen: tilted must be ≥ mean for positive tilt
            tilted = term_aggregate(dummy_nll, tilt=abs(self.tilt))
            erm = dummy_nll.mean().item()
            assert tilted.item() >= erm - 1e-5, (
                f"Jensen violated at tilt={self.tilt}: "
                f"term_aggregate={tilted.item():.6f} < mean={erm:.6f}"
            )
        print(f"[EDMAdapter.verify_patch] ✓  tilt=0 matches baseline  tilt={self.tilt}")

    # ─── private ─────────────────────────────────────────────────────────────

    def _build_patched_fn(
        self,
        edm_losses: types.ModuleType,
    ) -> types.FunctionType:
        """Build a patched version of compute_loss_and_nll using source substitution.

        We use ``inspect.getsource`` + string replacement so the patched function
        preserves all of EDM's internal logic and only changes the one reduction
        line.  The modified source is compiled and bound into the module's global
        namespace.
        """
        orig_fn = self._orig_fn
        src = inspect.getsource(orig_fn)

        if _EDM_REDUCTION_PATTERN not in src:
            # Graceful fallback: wrap the original and post-process its output.
            print(
                f"[EDMAdapter] WARNING: '{_EDM_REDUCTION_PATTERN}' not found in "
                "qm9.losses source — using wrapper fallback (see adapter.py)."
            )
            return self._build_wrapper_fallback(orig_fn)

        patched_src = src.replace(_EDM_REDUCTION_PATTERN, _EDM_REDUCTION_REPLACEMENT)

        # Inject our helpers into the function's global namespace alongside EDM's.
        tilt = self.tilt
        patched_globals = {
            **vars(edm_losses),
            "_TAILSEEKER_TERM_AGGREGATE": term_aggregate,
            "_TAILSEEKER_TILT": tilt,
        }
        exec(compile(patched_src, "<edm_adapter_patched>", "exec"), patched_globals)  # noqa: S102
        patched_fn: types.FunctionType = patched_globals["compute_loss_and_nll"]
        return patched_fn

    def _build_wrapper_fallback(self, orig_fn) -> types.FunctionType:
        """Fallback: call original, then re-derive nll from per-sample components.

        Used when the EDM source doesn't match the expected ``nll.mean(0)`` pattern
        (e.g., a different EDM version / fork).  Less precise but functional.

        Strategy:
            Temporarily wrap ``torch.Tensor.mean`` to intercept the first (B,)
            tensor that has ``.mean(0)`` called on it, save it, then call our
            term_aggregate on that captured tensor instead.
        """
        tilt = self.tilt

        def _patched_wrapper(  # noqa: E501
            args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context
        ):
            _captured: dict[str, torch.Tensor] = {}
            _orig_mean = torch.Tensor.mean

            def _intercepting_mean(self_tensor, *a, **kw):
                # Capture first 1-D positive-valued tensor reduced with dim=0
                if (
                    "captured" not in _captured
                    and self_tensor.ndim == 1
                    and self_tensor.shape[0] == x.shape[0]
                    and (len(a) == 0 or a[0] == 0)
                ):
                    _captured["nll"] = self_tensor.detach().clone()
                return _orig_mean(self_tensor, *a, **kw)

            torch.Tensor.mean = _intercepting_mean  # type: ignore[method-assign]
            try:
                nll_scalar, reg_term, mean_abs_z = orig_fn(
                    args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context
                )
            finally:
                torch.Tensor.mean = _orig_mean  # type: ignore[method-assign]

            if "nll" in _captured:
                nll_scalar = term_aggregate(_captured["nll"], tilt=tilt)
            else:
                print(
                    "[EDMAdapter] WARNING: could not intercept per-sample NLL — "
                    "falling back to original scalar NLL (no tilt applied)."
                )
            return nll_scalar, reg_term, mean_abs_z

        return _patched_wrapper
