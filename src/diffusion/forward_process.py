"""DDPM forward (noising) process q(x_t | x_0)."""

from __future__ import annotations

import torch


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x_t ~ q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I).

    Args:
        x0:   clean input, shape (B, ...).
        t:    integer timestep indices, shape (B,).
        sqrt_alphas_cumprod:          precomputed, shape (T,).
        sqrt_one_minus_alphas_cumprod: precomputed, shape (T,).
        noise: if None, sampled from N(0, I).

    Returns:
        (x_t, noise) — noisy sample and the noise that was added.
    """
    assert x0.shape[0] == t.shape[0], (
        f"Batch size mismatch: x0={x0.shape[0]}, t={t.shape[0]}"
    )
    if noise is None:
        noise = torch.randn_like(x0)

    # broadcast schedule scalars to input shape
    B = x0.shape[0]
    extra_dims = (1,) * (x0.ndim - 1)
    sqrt_alpha = sqrt_alphas_cumprod[t].view(B, *extra_dims)
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].view(B, *extra_dims)

    x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
    return x_t, noise
