"""Linear and cosine beta schedules for DDPM."""

from __future__ import annotations

import torch


def linear_beta_schedule(
    num_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Return linearly-spaced betas, shape (T,)."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps: int = 1000, s: float = 0.008) -> torch.Tensor:
    """Improved cosine schedule (Nichol & Dhariwal 2021), shape (T,)."""
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999)


def get_schedule(name: str, num_timesteps: int, **kwargs) -> torch.Tensor:
    """Registry: return beta schedule tensor by name."""
    if name == "linear":
        return linear_beta_schedule(num_timesteps, **kwargs)
    if name == "cosine":
        return cosine_beta_schedule(num_timesteps, **kwargs)
    raise ValueError(f"Unknown schedule: {name!r}. Choose 'linear' or 'cosine'.")


def precompute_schedule(betas: torch.Tensor) -> dict[str, torch.Tensor]:
    """Precompute all derived quantities from betas.

    Returns a dict with keys:
        betas, alphas, alphas_cumprod, alphas_cumprod_prev,
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
        log_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod, posterior_variance,
        posterior_log_variance_clipped, posterior_mean_coef1, posterior_mean_coef2
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

    posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": alphas_cumprod.sqrt(),
        "sqrt_one_minus_alphas_cumprod": (1 - alphas_cumprod).sqrt(),
        "log_one_minus_alphas_cumprod": (1 - alphas_cumprod).log(),
        "sqrt_recip_alphas_cumprod": alphas_cumprod.rsqrt(),
        "sqrt_recipm1_alphas_cumprod": (1 / alphas_cumprod - 1).sqrt(),
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_variance.clamp(min=1e-20).log(),
        "posterior_mean_coef1": betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod),
        "posterior_mean_coef2": (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod),
    }
