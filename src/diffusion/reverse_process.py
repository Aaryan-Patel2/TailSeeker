"""DDPM reverse (denoising) process p_θ(x_{t-1} | x_t)."""

from __future__ import annotations

import torch


@torch.no_grad()
def p_sample(
    model,
    x_t: torch.Tensor,
    t: torch.Tensor,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    """One DDPM reverse step: sample x_{t-1} ~ p_θ(x_{t-1} | x_t).

    Args:
        model:    score network; forward(x_t, t) → ModelOutput.
        x_t:      noisy sample at timestep t, shape (B, ...).
        t:        integer timestep indices, shape (B,).
        schedule: precomputed schedule dict from noise_schedule.precompute_schedule.

    Returns:
        x_{t-1}, shape (B, ...).
    """
    assert x_t.shape[0] == t.shape[0], (
        f"Batch size mismatch: x_t={x_t.shape[0]}, t={t.shape[0]}"
    )
    B = x_t.shape[0]
    extra = (1,) * (x_t.ndim - 1)

    output = model(x_t, t)
    pred_noise = output.pred_noise
    assert pred_noise is not None, "model must return pred_noise for DDPM sampling"
    assert pred_noise.shape == x_t.shape, (
        f"pred_noise shape {pred_noise.shape} != x_t shape {x_t.shape}"
    )

    def _get(key):
        return schedule[key][t].view(B, *extra).to(x_t.device)

    # predict x_0 from noise prediction
    pred_x0 = (
        _get("sqrt_recip_alphas_cumprod") * x_t
        - _get("sqrt_recipm1_alphas_cumprod") * pred_noise
    )

    # posterior mean μ_θ(x_t, t)
    mean = (
        _get("posterior_mean_coef1") * pred_x0
        + _get("posterior_mean_coef2") * x_t
    )

    # add noise unless at t=0
    noise = torch.randn_like(x_t)
    nonzero = (t > 0).float().view(B, *extra)
    log_var = _get("posterior_log_variance_clipped")
    return mean + nonzero * (0.5 * log_var).exp() * noise


@torch.no_grad()
def p_sample_loop(
    model,
    shape: tuple[int, ...],
    schedule: dict[str, torch.Tensor],
    device: torch.device,
    num_timesteps: int = 1000,
) -> torch.Tensor:
    """Full DDPM reverse chain: sample from T → 0.

    Returns:
        x_0, shape *shape*.
    """
    x = torch.randn(shape, device=device)
    T = num_timesteps
    for i in reversed(range(T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, schedule)
    return x
