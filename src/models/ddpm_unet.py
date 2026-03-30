"""Minimal DDPM U-Net score network."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseModel, ModelOutput


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding, shape (B, dim)."""
    assert t.ndim == 1, f"Expected 1-D timestep tensor, got {t.shape}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    """Residual block with timestep embedding injection."""

    def __init__(self, channels: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(self.act(emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.drop(self.conv2(h))
        return x + h


class DDPMUNet(BaseModel):
    """Small DDPM U-Net for 2-D molecular graph embeddings.

    Input:  (B, in_ch, H, W) — noisy node-feature grid at timestep t
    Output: ModelOutput with pred_noise of same spatial shape
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 64,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        emb_dim = model_channels * 4
        self.emb_mlp = nn.Sequential(
            nn.Linear(model_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ResBlock(model_channels, emb_dim, dropout) for _ in range(num_res_blocks)]
        )
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> ModelOutput:
        assert x.ndim == 4, f"Expected (B,C,H,W), got {x.shape}"
        assert t.ndim == 1 and t.shape[0] == x.shape[0], (
            f"Expected t shape ({x.shape[0]},), got {t.shape}"
        )
        emb = self.emb_mlp(_sinusoidal_embedding(t, self.emb_mlp[0].in_features))
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, emb)
        pred_noise = self.output_proj(h)
        assert pred_noise.shape == x.shape, (
            f"Output shape {pred_noise.shape} != input shape {x.shape}"
        )
        return ModelOutput(pred_noise=pred_noise)


def get_model(cfg) -> DDPMUNet:
    """Factory: instantiate DDPMUNet from Hydra config node."""
    return DDPMUNet(
        in_channels=cfg.get("in_channels", 4),
        out_channels=cfg.get("out_channels", 4),
        model_channels=cfg.get("model_channels", 64),
        num_res_blocks=cfg.get("num_res_blocks", 2),
        dropout=cfg.get("dropout", 0.0),
    )
