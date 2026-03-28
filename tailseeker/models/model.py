"""Model factory and UNet-style score network."""

from __future__ import annotations

import torch
import torch.nn as nn
from dotmap import DotMap

from .nn import TimestepEmbedSequential, conv_nd


class UNetModel(nn.Module):
    """Minimal UNet score network for diffusion / score matching.

    TODO: add encoder/decoder blocks, attention, and timestep embedding MLP.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int,
        num_res_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout

        # Minimal stub — replace with real encoder/bottleneck/decoder
        self.input_proj = conv_nd(2, in_channels, model_channels, 3, padding=1)
        self.blocks = TimestepEmbedSequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )
        self.output_proj = conv_nd(2, model_channels, out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the UNet.

        Args:
            x: noisy input tensor of shape (B, C, H, W).
            timestep: integer timestep tensor of shape (B,), or None.

        TODO: add timestep embedding and full encoder/decoder path.
        """
        assert x.ndim == 4, f"Expected 4-D input (B,C,H,W), got shape {x.shape}"
        h = self.input_proj(x)
        # emb is a dummy zero tensor until timestep embedding is implemented
        emb = torch.zeros(x.shape[0], self.model_channels, device=x.device)
        h = self.blocks(h, emb)
        return self.output_proj(h)


def get_model(config: DotMap) -> nn.Module:
    """Instantiate and return the model specified by *config*.

    Never instantiate a model class directly — always go through this factory.

    Raises:
        ValueError: if config["model_type"] is not recognised.
    """
    model_type: str = config.get("model_type", "unet")
    in_channels: int = config.get("in_channels", 4)
    out_channels: int = config.get("out_channels", 4)
    model_channels: int = config.get("model_channels", 128)
    num_res_blocks: int = config.get("num_res_blocks", 2)
    dropout: float = config.get("dropout", 0.0)

    if model_type == "unet":
        return UNetModel(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )
    raise ValueError(
        f"Unknown model_type={model_type!r}. Expected one of ['unet']."
    )
