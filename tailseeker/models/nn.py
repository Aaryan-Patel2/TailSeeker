"""Low-level neural network primitives for UNet-style architectures."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TimestepBlock(nn.Module, ABC):
    """Abstract base: any module whose forward takes (x, emb)."""

    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward pass conditioned on timestep embedding *emb*."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential container that passes timestep embeddings to TimestepBlock children.

    Non-TimestepBlock children receive only *x*.
    """

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward, dispatching *emb* only to TimestepBlock layers."""
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    """Return a Conv{dims}d layer.

    Args:
        dims: spatial dimensionality — 1, 2, or 3.
        *args, **kwargs: forwarded to the Conv class.

    Raises:
        ValueError: if dims is not 1, 2, or 3.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported dims={dims!r}. Expected 1, 2, or 3.")


def avg_pool_nd(dims: int, *args, **kwargs) -> nn.Module:
    """Return an AvgPool{dims}d layer.

    Args:
        dims: spatial dimensionality — 1, 2, or 3.
        *args, **kwargs: forwarded to the AvgPool class.

    Raises:
        ValueError: if dims is not 1, 2, or 3.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"Unsupported dims={dims!r}. Expected 1, 2, or 3.")
