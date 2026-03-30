"""Shared utilities: set_seed, log-sum-exp numerical trick."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all RNG sources for bit-for-bit reproducibility.

    Must be the first substantive call in every entrypoint.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # benchmark=True is non-deterministic


def log_sum_exp(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Numerically stable log-sum-exp along *dim*.

    Used internally by TiltedScoreMatchingLoss.
    log Σ_i exp(x_i) = max(x) + log Σ_i exp(x_i - max(x))
    """
    x_max = x.max(dim=dim, keepdim=True).values
    return x_max.squeeze(dim) + (x - x_max).exp().sum(dim=dim).log()
