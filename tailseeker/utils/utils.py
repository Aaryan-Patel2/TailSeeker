"""General utilities: reproducible seeding and batch formatting."""

import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG source for full cross-run reproducibility.

    Seeded sources: PYTHONHASHSEED env var, Python random, NumPy, PyTorch
    (CPU + CUDA), and cuDNN deterministic/benchmark flags.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_batch(batch: dict[str, Any]) -> tuple[Any, Any]:
    """Unpack a batch dict into (inputs, targets).

    Always call this function in training_step / validation_step — never
    unpack a batch inline.

    TODO: adapt the keys "x" and "y" to match the actual dataset structure.
    """
    x = batch["x"]
    y = batch["y"]
    return x, y
