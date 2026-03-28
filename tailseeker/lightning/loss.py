"""Loss functions for TailSeeker."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap


class TailSeekerLoss(nn.Module):
    """Tilted score matching loss (TERM-style log-sum-exp tilt).

    When tilt == 0.0 this reduces to standard MSE (ERM baseline).
    """

    def __init__(self, config: DotMap) -> None:
        super().__init__()
        self.tilt: float = config.get("tilt", 1.0)
        # TODO: register additional loss hyperparameters from config here

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute tilted score matching loss.

        Args:
            pred:   model predictions, shape (B, ...).
            target: ground-truth scores,  shape (B, ...).

        Returns:
            Scalar loss tensor.

        TODO: replace MSE stub with TERM log-sum-exp tilt over the batch.
              The tilt temperature is self.tilt.
        """
        assert pred.shape == target.shape, (
            f"Shape mismatch: pred={pred.shape}, target={target.shape}"
        )
        if self.tilt == 0.0:
            return F.mse_loss(pred, target)
        # TODO: implement tilted loss:
        #   per_sample = mse per sample (mean over non-batch dims)
        #   return (1/tilt) * log( mean( exp(tilt * per_sample) ) )
        return F.mse_loss(pred, target)
