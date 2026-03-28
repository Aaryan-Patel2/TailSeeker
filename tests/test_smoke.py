"""Smoke tests — one per module, verifying imports and basic shapes."""

import pytest
import torch
from dotmap import DotMap


@pytest.fixture
def cfg() -> DotMap:
    """Minimal config sufficient for all smoke tests."""
    return DotMap({
        "model_type": "unet",
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 32,   # small for fast tests
        "num_res_blocks": 1,
        "dropout": 0.0,
        "learning_rate": 1e-4,
        "batch_size": 2,
        "use_ema": False,
        "tilt": 1.0,
    })


# ------------------------------------------------------------------
# tailseeker.utils.utils
# ------------------------------------------------------------------

def test_set_seed_is_deterministic() -> None:
    """set_seed produces identical outputs across calls."""
    from tailseeker.utils.utils import set_seed

    set_seed(0)
    a = torch.rand(5)
    set_seed(0)
    b = torch.rand(5)
    assert torch.allclose(a, b), f"Expected identical tensors, got {a} vs {b}"


# ------------------------------------------------------------------
# tailseeker.models.nn
# ------------------------------------------------------------------

def test_conv_nd_2d() -> None:
    """conv_nd(2, ...) returns a Conv2d and produces the expected output shape."""
    from tailseeker.models.nn import conv_nd

    layer = conv_nd(2, 4, 8, 3, padding=1)
    assert isinstance(layer, torch.nn.Conv2d)
    x = torch.randn(1, 4, 16, 16)
    out = layer(x)
    assert out.shape == (1, 8, 16, 16), f"Unexpected output shape: {out.shape}"


def test_conv_nd_invalid_dims() -> None:
    """conv_nd raises ValueError for unsupported dims."""
    from tailseeker.models.nn import conv_nd

    with pytest.raises(ValueError, match="Unsupported dims"):
        conv_nd(4, 4, 8, 3)


def test_avg_pool_nd_2d() -> None:
    """avg_pool_nd(2, ...) returns an AvgPool2d."""
    from tailseeker.models.nn import avg_pool_nd

    layer = avg_pool_nd(2, 2)
    assert isinstance(layer, torch.nn.AvgPool2d)


def test_avg_pool_nd_invalid_dims() -> None:
    """avg_pool_nd raises ValueError for unsupported dims."""
    from tailseeker.models.nn import avg_pool_nd

    with pytest.raises(ValueError, match="Unsupported dims"):
        avg_pool_nd(5, 2)


def test_timestep_embed_sequential() -> None:
    """TimestepEmbedSequential dispatches emb only to TimestepBlock children."""
    from tailseeker.models.nn import TimestepBlock, TimestepEmbedSequential

    class _IdentityBlock(TimestepBlock):
        def forward(self, x, emb):
            return x + 0 * emb.mean()  # use emb so it's not ignored

    seq = TimestepEmbedSequential(_IdentityBlock(), torch.nn.Identity())
    x = torch.randn(2, 4, 8, 8)
    emb = torch.zeros(2, 4)
    out = seq(x, emb)
    assert out.shape == x.shape, f"Shape changed: {out.shape}"


# ------------------------------------------------------------------
# tailseeker.models.model
# ------------------------------------------------------------------

def test_get_model_returns_module(cfg: DotMap) -> None:
    """get_model returns an nn.Module."""
    from tailseeker.models.model import get_model

    model = get_model(cfg)
    assert isinstance(model, torch.nn.Module)


def test_get_model_forward(cfg: DotMap) -> None:
    """get_model output has the expected shape."""
    from tailseeker.models.model import get_model

    model = get_model(cfg)
    x = torch.randn(2, cfg.in_channels, 16, 16)
    out = model(x)
    assert out.shape == (2, cfg.out_channels, 16, 16), (
        f"Unexpected output shape: {out.shape}"
    )


def test_get_model_unknown_type(cfg: DotMap) -> None:
    """get_model raises ValueError for an unknown model_type."""
    from tailseeker.models.model import get_model

    cfg["model_type"] = "transformer"
    with pytest.raises(ValueError, match="Unknown model_type"):
        get_model(cfg)


# ------------------------------------------------------------------
# tailseeker.lightning.loss
# ------------------------------------------------------------------

def test_loss_returns_scalar(cfg: DotMap) -> None:
    """TailSeekerLoss.forward returns a 0-D tensor."""
    from tailseeker.lightning.loss import TailSeekerLoss

    loss_fn = TailSeekerLoss(cfg)
    pred = torch.randn(2, 4, 8, 8)
    target = torch.randn(2, 4, 8, 8)
    loss = loss_fn(pred, target)
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"


def test_loss_shape_mismatch(cfg: DotMap) -> None:
    """TailSeekerLoss raises AssertionError on shape mismatch."""
    from tailseeker.lightning.loss import TailSeekerLoss

    loss_fn = TailSeekerLoss(cfg)
    with pytest.raises(AssertionError, match="Shape mismatch"):
        loss_fn(torch.randn(2, 4), torch.randn(2, 8))


# ------------------------------------------------------------------
# tailseeker.utils.metrics
# ------------------------------------------------------------------

def test_top_k_tail_correctness() -> None:
    """top_k_tail returns the mean of the top-k values."""
    from tailseeker.utils.metrics import top_k_tail

    values = torch.arange(10, dtype=torch.float)  # [0..9]
    result = top_k_tail(values, k=3)               # mean(7,8,9) = 8.0
    assert abs(result - 8.0) < 1e-5, f"Expected 8.0, got {result}"


def test_top_k_tail_k_larger_than_n() -> None:
    """top_k_tail clips k to the tensor length."""
    from tailseeker.utils.metrics import top_k_tail

    values = torch.tensor([1.0, 2.0, 3.0])
    result = top_k_tail(values, k=1000)
    assert abs(result - 2.0) < 1e-5, f"Expected 2.0, got {result}"
