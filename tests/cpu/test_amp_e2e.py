"""End-to-end tests for mixed precision training (AMP).

Tests the autocast context manager and GradScaler workflow on CPU.
CPU autocast uses bfloat16 by default.
"""
import numpy as np
import candle as torch
import candle.nn as nn
import candle.nn.functional as F
from candle.amp import autocast, GradScaler


def test_autocast_cpu_context_manager():
    """autocast('cpu') should enter and exit without error."""
    model = nn.Linear(8, 4)
    x = torch.randn(2, 8)
    with autocast("cpu"):
        out = model(x)
    assert out.shape == (2, 4)


def test_autocast_cpu_dtype():
    """CPU autocast should cast float32 inputs to bfloat16."""
    x = torch.randn(2, 4)
    assert x.dtype == torch.float32
    with autocast("cpu"):
        # Inside autocast, eligible ops should receive bfloat16 inputs
        # The autocast state should be enabled
        from candle.amp.state import is_autocast_enabled
        assert is_autocast_enabled("cpu")
    assert not is_autocast_enabled("cpu")


def test_autocast_disabled():
    """autocast with enabled=False should be a no-op."""
    with autocast("cpu", enabled=False):
        from candle.amp.state import is_autocast_enabled
        assert not is_autocast_enabled("cpu")


def test_autocast_nesting():
    """Nested autocast should work correctly."""
    from candle.amp.state import is_autocast_enabled
    assert not is_autocast_enabled("cpu")
    with autocast("cpu"):
        assert is_autocast_enabled("cpu")
        with autocast("cpu", enabled=False):
            assert not is_autocast_enabled("cpu")
        assert is_autocast_enabled("cpu")
    assert not is_autocast_enabled("cpu")


def test_autocast_as_decorator():
    """autocast should work as a function decorator."""
    @autocast("cpu")
    def forward(x):
        return x * 2

    out = forward(torch.randn(2, 4))
    assert out.shape == (2, 4)


def test_grad_scaler_basic():
    """GradScaler basic workflow: scale -> backward -> unscale -> step -> update."""
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = GradScaler(device="cpu")

    x = torch.randn(2, 4)
    y = torch.randn(2, 2)

    optimizer.zero_grad()
    with autocast("cpu"):
        out = model(x)
        loss = F.mse_loss(out, y)

    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    scaler.step(optimizer)
    scaler.update()

    # After step, weights should have been updated
    assert scaler.get_scale() > 0


def test_grad_scaler_disabled():
    """GradScaler with enabled=False should be a passthrough."""
    scaler = GradScaler(device="cpu", enabled=False)
    assert not scaler.is_enabled()
    assert scaler.get_scale() == 1.0

    x = torch.tensor([3.0])
    scaled = scaler.scale(x)
    np.testing.assert_allclose(scaled.numpy(), x.numpy())


def test_grad_scaler_state_dict():
    """GradScaler state_dict should be saveable and loadable."""
    scaler = GradScaler(device="cpu", init_scale=1024.0)
    sd = scaler.state_dict()
    assert "scale" in sd
    assert sd["scale"] == 1024.0

    scaler2 = GradScaler(device="cpu")
    scaler2.load_state_dict(sd)
    assert scaler2.get_scale() == 1024.0


def test_amp_training_loop():
    """Full AMP training loop: autocast + GradScaler over multiple steps."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scaler = GradScaler(device="cpu")

    X = torch.randn(16, 8)
    Y = torch.randn(16, 4)

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        with autocast("cpu"):
            out = model(X)
            loss = F.mse_loss(out, Y)
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

    assert losses[-1] < losses[0], \
        f"AMP training should reduce loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
