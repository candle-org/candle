"""Tests for gradient checkpointing (torch.utils.checkpoint).

Verifies that checkpointed forward passes produce correct outputs and
gradients, trading memory for compute.
"""
import numpy as np
import candle as torch
import candle.nn as nn
import candle.nn.functional as F
from candle.utils.checkpoint import checkpoint, checkpoint_sequential


def _make_mlp():
    """Create a small MLP for testing."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )


def test_checkpoint_output_matches():
    """Checkpointed output should match non-checkpointed output."""
    torch.manual_seed(42)
    model = _make_mlp()
    x = torch.randn(4, 8)
    x.requires_grad = True

    out_normal = model(x)
    out_ckpt = checkpoint(model, x)

    np.testing.assert_allclose(
        out_normal.detach().numpy(),
        out_ckpt.detach().numpy(),
        atol=1e-6,
    )


def test_checkpoint_gradients_match():
    """Gradients through checkpoint should match non-checkpointed gradients."""
    torch.manual_seed(42)
    model_a = _make_mlp()

    torch.manual_seed(42)
    model_b = _make_mlp()

    torch.manual_seed(0)
    x_data = torch.randn(4, 8).detach().numpy()

    # Non-checkpointed
    x_a = torch.tensor(x_data)
    x_a.requires_grad = True
    out_a = model_a(x_a)
    out_a.sum().backward()

    # Checkpointed
    x_b = torch.tensor(x_data)
    x_b.requires_grad = True
    out_b = checkpoint(model_b, x_b)
    out_b.sum().backward()

    np.testing.assert_allclose(
        x_a.grad.numpy(), x_b.grad.numpy(), atol=1e-5,
        err_msg="Input gradients should match"
    )

    for (name_a, p_a), (name_b, p_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert p_a.grad is not None, f"{name_a} has no grad (normal)"
        assert p_b.grad is not None, f"{name_b} has no grad (checkpoint)"
        np.testing.assert_allclose(
            p_a.grad.numpy(), p_b.grad.numpy(), atol=1e-5,
            err_msg=f"Gradient mismatch for {name_a}"
        )


def test_checkpoint_no_grad_input():
    """checkpoint with non-grad input should still produce correct output."""
    model = _make_mlp()
    x = torch.randn(4, 8)  # requires_grad=False
    out = checkpoint(model, x)
    assert out.shape == (4, 4)


def test_checkpoint_with_custom_function():
    """checkpoint should work with arbitrary callables, not just nn.Module."""
    def my_fn(x):
        return x * 2 + 1

    x = torch.randn(4, 8)
    x.requires_grad = True
    out = checkpoint(my_fn, x)
    out.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.full((4, 8), 2.0), atol=1e-6)


def test_checkpoint_sequential_basic():
    """checkpoint_sequential should split modules into segments."""
    torch.manual_seed(42)
    layers = [nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)]
    x = torch.randn(4, 8)
    x.requires_grad = True

    out = checkpoint_sequential(layers, 2, x)
    assert out.shape == (4, 4)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == (4, 8)


def test_checkpoint_training_convergence():
    """Training with checkpoint should converge like normal training."""
    torch.manual_seed(42)
    X = torch.randn(16, 8)
    X.requires_grad = True
    Y = torch.randn(16, 4)

    model = _make_mlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        out = checkpoint(model, X)
        loss = F.mse_loss(out, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, \
        f"Checkpointed training should converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
