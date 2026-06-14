"""Test training-mode forward fast paths for NPU operators.

Training-mode forward ops (requires_grad=True) should stay in Cython and attach
autograd nodes directly, rather than falling back to Python dispatch.

This tests the implementation in _tensor_impl.pyx __add__/__mul__ slots with
_can_use_npu_binary_slot_train guard.
"""
import numpy as np
import pytest


def test_npu_add_training_mode_stays_in_cython(npu_device, monkeypatch):
    """Training-mode add should use Cython fast path, not Python dispatch."""
    import candle as torch

    # Monkeypatch to ensure Python dispatch is NOT called
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey

    calls = {"python_add": 0}

    def fail_python_add(*args, **kwargs):
        calls["python_add"] += 1
        raise AssertionError("Training-mode add should use Cython fast path, not Python dispatch")

    monkeypatch.setitem(registry.get("add").kernels, DispatchKey.NPU, fail_python_add)

    # Training mode: requires_grad=True
    x = torch.ones(2, 2, device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.ones(2, 2, device=npu_device, dtype=torch.float32, requires_grad=True)
    z = x + y

    # Verify autograd attached
    assert z.requires_grad
    assert z.grad_fn is not None
    assert "Add" in str(type(z.grad_fn))

    # Verify backward works
    loss = z.sum()
    loss.backward()

    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad.cpu().numpy(), np.ones((2, 2)), rtol=1e-6)
    np.testing.assert_allclose(y.grad.cpu().numpy(), np.ones((2, 2)), rtol=1e-6)

    # Verify Python dispatch was NOT called
    assert calls["python_add"] == 0, "Training-mode add went through Python dispatch"


def test_npu_mul_training_mode_stays_in_cython(npu_device, monkeypatch):
    """Training-mode mul should use Cython fast path, not Python dispatch."""
    import candle as torch
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey

    calls = {"python_mul": 0}

    def fail_python_mul(*args, **kwargs):
        calls["python_mul"] += 1
        raise AssertionError("Training-mode mul should use Cython fast path, not Python dispatch")

    monkeypatch.setitem(registry.get("mul").kernels, DispatchKey.NPU, fail_python_mul)

    x = torch.tensor([1.0, 2.0, 3.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    z = x * y

    assert z.requires_grad
    assert z.grad_fn is not None
    assert "Mul" in str(type(z.grad_fn))

    loss = z.sum()
    loss.backward()

    np.testing.assert_allclose(x.grad.cpu().numpy(), [4.0, 5.0, 6.0], rtol=1e-6)
    np.testing.assert_allclose(y.grad.cpu().numpy(), [1.0, 2.0, 3.0], rtol=1e-6)

    assert calls["python_mul"] == 0


def test_npu_training_forward_correctness_matches_inference(npu_device):
    """Training-mode forward results should match inference mode."""
    import candle as torch

    x_vals = np.random.randn(4, 4).astype(np.float32)
    y_vals = np.random.randn(4, 4).astype(np.float32)

    # Inference mode
    with torch.no_grad():
        x_inf = torch.tensor(x_vals, device=npu_device, dtype=torch.float32)
        y_inf = torch.tensor(y_vals, device=npu_device, dtype=torch.float32)
        z_inf_add = x_inf + y_inf
        z_inf_mul = x_inf * y_inf

    # Training mode
    x_train = torch.tensor(x_vals, device=npu_device, dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor(y_vals, device=npu_device, dtype=torch.float32, requires_grad=True)
    z_train_add = x_train + y_train
    z_train_mul = x_train * y_train

    # Results should match
    np.testing.assert_allclose(z_train_add.detach().cpu().numpy(), z_inf_add.cpu().numpy(), rtol=1e-6)
    np.testing.assert_allclose(z_train_mul.detach().cpu().numpy(), z_inf_mul.cpu().numpy(), rtol=1e-6)


def test_npu_training_forward_mixed_requires_grad(npu_device):
    """Training-mode fast path should work when only one operand requires grad."""
    import candle as torch

    # Only x requires grad
    x = torch.ones(3, 3, device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.ones(3, 3, device=npu_device, dtype=torch.float32, requires_grad=False)
    z = x + y

    assert z.requires_grad
    assert z.grad_fn is not None

    loss = z.sum()
    loss.backward()

    assert x.grad is not None
    assert y.grad is None
    np.testing.assert_allclose(x.grad.cpu().numpy(), np.ones((3, 3)), rtol=1e-6)


def test_npu_inference_mode_still_fast(npu_device, monkeypatch):
    """Inference mode should still use the original inference-only fast path."""
    import candle as torch
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey

    calls = {"python_add": 0}

    def fail_python_add(*args, **kwargs):
        calls["python_add"] += 1
        raise AssertionError("Inference-mode add should use Cython fast path")

    monkeypatch.setitem(registry.get("add").kernels, DispatchKey.NPU, fail_python_add)

    # Inference mode
    with torch.no_grad():
        x = torch.ones(2, 2, device=npu_device, dtype=torch.float32)
        y = torch.ones(2, 2, device=npu_device, dtype=torch.float32)
        z = x + y

    assert not z.requires_grad
    assert z.grad_fn is None
    np.testing.assert_allclose(z.cpu().numpy(), 2 * np.ones((2, 2)), rtol=1e-6)

    assert calls["python_add"] == 0
