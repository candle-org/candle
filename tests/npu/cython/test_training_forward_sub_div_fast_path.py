"""Test training-mode forward fast paths for NPU sub/div operators.

Training-mode forward sub/div ops (requires_grad=True) should stay in Cython and
attach autograd nodes directly, rather than falling back to Python dispatch.

This tests the implementation in _tensor_impl.pyx __sub__/__truediv__ slots with
the _can_use_npu_binary_slot_train guard plus the attach_npu_sub_grad/
attach_npu_div_grad autograd attach functions in _functional_ops.pyx.
"""
import numpy as np
import pytest


def test_npu_sub_training_mode_stays_in_cython(npu_device, monkeypatch):
    """Training-mode sub should use Cython fast path, not Python dispatch."""
    import candle as torch
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey

    calls = {"python_sub": 0}

    def fail_python_sub(*args, **kwargs):
        calls["python_sub"] += 1
        raise AssertionError("Training-mode sub should use Cython fast path, not Python dispatch")

    monkeypatch.setitem(registry.get("sub").kernels, DispatchKey.NPU, fail_python_sub)

    x = torch.tensor([5.0, 8.0, 11.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([2.0, 3.0, 4.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    z = x - y

    assert z.requires_grad
    assert z.grad_fn is not None
    assert "Sub" in str(type(z.grad_fn))

    loss = z.sum()
    loss.backward()

    # d(x - y)/dx = 1, d(x - y)/dy = -1
    np.testing.assert_allclose(x.grad.cpu().numpy(), [1.0, 1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(y.grad.cpu().numpy(), [-1.0, -1.0, -1.0], rtol=1e-6)

    assert calls["python_sub"] == 0, "Training-mode sub went through Python dispatch"


def test_npu_div_training_mode_stays_in_cython(npu_device, monkeypatch):
    """Training-mode div should use Cython fast path, not Python dispatch."""
    import candle as torch
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey

    calls = {"python_div": 0}

    def fail_python_div(*args, **kwargs):
        calls["python_div"] += 1
        raise AssertionError("Training-mode div should use Cython fast path, not Python dispatch")

    # true_divide is the registered op name for tensor-tensor division
    monkeypatch.setitem(registry.get("true_divide").kernels, DispatchKey.NPU, fail_python_div)

    x = torch.tensor([4.0, 6.0, 8.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([2.0, 2.0, 4.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    z = x / y

    assert z.requires_grad
    assert z.grad_fn is not None
    assert "Div" in str(type(z.grad_fn))

    loss = z.sum()
    loss.backward()

    # d(x / y)/dx = 1/y ; d(x / y)/dy = -x/y^2
    np.testing.assert_allclose(x.grad.cpu().numpy(), [0.5, 0.5, 0.25], rtol=1e-6)
    np.testing.assert_allclose(y.grad.cpu().numpy(), [-1.0, -1.5, -0.5], rtol=1e-6)

    assert calls["python_div"] == 0, "Training-mode div went through Python dispatch"


def test_npu_training_forward_sub_div_correctness_matches_inference(npu_device):
    """Training-mode sub/div forward results should match inference mode."""
    import candle as torch

    x_vals = np.random.randn(4, 4).astype(np.float32)
    y_vals = (np.random.randn(4, 4) + 3.0).astype(np.float32)  # avoid div by ~0

    with torch.no_grad():
        x_inf = torch.tensor(x_vals, device=npu_device, dtype=torch.float32)
        y_inf = torch.tensor(y_vals, device=npu_device, dtype=torch.float32)
        z_inf_sub = x_inf - y_inf
        z_inf_div = x_inf / y_inf

    x_train = torch.tensor(x_vals, device=npu_device, dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor(y_vals, device=npu_device, dtype=torch.float32, requires_grad=True)
    z_train_sub = x_train - y_train
    z_train_div = x_train / y_train

    np.testing.assert_allclose(z_train_sub.detach().cpu().numpy(), z_inf_sub.cpu().numpy(), rtol=1e-6)
    np.testing.assert_allclose(z_train_div.detach().cpu().numpy(), z_inf_div.cpu().numpy(), rtol=1e-5)


def test_npu_training_forward_sub_div_mixed_requires_grad(npu_device):
    """Sub/div fast path should work when only one operand requires grad."""
    import candle as torch

    # Only x requires grad
    x = torch.tensor([6.0, 9.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([2.0, 3.0], device=npu_device, dtype=torch.float32, requires_grad=False)

    z_sub = x - y
    assert z_sub.requires_grad and z_sub.grad_fn is not None
    z_sub.sum().backward()
    np.testing.assert_allclose(x.grad.cpu().numpy(), [1.0, 1.0], rtol=1e-6)
    assert y.grad is None

    x2 = torch.tensor([6.0, 9.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    z_div = x2 / y
    assert z_div.requires_grad and z_div.grad_fn is not None
    z_div.sum().backward()
    np.testing.assert_allclose(x2.grad.cpu().numpy(), [0.5, 1.0 / 3.0], rtol=1e-6)
    assert y.grad is None

    # Only y requires grad
    a = torch.tensor([6.0, 9.0], device=npu_device, dtype=torch.float32, requires_grad=False)
    b = torch.tensor([2.0, 3.0], device=npu_device, dtype=torch.float32, requires_grad=True)

    z_sub2 = a - b
    assert z_sub2.requires_grad and z_sub2.grad_fn is not None
    z_sub2.sum().backward()
    np.testing.assert_allclose(b.grad.cpu().numpy(), [-1.0, -1.0], rtol=1e-6)

    b2 = torch.tensor([2.0, 3.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    z_div2 = a / b2
    assert z_div2.requires_grad and z_div2.grad_fn is not None
    z_div2.sum().backward()
    # d(a / b)/db = -a/b^2 = [-6/4, -9/9] = [-1.5, -1.0]
    np.testing.assert_allclose(b2.grad.cpu().numpy(), [-1.5, -1.0], rtol=1e-6)


def test_npu_inference_mode_sub_div_still_fast(npu_device, monkeypatch):
    """Inference mode sub/div should still use the inference-only fast path."""
    import candle as torch
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey

    calls = {"python": 0}

    def fail_python(*args, **kwargs):
        calls["python"] += 1
        raise AssertionError("Inference-mode sub/div should use Cython fast path")

    monkeypatch.setitem(registry.get("sub").kernels, DispatchKey.NPU, fail_python)
    monkeypatch.setitem(registry.get("true_divide").kernels, DispatchKey.NPU, fail_python)

    with torch.no_grad():
        x = torch.tensor([4.0, 6.0], device=npu_device, dtype=torch.float32)
        y = torch.tensor([2.0, 3.0], device=npu_device, dtype=torch.float32)
        z_sub = x - y
        z_div = x / y

    assert not z_sub.requires_grad and z_sub.grad_fn is None
    assert not z_div.requires_grad and z_div.grad_fn is None
    np.testing.assert_allclose(z_sub.cpu().numpy(), [2.0, 3.0], rtol=1e-6)
    np.testing.assert_allclose(z_div.cpu().numpy(), [2.0, 2.0], rtol=1e-6)

    assert calls["python"] == 0


def test_npu_div_backward_create_graph(npu_device):
    """Div backward under create_graph should build a differentiable graph (second-order)."""
    import candle as torch
    import candle.autograd as ag

    x = torch.tensor([4.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([2.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    z = x / y

    g = ag.grad(z, [x, y], grad_outputs=torch.ones_like(z), create_graph=True)
    # dz/dx = 1/y = 0.5 ; dz/dy = -x/y^2 = -1.0
    np.testing.assert_allclose(g[0].detach().cpu().numpy(), [0.5], rtol=1e-6)
    np.testing.assert_allclose(g[1].detach().cpu().numpy(), [-1.0], rtol=1e-6)

    # First-order grads must remain differentiable.
    assert g[0].grad_fn is not None
    assert g[1].grad_fn is not None

    # Second-order: d/dx(dz/dy) = d/dx(-x/y^2) = -1/y^2 = -0.25
    g2 = ag.grad(g[1], [x], grad_outputs=torch.ones_like(g[1]), retain_graph=True)
    np.testing.assert_allclose(g2[0].detach().cpu().numpy(), [-0.25], rtol=1e-6)


def test_npu_sub_backward_create_graph(npu_device):
    """Sub backward under create_graph should produce correct gradients.

    The gradient of subtraction is constant (+1 / -1), so like PyTorch the
    resulting gradient tensors carry no grad_fn — but the call must succeed
    under create_graph and yield correct values without raising.
    """
    import candle as torch
    import candle.autograd as ag

    x = torch.tensor([5.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([2.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    z = x - y

    g = ag.grad(z, [x, y], grad_outputs=torch.ones_like(z), create_graph=True)
    # dz/dx = 1 ; dz/dy = -1
    np.testing.assert_allclose(g[0].detach().cpu().numpy(), [1.0], rtol=1e-6)
    np.testing.assert_allclose(g[1].detach().cpu().numpy(), [-1.0], rtol=1e-6)


