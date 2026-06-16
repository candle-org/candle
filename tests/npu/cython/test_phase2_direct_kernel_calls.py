"""Test Phase 2: Backward formulas use direct Cython kernel calls, not Python redispatch.

Phase 2 goal: Replace _redispatch() calls in Cython backward formulas with direct
Cython kernel calls to eliminate the 27ms backward overhead.
"""
import numpy as np
import pytest


def test_mul_backward_bypasses_python_redispatch(npu_device, monkeypatch):
    """MulBackward should call Cython kernel directly, not Python redispatch."""
    import candle as torch
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey
    
    calls = {"python_mul": 0}
    
    def fail_python_mul(*args, **kwargs):
        calls["python_mul"] += 1
        raise AssertionError("MulBackward should use direct Cython kernel, not Python redispatch")
    
    # Monkey-patch Python dispatch for mul
    monkeypatch.setitem(registry.get("mul").kernels, DispatchKey.NPU, fail_python_mul)
    
    x = torch.tensor([2.0, 3.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([4.0, 5.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    
    # Forward: x * y
    z = x * y
    
    # Backward: triggers MulBackward which internally does grad * other
    loss = z.sum()
    loss.backward()
    
    # Verify gradients are correct
    np.testing.assert_allclose(x.grad.cpu().numpy(), [4.0, 5.0], rtol=1e-6)
    np.testing.assert_allclose(y.grad.cpu().numpy(), [2.0, 3.0], rtol=1e-6)
    
    # The critical check: backward should NOT have called Python redispatch
    assert calls["python_mul"] == 0, f"MulBackward called Python redispatch {calls['python_mul']} times"


def test_add_backward_bypasses_python_redispatch(npu_device, monkeypatch):
    """AddBackward should call Cython kernel directly, not Python redispatch."""
    import candle as torch
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey
    
    calls = {"python_add": 0}
    
    def fail_python_add(*args, **kwargs):
        calls["python_add"] += 1
        raise AssertionError("AddBackward should use direct Cython kernel, not Python redispatch")
    
    monkeypatch.setitem(registry.get("add").kernels, DispatchKey.NPU, fail_python_add)
    
    x = torch.tensor([2.0, 3.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([4.0, 5.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    
    z = x + y
    loss = z.sum()
    loss.backward()
    
    np.testing.assert_allclose(x.grad.cpu().numpy(), [1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(y.grad.cpu().numpy(), [1.0, 1.0], rtol=1e-6)
    
    assert calls["python_add"] == 0, f"AddBackward called Python redispatch {calls['python_add']} times"


def test_neg_backward_bypasses_python_redispatch(npu_device, monkeypatch):
    """NegBackward should call Cython kernel directly, not Python redispatch."""
    import candle as torch
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey

    x = torch.tensor([2.0, -3.0], device=npu_device, dtype=torch.float32, requires_grad=True)

    # Forward: -x
    z = -x

    # Patch AFTER forward, so only backward is tested
    calls = {"python_neg": 0}
    original_neg = registry.get("neg").kernels[DispatchKey.NPU]

    def fail_python_neg(*args, **kwargs):
        calls["python_neg"] += 1
        raise AssertionError("NegBackward should use direct Cython kernel, not Python redispatch")

    monkeypatch.setitem(registry.get("neg").kernels, DispatchKey.NPU, fail_python_neg)

    try:
        loss = z.sum()
        loss.backward()

        np.testing.assert_allclose(x.grad.cpu().numpy(), [-1.0, -1.0], rtol=1e-6)

        assert calls["python_neg"] == 0, f"NegBackward called Python redispatch {calls['python_neg']} times"
    finally:
        registry.get("neg").kernels[DispatchKey.NPU] = original_neg


def test_combined_ops_bypass_python_redispatch(npu_device, monkeypatch):
    """Combined ops (mul + add) should all bypass Python redispatch in backward."""
    import candle as torch
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey
    
    calls = {"python_mul": 0, "python_add": 0}
    
    def fail_python_mul(*args, **kwargs):
        calls["python_mul"] += 1
        raise AssertionError("Backward should not call Python mul dispatch")
    
    def fail_python_add(*args, **kwargs):
        calls["python_add"] += 1
        raise AssertionError("Backward should not call Python add dispatch")
    
    monkeypatch.setitem(registry.get("mul").kernels, DispatchKey.NPU, fail_python_mul)
    monkeypatch.setitem(registry.get("add").kernels, DispatchKey.NPU, fail_python_add)
    
    x = torch.tensor([2.0, 3.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([4.0, 5.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    
    # z = x * y + x
    # dz/dx = y + 1, dz/dy = x
    z = x * y + x
    loss = z.sum()
    loss.backward()
    
    np.testing.assert_allclose(x.grad.cpu().numpy(), [5.0, 6.0], rtol=1e-6)  # y + 1
    np.testing.assert_allclose(y.grad.cpu().numpy(), [2.0, 3.0], rtol=1e-6)  # x
    
    assert calls["python_mul"] == 0, f"Backward called Python mul {calls['python_mul']} times"
    assert calls["python_add"] == 0, f"Backward called Python add {calls['python_add']} times"
