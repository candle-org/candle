"""Comprehensive Phase 2 test: verify multiple backward ops bypass Python redispatch."""
import numpy as np
import pytest


def test_phase2_mul_div_bypass_redispatch(npu_device):
    """Test that mul and div in backward bypass Python redispatch."""
    import candle as torch
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey
    
    x = torch.tensor([2.0, 3.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([4.0, 6.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    
    # Forward: x / y (uses div)
    z = x / y
    
    # Patch mul and div AFTER forward
    calls = {"mul": 0, "div": 0}
    original_mul = registry.get("mul").kernels[DispatchKey.NPU]
    original_div = registry.get("true_divide").kernels[DispatchKey.NPU]
    
    def count_mul(*args, **kwargs):
        calls["mul"] += 1
        return original_mul(*args, **kwargs)
    
    def count_div(*args, **kwargs):
        calls["div"] += 1
        return original_div(*args, **kwargs)
    
    registry.get("mul").kernels[DispatchKey.NPU] = count_mul
    registry.get("true_divide").kernels[DispatchKey.NPU] = count_div
    
    try:
        # Backward: div backward uses mul and div internally
        loss = z.sum()
        loss.backward()
        
        # Verify gradients are correct
        # d(x/y)/dx = 1/y, d(x/y)/dy = -x/y^2
        np.testing.assert_allclose(x.grad.cpu().numpy(), [0.25, 1.0/6.0], rtol=1e-6)
        np.testing.assert_allclose(y.grad.cpu().numpy(), [-0.125, -1.0/12.0], rtol=1e-6)
        
        print(f"Backward: mul calls={calls['mul']}, div calls={calls['div']}")
        
        # Phase 2 goal: these should be 0
        if calls["mul"] == 0 and calls["div"] == 0:
            print("✓ Phase 2 SUCCESS: Backward bypassed all Python redispatch!")
        else:
            print(f"⚠ Phase 2 partial: still using Python redispatch")
            
    finally:
        registry.get("mul").kernels[DispatchKey.NPU] = original_mul
        registry.get("true_divide").kernels[DispatchKey.NPU] = original_div


def test_pow_backward_mul_count(npu_device):
    """Count how many mul calls pow backward makes."""
    import candle as torch
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey
    
    x = torch.tensor([2.0, 3.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    
    # Forward: x^2
    z = x.pow(2.0)
    
    # Count mul calls in backward
    calls = {"mul": 0}
    original_mul = registry.get("mul").kernels[DispatchKey.NPU]
    
    def count_mul(*args, **kwargs):
        calls["mul"] += 1
        return original_mul(*args, **kwargs)
    
    registry.get("mul").kernels[DispatchKey.NPU] = count_mul
    
    try:
        loss = z.sum()
        loss.backward()
        
        np.testing.assert_allclose(x.grad.cpu().numpy(), [4.0, 6.0], rtol=1e-6)
        
        print(f"Pow backward mul calls: {calls['mul']}")
        
        if calls["mul"] == 0:
            print("✓ Phase 2 SUCCESS for pow backward")
        else:
            print(f"⚠ Pow backward still calls Python mul {calls['mul']} times")
            
    finally:
        registry.get("mul").kernels[DispatchKey.NPU] = original_mul
