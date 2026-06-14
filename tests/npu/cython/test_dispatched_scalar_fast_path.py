"""RED tests: dispatched NPU scalar div/sub/pow must use the cached-scalar
Cython fast path, not the slow per-call _scalar_to_npu_tensor host-malloc +
byte-pack + H2D-memcpy materialization.

Backward helpers (e.g. MeanBackward0 -> div by numel, PowTensorScalarBackward0
-> pow(x, 1.0)) call these ops via redispatch with a Python scalar operand.
Each such call currently materializes a *full-shape* scalar tensor on device.
add/mul already avoid this via cached 0-d scalar kernels; div/sub/pow should
match. These tests assert the slow helper is NOT used and results stay correct.
"""
import numpy as np
import pytest


def _patch_scalar_counters(monkeypatch):
    """Patch every module binding of _scalar_to_npu_tensor with a counter.

    math.py binds the helper into its own namespace at import (used by div/sub);
    _npu_scalar_like (used by pow) imports it from _helpers at call time. Patch
    both so any slow-path use is detected regardless of which op triggers it.
    """
    import candle._backends.npu.ops.math as math_mod
    import candle._backends.npu.ops._helpers as helpers_mod

    calls = {"n": 0}

    def make_counter(orig):
        def counter(scalar, ref):
            calls["n"] += 1
            return orig(scalar, ref)
        return counter

    monkeypatch.setattr(math_mod, "_scalar_to_npu_tensor",
                        make_counter(math_mod._scalar_to_npu_tensor))
    monkeypatch.setattr(helpers_mod, "_scalar_to_npu_tensor",
                        make_counter(helpers_mod._scalar_to_npu_tensor))
    return calls


def test_dispatched_div_scalar_avoids_slow_helper(npu_device, monkeypatch):
    """div(tensor, python_scalar) must not materialize a full-shape scalar tensor."""
    import candle._backends.npu.ops.math as math_mod
    import candle as torch

    calls = _patch_scalar_counters(monkeypatch)

    x = torch.tensor([4.0, 6.0, 8.0], device=npu_device, dtype=torch.float32)
    out = math_mod.div(x, 2.0)

    np.testing.assert_allclose(out.cpu().numpy(), [2.0, 3.0, 4.0], rtol=1e-6)
    assert calls["n"] == 0, "div(tensor, scalar) used the slow _scalar_to_npu_tensor path"


def test_dispatched_sub_scalar_avoids_slow_helper(npu_device, monkeypatch):
    """sub(tensor, python_scalar) must not materialize a full-shape scalar tensor."""
    import candle._backends.npu.ops.math as math_mod
    import candle as torch

    calls = _patch_scalar_counters(monkeypatch)

    x = torch.tensor([4.0, 6.0, 8.0], device=npu_device, dtype=torch.float32)
    out = math_mod.sub(x, 1.0)

    np.testing.assert_allclose(out.cpu().numpy(), [3.0, 5.0, 7.0], rtol=1e-6)
    assert calls["n"] == 0, "sub(tensor, scalar) used the slow _scalar_to_npu_tensor path"


def test_dispatched_pow_scalar_avoids_slow_helper(npu_device, monkeypatch):
    """pow(tensor, python_scalar) must not materialize a full-shape scalar tensor.

    Uses exponent 3.0 (not 2.0, which has its own mul(a, a) shortcut) and 1.0
    (the exact exponent PowTensorScalarBackward0 produces for the x**2 backward).
    """
    import candle._backends.npu.ops.math as math_mod
    import candle as torch

    calls = _patch_scalar_counters(monkeypatch)

    x = torch.tensor([2.0, 3.0, 4.0], device=npu_device, dtype=torch.float32)
    out3 = math_mod.pow(x, 3.0)
    np.testing.assert_allclose(out3.cpu().numpy(), [8.0, 27.0, 64.0], rtol=1e-5)

    out1 = math_mod.pow(x, 1.0)
    np.testing.assert_allclose(out1.cpu().numpy(), [2.0, 3.0, 4.0], rtol=1e-6)

    assert calls["n"] == 0, "pow(tensor, scalar) used the slow _scalar_to_npu_tensor path"


def test_dispatched_scalar_ops_match_inference_dtypes(npu_device):
    """fp16/fp32 dispatched div/sub/pow scalar results stay numerically correct."""
    import candle._backends.npu.ops.math as math_mod
    import candle as torch

    for dtype, rtol in ((torch.float32, 1e-6), (torch.float16, 3e-3)):
        x = torch.tensor([4.0, 6.0, 8.0], device=npu_device, dtype=dtype)
        np.testing.assert_allclose(
            math_mod.div(x, 2.0).cpu().float().numpy(), [2.0, 3.0, 4.0], rtol=rtol)
        np.testing.assert_allclose(
            math_mod.sub(x, 1.0).cpu().float().numpy(), [3.0, 5.0, 7.0], rtol=rtol)
        np.testing.assert_allclose(
            math_mod.pow(x, 3.0).cpu().float().numpy(), [64.0, 216.0, 512.0], rtol=rtol)


def test_rmsnorm_backward_scalar_path_correct(npu_device):
    """End-to-end: mean/pow backward (RMSNorm-shaped) gradients stay correct.

    Exercises MeanBackward0 (div by numel) and PowTensorScalarBackward0
    (pow(x, 1.0)) through the dispatched scalar fast path during .backward().
    """
    import candle as torch

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=npu_device,
                     dtype=torch.float32, requires_grad=True)
    # variance = mean(x^2, -1, keepdim) ; loss = variance.sum()
    var = x.pow(2).mean(-1, keepdim=True)
    var.sum().backward()

    # d/dx mean(x^2)/1 over last dim of size n: 2x/n  (n=4)
    expected = (2.0 * np.array([[1.0, 2.0, 3.0, 4.0]])) / 4.0
    np.testing.assert_allclose(x.grad.cpu().numpy(), expected, rtol=1e-5)
