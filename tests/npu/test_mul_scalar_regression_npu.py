import pytest

import candle as torch
from candle._backends.npu import aclnn
from candle._backends.npu import ops as npu_ops


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_mul_scalar_after_randn_on_npu():
    torch.manual_seed(1234)

    x = torch.randn(128, device="npu")
    scale = torch.tensor(3.0, device="npu")

    y = torch.mul(x, scale)
    assert y.device.type == "npu"
    assert y.shape == x.shape
    expected = x.to("cpu").numpy() * 3.0
    assert y.to("cpu").shape == x.to("cpu").shape
    assert pytest.approx(expected, rel=1e-6, abs=1e-6) == y.to("cpu").numpy()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_mul_supports_zero_dim_npu_tensors_without_fallback():
    a = torch.tensor(2.0, device="npu")
    b = torch.tensor(3.0, device="npu")

    y = torch.mul(a, b)
    assert y.device.type == "npu"
    assert y.shape == ()
    assert y.item() == pytest.approx(6.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_mul_zero_dim_npu_parameter_keeps_gradients():
    x = torch.randn(16, device="npu")
    weight = torch.tensor(3.0, device="npu", requires_grad=True)

    y = torch.mul(x, weight).sum()
    y.backward()

    assert weight.grad is not None
    assert weight.grad.shape == ()
    assert weight.grad.device.type == "npu"
    assert weight.grad.item() == pytest.approx(x.to("cpu").sum().item(), rel=1e-5, abs=1e-5)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_mul_propagates_kernel_error_instead_of_falling_back(monkeypatch):
    x = torch.tensor([2.0, 3.0], device="npu")
    y = torch.tensor([4.0, 5.0], device="npu")

    def fail_mul(*args, **kwargs):
        raise RuntimeError("sentinel mul failure")

    def fail_div(*args, **kwargs):
        raise AssertionError("mul should not fall back to div")

    # When the Cython fast_mul path is active, aclnn.mul is bypassed entirely.
    # Patch the FFI-level binary_op_no_alpha to verify error propagation from
    # the actual execution path.
    try:
        import candle._C._aclnn_ffi as ffi_mod  # pylint: disable=import-error
        monkeypatch.setattr(ffi_mod, "binary_op_no_alpha", fail_mul)
    except (ImportError, AttributeError):
        # Fall back to patching the Python aclnn wrapper if Cython FFI unavailable
        monkeypatch.setattr(aclnn, "mul", fail_mul)
    monkeypatch.setattr(npu_ops, "div", fail_div)

    with pytest.raises(RuntimeError, match="sentinel mul failure"):
        torch.mul(x, y)
