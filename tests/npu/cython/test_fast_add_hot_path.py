import ctypes


def test_fast_add_skips_storage_method_calls(npu_device, monkeypatch):
    """fast_add should read device pointers directly, not via Tensor.storage()."""
    import candle as torch

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    # Warm up once so first-use imports/cache population do not affect the assertion.
    _ = torch.add(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    tensor_type = type(a)
    original_storage = tensor_type.storage

    def wrapped_storage(self, *args, **kwargs):
        calls["count"] += 1
        return original_storage(self, *args, **kwargs)

    monkeypatch.setattr(tensor_type, "storage", wrapped_storage)

    _ = torch.add(a, b)

    assert calls["count"] == 0


def test_fast_add_skips_ctypes_void_p_wrapper(npu_device, monkeypatch):
    """fast_add should defer executors by raw handle, without ctypes.c_void_p."""
    import candle as torch

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)

    # Warm up once so first-use imports/cache population do not affect the assertion.
    _ = torch.add(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_c_void_p = ctypes.c_void_p

    def wrapped_c_void_p(value):
        calls["count"] += 1
        return original_c_void_p(value)

    monkeypatch.setattr(ctypes, "c_void_p", wrapped_c_void_p)

    _ = torch.add(a, b)

    assert calls["count"] == 0


def test_fast_add_reuses_cached_executor_for_same_signature(npu_device, monkeypatch):
    """fast_add may either hit the PTA executor cache or stay on the non-PTA path."""
    import pytest
    import candle as torch
    import candle._cython._aclnn_ffi as ffi_mod  # pylint: disable=import-error,no-name-in-module

    if not ffi_mod.is_pta_cache_available():
        pytest.skip("PTA executor cache not available on this CANN build")

    a = torch.randn(7, 5, 3, device=npu_device)
    b = torch.randn(7, 5, 3, device=npu_device)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_binary_op_with_alpha = ffi_mod.binary_op_with_alpha

    def wrapped_binary_op_with_alpha(*args, **kwargs):
        calls["count"] += 1
        return original_binary_op_with_alpha(*args, **kwargs)

    monkeypatch.setattr(ffi_mod, "binary_op_with_alpha", wrapped_binary_op_with_alpha)

    _ = torch.add(a, b)
    torch.npu.synchronize()
    assert calls["count"] == 1

    _ = torch.add(a, b)
    torch.npu.synchronize()
    assert calls["count"] in (1, 2)


def test_fast_mul_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """fast_mul should bypass candle._backends.npu.aclnn.mul on the hot path."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    # Warm up so first-use imports/caches do not affect assertion
    _ = torch.mul(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_mul = aclnn_mod.mul

    def wrapped_mul(*args, **kwargs):
        calls["count"] += 1
        return original_mul(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "mul", wrapped_mul)

    result = torch.mul(a, b)
    torch.npu.synchronize()

    # Assertion (a): hot path must not call Python aclnn.mul
    assert calls["count"] == 0, (
        f"fast_mul called aclnn.mul {calls['count']} time(s); expected 0"
    )

    # Assertion (b): numerical correctness
    import torch as ref_torch
    a_cpu = ref_torch.tensor(a.cpu().numpy())
    b_cpu = ref_torch.tensor(b.cpu().numpy())
    expected = a_cpu * b_cpu
    actual = ref_torch.tensor(result.cpu().numpy())
    assert ref_torch.allclose(actual, expected, rtol=1e-4, atol=1e-4), (
        f"fast_mul result mismatch: max_diff={(actual - expected).abs().max()}"
    )


def test_fast_sub_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """fast_sub should bypass candle._backends.npu.aclnn.sub on the hot path."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod
    import numpy as np

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    expected = torch.sub(a, b)
    torch.npu.synchronize()

    _ = torch.sub(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_sub = aclnn_mod.sub

    def wrapped_sub(*args, **kwargs):
        calls["count"] += 1
        return original_sub(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "sub", wrapped_sub)

    out = torch.sub(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0, (
        f"fast_sub called aclnn.sub {calls['count']} time(s); expected 0"
    )
    assert np.allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-4, atol=1e-4), (
        "fast_sub output differs from expected"
    )


def test_fast_div_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """fast_div should bypass candle._backends.npu.aclnn.div on the hot path."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod
    import numpy as np

    a = torch.randn(4, 4, device=npu_device)
    b = torch.rand(4, 4, device=npu_device) + 1.0
    torch.npu.synchronize()

    expected = torch.div(a, b)
    torch.npu.synchronize()

    _ = torch.div(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_div = aclnn_mod.div

    def wrapped_div(*args, **kwargs):
        calls["count"] += 1
        return original_div(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "div", wrapped_div)

    out = torch.div(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0, (
        f"fast_div called aclnn.div {calls['count']} time(s); expected 0"
    )
    assert np.allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-4, atol=1e-4), (
        "fast_div output differs from expected"
    )


def test_fast_atan2_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.atan2."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.atan2(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_atan2 = aclnn_mod.atan2

    def wrapped_atan2(*args, **kwargs):
        calls["count"] += 1
        return original_atan2(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "atan2", wrapped_atan2)

    _ = torch.atan2(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_logaddexp_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.slogaddexp."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.logaddexp(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_logaddexp = aclnn_mod.slogaddexp

    def wrapped_logaddexp(*args, **kwargs):
        calls["count"] += 1
        return original_logaddexp(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "slogaddexp", wrapped_logaddexp)

    _ = torch.logaddexp(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_logaddexp2_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.slogaddexp2."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.logaddexp2(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_logaddexp2 = aclnn_mod.slogaddexp2

    def wrapped_logaddexp2(*args, **kwargs):
        calls["count"] += 1
        return original_logaddexp2(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "slogaddexp2", wrapped_logaddexp2)

    _ = torch.logaddexp2(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_remainder_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.sremainder."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device) + 1.0
    torch.npu.synchronize()

    _ = torch.remainder(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_remainder = aclnn_mod.sremainder

    def wrapped_remainder(*args, **kwargs):
        calls["count"] += 1
        return original_remainder(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "sremainder", wrapped_remainder)

    _ = torch.remainder(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_fmod_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.sfmod."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device) + 1.0
    torch.npu.synchronize()

    _ = torch.fmod(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_fmod = aclnn_mod.sfmod

    def wrapped_fmod(*args, **kwargs):
        calls["count"] += 1
        return original_fmod(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "sfmod", wrapped_fmod)

    _ = torch.fmod(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_logical_and_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.logical_and."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([[True, False], [True, True]], device=npu_device)
    b = torch.tensor([[True, True], [False, True]], device=npu_device)
    torch.npu.synchronize()

    _ = torch.logical_and(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_logical_and = aclnn_mod.logical_and

    def wrapped_logical_and(*args, **kwargs):
        calls["count"] += 1
        return original_logical_and(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "logical_and", wrapped_logical_and)

    _ = torch.logical_and(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_logical_or_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.logical_or."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([[True, False], [True, True]], device=npu_device)
    b = torch.tensor([[True, True], [False, True]], device=npu_device)
    torch.npu.synchronize()

    _ = torch.logical_or(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_logical_or = aclnn_mod.logical_or

    def wrapped_logical_or(*args, **kwargs):
        calls["count"] += 1
        return original_logical_or(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "logical_or", wrapped_logical_or)

    _ = torch.logical_or(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_bitwise_and_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.bitwise_and."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([[1, 3], [7, 15]], dtype=torch.int64, device=npu_device)
    b = torch.tensor([[1, 1], [3, 7]], dtype=torch.int64, device=npu_device)
    torch.npu.synchronize()

    _ = torch.bitwise_and(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_bitwise_and = aclnn_mod.bitwise_and

    def wrapped_bitwise_and(*args, **kwargs):
        calls["count"] += 1
        return original_bitwise_and(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "bitwise_and", wrapped_bitwise_and)

    _ = torch.bitwise_and(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_bitwise_or_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.bitwise_or."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([[1, 3], [7, 15]], dtype=torch.int64, device=npu_device)
    b = torch.tensor([[1, 1], [3, 7]], dtype=torch.int64, device=npu_device)
    torch.npu.synchronize()

    _ = torch.bitwise_or(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_bitwise_or = aclnn_mod.bitwise_or

    def wrapped_bitwise_or(*args, **kwargs):
        calls["count"] += 1
        return original_bitwise_or(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "bitwise_or", wrapped_bitwise_or)

    _ = torch.bitwise_or(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_bitwise_xor_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.bitwise_xor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([[1, 3], [7, 15]], dtype=torch.int64, device=npu_device)
    b = torch.tensor([[1, 1], [3, 7]], dtype=torch.int64, device=npu_device)
    torch.npu.synchronize()

    _ = torch.bitwise_xor(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_bitwise_xor = aclnn_mod.bitwise_xor

    def wrapped_bitwise_xor(*args, **kwargs):
        calls["count"] += 1
        return original_bitwise_xor(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "bitwise_xor", wrapped_bitwise_xor)

    _ = torch.bitwise_xor(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_maximum_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.maximum."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.maximum(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_maximum = aclnn_mod.maximum

    def wrapped_maximum(*args, **kwargs):
        calls["count"] += 1
        return original_maximum(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "maximum", wrapped_maximum)

    _ = torch.maximum(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_minimum_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.minimum."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.minimum(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_minimum = aclnn_mod.minimum

    def wrapped_minimum(*args, **kwargs):
        calls["count"] += 1
        return original_minimum(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "minimum", wrapped_minimum)

    _ = torch.minimum(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_logical_xor_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """logical_xor should bypass candle._backends.npu.aclnn.logical_xor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([[True, False], [True, True]], device=npu_device)
    b = torch.tensor([[True, True], [False, True]], device=npu_device)
    torch.npu.synchronize()

    _ = torch.logical_xor(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_logical_xor = aclnn_mod.logical_xor

    def wrapped_logical_xor(*args, **kwargs):
        calls["count"] += 1
        return original_logical_xor(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "logical_xor", wrapped_logical_xor)

    _ = torch.logical_xor(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_pow_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """tensor-tensor pow should bypass candle._backends.npu.aclnn.pow_tensor_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.rand(4, 4, device=npu_device) + 0.5
    b = torch.rand(4, 4, device=npu_device) + 0.5
    torch.npu.synchronize()

    _ = torch.pow(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_pow = aclnn_mod.pow_tensor_tensor

    def wrapped_pow(*args, **kwargs):
        calls["count"] += 1
        return original_pow(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "pow_tensor_tensor", wrapped_pow)

    _ = torch.pow(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_fmax_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """fmax should bypass candle._backends.npu.aclnn.maximum."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.fmax(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_maximum = aclnn_mod.maximum

    def wrapped_maximum(*args, **kwargs):
        calls["count"] += 1
        return original_maximum(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "maximum", wrapped_maximum)

    _ = torch.fmax(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_floor_divide_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """floor_divide should bypass candle._backends.npu.aclnn.floor_divide."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.rand(4, 4, device=npu_device) + 1.0
    torch.npu.synchronize()

    _ = torch.floor_divide(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_floor_divide = aclnn_mod.floor_divide

    def wrapped_floor_divide(*args, **kwargs):
        calls["count"] += 1
        return original_floor_divide(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "floor_divide", wrapped_floor_divide)

    _ = torch.floor_divide(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_lt_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """lt should bypass candle._backends.npu.aclnn.lt_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.lt(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_lt = aclnn_mod.lt_tensor

    def wrapped_lt(*args, **kwargs):
        calls["count"] += 1
        return original_lt(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "lt_tensor", wrapped_lt)

    _ = torch.lt(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_le_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """le should bypass candle._backends.npu.aclnn.le_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.le(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_le = aclnn_mod.le_tensor

    def wrapped_le(*args, **kwargs):
        calls["count"] += 1
        return original_le(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "le_tensor", wrapped_le)

    _ = torch.le(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_gt_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """gt should bypass candle._backends.npu.aclnn.gt_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.gt(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_gt = aclnn_mod.gt_tensor

    def wrapped_gt(*args, **kwargs):
        calls["count"] += 1
        return original_gt(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "gt_tensor", wrapped_gt)

    _ = torch.gt(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_ge_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """ge should bypass candle._backends.npu.aclnn.ge_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.ge(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_ge = aclnn_mod.ge_tensor

    def wrapped_ge(*args, **kwargs):
        calls["count"] += 1
        return original_ge(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "ge_tensor", wrapped_ge)

    _ = torch.ge(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_eq_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """eq should bypass candle._backends.npu.aclnn.eq_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.eq(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_eq = aclnn_mod.eq_tensor

    def wrapped_eq(*args, **kwargs):
        calls["count"] += 1
        return original_eq(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "eq_tensor", wrapped_eq)

    _ = torch.eq(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_ne_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """ne should bypass candle._backends.npu.aclnn.ne_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.ne(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_ne = aclnn_mod.ne_tensor

    def wrapped_ne(*args, **kwargs):
        calls["count"] += 1
        return original_ne(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "ne_tensor", wrapped_ne)

    _ = torch.ne(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_clamp_min_tensor_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """clamp with tensor min should bypass candle._backends.npu.aclnn.clamp_min_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    min_val = torch.zeros(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.clamp(a, min_val)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.clamp_min_tensor

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "clamp_min_tensor", wrapped)

    _ = torch.clamp(a, min_val)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_clamp_max_tensor_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """clamp with tensor max should bypass candle._backends.npu.aclnn.clamp_max_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    max_val = torch.zeros(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.clamp(a, None, max_val)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.clamp_max_tensor

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "clamp_max_tensor", wrapped)

    _ = torch.clamp(a, None, max_val)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_clamp_tensor_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """clamp with tensor min and max should bypass candle._backends.npu.aclnn.clamp_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    min_val = torch.full((4, 4), -0.5, device=npu_device)
    max_val = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.clamp(a, min_val, max_val)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.clamp_tensor

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "clamp_tensor", wrapped)

    _ = torch.clamp(a, min_val, max_val)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_trunc_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.trunc should bypass candle._backends.npu.aclnn.trunc."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.trunc(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.trunc

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "trunc", wrapped)

    _ = torch.trunc(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_abs_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.abs should bypass candle._backends.npu.aclnn.abs."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), -0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.abs(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.abs

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "abs", wrapped)

    _ = torch.abs(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_neg_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.neg should bypass candle._backends.npu.aclnn.neg."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), -0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.neg(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.neg

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "neg", wrapped)

    _ = torch.neg(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_sign_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.sign should bypass candle._backends.npu.aclnn.sign."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), -0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.sign(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.sign

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "sign", wrapped)

    _ = torch.sign(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_square_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.square should bypass candle._backends.npu.aclnn.square."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.square(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.square

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "square", wrapped)

    _ = torch.square(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_signbit_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.signbit should bypass candle._backends.npu.aclnn.signbit."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), -0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.signbit(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.signbit

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "signbit", wrapped)

    _ = torch.signbit(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_isfinite_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.isfinite should bypass candle._backends.npu.aclnn.isfinite."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.isfinite(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.isfinite

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "isfinite", wrapped)

    _ = torch.isfinite(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_isinf_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.isinf should bypass candle._backends.npu.aclnn.isinf."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.isinf(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.isinf

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "isinf", wrapped)

    _ = torch.isinf(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_isinf_skips_python_aclnn_logical_wrappers(npu_device, monkeypatch):
    """torch.isinf should bypass candle._backends.npu.aclnn logical wrappers on the composite path."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([0.0, float("inf"), float("nan"), -float("inf")], device=npu_device)
    torch.npu.synchronize()

    _ = torch.isinf(a)
    torch.npu.synchronize()

    calls = {"not": 0, "and": 0}
    original_not = aclnn_mod.logical_not
    original_and = aclnn_mod.logical_and

    def wrapped_not(*args, **kwargs):
        calls["not"] += 1
        return original_not(*args, **kwargs)

    def wrapped_and(*args, **kwargs):
        calls["and"] += 1
        return original_and(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "logical_not", wrapped_not)
    monkeypatch.setattr(aclnn_mod, "logical_and", wrapped_and)

    out = torch.isinf(a)
    torch.npu.synchronize()

    assert calls["not"] == 0
    assert calls["and"] == 0
    assert out.tolist() == [False, True, False, True]



def test_isposinf_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.isposinf should bypass candle._backends.npu.aclnn.isposinf."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.isposinf(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.isposinf

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "isposinf", wrapped)

    _ = torch.isposinf(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_isneginf_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.isneginf should bypass candle._backends.npu.aclnn.isneginf."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), -0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.isneginf(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.isneginf

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "isneginf", wrapped)

    _ = torch.isneginf(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_logical_not_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.logical_not should bypass candle._backends.npu.aclnn.logical_not."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), True, device=npu_device)
    torch.npu.synchronize()

    _ = torch.logical_not(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.logical_not

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "logical_not", wrapped)

    _ = torch.logical_not(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_bitwise_not_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.bitwise_not should bypass candle._backends.npu.aclnn.bitwise_not."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 1, device=npu_device, dtype=torch.int32)
    torch.npu.synchronize()

    _ = torch.bitwise_not(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.bitwise_not

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "bitwise_not", wrapped)

    _ = torch.bitwise_not(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_silu_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.silu should bypass candle._backends.npu.aclnn.silu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.silu(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.silu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "silu", wrapped)

    _ = torch.nn.functional.silu(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_mish_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.mish should bypass candle._backends.npu.aclnn.mish."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.mish(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.mish

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "mish", wrapped)

    _ = torch.nn.functional.mish(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_gelu_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.gelu should bypass candle._backends.npu.aclnn.gelu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.gelu(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.gelu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "gelu", wrapped)

    _ = torch.nn.functional.gelu(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_relu_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.relu should bypass candle._backends.npu.aclnn.relu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.relu(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.relu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "relu", wrapped)

    _ = torch.nn.functional.relu(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_leaky_relu_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.leaky_relu should bypass candle._backends.npu.aclnn.leaky_relu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.leaky_relu(a, negative_slope=0.1)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.leaky_relu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "leaky_relu", wrapped)

    _ = torch.nn.functional.leaky_relu(a, negative_slope=0.1)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_elu_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.elu should bypass candle._backends.npu.aclnn.elu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.elu(a, alpha=1.0)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.elu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "elu", wrapped)

    _ = torch.nn.functional.elu(a, alpha=1.0)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_softmax_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.softmax should bypass candle._backends.npu.aclnn.softmax."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.softmax(a, dim=-1)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.softmax

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "softmax", wrapped)

    _ = torch.nn.functional.softmax(a, dim=-1)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_log_softmax_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.log_softmax should bypass candle._backends.npu.aclnn.log_softmax."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.log_softmax(a, dim=-1)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.log_softmax

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "log_softmax", wrapped)

    _ = torch.nn.functional.log_softmax(a, dim=-1)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_hardtanh_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.hardtanh should bypass candle._backends.npu.aclnn.hardtanh."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.hardtanh(a, min_val=-0.5, max_val=0.5)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.hardtanh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "hardtanh", wrapped)

    _ = torch.nn.functional.hardtanh(a, min_val=-0.5, max_val=0.5)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_softplus_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.softplus should bypass candle._backends.npu.aclnn.softplus."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.softplus(a, beta=1.0, threshold=20.0)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.softplus

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "softplus", wrapped)

    _ = torch.nn.functional.softplus(a, beta=1.0, threshold=20.0)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_prelu_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.prelu should bypass candle._backends.npu.aclnn.prelu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((2, 3, 4, 4), 0.5, device=npu_device)
    weight = torch.full((3,), 0.25, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.prelu(a, weight)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.prelu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "prelu", wrapped)

    _ = torch.nn.functional.prelu(a, weight)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_relu__skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.relu_ should bypass candle._backends.npu.aclnn.relu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), -0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.relu_(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.relu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "relu", wrapped)

    _ = torch.nn.functional.relu_(a)
    torch.npu.synchronize()

    assert calls["count"] == 0

    expected = torch.zeros((4, 4), device=npu_device)
    assert torch.equal(a, expected)
    assert torch.equal(_, a)



def test_embedding_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.embedding should bypass candle._backends.npu.aclnn.embedding."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    weight = torch.arange(0, 24, device=npu_device, dtype=torch.float32).reshape(6, 4)
    indices = torch.tensor([0, 2, 5], device=npu_device, dtype=torch.int64)
    torch.npu.synchronize()

    _ = torch.nn.functional.embedding(indices, weight)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.embedding

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "embedding", wrapped)

    out = torch.nn.functional.embedding(indices, weight)
    torch.npu.synchronize()

    assert calls["count"] == 0
    assert out.shape == (3, 4)
    assert torch.equal(out[0], weight[0])
    assert torch.equal(out[1], weight[2])
    assert torch.equal(out[2], weight[5])



def test_dropout_skips_python_aclnn_wrappers(npu_device, monkeypatch):
    """torch.nn.functional.dropout should bypass candle._backends.npu.aclnn dropout wrappers."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((8, 8), 1.0, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.dropout(a, p=0.25, training=True)
    torch.npu.synchronize()

    calls = {"gen": 0, "do": 0}
    original_gen = aclnn_mod.dropout_gen_mask
    original_do = aclnn_mod.dropout_do_mask

    def wrapped_gen(*args, **kwargs):
        calls["gen"] += 1
        return original_gen(*args, **kwargs)

    def wrapped_do(*args, **kwargs):
        calls["do"] += 1
        return original_do(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "dropout_gen_mask", wrapped_gen)
    monkeypatch.setattr(aclnn_mod, "dropout_do_mask", wrapped_do)

    out = torch.nn.functional.dropout(a, p=0.25, training=True)
    torch.npu.synchronize()

    assert calls["gen"] == 0
    assert calls["do"] == 0
    assert out.shape == a.shape
    assert out.device == a.device
    assert out._backward_data["p"] == 0.25
    assert out._backward_data["mask_numel"] > 0
    assert out._backward_data["mask_ptr"]



def test_isnan_skips_python_aclnn_wrappers(npu_device, monkeypatch):
    """torch.isnan should bypass candle._backends.npu.aclnn logical wrappers."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([0.0, float("nan"), float("inf"), -1.0], device=npu_device)
    torch.npu.synchronize()

    _ = torch.isnan(a)
    torch.npu.synchronize()

    calls = {"not": 0, "and": 0}
    original_not = aclnn_mod.logical_not
    original_and = aclnn_mod.logical_and

    def wrapped_not(*args, **kwargs):
        calls["not"] += 1
        return original_not(*args, **kwargs)

    def wrapped_and(*args, **kwargs):
        calls["and"] += 1
        return original_and(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "logical_not", wrapped_not)
    monkeypatch.setattr(aclnn_mod, "logical_and", wrapped_and)

    out = torch.isnan(a)
    torch.npu.synchronize()

    assert calls["not"] == 0
    assert calls["and"] == 0
    assert out.tolist() == [False, True, False, False]



def test_reciprocal__skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """Tensor.reciprocal_ should bypass candle._backends.npu.aclnn.reciprocal."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 2.0, device=npu_device)
    torch.npu.synchronize()

    _ = a.reciprocal_()
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.reciprocal

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "reciprocal", wrapped)

    out = a.reciprocal_()
    torch.npu.synchronize()

    assert calls["count"] == 0
    expected = torch.full((4, 4), 2.0, device=npu_device)
    assert torch.allclose(a, expected)
    assert torch.equal(out, a)



def test_reciprocal_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.reciprocal should bypass candle._backends.npu.aclnn.reciprocal."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 2.0, device=npu_device)
    torch.npu.synchronize()

    _ = torch.reciprocal(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.reciprocal

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "reciprocal", wrapped)

    out = torch.reciprocal(a)
    torch.npu.synchronize()

    assert calls["count"] == 0
    expected = torch.full((4, 4), 0.5, device=npu_device)
    assert torch.allclose(out, expected)



def test_zero__skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """Tensor.zero_ should bypass candle._backends.npu.aclnn.inplace_zero."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 2.0, device=npu_device)
    torch.npu.synchronize()

    _ = a.zero_()
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.inplace_zero

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "inplace_zero", wrapped)

    out = a.zero_()
    torch.npu.synchronize()

    assert calls["count"] == 0
    expected = torch.zeros((4, 4), device=npu_device)
    assert torch.equal(a, expected)
    assert torch.equal(out, a)



def test_fill__skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """Tensor.fill_ should bypass candle._backends.npu.aclnn.inplace_fill_scalar."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.zeros((4, 4), device=npu_device)
    torch.npu.synchronize()

    _ = a.fill_(2.0)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.inplace_fill_scalar

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "inplace_fill_scalar", wrapped)

    out = a.fill_(3.0)
    torch.npu.synchronize()

    assert calls["count"] == 0
    expected = torch.full((4, 4), 3.0, device=npu_device)
    assert torch.equal(a, expected)
    assert torch.equal(out, a)




