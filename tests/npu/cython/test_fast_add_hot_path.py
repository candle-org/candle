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
    """fast_add should miss once, then hit the PTA executor cache for the same signature."""
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
    assert calls["count"] == 1


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
