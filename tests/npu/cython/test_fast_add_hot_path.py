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
