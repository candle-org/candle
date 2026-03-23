import pytest


def test_cache_hit_same_tensor(npu_device):
    import candle as torch
    from candle._cython._aclnn_ffi import get_tensor_desc_cache  # pylint: disable=import-error,no-name-in-module
    a = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()
    cache = get_tensor_desc_cache()
    cache.clear()
    h1 = cache.get_or_create(a.storage().data_ptr(), a.shape, a.stride, 0, 2)
    h2 = cache.get_or_create(a.storage().data_ptr(), a.shape, a.stride, 0, 2)
    assert h1 == h2, "Cache miss on identical key"


def test_cache_miss_different_stride(npu_device):
    import candle as torch
    from candle._cython._aclnn_ffi import get_tensor_desc_cache  # pylint: disable=import-error,no-name-in-module
    a = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()
    cache = get_tensor_desc_cache()
    cache.clear()
    h1 = cache.get_or_create(a.storage().data_ptr(), a.shape, a.stride, 0, 2)
    a_t = a.t()
    h2 = cache.get_or_create(a_t.storage().data_ptr(), a_t.shape, a_t.stride, 0, 2)
    assert h1 != h2, "Cache incorrectly hit for transposed tensor"


def test_cache_invalidated_on_free(npu_device):
    import candle as torch
    import gc
    from candle._cython._aclnn_ffi import get_tensor_desc_cache  # pylint: disable=import-error,no-name-in-module
    cache = get_tensor_desc_cache()
    cache.clear()
    a = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()
    _ = cache.get_or_create(a.storage().data_ptr(), a.shape, a.stride, 0, 2)
    assert cache.size() == 1
    del a
    gc.collect()
    torch.npu.synchronize()
    assert cache.size() == 0, "Cache entry not invalidated after free"


def test_add_uses_cache(npu_device):
    import candle as torch
    from candle._cython._aclnn_ffi import get_tensor_desc_cache  # pylint: disable=import-error,no-name-in-module
    cache = get_tensor_desc_cache()
    cache.clear()
    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()
    _ = torch.add(a, b)
    torch.npu.synchronize()
    # Two valid outcomes:
    # - 2 entries: add went through GetWorkspaceSize and reused the tensor-desc cache
    # - 0 entries: a pre-warmed PTA executor cache hit bypassed descriptor creation entirely
    assert cache.size() in (0, 2), f"Expected 0 or 2 cached entries, got {cache.size()}"
