import pytest


@pytest.fixture
def npu_device():
    import candle as torch
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    return torch.device("npu:0")


def test_storage_dealloc_returns_to_pool(npu_device):
    """FastNPUStorage.__dealloc__ returns device memory to pool synchronously (no gc.collect needed)."""
    import candle as torch
    from candle._backends.npu import allocator as am
    alloc = am.get_allocator(0)
    torch.npu.synchronize()

    a = torch.randn((64, 64), dtype=torch.float32, device=npu_device)
    active_mid = alloc.memory_stats()["active_bytes.all.current"]
    assert active_mid > 0

    # Drop the only reference — CPython refcount should call __dealloc__ synchronously
    del a
    # No gc.collect() needed — if dealloc is truly synchronous, stats drop immediately
    active_after = alloc.memory_stats()["active_bytes.all.current"]
    assert active_after < active_mid, (
        "Expected active bytes to drop after del — "
        "__dealloc__ may not be called synchronously"
    )


def test_storage_data_ptr_nonzero(npu_device):
    import candle as torch
    a = torch.randn((4, 4), dtype=torch.float32, device=npu_device)
    s = a.storage()
    assert s.data_ptr() != 0
    assert s.nbytes() == 4 * 4 * 4
    assert s.device.type == "npu"
    assert not s.is_pinned()


def test_typed_storage_interface(npu_device):
    import candle as torch
    a = torch.randn((8, 8), dtype=torch.float32, device=npu_device)
    ts = a._storage
    assert ts.data_ptr() != 0
    assert ts.size() == 64
    assert ts.device.type == "npu"
    assert ts.dtype is not None


def test_fast_synchronize_is_callable(npu_device):
    """cy_npu_synchronize must exist and complete without error."""
    from candle._cython._npu_ops import cy_npu_synchronize
    import candle as torch
    torch.add(
        torch.randn((64, 64), dtype=torch.float32, device=npu_device),
        torch.randn((64, 64), dtype=torch.float32, device=npu_device),
    )
    cy_npu_synchronize(0)  # must not raise
