import candle as torch
import pytest


@pytest.fixture(autouse=True)
def _reset_allocator_state():
    import candle._backends.npu.allocator as npu_allocator

    saved = dict(npu_allocator._ALLOCATORS)
    try:
        from candle._C import _npu_ops, _npu_storage
    except ImportError:
        _npu_ops = None
        _npu_storage = None
    if _npu_ops is not None:
        _npu_ops.invalidate_allocator_cache_dev0()
    if _npu_storage is not None:
        _npu_storage.invalidate_allocator_cache_dev0()
    npu_allocator._ALLOCATORS.clear()
    yield
    npu_allocator._ALLOCATORS.clear()
    npu_allocator._ALLOCATORS.update(saved)
    if _npu_ops is not None:
        _npu_ops.invalidate_allocator_cache_dev0()
    if _npu_storage is not None:
        _npu_storage.invalidate_allocator_cache_dev0()



def test_allocator_stats_defaults():
    from candle._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    stats = alloc.memory_stats()
    assert stats["allocated_bytes.all.current"] == 0
    assert stats["reserved_bytes.all.current"] == 0
    assert stats["active_bytes.all.current"] == 0
    assert stats["allocated.all.current"] == 0


def test_get_allocator_singleton():
    from candle._backends.npu import allocator

    a1 = allocator.get_allocator(0)
    a2 = allocator.get_allocator(0)
    assert a1 is a2


def test_allocator_allocates_and_tracks(monkeypatch):
    from candle._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    monkeypatch.setattr(alloc, "_raw_malloc", lambda size: (1234, size))

    ptr = alloc.malloc(512)
    stats = alloc.memory_stats()
    assert ptr == 1234
    assert stats["allocated_bytes.all.current"] >= 512
    assert stats["allocated.all.current"] == 1
    assert stats["segment.all.current"] == 1


def test_allocator_free_uses_pending_events(monkeypatch):
    from candle._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    monkeypatch.setattr(alloc, "_raw_malloc", lambda size: (1234, size))
    fake_event = object()
    monkeypatch.setattr(alloc, "_record_event", lambda stream: fake_event)
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    ptr = alloc.malloc(512, stream="s0")
    alloc.free(ptr, stream="s0")
    alloc.synchronize()

    stats = alloc.memory_stats()
    assert stats["active_bytes.all.current"] == 0


def test_allocator_synchronize_reuses_completed_events(monkeypatch):
    from candle._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    monkeypatch.setattr(alloc, "_raw_malloc", lambda size: (1234, size))
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    event = object()
    record_calls = []

    def record_event(stream):
        record_calls.append(stream)
        return event

    monkeypatch.setattr(alloc, "_record_event", record_event)

    ptr = alloc.malloc(512, stream="s0")
    alloc.free(ptr, stream="s0")
    alloc.synchronize()

    assert alloc._event_pool == [event]

    next_event = alloc._event_pool.pop()
    assert next_event is event
    assert record_calls == ["s0"]


def test_empty_cache_releases_cached(monkeypatch):
    from candle._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    monkeypatch.setattr(alloc, "_raw_malloc", lambda size: (1234, size))
    freed = []
    monkeypatch.setattr(alloc, "_raw_free", lambda ptr: freed.append(ptr))
    monkeypatch.setattr(alloc, "_record_event", lambda stream: object())
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    ptr = alloc.malloc(512)
    alloc.free(ptr, stream=None)
    alloc.synchronize()
    alloc.empty_cache()

    stats = alloc.memory_stats()
    assert stats["reserved_bytes.all.current"] == 0
    assert freed


def test_allocator_reuses_cached_block(monkeypatch):
    from candle._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    calls = []

    def fake_raw_malloc(size):
        calls.append(size)
        return (1000 + len(calls), size)

    monkeypatch.setattr(alloc, "_raw_malloc", fake_raw_malloc)
    monkeypatch.setattr(alloc, "_record_event", lambda stream: object())
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    ptr = alloc.malloc(512, stream="s0")
    alloc.free(ptr, stream="s0")
    alloc.synchronize()
    calls.clear()

    ptr2 = alloc.malloc(512, stream="s1")
    stats = alloc.memory_stats()

    assert ptr2 == ptr
    assert calls == []
    assert stats["reserved_bytes.all.current"] == 512


def test_allocator_splits_cached_block(monkeypatch):
    from candle._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    calls = []

    def fake_raw_malloc(size):
        calls.append(size)
        return (2000 + len(calls), size)

    monkeypatch.setattr(alloc, "_raw_malloc", fake_raw_malloc)
    monkeypatch.setattr(alloc, "_record_event", lambda stream: object())
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    ptr = alloc.malloc(2048, stream="s0")
    alloc.free(ptr, stream="s0")
    alloc.synchronize()
    calls.clear()

    ptr2 = alloc.malloc(512, stream="s1")
    stats = alloc.memory_stats()

    assert calls == []
    assert ptr2 == ptr
    assert stats["inactive_split_bytes.all.current"] == 1536


def test_empty_cache_frees_only_complete_python_segments_after_split(monkeypatch):
    from candle._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    base_ptr = 2000
    freed = []

    monkeypatch.setattr(alloc, "_raw_malloc", lambda size: (base_ptr, size))

    def raw_free(ptr):
        if ptr != base_ptr:
            raise RuntimeError(f"attempted to free interior split pointer {ptr}")
        freed.append(ptr)

    monkeypatch.setattr(alloc, "_raw_free", raw_free)
    monkeypatch.setattr(alloc, "_record_event", lambda stream: object())
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    ptr = alloc.malloc(2048, stream="s0")
    alloc.free(ptr, stream="s0")
    alloc.synchronize()

    ptr2 = alloc.malloc(512, stream="s1")
    alloc.free(ptr2, stream="s1")
    alloc.synchronize()
    alloc.empty_cache()

    stats = alloc.memory_stats()
    assert freed == [base_ptr]
    assert stats["reserved_bytes.all.current"] == 0
    assert stats["segment.all.current"] == 0


def test_empty_cache_frees_only_complete_fast_segments_after_split(monkeypatch):
    from candle._C._allocator import FastNpuAllocator

    alloc = FastNpuAllocator(0)
    base_ptr = 4096
    freed = []

    monkeypatch.setattr(alloc, "_raw_malloc", lambda size: (base_ptr, size))

    def raw_free(ptr):
        if ptr != base_ptr:
            raise RuntimeError(f"attempted to free interior split pointer {ptr}")
        freed.append(ptr)

    monkeypatch.setattr(alloc, "_raw_free", raw_free)
    monkeypatch.setattr(alloc, "_record_event", lambda stream: object())
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    ptr = alloc.malloc(2048, stream="s0")
    alloc.free(ptr, stream="s0")
    alloc.synchronize()

    ptr2 = alloc.malloc(512, stream="s1")
    alloc.free(ptr2, stream="s1")
    alloc.synchronize()
    alloc.empty_cache()

    stats = alloc.memory_stats()
    assert freed == [base_ptr]
    assert stats["reserved_bytes.all.current"] == 0
    assert stats["segment.all.current"] == 0


def test_npu_memory_stats_api(monkeypatch):
    class DummyAlloc:
        def memory_stats(self):
            return {
                "allocated_bytes.all.current": 12,
                "allocated_bytes.all.peak": 34,
                "reserved_bytes.all.current": 56,
                "reserved_bytes.all.peak": 78,
            }

        def reset_peak_memory_stats(self):
            self.peak_reset = True

        def reset_accumulated_memory_stats(self):
            self.accum_reset = True

        def empty_cache(self):
            self.cache_emptied = True

    dummy = DummyAlloc()

    monkeypatch.setattr(torch.npu, "_get_allocator", lambda device=None: dummy)

    assert torch.npu.memory_allocated() == 12
    assert torch.npu.max_memory_allocated() == 34
    assert torch.npu.memory_reserved() == 56
    assert torch.npu.max_memory_reserved() == 78

    torch.npu.reset_peak_memory_stats()
    torch.npu.reset_accumulated_memory_stats()
    torch.npu.empty_cache()

    assert dummy.peak_reset is True
    assert dummy.accum_reset is True
    assert dummy.cache_emptied is True


def test_allocator_record_stream(monkeypatch):
    from candle._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    monkeypatch.setattr(alloc, "_raw_malloc", lambda size: (1234, size))

    ptr = alloc.malloc(512, stream="s0")
    alloc.record_stream(ptr, stream="s1")

    assert alloc._active[ptr].stream == "s1"


def test_npu_storage_free_updates_allocator(monkeypatch):
    import gc

    from candle._backends.npu import allocator
    from candle._C import npu_typed_storage_from_ptr

    alloc = allocator.get_allocator(0)
    monkeypatch.setattr(alloc, "_raw_malloc", lambda size: (1234, size))
    monkeypatch.setattr(alloc, "_record_event", lambda stream: object())
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    ptr = alloc.malloc(512)
    storage = npu_typed_storage_from_ptr(ptr, 128, dtype=None)
    del storage
    gc.collect()

    alloc.synchronize()
    assert alloc.memory_stats()["active_bytes.all.current"] == 0


def test_alloc_conf_precedence(monkeypatch):
    from candle._backends.npu import allocator

    allocator._reset_alloc_conf_for_test()
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:4")
    monkeypatch.setenv("CANDLE_NPU_ALLOC_CONF", "max_split_size_mb:8")

    conf = allocator._load_alloc_conf(force=True)
    assert conf["max_split_size_mb"] == 8


def test_alloc_conf_unsupported_key_warns(monkeypatch):
    from candle._backends.npu import allocator

    allocator._reset_alloc_conf_for_test()
    monkeypatch.setenv("CANDLE_NPU_ALLOC_CONF", "unknown_key:1")

    with pytest.warns(UserWarning):
        conf = allocator._load_alloc_conf(force=True)
    assert conf == {}


def test_alloc_conf_max_split_size_mb(monkeypatch):
    from candle._backends.npu import allocator

    allocator._reset_alloc_conf_for_test()
    monkeypatch.setenv("CANDLE_NPU_ALLOC_CONF", "max_split_size_mb:1")
    allocator._load_alloc_conf(force=True)

    alloc = allocator.NpuAllocator(device_id=0)
    assert alloc.max_split_size == 1 * 1024 * 1024


def test_alloc_conf_gc_threshold_triggers(monkeypatch):
    from candle._backends.npu import allocator

    allocator._reset_alloc_conf_for_test()
    monkeypatch.setenv("CANDLE_NPU_ALLOC_CONF", "garbage_collection_threshold:0.5")
    allocator._load_alloc_conf(force=True)

    alloc = allocator.NpuAllocator(device_id=0)
    alloc._stats["reserved_bytes.all.current"] = 80
    monkeypatch.setattr(alloc, "_mem_get_info", lambda: (20, 100))

    freed = []
    monkeypatch.setattr(alloc, "_raw_free", lambda ptr: freed.append(ptr))

    block = allocator.Block(1234, 16, 16, "small_pool", None)
    alloc._cached["small_pool"].append(block)

    alloc._maybe_collect_garbage()
    assert freed


def test_oom_retry_increments_stats(monkeypatch):
    from candle._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    calls = []

    def raw_malloc(size):
        calls.append(size)
        if len(calls) == 1:
            raise RuntimeError("acl.rt.malloc failed: 100")
        return (999, size)

    monkeypatch.setattr(alloc, "_raw_malloc", raw_malloc)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)
    monkeypatch.setattr(alloc, "_record_event", lambda stream: object())
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)

    ptr = alloc.malloc(512)
    assert ptr == 999
    assert alloc._stats["num_ooms"] == 1
    assert alloc._stats["num_alloc_retries"] == 1


def test_fast_allocator_empty_cache_resolves_desc_cache_once(monkeypatch):
    """empty_cache() must not re-resolve the desc-cache singleton per block.

    TensorDescCache is a module-level singleton that never rebinds. Cached blocks
    keep descriptors valid while memory stays reserved, so descriptor eviction is
    deferred until empty_cache() actually releases memory to ACL. The allocator
    must resolve the descriptor cache at most once and clear it once for the
    empty_cache pass.
    """
    from candle._C import _aclnn_ffi
    from candle._C._allocator import FastNpuAllocator

    real_cache = _aclnn_ffi.get_tensor_desc_cache()
    resolved = {"count": 0}
    cleared = {"count": 0}

    class _CountingCache:
        def clear(self):
            cleared["count"] += 1
            real_cache.clear()

    def _counting_get():
        resolved["count"] += 1
        return _CountingCache()

    # Patch BEFORE constructing the allocator so an init-time or first-call
    # resolution is observed either way.
    monkeypatch.setattr(_aclnn_ffi, "get_tensor_desc_cache", _counting_get)

    alloc = FastNpuAllocator(0)
    monkeypatch.setattr(alloc, "_raw_malloc", lambda size: (4096, size))
    monkeypatch.setattr(alloc, "_raw_free", lambda ptr: None)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    for _ in range(5):
        ptr = alloc.malloc(512)
        alloc.free(ptr)

    alloc.empty_cache()

    assert cleared["count"] == 1
    # Performance contract: the singleton was resolved at most once.
    assert resolved["count"] <= 1, (
        f"desc cache resolved {resolved['count']} times across empty_cache; "
        "expected <= 1 (cached handle)"
    )


def test_npu_temporary_matmul_storage_returns_to_allocator():
    """Dropping an NPU temporary must release its active allocator block.

    Real eager model blocks create matmul outputs that are immediately consumed
    by view/reshape and elementwise ops. If those temporaries keep their storage
    active after the Python Tensor is dropped, reserved NPU memory grows every
    iteration instead of reusing cached blocks, causing latency spikes and OOM.
    """
    import gc

    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    from candle._backends.npu import allocator

    alloc = allocator.get_allocator(0)
    x = torch.randn((1, 4, 8), device="npu", dtype=torch.float16)
    w = torch.randn((8, 16), device="npu", dtype=torch.float16)
    torch.npu.synchronize()
    gc.collect()

    before_active = len(alloc._active)
    before_allocs = alloc.memory_stats()["num_device_alloc"]

    y = torch.matmul(x, w)
    torch.npu.synchronize()
    ptr = y.data_ptr()
    assert ptr in alloc._active
    assert not y._is_view()

    del y
    gc.collect()
    torch.npu.synchronize()
    gc.collect()

    assert len(alloc._active) == before_active

    z = torch.matmul(x, w)
    torch.npu.synchronize()
    try:
        after_allocs = alloc.memory_stats()["num_device_alloc"]
        assert z.data_ptr() == ptr
        assert after_allocs == before_allocs + 1
    finally:
        del z
        gc.collect()
