import pytest


@pytest.fixture
def npu_device():
    import candle as torch
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    return torch.device("npu:0")


def test_storage_dealloc_returns_to_pool(npu_device):
    """FastNPUStorage.__dealloc__ must be safe and memory reclaimed after sync."""
    import candle as torch
    from candle._backends.npu import allocator as am
    alloc = am.get_allocator(0)
    torch.npu.synchronize()

    a = torch.randn((64, 64), dtype=torch.float32, device=npu_device)
    active_mid = alloc.memory_stats()["active_bytes.all.current"]
    assert active_mid > 0

    del a
    # We don't assert immediate allocator state here because Cython __dealloc__ timing
    # and runtime deferred-free bookkeeping are implementation details. The contract
    # we care about is: after synchronize(), the memory is reclaimed safely.
    torch.npu.synchronize()
    active_after = alloc.memory_stats()["active_bytes.all.current"]
    assert active_after < active_mid


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




def test_fast_synchronize_matches_runtime_for_string_device(monkeypatch):
    """npu.synchronize('npu:0') must preserve full runtime semantics."""
    import candle.npu as npu

    seen = {"runtime": 0, "fast": 0}

    class FakeRuntime:
        def synchronize(self):
            seen["runtime"] += 1

    monkeypatch.setattr(npu, "_cy_npu_sync", lambda dev_idx: seen.__setitem__("fast", seen["fast"] + 1))
    monkeypatch.setattr(npu, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr("candle._backends.npu.runtime.get_runtime", lambda idx: FakeRuntime())

    npu.synchronize("npu:0")
    assert seen["runtime"] == 1
    assert seen["fast"] == 0


def test_fast_synchronize_skips_fast_path_during_capture(monkeypatch):
    """Fast path must be disabled while the current stream is under capture."""
    import candle.npu as npu
    from candle._backends.npu import runtime as npu_runtime

    seen = {"runtime": 0, "fast": 0}

    class FakeRuntime:
        def synchronize(self):
            seen["runtime"] += 1

    monkeypatch.setattr(npu, "_NPU_INITIALIZED", True)
    monkeypatch.setattr(npu_runtime.cann_discovery, "get_cann_version", lambda: (8, 5, 0))
    monkeypatch.setattr("candle._backends.npu.ops_soc.aclgraph_supported", lambda profile=None: True)
    monkeypatch.setattr(npu, "_cy_npu_sync", lambda dev_idx: seen.__setitem__("fast", seen["fast"] + 1))
    monkeypatch.setattr(npu, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(npu_runtime, "get_runtime", lambda idx: FakeRuntime())

    npu.synchronize(None)
    assert seen["runtime"] == 1
    assert seen["fast"] == 0


def test_fast_synchronize_skips_aclgraph_probe_on_unsupported_soc(monkeypatch):
    """Fast synchronize should not query aclgraph capture state on unsupported SoCs."""
    import candle.npu as npu
    from candle._backends.npu import runtime as npu_runtime

    seen = {"fast": 0}

    monkeypatch.setattr(npu, "_NPU_INITIALIZED", True)
    monkeypatch.setattr(npu_runtime.cann_discovery, "get_cann_version", lambda: (8, 5, 0))
    monkeypatch.setattr("candle._backends.npu.ops_soc.aclgraph_supported", lambda profile=None: False)
    monkeypatch.setattr(npu, "_cy_npu_sync", lambda dev_idx: seen.__setitem__("fast", seen["fast"] + 1))

    def fail_probe():
        raise AssertionError("should not query aclgraph capture state on unsupported SoC")

    monkeypatch.setattr(npu, "is_current_stream_capturing", fail_probe)

    npu.synchronize()

    assert seen["fast"] == 1


def test_fast_synchronize_skips_aclgraph_probe_on_old_cann(monkeypatch):
    """Fast synchronize should not query aclgraph capture state on old CANN."""
    import candle.npu as npu
    from candle._backends.npu import runtime as npu_runtime

    seen = {"fast": 0}

    monkeypatch.setattr(npu_runtime.cann_discovery, "get_cann_version", lambda: (8, 3, 2))
    monkeypatch.setattr(npu, "_cy_npu_sync", lambda dev_idx: seen.__setitem__("fast", seen["fast"] + 1))

    def fail_probe():
        raise AssertionError("should not query aclgraph capture state on old CANN")

    monkeypatch.setattr(npu, "is_current_stream_capturing", fail_probe)

    npu.synchronize()

    assert seen["fast"] == 1


def test_fast_storage_clone_is_independent(npu_device):
    """FastTypedStorage.clone() should allocate independent device memory."""
    import candle as torch
    a = torch.randn((4, 4), dtype=torch.float32, device=npu_device)
    ts = a._storage
    cloned = ts.clone()
    assert cloned.data_ptr() != ts.data_ptr()
    assert cloned.size() == ts.size()
    assert cloned.dtype == ts.dtype


def test_fast_storage_resize_updates_size(npu_device):
    """FastTypedStorage.resize_ should update logical size and keep storage valid."""
    import candle as torch
    a = torch.randn((4, 4), dtype=torch.float32, device=npu_device)
    ts = a._storage
    ts.resize_(32)
    assert ts.size() == 32
    assert ts.data_ptr() != 0
    assert ts.nbytes() == 32 * 4
