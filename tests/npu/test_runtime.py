import atexit
import types
import warnings

import candle._backends.npu.runtime as ascend
import candle as torch


def test_runtime_init_registers_cleanup_once(monkeypatch):
    calls = []

    class DummyRT:
        def __init__(self, log):
            self.log = log

        def set_device(self, device_id):
            self.log.append(("set_device", device_id))
            return 0

        def create_context(self, device_id):
            self.log.append(("create_context", device_id))
            return "ctx", 0

        def create_stream(self):
            self.log.append(("create_stream",))
            return "stream", 0

        def destroy_stream(self, stream):
            self.log.append(("destroy_stream", stream))
            return 0

        def destroy_context(self, ctx):
            self.log.append(("destroy_context", ctx))
            return 0

        def reset_device(self, device_id):
            self.log.append(("reset_device", device_id))
            return 0

    def init():
        calls.append("init")
        return 0

    def finalize():
        calls.append("finalize")
        return 0

    dummy_acl = types.SimpleNamespace(init=init, finalize=finalize, rt=DummyRT(calls))

    monkeypatch.setattr(ascend, "acl", dummy_acl)
    runtime = ascend._Runtime()
    monkeypatch.setattr(ascend, "_runtime", runtime)
    monkeypatch.setattr(ascend, "_RUNTIME_CLEANUP_REGISTERED", False, raising=False)

    registered = []

    def register(func):
        registered.append(func)

    monkeypatch.setattr(atexit, "register", register)

    runtime.init(0)
    runtime.init(0)

    assert len(registered) == 1
    assert registered[0].__self__ is runtime
    assert calls.count("init") == 1


def test_runtime_synchronize_drains_deferred(monkeypatch):
    calls = []

    class DummyRT:
        def set_device(self, device_id):
            calls.append(("set_device", device_id))
            return 0

        def set_context(self, ctx):
            calls.append(("set_context", ctx))
            return 0

    class DummyAlloc:
        def synchronize(self):
            calls.append("alloc_sync")

        def free(self, ptr):
            calls.append(("alloc_free", ptr))

    dummy_acl = types.SimpleNamespace(rt=DummyRT())
    runtime = ascend._Runtime()
    runtime.initialized = True
    runtime.stream = "stream"
    runtime.device_id = 0
    runtime.context = "ctx"

    monkeypatch.setattr(ascend, "acl", dummy_acl)

    from candle._backends.npu import allocator as npu_allocator

    dummy_alloc = DummyAlloc()
    monkeypatch.setattr(npu_allocator, "get_allocator", lambda device_id=0: dummy_alloc)

    runtime.defer_free(111)
    runtime.defer_free(222)
    runtime.synchronize()

    assert "alloc_sync" in calls
    assert ("alloc_free", 111) in calls
    assert ("alloc_free", 222) in calls


def test_acl_launch_blocking_forces_sync(monkeypatch):
    calls = []

    class DummyRT:
        def set_device(self, device_id):
            return 0

        def set_context(self, ctx):
            return 0

    class DummyAlloc:
        def synchronize(self):
            calls.append("alloc_sync")

        def free(self, ptr):
            return None

    class DummyAcl:
        def __init__(self):
            self.rt = DummyRT()

    dummy_acl = DummyAcl()
    runtime = ascend._Runtime()
    runtime.initialized = True
    runtime.stream = "stream"
    runtime.device_id = 0
    runtime.context = "ctx"
    runtime._deferred_frees = []

    monkeypatch.setenv("ACL_LAUNCH_BLOCKING", "1")
    monkeypatch.setattr(ascend, "acl", dummy_acl)
    monkeypatch.setattr(ascend, "get_runtime", lambda device_id=0: runtime)

    from candle._backends.npu import allocator as npu_allocator

    dummy_alloc = DummyAlloc()
    monkeypatch.setattr(npu_allocator, "get_allocator", lambda device_id=0: dummy_alloc)

    from candle._backends.npu import aclnn
    # call internal helper to trigger sync in op wrapper
    aclnn._maybe_sync(runtime)

    assert calls == ["alloc_sync"]


def test_npu_synchronize_uses_runtime(monkeypatch):
    calls = []

    class DummyRuntime:
        def synchronize(self):
            calls.append("sync")

    dummy_runtime = DummyRuntime()

    from candle._backends.npu import runtime as npu_runtime
    monkeypatch.setattr(npu_runtime, "get_runtime", lambda device_id=0: dummy_runtime)

    import candle.npu as npu
    npu.synchronize("npu:0")

    assert calls == ["sync"]


def test_npu_synchronize_prefers_full_runtime_sync_over_device_only(monkeypatch):
    calls = []

    class DummyRuntime:
        def synchronize_device(self):
            calls.append("sync_device")

        def synchronize(self):
            calls.append("sync")

    from candle._backends.npu import runtime as npu_runtime

    monkeypatch.setattr(npu_runtime, "get_runtime", lambda device_id=0: DummyRuntime())
    monkeypatch.setattr(npu_runtime.cann_discovery, "get_cann_version", lambda: (8, 5, 0))

    import candle.npu as npu

    monkeypatch.setattr(npu, "_NPU_INITIALIZED", True)

    def fail_probe():
        raise AssertionError("string device synchronize should not query current capture state")

    monkeypatch.setattr(npu, "is_current_stream_capturing", fail_probe)

    npu.synchronize("npu:0")

    assert calls == ["sync"]


def test_npu_synchronize_string_device_skips_aclgraph_probe_on_old_cann(monkeypatch):
    calls = []

    class DummyRuntime:
        def synchronize(self):
            calls.append("sync")

    from candle._backends.npu import runtime as npu_runtime

    monkeypatch.setattr(npu_runtime.cann_discovery, "get_cann_version", lambda: (8, 3, 2))
    monkeypatch.setattr(npu_runtime, "get_runtime", lambda device_id=0: DummyRuntime())

    import candle.npu as npu

    def fail_probe():
        raise AssertionError("should not query aclgraph capture state on old CANN")

    monkeypatch.setattr(npu, "is_current_stream_capturing", fail_probe)

    npu.synchronize("npu:0")

    assert calls == ["sync"]


def test_npu_mem_get_info(monkeypatch):
    from candle._backends.npu import runtime as npu_runtime

    class DummyRT:
        def set_device(self, device_id):
            return 0

        def set_context(self, ctx):
            return 0

        def get_mem_info(self, attr):
            return 10, 20, 0

    dummy_acl = types.SimpleNamespace(rt=DummyRT())
    monkeypatch.setattr(npu_runtime, "acl", dummy_acl)
    monkeypatch.setattr(
        npu_runtime,
        "get_runtime",
        lambda device_id=0: types.SimpleNamespace(activate=lambda: None),
    )

    import candle.npu as npu
    free, total = npu.mem_get_info("npu:0")
    assert (free, total) == (10, 20)



def test_npu_is_available_verbose_reports_acl_missing(monkeypatch):
    def fake_get_runtime(device_id=0):
        raise ModuleNotFoundError("No module named acl")

    monkeypatch.setattr(ascend, "get_runtime", fake_get_runtime)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = torch.npu.is_available(verbose=True)

    assert ok is False
    assert any("acl" in str(w.message) for w in caught)
