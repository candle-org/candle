import threading

import pytest

import candle as torch


@pytest.fixture(autouse=True)
def _reset_npu_state():
    import candle._backends.npu.state as npu_state

    state = npu_state._state()
    state.current_device = 0
    state.current_streams = {}
    state.default_streams = {}
    yield
    state.current_device = 0
    state.current_streams = {}
    state.default_streams = {}


def _stub_runtime(monkeypatch):
    import candle._backends.npu.runtime as npu_runtime
    import candle._backends.npu.state as npu_state

    class FakeRuntime:
        def __init__(self, device_id):
            self.device_id = device_id
            self.wait_calls = []
            self.record_calls = []

        def create_stream(self, priority=0):
            return int(self.device_id * 1000 + int(priority))

        def synchronize_stream(self, stream):
            return None

        def create_event(self, enable_timing, blocking, interprocess):
            return (bool(enable_timing), bool(blocking), bool(interprocess))

        def record_event(self, event, stream):
            self.record_calls.append((event, stream))
            return None

        def synchronize_event(self, event):
            return None

        def query_event(self, event):
            return True

        def event_elapsed_time(self, start, end):
            return 0.0

        def stream_wait_event(self, stream, event):
            self.wait_calls.append((stream, event))
            return None

    runtime = FakeRuntime(0)

    def fake_get_runtime(device_id=0):
        runtime.device_id = int(device_id)
        return runtime

    monkeypatch.setattr(npu_runtime, "get_runtime", fake_get_runtime)
    monkeypatch.setattr(npu_state, "get_runtime", fake_get_runtime)
    monkeypatch.setattr(npu_runtime, "_RUNTIMES", {})
    return runtime


def test_npu_device_guard_restores_device(monkeypatch):
    _stub_runtime(monkeypatch)
    torch.npu.set_device(0)
    assert torch.npu.current_device() == 0
    with torch.npu.device(1):
        assert torch.npu.current_device() == 1
    assert torch.npu.current_device() == 0


def test_npu_current_device_thread_local(monkeypatch):
    _stub_runtime(monkeypatch)
    torch.npu.set_device(0)
    assert torch.npu.current_device() == 0
    result = {}

    def worker():
        torch.npu.set_device(1)
        result["dev"] = torch.npu.current_device()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["dev"] == 1
    assert torch.npu.current_device() == 0


def test_npu_default_stream_is_device_global(monkeypatch):
    runtime = _stub_runtime(monkeypatch)
    runtime._stream_id = 100

    def _create_stream(priority=0):
        runtime._stream_id += 1
        return runtime._stream_id

    runtime.create_stream = _create_stream

    s0 = torch.npu.default_stream()
    got = {}

    def worker():
        got["s1"] = torch.npu.default_stream()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert s0.stream == got["s1"].stream
    assert runtime._stream_id == 101


def test_npu_stream_context_switches_device_and_stream(monkeypatch):
    _stub_runtime(monkeypatch)
    s = torch.npu.Stream()
    cur = torch.npu.current_stream()
    with torch.npu.stream(s):
        assert torch.npu.current_stream() is s
        assert torch.npu.current_device() == s.device.index
    assert torch.npu.current_stream() is cur


def test_npu_event_record_query(monkeypatch):
    _stub_runtime(monkeypatch)
    evt = torch.npu.Event(enable_timing=True)
    s = torch.npu.current_stream()
    evt.record(s)
    assert evt.query() in (True, False)


def test_npu_stream_record_event(monkeypatch):
    runtime = _stub_runtime(monkeypatch)
    s = torch.npu.Stream()
    evt = s.record_event()
    assert runtime.record_calls == [(evt.event, s.stream)]


def test_npu_event_wait_stream(monkeypatch):
    runtime = _stub_runtime(monkeypatch)
    s = torch.npu.Stream()
    evt = torch.npu.Event()
    evt.wait(s)
    assert runtime.wait_calls == [(s.stream, evt.event)]


def test_npu_runtime_primitives_exist():
    import candle._backends.npu.runtime as npu_runtime

    runtime = npu_runtime._Runtime()
    assert hasattr(runtime, "create_stream")
    assert hasattr(runtime, "synchronize_stream")
    assert hasattr(runtime, "create_event")
    assert hasattr(runtime, "record_event")
    assert hasattr(runtime, "query_event")
    assert hasattr(runtime, "event_elapsed_time")
    assert hasattr(runtime, "stream_wait_event")
    assert hasattr(runtime, "synchronize_device")


def test_npu_op_uses_current_stream(monkeypatch):
    _stub_runtime(monkeypatch)
    import candle._backends.npu.aclnn as aclnn
    import candle._backends.npu.ops as npu_ops
    from candle import float32

    seen = {}

    def fake_add(*args, **kwargs):
        seen["stream"] = kwargs.get("stream")
        return None

    monkeypatch.setattr(aclnn, "add", fake_add)

    class DummyStorage:
        def __init__(self, ptr, device):
            self._ptr = ptr
            self.device = device

        def data_ptr(self):
            return self._ptr

    class DummyTensor:
        def __init__(self, device):
            self.device = device
            self.shape = (1,)
            self.stride = (1,)
            self.dtype = float32
            self._storage = DummyStorage(123, device)

        def storage(self):
            return self._storage

    def fake_alloc(size, runtime=None):
        return 456

    def fake_wrap(storage, shape, stride):
        return None

    def fake_storage_from_ptr(ptr, size, dtype, device=None):
        return DummyStorage(ptr, device)

    monkeypatch.setattr(npu_ops.npu_runtime, "_alloc_device", fake_alloc)
    monkeypatch.setattr(npu_ops, "_wrap_tensor", fake_wrap)
    monkeypatch.setattr(npu_ops, "npu_typed_storage_from_ptr", fake_storage_from_ptr)

    s = torch.npu.Stream()
    with torch.npu.stream(s):
        a = DummyTensor(s.device)
        b = DummyTensor(s.device)
        npu_ops.add(a, b)

    assert seen["stream"] == s.stream


def test_npu_stream_wait_event(monkeypatch):
    runtime = _stub_runtime(monkeypatch)
    s = torch.npu.Stream()
    evt = torch.npu.Event()
    s.wait_event(evt)
    assert runtime.wait_calls == [(s.stream, evt.event)]


def test_npu_set_stream_changes_current(monkeypatch):
    _stub_runtime(monkeypatch)
    s = torch.npu.Stream()
    torch.npu.set_stream(s)
    assert torch.npu.current_stream() is s


def test_npu_stream_wait_stream(monkeypatch):
    runtime = _stub_runtime(monkeypatch)
    s0 = torch.npu.Stream()
    s1 = torch.npu.Stream()
    s0.wait_stream(s1)
    assert runtime.wait_calls == [(s0.stream, s0._event.event)]


def test_runtime_memcpy_helpers_use_current_stream(monkeypatch):
    import candle._backends.npu.runtime as npu_runtime
    import candle._backends.npu.state as npu_state

    calls = []

    class FakeRt:
        def memcpy_async(self, dst, dst_size, src, src_size, kind, stream):
            calls.append((kind, stream))
            return 0

        def memcpy(self, dst, dst_size, src, src_size, kind):
            calls.append((kind, None))
            return 0

    class FakeAcl:
        def __init__(self):
            self.rt = FakeRt()

    class DummyRuntime:
        device_id = 0

        def activate(self):
            return None

    class DummyStream:
        stream = 123

    monkeypatch.setattr(npu_runtime, "acl", FakeAcl())
    monkeypatch.setattr(npu_runtime, "ensure_acl", lambda: npu_runtime.acl)
    monkeypatch.setattr(npu_state, "current_stream", lambda device_id=0: DummyStream())

    npu_runtime.memcpy_h2d(1, 4, 2, runtime=DummyRuntime(), stream=None, non_blocking=True)
    npu_runtime.memcpy_d2h(1, 4, 2, runtime=DummyRuntime(), stream=None, non_blocking=True)
    npu_runtime.memcpy_d2d(1, 4, 2, runtime=DummyRuntime(), stream=None, non_blocking=True)

    assert [entry[1] for entry in calls] == [123, 123, 123]


def test_copy_cpu_to_npu_uses_current_stream_when_none(monkeypatch):
    _stub_runtime(monkeypatch)
    import numpy as np
    import candle._backends.npu.runtime as npu_runtime

    calls = {}

    class DummyRuntime:
        stream = 999
        device_id = 0

        def activate(self):
            return None

        def defer_host_free(self, ptr):
            return None

    import candle._backends.npu.state as npu_state

    class DummyStream:
        stream = 111

    monkeypatch.setattr(npu_state, "current_stream", lambda device_id=0: DummyStream())
    monkeypatch.setattr(
        npu_runtime,
        "_memcpy_h2d",
        lambda dst, size, src_ptr, runtime=None: calls.setdefault("stream", "sync"),
    )

    class FakeAcl:
        class Rt:
            def malloc_host(self, size):
                return (1234, 0)

            def free_host(self, ptr):
                return 0

            def memcpy_async(self, dst, dst_size, src, src_size, kind, stream):
                calls["stream"] = stream
                return 0

        def __init__(self):
            self.rt = self.Rt()

    fake_acl = FakeAcl()
    monkeypatch.setattr(npu_runtime, "acl", fake_acl)
    monkeypatch.setattr(npu_runtime, "ensure_acl", lambda: fake_acl)
    monkeypatch.setattr(npu_runtime, "_alloc_device", lambda size, runtime=None: 5678)
    monkeypatch.setattr(npu_runtime.ctypes, "memmove", lambda *args, **kwargs: None)

    arr = np.zeros((2,), dtype=np.float32)
    npu_runtime._copy_cpu_to_npu(arr, runtime=DummyRuntime(), non_blocking=True, stream=None)

    assert calls.get("stream") == 111


def test_copy_npu_to_cpu_uses_current_stream_when_none(monkeypatch):
    _stub_runtime(monkeypatch)
    import numpy as np
    import candle._backends.npu.runtime as npu_runtime

    calls = {}

    class DummyRuntime:
        stream = 999
        device_id = 0

        def activate(self):
            return None

        def synchronize_stream(self, stream):
            calls["sync_stream"] = stream

    class FakeAcl:
        class Rt:
            def memcpy_async(self, dst, dst_size, src, src_size, kind, stream):
                calls["stream"] = stream
                return 0

            def memcpy(self, dst, dst_size, src, src_size, kind):
                calls["stream"] = "sync"
                return 0

            def malloc_host(self, size):
                return (1234, 0)

            def free_host(self, ptr):
                return 0

        def __init__(self):
            self.rt = self.Rt()

    fake_acl = FakeAcl()
    monkeypatch.setattr(npu_runtime, "acl", fake_acl)
    monkeypatch.setattr(npu_runtime, "ensure_acl", lambda: fake_acl)
    monkeypatch.setattr(
        npu_runtime,
        "_numpy_from_ptr",
        lambda ptr, shape, dtype: np.zeros(shape, dtype=np.float32),
    )

    import candle._backends.npu.state as npu_state

    class DummyStream:
        stream = 111

    monkeypatch.setattr(npu_state, "current_stream", lambda device_id=0: DummyStream())

    arr = npu_runtime._copy_npu_to_cpu(
        123,
        8,
        (2,),
        "float32",
        runtime=DummyRuntime(),
        non_blocking=True,
        stream=None,
    )

    assert calls.get("stream") == 111
    assert calls.get("sync_stream") == 111
    assert arr.shape == (2,)
