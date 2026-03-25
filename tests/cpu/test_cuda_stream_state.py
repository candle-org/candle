import threading

import pytest

import candle as torch


@pytest.fixture(autouse=True)
def _reset_cuda_state():
    import candle._backends.cuda.state as cuda_state

    state = cuda_state._state()
    state.current_device = 0
    state.current_streams = {}
    cuda_state._default_streams.clear()
    state.default_streams = cuda_state._default_streams
    yield
    state.current_device = 0
    state.current_streams = {}
    cuda_state._default_streams.clear()
    state.default_streams = cuda_state._default_streams



def _stub_runtime(monkeypatch):
    import candle._backends.cuda.runtime as cuda_runtime
    import candle._backends.cuda.state as cuda_state

    class FakeRuntime:
        def __init__(self):
            self.device_id = 0
            self.stream_id = 100
            self.wait_calls = []
            self.record_calls = []

        def create_stream(self):
            self.stream_id += 1
            return self.stream_id

        def destroy_stream(self, stream):
            return None

        def synchronize_stream(self, stream):
            return None

        def create_event(self):
            return object()

        def record_event(self, event, stream):
            self.record_calls.append((event, stream))
            return None

        def synchronize_event(self, event):
            return None

        def destroy_event(self, event):
            return None

        def current_device(self):
            return self.device_id

        def set_device(self, index):
            self.device_id = int(index)

    runtime = FakeRuntime()

    monkeypatch.setattr(cuda_runtime, "create_stream", runtime.create_stream)
    monkeypatch.setattr(cuda_runtime, "destroy_stream", runtime.destroy_stream)
    monkeypatch.setattr(cuda_runtime, "synchronize_stream", runtime.synchronize_stream)
    monkeypatch.setattr(cuda_runtime, "create_event", runtime.create_event)
    monkeypatch.setattr(cuda_runtime, "record_event", runtime.record_event)
    monkeypatch.setattr(cuda_runtime, "synchronize_event", runtime.synchronize_event)
    monkeypatch.setattr(cuda_runtime, "destroy_event", runtime.destroy_event)
    monkeypatch.setattr(cuda_runtime, "current_device", runtime.current_device)
    monkeypatch.setattr(cuda_runtime, "set_device", runtime.set_device)
    monkeypatch.setattr(cuda_runtime, "is_available", lambda: True)
    monkeypatch.setattr(cuda_runtime, "device_count", lambda: 2)
    monkeypatch.setattr(cuda_state, "cuda_runtime", cuda_runtime)
    return runtime



def test_cuda_default_stream_is_device_global(monkeypatch):
    runtime = _stub_runtime(monkeypatch)
    s0 = torch.cuda.default_stream()
    got = {}

    def worker():
        got["s1"] = torch.cuda.default_stream()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert s0.stream == got["s1"].stream
    assert runtime.stream_id == 101



def test_cuda_current_stream_thread_local(monkeypatch):
    _stub_runtime(monkeypatch)
    base = torch.cuda.current_stream()
    result = {}

    def worker():
        s = torch.cuda.Stream(device=1)
        torch.cuda.set_stream(s)
        result["stream"] = torch.cuda.current_stream(device=1).stream
        result["device"] = torch.cuda.current_device()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert result["stream"] != torch.cuda.default_stream(device=1).stream
    assert result["device"] == 0
    assert torch.cuda.current_stream().stream == base.stream



def test_cuda_stream_context_switches_current_stream(monkeypatch):
    _stub_runtime(monkeypatch)
    s = torch.cuda.Stream(device=1)
    cur = torch.cuda.current_stream()
    with torch.cuda.stream(s):
        assert torch.cuda.current_stream() is s
        assert torch.cuda.current_device() == s.device.index
    assert torch.cuda.current_stream() is cur





def test_cuda_add_uses_current_stream_for_transfers(monkeypatch):
    _stub_runtime(monkeypatch)
    import numpy as np
    import candle._backends.cuda.ops as cuda_ops
    import candle._backends.cuda.state as cuda_state

    class DummyTensor:
        def __init__(self):
            self.shape = (2,)
            self.dtype = torch.float32
            self.device = torch.device("cuda", 0)
            self._storage = object()

        def storage(self):
            return self._storage

    seen = {}

    class DummyStream:
        def __init__(self, stream):
            self.stream = stream

    monkeypatch.setattr(cuda_state, "current_stream", lambda device_id=0: DummyStream(777))

    def fake_to_numpy(storage, shape, dtype, stream=None):
        seen["to_stream"] = stream
        return np.array([1.0, 2.0], dtype=np.float32)

    def fake_from_numpy(arr, dtype, device, stream=None):
        seen["from_stream"] = stream
        return arr

    monkeypatch.setattr(cuda_ops, "cuda_typed_storage_to_numpy", fake_to_numpy)
    monkeypatch.setattr(cuda_ops, "_from_numpy_on_current_stream", fake_from_numpy)

    out = cuda_ops.add(DummyTensor(), 1.0)

    assert seen["to_stream"] == 777
    assert seen["from_stream"] == 777
    np.testing.assert_allclose(out, np.array([2.0, 3.0], dtype=np.float32))
