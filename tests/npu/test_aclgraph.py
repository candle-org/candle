import types

import pytest

import candle as torch


class _FakeStream:
    def __init__(self, handle=123, device_index=0):
        self._handle = handle
        self.stream = handle
        self.device = types.SimpleNamespace(index=device_index)
        self.sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1


class _FakeGraphImpl:
    def __init__(self):
        self.calls = []
        self.capture_stream = 999

    def capture_begin(self, stream, mode):
        self.calls.append(("capture_begin", stream, mode))
        self.capture_stream = stream

    def capture_end(self):
        self.calls.append(("capture_end",))

    def replay_async(self, stream):
        self.calls.append(("replay_async", stream))

    def reset(self):
        self.calls.append(("reset",))

    def abort(self):
        self.calls.append(("abort",))

    def debug_dump(self, path):
        self.calls.append(("debug_dump", path))


@pytest.fixture
def fake_graph_env(monkeypatch):
    import candle._cython._aclgraph as aclgraph_mod

    fake_stream = _FakeStream()
    sync_calls = []

    monkeypatch.setattr(aclgraph_mod, "_NPUGraphImpl", _FakeGraphImpl)
    monkeypatch.setattr(torch.npu, "current_stream", lambda device=None: fake_stream)
    monkeypatch.setattr(torch.npu, "synchronize", lambda device=None: sync_calls.append(device))
    monkeypatch.setattr(torch.npu, "_ensure_initialized", lambda: None)
    return fake_stream, sync_calls


def test_npu_graph_api_symbols_exist():
    assert hasattr(torch.npu, "NPUGraph")
    assert hasattr(torch.npu, "graph")
    assert hasattr(torch.npu, "is_current_stream_capturing")


def test_npu_graph_capture_begin_uses_current_stream(fake_graph_env):
    fake_stream, _ = fake_graph_env
    g = torch.npu.NPUGraph()

    g.capture_begin(capture_error_mode="thread_local")

    assert g._impl.calls == [("capture_begin", fake_stream._handle, 1)]


def test_npu_graph_replay_uses_capture_stream(fake_graph_env):
    g = torch.npu.NPUGraph()
    g._impl.capture_stream = 456

    g.replay()

    assert g._impl.calls == [("replay_async", 456)]




def test_npu_graph_context_manager_syncs_capture_stream(fake_graph_env):
    fake_stream, sync_calls = fake_graph_env
    g = torch.npu.NPUGraph()

    with torch.npu.graph(g):
        pass

    assert sync_calls == [None]
    assert g._impl.calls == [
        ("capture_begin", fake_stream._handle, 0),
        ("capture_end",),
    ]
    assert fake_stream.sync_calls == 1


def test_npu_graph_context_abort_on_exception(fake_graph_env):
    _, sync_calls = fake_graph_env
    g = torch.npu.NPUGraph()

    with pytest.raises(ValueError):
        with torch.npu.graph(g):
            raise ValueError("boom")

    assert sync_calls == [None]
    assert ("abort",) in g._impl.calls


def test_npu_graph_enter_failure_restores_stream_context(fake_graph_env, monkeypatch):
    class RaisingGraphImpl(_FakeGraphImpl):
        def capture_begin(self, stream, mode):
            super().capture_begin(stream, mode)
            raise RuntimeError("capture failed")

    import candle._cython._aclgraph as aclgraph_mod

    exit_calls = []

    class FakeStreamCtx:
        def __init__(self, s):
            self.s = s

        def __enter__(self):
            return self.s

        def __exit__(self, exc_type, exc, tb):
            exit_calls.append((exc_type, exc, tb))
            return False

    monkeypatch.setattr(aclgraph_mod, "_NPUGraphImpl", RaisingGraphImpl)
    monkeypatch.setattr(torch.npu, "stream", FakeStreamCtx)

    g = torch.npu.NPUGraph()
    explicit_stream = _FakeStream(handle=777)

    with pytest.raises(RuntimeError, match="capture failed"):
        with torch.npu.graph(g, stream=explicit_stream):
            pass

    assert len(exit_calls) == 1




def test_npu_graph_exit_sync_failure_still_restores_stream_context(fake_graph_env, monkeypatch):
    import candle._cython._aclgraph as aclgraph_mod

    exit_calls = []

    class FakeStreamCtx:
        def __init__(self, s):
            self.s = s

        def __enter__(self):
            return self.s

        def __exit__(self, exc_type, exc, tb):
            exit_calls.append((exc_type, exc, tb))
            return False

    fake_stream, _ = fake_graph_env

    def _raise_sync():
        raise RuntimeError("sync failed")

    fake_stream.synchronize = _raise_sync
    monkeypatch.setattr(torch.npu, "stream", FakeStreamCtx)
    monkeypatch.setattr(aclgraph_mod, "_NPUGraphImpl", _FakeGraphImpl)

    g = torch.npu.NPUGraph()
    explicit_stream = _FakeStream(handle=777)

    with pytest.raises(RuntimeError, match="sync failed"):
        with torch.npu.graph(g, stream=explicit_stream):
            pass

    assert len(exit_calls) == 1




def test_npu_graph_reset_and_debug_dump(fake_graph_env):
    g = torch.npu.NPUGraph()

    g.reset()
    g.debug_dump("/tmp/aclgraph.json")

    assert g._impl.calls == [
        ("reset",),
        ("debug_dump", "/tmp/aclgraph.json"),
    ]


def test_npu_is_current_stream_capturing(fake_graph_env, monkeypatch):
    import candle._cython._aclrt_ffi as aclrt_ffi

    monkeypatch.setattr("candle._backends.npu.ops_soc.aclgraph_supported", lambda profile=None: True)
    monkeypatch.setattr(
        aclrt_ffi,
        "capture_get_info",
        lambda handle: (aclrt_ffi.ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE, 0),
    )

    assert torch.npu.is_current_stream_capturing() is True


def test_npu_is_current_stream_not_capturing(fake_graph_env, monkeypatch):
    import candle._cython._aclrt_ffi as aclrt_ffi

    monkeypatch.setattr("candle._backends.npu.ops_soc.aclgraph_supported", lambda profile=None: True)
    monkeypatch.setattr(
        aclrt_ffi,
        "capture_get_info",
        lambda handle: (aclrt_ffi.ACL_MODEL_RI_CAPTURE_STATUS_NONE, 0),
    )

    assert torch.npu.is_current_stream_capturing() is False


def test_npu_is_current_stream_capturing_returns_false_when_aclgraph_unsupported(fake_graph_env, monkeypatch):
    import candle._cython._aclrt_ffi as aclrt_ffi

    monkeypatch.setattr("candle._backends.npu.ops_soc.aclgraph_supported", lambda profile=None: False)

    def fail_probe(handle):
        raise AssertionError("capture_get_info should not run when aclgraph is unsupported")

    monkeypatch.setattr(aclrt_ffi, "capture_get_info", fail_probe)

    assert torch.npu.is_current_stream_capturing() is False


def test_npu_graph_invalid_capture_error_mode(fake_graph_env):
    g = torch.npu.NPUGraph()
    with pytest.raises(ValueError, match="Invalid capture_error_mode"):
        g.capture_begin(capture_error_mode="unknown_mode")


# ---- Direct _NPUGraphImpl state-machine tests (no real hardware) ----


def test_npu_graph_impl_replay_from_idle_raises():
    from candle._cython._aclgraph import _NPUGraphImpl

    impl = _NPUGraphImpl()
    with pytest.raises(RuntimeError):
        impl.replay_async(999)


def test_npu_graph_impl_capture_end_without_begin_raises():
    from candle._cython._aclgraph import _NPUGraphImpl

    impl = _NPUGraphImpl()
    with pytest.raises(RuntimeError):
        impl.capture_end()


def test_npu_graph_impl_abort_from_idle_raises():
    from candle._cython._aclgraph import _NPUGraphImpl

    impl = _NPUGraphImpl()
    with pytest.raises(RuntimeError):
        impl.abort()


