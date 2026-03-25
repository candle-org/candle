"""Task 3 TDD: HCCL-specific async completion via ACL events.

Contracts being locked:

  E1: Work created with an ``_aclrt_event`` stays is_completed()==False
      until wait() is called (event-based laziness, not stream sync).

  E2: wait() on an event-backed Work calls synchronize_event(), NOT
      synchronize_stream(). The stream is NOT blocked for unrelated work.

  E3: is_completed() can be polled cheaply via query_event() without
      blocking.  Returns True as soon as the event fires.

  E4: _make_work() in ProcessGroupHCCL records an ACL event on the
      stream immediately after the collective is enqueued, and passes
      it to Work.  The event handle is stored as Work._aclrt_event.

  E5: When _completed is already True, is_completed() returns True
      without calling query_event.

  E6: get_future() on an event-backed pending Work returns a Future
      that is not yet done.  After wait(), the future is resolved.

Tests use mock runtimes — no real HCCL hardware required.
"""

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Minimal pure-Python Future (same pattern as test_work_async.py)
# ---------------------------------------------------------------------------

class _PurePyFuture:
    def __init__(self):
        self._result = None
        self._exception = None
        self._done = False
        self._callbacks = []

    def set_result(self, value):
        if self._done:
            raise RuntimeError("Future already resolved")
        self._result = value
        self._done = True
        for cb in self._callbacks:
            cb(self)

    def set_exception(self, exc):
        if self._done:
            raise RuntimeError("Future already resolved")
        self._exception = exc
        self._done = True
        for cb in self._callbacks:
            cb(self)

    def done(self):
        return self._done

    def result(self):
        if not self._done:
            raise RuntimeError("Future not done")
        if self._exception is not None:
            raise self._exception
        return self._result

    def value(self):
        return self.result()

    def wait(self):
        return self.result()

    def add_done_callback(self, fn):
        if self._done:
            fn(self)
        else:
            self._callbacks.append(fn)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

WORKTREE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
SRC = os.path.join(WORKTREE, "src")

_STUB_NAMES = (
    "candle", "candle.distributed", "candle.futures",
    "candle._backends", "candle._backends.npu",
)


def _make_mock_runtime(event_done=False):
    """Return a mock npu_runtime with controllable event state."""
    rt = MagicMock()
    rt.query_event.return_value = event_done
    rt.synchronize_event.return_value = None
    rt.synchronize_stream.return_value = None
    rt.record_event.return_value = None
    sentinel = object()
    rt.create_event.return_value = sentinel
    rt._sentinel_event = sentinel
    return rt


def _inject_runtime(mock_rt, saved, device_id=0):
    """Inject mock runtime into sys.modules so `from candle._backends.npu
    import runtime` resolves to it during wait() / is_completed() calls."""
    stub_npu_rt = types.ModuleType("candle._backends.npu.runtime")
    stub_npu_rt.get_runtime = lambda dev=device_id: mock_rt  # noqa: E731

    stub_backends = types.ModuleType("candle._backends")
    stub_npu = types.ModuleType("candle._backends.npu")
    stub_npu.runtime = stub_npu_rt
    stub_backends.npu = stub_npu

    for key in (
        "candle._backends",
        "candle._backends.npu",
        "candle._backends.npu.runtime",
    ):
        saved[key] = sys.modules.get(key)

    sys.modules["candle._backends"] = stub_backends
    sys.modules["candle._backends.npu"] = stub_npu
    sys.modules["candle._backends.npu.runtime"] = stub_npu_rt


def _restore(saved):
    for key, val in saved.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


def _ensure_work_importable():
    """Load _work.py registered as candle.distributed._work so that
    relative imports inside it resolve correctly via sys.modules.
    Mirrors the bootstrap pattern from test_work_async.py and restores
    any pre-existing real candle.futures module after bootstrap."""
    if "candle.distributed._work" in sys.modules:
        return

    injected = []
    had_real_futures = 'candle.futures' in sys.modules
    real_futures_mod = sys.modules.get('candle.futures')
    _MISSING = object()
    real_future_attr = getattr(real_futures_mod, 'Future', _MISSING) if had_real_futures else _MISSING
    for name in _STUB_NAMES:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            injected.append(name)

    sys.modules["candle.futures"].Future = _PurePyFuture

    work_path = os.path.join(SRC, "candle/distributed/_work.py")
    spec = importlib.util.spec_from_file_location(
        "candle.distributed._work", work_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candle.distributed._work"] = mod
    spec.loader.exec_module(mod)

    for name in injected:
        del sys.modules[name]

    if had_real_futures:
        sys.modules['candle.futures'] = real_futures_mod
        if real_future_attr is _MISSING:
            try:
                delattr(real_futures_mod, 'Future')
            except AttributeError:
                pass
        else:
            real_futures_mod.Future = real_future_attr
    else:
        sys.modules.pop('candle.futures', None)


_ensure_work_importable()
Work = sys.modules["candle.distributed._work"].Work


# ---------------------------------------------------------------------------
# E1: event-backed Work is pending until wait() is called
# ---------------------------------------------------------------------------

class TestEventBackedWorkIsPendingUntilWait:
    """E1: is_completed() == False until wait() is called."""

    def test_is_completed_false_before_wait_with_event(self):
        w = Work(stream=None, device_id=0)
        w._aclrt_event = object()
        assert not w.is_completed()

    def test_is_completed_true_after_wait_with_event(self):
        mock_rt = _make_mock_runtime(event_done=False)
        sentinel_event = object()
        w = Work(stream=None, device_id=0)
        w._aclrt_event = sentinel_event

        saved = {}
        _inject_runtime(mock_rt, saved)
        # Override create_event/record_event as irrelevant here
        try:
            w.wait()
        finally:
            _restore(saved)

        assert w.is_completed()


# ---------------------------------------------------------------------------
# E2: wait() uses synchronize_event, NOT synchronize_stream
# ---------------------------------------------------------------------------

class TestWaitUsesSynchronizeEventNotStream:
    """E2: wait() on an event-backed Work must call synchronize_event,
    not synchronize_stream."""

    def test_wait_calls_synchronize_event(self):
        mock_rt = _make_mock_runtime()
        sentinel_event = object()
        w = Work(stream=12345, device_id=0)
        w._aclrt_event = sentinel_event

        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            w.wait()
        finally:
            _restore(saved)

        mock_rt.synchronize_event.assert_called_once_with(sentinel_event)

    def test_wait_does_not_call_synchronize_stream_when_event_present(self):
        mock_rt = _make_mock_runtime()
        sentinel_event = object()
        w = Work(stream=12345, device_id=0)
        w._aclrt_event = sentinel_event

        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            w.wait()
        finally:
            _restore(saved)

        mock_rt.synchronize_stream.assert_not_called()

    def test_wait_still_calls_synchronize_stream_when_no_event(self):
        """Fallback: without an event, the old stream sync path still works."""
        mock_rt = _make_mock_runtime()
        w = Work(stream=12345, device_id=0)
        # No _aclrt_event set

        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            w.wait()
        finally:
            _restore(saved)

        mock_rt.synchronize_stream.assert_called_once_with(12345)


# ---------------------------------------------------------------------------
# E3: is_completed() polls via query_event without blocking
# ---------------------------------------------------------------------------

class TestIsCompletedPollsQueryEvent:
    """E3: is_completed() uses query_event() for cheap non-blocking check."""

    def test_is_completed_returns_false_when_event_not_done(self):
        mock_rt = _make_mock_runtime(event_done=False)
        sentinel_event = object()
        w = Work(stream=None, device_id=0)
        w._aclrt_event = sentinel_event

        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            result = w.is_completed()
        finally:
            _restore(saved)

        assert result is False
        mock_rt.query_event.assert_called_once_with(sentinel_event)

    def test_is_completed_returns_true_when_event_done(self):
        mock_rt = _make_mock_runtime(event_done=True)
        sentinel_event = object()
        w = Work(stream=None, device_id=0)
        w._aclrt_event = sentinel_event

        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            result = w.is_completed()
        finally:
            _restore(saved)

        assert result is True
        mock_rt.query_event.assert_called_once_with(sentinel_event)

    def test_is_completed_does_not_call_synchronize_event(self):
        """is_completed() must be non-blocking."""
        mock_rt = _make_mock_runtime(event_done=False)
        sentinel_event = object()
        w = Work(stream=None, device_id=0)
        w._aclrt_event = sentinel_event

        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            w.is_completed()
        finally:
            _restore(saved)

        mock_rt.synchronize_event.assert_not_called()


# ---------------------------------------------------------------------------
# E4: _make_work records ACL event and stores it on Work._aclrt_event
# ---------------------------------------------------------------------------

class TestMakeWorkRecordsEvent:
    """E4: _make_work(stream) must record an ACL event on the stream and
    attach it to the returned Work as _aclrt_event."""

    def _load_pg_mod(self):
        """Load _process_group.py registered as candle.distributed._process_group
        so that relative imports inside it resolve via sys.modules."""
        extra_stubs = {
            "candle.distributed._reduce_op": types.ModuleType(
                "candle.distributed._reduce_op"
            ),
        }
        extra_stubs["candle.distributed._reduce_op"].ReduceOp = MagicMock()
        # Ensure Work module is already in sys.modules (set up by bootstrap)
        extra_stubs["candle.distributed._work"] = sys.modules.get(
            "candle.distributed._work"
        ) or types.ModuleType("candle.distributed._work")
        if not hasattr(extra_stubs["candle.distributed._work"], "Work"):
            extra_stubs["candle.distributed._work"].Work = Work

        # Also ensure the base stub names are present during load
        injected = []
        had_real_futures = 'candle.futures' in sys.modules
        real_futures_mod = sys.modules.get('candle.futures')
        _MISSING = object()
        real_future_attr = getattr(real_futures_mod, 'Future', _MISSING) if had_real_futures else _MISSING
        for name in _STUB_NAMES:
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)
                injected.append(name)
        sys.modules["candle.futures"].Future = _PurePyFuture

        saved_extra = {k: sys.modules.get(k) for k in extra_stubs}
        sys.modules.update(extra_stubs)

        try:
            spec = importlib.util.spec_from_file_location(
                "candle.distributed._process_group",
                os.path.join(SRC, "candle/distributed/_process_group.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["candle.distributed._process_group"] = mod
            spec.loader.exec_module(mod)
        finally:
            _restore(saved_extra)
            for name in injected:
                sys.modules.pop(name, None)
            if had_real_futures:
                sys.modules['candle.futures'] = real_futures_mod
                if real_future_attr is _MISSING:
                    try:
                        delattr(real_futures_mod, 'Future')
                    except AttributeError:
                        pass
                else:
                    real_futures_mod.Future = real_future_attr
            else:
                sys.modules.pop('candle.futures', None)
            sys.modules.pop("candle.distributed._process_group", None)

        return mod

    def test_make_work_attaches_event_to_work(self):
        pg_mod = self._load_pg_mod()
        mock_rt = _make_mock_runtime()
        sentinel_event = object()
        mock_rt.create_event.return_value = sentinel_event

        pg = pg_mod.ProcessGroupHCCL.__new__(pg_mod.ProcessGroupHCCL)
        pg._device_id = 0
        pg._rank = 0
        pg._size = 1

        fake_stream = 99999
        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            work = pg._make_work(fake_stream)
        finally:
            _restore(saved)

        assert hasattr(work, "_aclrt_event"), (
            "_make_work() must attach _aclrt_event to returned Work"
        )
        assert work._aclrt_event is sentinel_event

    def test_make_work_calls_record_event_on_stream(self):
        pg_mod = self._load_pg_mod()
        mock_rt = _make_mock_runtime()
        sentinel_event = object()
        mock_rt.create_event.return_value = sentinel_event

        pg = pg_mod.ProcessGroupHCCL.__new__(pg_mod.ProcessGroupHCCL)
        pg._device_id = 0
        pg._rank = 0
        pg._size = 1

        fake_stream = 99999
        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            pg._make_work(fake_stream)
        finally:
            _restore(saved)

        mock_rt.record_event.assert_called_once_with(sentinel_event, fake_stream)

    def test_make_work_no_event_when_stream_is_none(self):
        """When stream is None, no event is created or recorded."""
        pg_mod = self._load_pg_mod()
        mock_rt = _make_mock_runtime()

        pg = pg_mod.ProcessGroupHCCL.__new__(pg_mod.ProcessGroupHCCL)
        pg._device_id = 0
        pg._rank = 0
        pg._size = 1

        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            work = pg._make_work(None)
        finally:
            _restore(saved)

        assert work._aclrt_event is None, (
            "No event should be recorded when stream is None"
        )
        mock_rt.create_event.assert_not_called()
        mock_rt.record_event.assert_not_called()


# ---------------------------------------------------------------------------
# E5: is_completed() returns True without query_event when _completed is True
# ---------------------------------------------------------------------------

class TestIsCompletedAlreadyDone:
    """E5: Once _completed flag is True, is_completed() returns True
    without touching the runtime."""

    def test_is_completed_true_after_explicit_completion_no_query(self):
        mock_rt = _make_mock_runtime(event_done=True)
        sentinel_event = object()

        w = Work(stream=None, device_id=0)
        w._aclrt_event = sentinel_event
        w._completed = True  # manually mark completed

        # Must return True without calling query_event
        result = w.is_completed()
        assert result is True
        mock_rt.query_event.assert_not_called()


# ---------------------------------------------------------------------------
# E6: get_future() on event-backed Work is lazy; resolves after wait()
# ---------------------------------------------------------------------------

class TestGetFutureWithEvent:
    """E6: get_future() stays pending until wait(); resolves on wait()."""

    def test_get_future_pending_before_wait(self):
        w = Work(stream=None, device_id=0)
        w._aclrt_event = object()
        fut = w.get_future()
        assert not fut.done(), "Future must be pending before wait()"

    def test_get_future_resolved_after_wait(self):
        mock_rt = _make_mock_runtime(event_done=False)
        sentinel_event = object()
        w = Work(stream=None, device_id=0)
        w._aclrt_event = sentinel_event

        fut = w.get_future()
        assert not fut.done()

        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            w.wait()
        finally:
            _restore(saved)

        assert fut.done(), "Future must be resolved after wait()"

    def test_get_future_exception_from_synchronize_event_propagated(self):
        """If synchronize_event raises, the future must carry the exception."""
        mock_rt = _make_mock_runtime()
        boom = RuntimeError("HCCL event sync failed")
        mock_rt.synchronize_event.side_effect = boom
        sentinel_event = object()

        w = Work(stream=None, device_id=0)
        w._aclrt_event = sentinel_event

        fut = w.get_future()
        assert not fut.done()

        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            w.wait()
        except RuntimeError:
            pass
        finally:
            _restore(saved)

        assert fut.done(), "Future must be done after a failed wait()"
        with pytest.raises(RuntimeError, match="HCCL event sync failed"):
            fut.wait()


# ---------------------------------------------------------------------------
# Gap 1: is_completed() must resolve the future when query_event fires
# ---------------------------------------------------------------------------

class TestIsCompletedResolvesFutureOnEventFire:
    """When is_completed() polls query_event and it returns True, the
    stored future (if any) must be resolved — not just _completed."""

    def test_future_resolved_when_is_completed_polls_true(self):
        """is_completed() observes event done → future must become done."""
        mock_rt = _make_mock_runtime(event_done=True)
        sentinel_event = object()

        w = Work(stream=None, device_id=0)
        w._aclrt_event = sentinel_event

        fut = w.get_future()
        assert not fut.done(), "Future must be pending before polling"

        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            result = w.is_completed()
        finally:
            _restore(saved)

        assert result is True, "is_completed() must return True when event fired"
        assert fut.done(), (
            "Future must be resolved when is_completed() observes event completion"
        )
        assert fut.wait() == [], "Future result must be the Work result ([])"

    def test_future_not_resolved_when_is_completed_polls_false(self):
        """is_completed() sees event not done → future stays pending."""
        mock_rt = _make_mock_runtime(event_done=False)
        sentinel_event = object()

        w = Work(stream=None, device_id=0)
        w._aclrt_event = sentinel_event

        fut = w.get_future()

        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            result = w.is_completed()
        finally:
            _restore(saved)

        assert result is False
        assert not fut.done(), "Future must still be pending when event is not done"


# ---------------------------------------------------------------------------
# Gap 2: _make_work() fallback — event creation failure → stream sync path
# ---------------------------------------------------------------------------

class TestMakeWorkFallbackOnEventFailure:
    """When create_event or record_event raises, _make_work() must:
    - return a Work whose _aclrt_event is None (stream-sync fallback)
    - that Work.wait() resolves via synchronize_stream, not synchronize_event
    - the future is still resolved correctly after wait()
    """

    def _load_pg_mod(self):
        """Load _process_group.py with minimal stubs (reuses the pattern
        from TestMakeWorkRecordsEvent)."""
        import ctypes
        import importlib.util as ilu

        extra_stubs = {
            "candle.distributed": types.ModuleType("candle.distributed"),
            "candle.distributed._work": sys.modules.get(
                "candle.distributed._work"
            ) or types.ModuleType("candle.distributed._work"),
        }
        if not hasattr(extra_stubs["candle.distributed._work"], "Work"):
            extra_stubs["candle.distributed._work"].Work = Work

        injected = []
        had_real_futures = 'candle.futures' in sys.modules
        real_futures_mod = sys.modules.get('candle.futures')
        _MISSING = object()
        real_future_attr = getattr(real_futures_mod, 'Future', _MISSING) if had_real_futures else _MISSING
        for name in _STUB_NAMES:
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)
                injected.append(name)
        sys.modules["candle.futures"].Future = _PurePyFuture

        saved_extra = {k: sys.modules.get(k) for k in extra_stubs}
        sys.modules.update(extra_stubs)

        try:
            spec = ilu.spec_from_file_location(
                "candle.distributed._process_group",
                os.path.join(SRC, "candle/distributed/_process_group.py"),
            )
            mod = ilu.module_from_spec(spec)
            sys.modules["candle.distributed._process_group"] = mod
            spec.loader.exec_module(mod)
        finally:
            _restore(saved_extra)
            for name in injected:
                sys.modules.pop(name, None)
            if had_real_futures:
                sys.modules['candle.futures'] = real_futures_mod
                if real_future_attr is _MISSING:
                    try:
                        delattr(real_futures_mod, 'Future')
                    except AttributeError:
                        pass
                else:
                    real_futures_mod.Future = real_future_attr
            else:
                sys.modules.pop('candle.futures', None)
            sys.modules.pop("candle.distributed._process_group", None)

        return mod

    def test_make_work_fallback_when_create_event_raises(self):
        """If create_event raises, _aclrt_event must be None (stream-sync path)."""
        pg_mod = self._load_pg_mod()
        mock_rt = _make_mock_runtime()
        mock_rt.create_event.side_effect = RuntimeError("ACL event unavailable")

        pg = pg_mod.ProcessGroupHCCL.__new__(pg_mod.ProcessGroupHCCL)
        pg._device_id = 0
        pg._rank = 0
        pg._size = 1

        fake_stream = 99999
        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            work = pg._make_work(fake_stream)
        finally:
            _restore(saved)

        assert work._aclrt_event is None, (
            "When create_event raises, _aclrt_event must be None (stream-sync fallback)"
        )

    def test_make_work_fallback_resolves_future_via_stream_sync(self):
        """Fallback Work (no event) must still resolve the future after wait()."""
        pg_mod = self._load_pg_mod()

        # create_event fails → no event attached
        mock_rt = _make_mock_runtime()
        mock_rt.create_event.side_effect = RuntimeError("ACL event unavailable")

        pg = pg_mod.ProcessGroupHCCL.__new__(pg_mod.ProcessGroupHCCL)
        pg._device_id = 0
        pg._rank = 0
        pg._size = 1

        fake_stream = 99999
        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            work = pg._make_work(fake_stream)
        finally:
            _restore(saved)

        assert work._aclrt_event is None

        fut = work.get_future()
        assert not fut.done(), "Future must be pending before wait()"

        # Now wait() — must use stream-sync path
        _inject_runtime(mock_rt, saved)
        try:
            work.wait()
        finally:
            _restore(saved)

        mock_rt.synchronize_stream.assert_called_once_with(fake_stream)
        mock_rt.synchronize_event.assert_not_called()
        assert fut.done(), "Future must be resolved after stream-sync wait()"
        assert fut.wait() == []

    def test_make_work_fallback_record_event_raises(self):
        """If record_event raises after create_event succeeds, still fallback."""
        pg_mod = self._load_pg_mod()
        mock_rt = _make_mock_runtime()
        # create_event succeeds but record_event fails
        mock_rt.record_event.side_effect = RuntimeError("ACL record failed")

        pg = pg_mod.ProcessGroupHCCL.__new__(pg_mod.ProcessGroupHCCL)
        pg._device_id = 0
        pg._rank = 0
        pg._size = 1

        fake_stream = 99999
        saved = {}
        _inject_runtime(mock_rt, saved)
        try:
            work = pg._make_work(fake_stream)
        finally:
            _restore(saved)

        assert work._aclrt_event is None, (
            "When record_event raises, _aclrt_event must be None (stream-sync fallback)"
        )
