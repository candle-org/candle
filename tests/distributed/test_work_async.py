"""Work/Future async semantics contracts.

All contracts are now GREEN (implemented in Task 2 / Cython Work hot path).

Contracts covered:

  1. get_future() must not call wait() eagerly; the returned Future must be
     unresolved until the caller explicitly calls wait().

  2. get_future() called twice on the same Work returns the same logical future;
     both see done()==False before wait() and done()==True after.

  3. add_done_callback fires after wait() completes, not at registration time.

  4. is_completed() is False immediately after Work creation with a pending
     _on_wait callback, and True after wait() returns.

  5. If _on_wait raises, the future carries the exception.

These tests exercise Work in isolation — no process group or network needed.
"""

import importlib.util
import os
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Bootstrap: load _work.py without compiled Cython extensions
# ---------------------------------------------------------------------------

class _PurePyFuture:
    """Minimal Future mirroring candle.futures.Future contract.

    Stored at module level so test helpers can reference it directly
    without relying on sys.modules stubs remaining in place after bootstrap.
    """
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
            raise RuntimeError("Future is not done yet")
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


def _ensure_work_importable():
    """Load _work.py with minimal transient stubs, then restore sys.modules.

    Stubs are injected only for the names that are not already present.
    After _work.py is loaded they are removed so the real candle package
    remains importable for other test files collected in the same process.
    Only candle.distributed._work (the target module) is kept permanently.
    """
    if 'candle.distributed._work' in sys.modules:
        return

    src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')

    _STUB_NAMES = ('candle', 'candle.distributed', 'candle.futures',
                   'candle._backends', 'candle._backends.npu')

    # Record which names we inject so we can clean them up afterward.
    injected = []
    had_real_futures = 'candle.futures' in sys.modules
    real_futures_mod = sys.modules.get('candle.futures')
    _MISSING = object()
    real_future_attr = getattr(real_futures_mod, 'Future', _MISSING) if had_real_futures else _MISSING
    for name in _STUB_NAMES:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            injected.append(name)

    # Attach the pure-Python Future to the (possibly stub) futures module so
    # _work.py's `from candle.futures import Future` resolves correctly.
    sys.modules['candle.futures'].Future = _PurePyFuture

    work_path = os.path.join(src, 'candle', 'distributed', '_work.py')
    spec = importlib.util.spec_from_file_location('candle.distributed._work', work_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['candle.distributed._work'] = mod
    spec.loader.exec_module(mod)

    # Remove the stubs we injected so real candle imports are not shadowed.
    # Keep candle.distributed._work — that is exactly what we loaded.
    for name in injected:
        del sys.modules[name]

    # If a real candle.futures module existed before bootstrap, restore both
    # the module object and its original Future attribute.
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


def _get_work_class():
    return sys.modules['candle.distributed._work'].Work


def _make_pending_work(callback=None):
    """Create a Work whose completion is deferred via _on_wait."""
    Work = _get_work_class()
    w = Work(stream=None)
    if callback is not None:
        w._on_wait = callback
    return w


# ---------------------------------------------------------------------------
# Contract 1: get_future() must NOT eagerly call wait()
# ---------------------------------------------------------------------------

class TestGetFutureDoesNotEagerlyWait:
    """get_future() must return without calling wait()."""

    def test_get_future_does_not_set_completed(self):
        """After get_future(), is_completed() must still be False for pending Work.

        EXPECTED: is_completed() == False after get_future() on pending Work.
        ACTUAL (bug): get_future() calls wait() synchronously, so _completed=True.
        """
        wait_called = []
        w = _make_pending_work(callback=lambda: wait_called.append(True))
        _fut = w.get_future()
        assert not w.is_completed(), (
            "get_future() must not call wait(); Work should still be incomplete"
        )

    def test_get_future_does_not_trigger_on_wait_callback(self):
        """get_future() must not trigger the _on_wait callback."""
        wait_called = []
        w = _make_pending_work(callback=lambda: wait_called.append(True))
        _fut = w.get_future()
        assert wait_called == [], (
            "get_future() must not trigger _on_wait callback; async work not yet done"
        )

    def test_future_is_not_done_immediately(self):
        """Future.done() must be False right after get_future() on pending Work."""
        w = _make_pending_work(callback=lambda: None)
        fut = w.get_future()
        assert not fut.done(), (
            "Future returned by get_future() must be unresolved until wait() is called"
        )

    def test_future_has_no_result_before_wait(self):
        """Future.result() must raise before wait() is called."""
        w = _make_pending_work(callback=lambda: None)
        fut = w.get_future()
        with pytest.raises(RuntimeError):
            fut.result()  # must raise — future not yet resolved


# ---------------------------------------------------------------------------
# Contract 2: wait() resolves the future exactly once
# ---------------------------------------------------------------------------

class TestWaitResolvesFutureExactlyOnce:
    """wait() must resolve the pending future and mark Work completed."""

    def test_future_becomes_done_after_wait(self):
        """After wait(), the future previously returned by get_future() is done()."""
        results = []
        w = _make_pending_work(callback=lambda: results.append('done'))
        fut = w.get_future()
        assert not fut.done()  # must be pending first
        w.wait()
        assert fut.done(), "Future must be done() after wait()"

    def test_wait_result_propagated_to_future(self):
        """The value returned by Work.result() must be carried by the future."""
        w = _make_pending_work(callback=lambda: None)
        fut = w.get_future()
        assert not fut.done()
        w.wait()
        assert fut.done()
        assert fut.result() == [], f"Expected [] from Work.result(), got {fut.result()!r}"

    def test_double_wait_callback_fires_once(self):
        """_on_wait callback fires exactly once; second wait() is a no-op."""
        fired = []
        w = _make_pending_work(callback=lambda: fired.append(1))
        fut = w.get_future()
        assert not fut.done(), "Future must be pending before explicit wait()"
        w.wait()
        assert fired == [1]
        w.wait()  # idempotent
        assert fired == [1], f"Callback must NOT fire again on second wait(); got {fired!r}"


# ---------------------------------------------------------------------------
# Contract 3: add_done_callback fires after wait()
# ---------------------------------------------------------------------------

class TestAddDoneCallbackFiring:
    """A callback added to the future before wait() must fire after wait()."""

    def test_callback_on_future_fires_after_wait(self):
        """add_done_callback registered before wait() fires when wait() completes.

        EXPECTED: callback fires once with the resolved future.
        ACTUAL (bug): future is already resolved when add_done_callback is called
            (due to eager wait), so callback fires immediately — wrong ordering.
        """
        cb_results = []
        w = _make_pending_work(callback=lambda: None)
        fut = w.get_future()
        assert not fut.done(), "Future must be pending; eager wait() bug detected"
        fut.add_done_callback(lambda f: cb_results.append(f.result()))
        assert cb_results == [], "Callback must not fire before wait()"
        w.wait()
        assert len(cb_results) == 1, f"Callback must fire once after wait(); got {cb_results!r}"
        assert cb_results[0] == [], f"Wrong result in callback: {cb_results[0]!r}"

    def test_multiple_callbacks_all_fire_in_order(self):
        """All callbacks registered before wait() fire in registration order."""
        fired = []
        w = _make_pending_work(callback=lambda: None)
        fut = w.get_future()
        assert not fut.done(), "Future must be pending; eager wait() bug detected"
        fut.add_done_callback(lambda f: fired.append('cb1'))
        fut.add_done_callback(lambda f: fired.append('cb2'))
        fut.add_done_callback(lambda f: fired.append('cb3'))
        assert fired == []
        w.wait()
        assert fired == ['cb1', 'cb2', 'cb3'], f"Wrong callback order: {fired!r}"


# ---------------------------------------------------------------------------
# Contract 4: is_completed() reflects actual completion state
# ---------------------------------------------------------------------------

class TestIsCompletedReflectsActualCompletion:
    """is_completed() must be False until wait() has been called."""

    def test_is_completed_false_on_new_work_with_callback(self):
        """Work with a pending _on_wait callback is not yet complete."""
        w = _make_pending_work(callback=lambda: None)
        assert not w.is_completed(), (
            "New Work with a pending callback must not be considered completed"
        )

    def test_is_completed_true_after_wait(self):
        """is_completed() becomes True only after wait()."""
        w = _make_pending_work(callback=lambda: None)
        assert not w.is_completed()
        w.wait()
        assert w.is_completed()

    def test_is_completed_true_on_work_with_no_callback_no_stream(self):
        """Work with no callback is complete after explicit wait()."""
        Work = _get_work_class()
        w = Work(stream=None)
        w.wait()
        assert w.is_completed()

    def test_double_wait_is_completed_stays_true(self):
        """is_completed() stays True after second wait()."""
        w = _make_pending_work(callback=lambda: None)
        w.wait()
        assert w.is_completed()
        w.wait()
        assert w.is_completed(), "is_completed() must remain True after double wait()"


# ---------------------------------------------------------------------------
# Contract 5: get_future() idempotency — two calls track the same completion
# ---------------------------------------------------------------------------

class TestGetFutureIdempotency:
    """get_future() called twice must both reflect completion after one wait()."""

    def test_second_get_future_also_pending_before_wait(self):
        """Both futures obtained from get_future() must be pending before wait()."""
        w = _make_pending_work(callback=lambda: None)
        fut1 = w.get_future()
        fut2 = w.get_future()
        assert not fut1.done(), "fut1 must be pending before wait()"
        assert not fut2.done(), "fut2 must be pending before wait()"
        w.wait()
        assert fut1.done(), "fut1 must be done after wait()"
        assert fut2.done(), "fut2 must be done after wait()"

    def test_callback_fired_exactly_once_across_two_get_future_calls(self):
        """_on_wait callback fires exactly once regardless of get_future() call count."""
        fired = []
        w = _make_pending_work(callback=lambda: fired.append(1))
        _fut1 = w.get_future()
        _fut2 = w.get_future()
        assert fired == [], "callback must not fire during get_future() calls"
        w.wait()
        assert fired == [1], f"callback must fire exactly once; got {fired!r}"


# ---------------------------------------------------------------------------
# Contract 6: get_future() on already-completed Work returns a resolved future
# ---------------------------------------------------------------------------

class TestGetFutureOnCompletedWork:
    """If Work is already complete, get_future() MAY return a resolved future."""

    def test_already_completed_work_returns_resolved_future(self):
        """Work that has already been wait()ed may return a done() future."""
        Work = _get_work_class()
        w = Work(stream=None)
        w.wait()
        assert w.is_completed()
        fut = w.get_future()
        # Calling result() must not raise
        value = fut.result()
        assert value == []


# ---------------------------------------------------------------------------
# Contract 7: get_future() result payload — future carries _on_wait output
# ---------------------------------------------------------------------------

class TestWorkResultPayload:
    """The future must carry whatever result() returns when wait() completes."""

    def test_result_populated_before_future_resolved(self):
        """_on_wait callback runs before future is resolved; result() reflects it.

        EXPECTED: future carries the result set by _on_wait.
        ACTUAL (bug): get_future() calls wait() eagerly so the future is
            already resolved before _on_wait can populate state in the correct
            async sequence.
        """
        payload = []

        class PayloadWork(_get_work_class()):
            def result(self):
                return payload[:]

        w = PayloadWork(stream=None)
        w._on_wait = lambda: payload.append(42)

        fut = w.get_future()
        assert not fut.done(), (
            "get_future() must not call wait(); future must be pending"
        )

        w.wait()
        assert fut.done()
        value = fut.result()
        assert value == [42], f"Future must carry result from _on_wait; got {value!r}"


# ---------------------------------------------------------------------------
# Contract 8: exception path — get_future() on failing work
# ---------------------------------------------------------------------------

class TestGetFutureExceptionPath:
    """If _on_wait raises, the future must carry the exception."""

    def test_exception_in_callback_propagated_to_future(self):
        """A callback that raises must set an exception on the future, not crash."""
        boom = RuntimeError("kernel exploded")

        def bad_callback():
            raise boom

        w = _make_pending_work(callback=bad_callback)
        fut = w.get_future()
        assert not fut.done(), "Future must be pending before wait()"

        try:
            w.wait()
        except RuntimeError:
            pass  # acceptable for wait() to propagate the exception

        assert fut.done(), "Future must be done() after a failed wait()"
        with pytest.raises(RuntimeError, match="kernel exploded"):
            fut.result()
