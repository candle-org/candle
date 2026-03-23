"""Focused direct tests for the compiled Cython candle.distributed._c10d.Work.

These tests exercise Work imported from the compiled extension (.so), not the
Python fallback _work.py. They verify the four async-semantics contracts that
Task 2 (Move Work hot path into Cython) claims to implement.

No process group, HCCL, or network is required.
"""

import pytest


# ---------------------------------------------------------------------------
# Guard: skip the whole module if the compiled extension is absent.
# ---------------------------------------------------------------------------

pytest.importorskip("candle.distributed._c10d", reason="Cython _c10d extension not built")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_work(on_wait=None):
    """Return a fresh Cython Work with stream=None and an optional _on_wait hook."""
    from candle.distributed._c10d import Work
    w = Work(stream=None)
    if on_wait is not None:
        w._on_wait = on_wait
    return w


# ---------------------------------------------------------------------------
# 1. get_future() returns a pending (not-done) future before wait()
# ---------------------------------------------------------------------------

def test_get_future_pending_before_wait():
    """get_future() must return a future that is NOT done before wait() is called."""
    called = []
    w = _make_work(on_wait=lambda: called.append(1))

    fut = w.get_future()

    # The on_wait hook must not have fired yet.
    assert called == [], "on_wait fired too early — wait() was called inside get_future()"
    # The future must be pending.
    assert not fut.done(), "Future is already done before wait() — Work completed too early"
    # The Work itself must not be marked complete.
    assert not w.is_completed(), "Work.is_completed() is True before wait()"


# ---------------------------------------------------------------------------
# 2. wait() resolves the future
# ---------------------------------------------------------------------------

def test_wait_resolves_future():
    """Calling wait() must mark the future as done."""
    w = _make_work()

    fut = w.get_future()
    assert not fut.done(), "precondition: future should be pending before wait()"

    w.wait()

    assert fut.done(), "Future is still pending after wait() returned"
    assert w.is_completed(), "Work.is_completed() is False after wait()"


# ---------------------------------------------------------------------------
# 3. get_future() is idempotent — same object returned on repeated calls
# ---------------------------------------------------------------------------

def test_get_future_idempotent():
    """Repeated calls to get_future() must return the same Future object."""
    w = _make_work()

    fut1 = w.get_future()
    fut2 = w.get_future()

    assert fut1 is fut2, (
        f"get_future() returned two different objects: {id(fut1):#x} vs {id(fut2):#x}. "
        "This breaks callers that cache the future before registering callbacks."
    )


# ---------------------------------------------------------------------------
# 4. get_future() after wait() returns an already-done future
# ---------------------------------------------------------------------------

def test_get_future_after_wait_already_done():
    """If wait() was called before get_future(), the returned future must already be done."""
    w = _make_work()
    w.wait()

    fut = w.get_future()

    assert fut.done(), "Future obtained after wait() should already be done"


# ---------------------------------------------------------------------------
# 5. add_done_callback fires after wait(), not before
# ---------------------------------------------------------------------------

def test_callback_fires_after_wait_not_before():
    """A callback registered before wait() must not fire until wait() is called."""
    called_at = []

    def _record(f):
        called_at.append("callback")

    w = _make_work()
    fut = w.get_future()

    fut.add_done_callback(_record)

    # Callback must NOT have fired yet.
    assert called_at == [], "add_done_callback fired immediately — future was already resolved"

    w.wait()

    # Now the callback must have fired.
    assert called_at == ["callback"], (
        f"Callback did not fire after wait(). Recorded events: {called_at}"
    )


# ---------------------------------------------------------------------------
# 6. on_wait exception propagates through future and re-raises from wait()
# ---------------------------------------------------------------------------

def test_on_wait_exception_carried_in_future():
    """If _on_wait raises, wait() re-raises and the future carries the exception."""
    boom = RuntimeError("collective failed")

    def _fail():
        raise boom

    w = _make_work(on_wait=_fail)
    fut = w.get_future()

    with pytest.raises(RuntimeError, match="collective failed"):
        w.wait()

    # The future must be done and carry the exception.
    assert fut.done(), "Future should be done even after on_wait raised"
    assert fut._exception is boom, (
        f"Future._exception is {fut._exception!r}, expected the RuntimeError"
    )
