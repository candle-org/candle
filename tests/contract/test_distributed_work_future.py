from candle.distributed._work import Work


def test_work_get_future_is_lazy_and_wait_resolves_result_list() -> None:
    state = {"wait_called": False}

    w = Work()

    def _on_wait() -> None:
        state["wait_called"] = True

    w._on_wait = _on_wait

    fut = w.get_future()
    assert fut is not None
    assert fut.done() is False
    assert state["wait_called"] is False

    # Current contract: get_future() is lazy. Explicit Work.wait() drives the
    # completion; Future.wait() then returns the resolved result payload.
    w.wait()
    value = fut.wait()

    assert state["wait_called"] is True
    assert value == w.result()


def test_work_get_future_returns_resolved_future_after_wait() -> None:
    w = Work()
    w.wait()

    fut = w.get_future()
    assert fut.done() is True
    assert fut.wait() == w.result()


def test_work_get_future_propagates_wait_exception_after_work_wait() -> None:
    w = Work()

    def _boom() -> None:
        raise RuntimeError("boom from on_wait")

    w._on_wait = _boom

    fut = w.get_future()
    assert fut.done() is False

    try:
        w.wait()
        raise AssertionError("expected w.wait() to raise")
    except RuntimeError as exc:
        assert "boom from on_wait" in str(exc)

    assert fut.done() is True
    # Future wait path should re-raise the original failure after Work.wait().
    try:
        fut.wait()
        raise AssertionError("expected fut.wait() to raise")
    except RuntimeError as exc:
        assert "boom from on_wait" in str(exc)
