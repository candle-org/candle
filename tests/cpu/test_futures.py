import threading

from candle.futures import Future, collect_all


def test_future_basic_result():
    fut = Future()
    fut.set_result(42)
    assert fut.done() is True
    assert fut.wait() == 42
    assert fut.value() == 42


def test_future_exception_propagates():
    fut = Future()
    fut.set_exception(RuntimeError("boom"))

    try:
        fut.wait()
        assert False, "expected exception"
    except RuntimeError as exc:
        assert str(exc) == "boom"


def test_future_then_chains_result():
    fut = Future()
    chained = fut.then(lambda src: src.wait() * 2)
    fut.set_result(21)
    assert chained.wait() == 42


def test_future_add_done_callback_after_completion_runs_immediately():
    fut = Future()
    seen = []
    fut.set_result("ok")
    fut.add_done_callback(lambda f: seen.append(f.wait()))
    assert seen == ["ok"]


def test_collect_all_returns_original_futures_in_order():
    futures = [Future(), Future(), Future()]
    agg = collect_all(futures)
    futures[0].set_result("a")
    futures[1].set_result("b")
    futures[2].set_result("c")
    assert agg.wait() == futures


def test_collect_all_propagates_first_exception_after_all_done():
    futures = [Future(), Future()]
    agg = collect_all(futures)
    futures[0].set_exception(ValueError("bad"))
    futures[1].set_result("ok")

    try:
        agg.wait()
        assert False, "expected exception"
    except ValueError as exc:
        assert str(exc) == "bad"


def test_future_wait_blocks_until_other_thread_sets_result():
    fut = Future()
    seen = []

    def waiter():
        seen.append(fut.wait())

    thread = threading.Thread(target=waiter)
    thread.start()
    fut.set_result("threaded")
    thread.join(timeout=2)
    assert seen == ["threaded"]


def test_future_module_reports_compiled_fast_path():
    import candle.futures as futures_mod

    assert hasattr(futures_mod, "_FUTURE_CYTHON")
    assert futures_mod._FUTURE_CYTHON is True
