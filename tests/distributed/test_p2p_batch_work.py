"""Tests for batch_isend_irecv aggregate Work semantics.

These tests focus on the prerequisite behavior Task 15A needs before a
minimal pipeline runtime can be built.  The key contract is that
``batch_isend_irecv`` returns a *batch-level* Work handle whose lifecycle
represents all P2P ops in the batch, instead of exposing only per-op works.
"""

import pytest


class _FakeWork:
    def __init__(self, name, *, done=False, fail_exc=None):
        self.name = name
        self._done = done
        self._fail_exc = fail_exc
        self.wait_calls = 0
        self.future = None

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self._fail_exc is not None:
            raise self._fail_exc
        self._done = True
        return True

    def is_completed(self):
        return self._done

    def is_success(self):
        return self._done and self._fail_exc is None

    def exception(self):
        return self._fail_exc

    def result(self):
        return []

    def synchronize(self):
        return self.wait()

    def get_future(self):
        if self.future is None:
            class _Future:
                def __init__(self):
                    self._done = False
                    self._exc = None
                    self._result = None
                def set_result(self, value):
                    self._done = True
                    self._result = value
                def set_exception(self, exc):
                    self._done = True
                    self._exc = exc
                def done(self):
                    return self._done
                def result(self):
                    if self._exc is not None:
                        raise self._exc
                    return self._result
            self.future = _Future()
        return self.future


class _Recorder:
    def __init__(self):
        self.calls = []

    def make_op(self, name, *, done=False, fail_exc=None):
        def _op(tensor, peer=None, **kwargs):
            self.calls.append((name, tensor, peer, kwargs))
            return _FakeWork(name, done=done, fail_exc=fail_exc)
        _op.__name__ = name
        return _op


def _p2p(op, tensor, peer=None, group=None, tag=0, group_peer=None):
    from candle.distributed._p2p import P2POp
    return P2POp(op, tensor, peer=peer, group=group, tag=tag, group_peer=group_peer)


def test_batch_returns_single_work_handle():
    from candle.distributed._p2p import batch_isend_irecv

    rec = _Recorder()
    ops = [
        _p2p(rec.make_op("isend"), "t0", peer=1),
        _p2p(rec.make_op("irecv"), "t1", peer=2),
    ]

    work = batch_isend_irecv(ops)

    assert hasattr(work, "wait")
    assert hasattr(work, "is_completed")
    assert hasattr(work, "get_future")


def test_batch_dispatches_all_ops_before_wait():
    from candle.distributed._p2p import batch_isend_irecv

    rec = _Recorder()
    ops = [
        _p2p(rec.make_op("isend"), "t0", peer=1),
        _p2p(rec.make_op("irecv"), "t1", peer=2),
        _p2p(rec.make_op("isend"), "t2", peer=3),
    ]

    work = batch_isend_irecv(ops)

    assert [c[0] for c in rec.calls] == ["isend", "irecv", "isend"]
    assert work.is_completed() is False


def test_batch_wait_waits_all_children():
    from candle.distributed._p2p import batch_isend_irecv

    rec = _Recorder()
    op1 = rec.make_op("isend")
    op2 = rec.make_op("irecv")
    work = batch_isend_irecv([
        _p2p(op1, "t0", peer=1),
        _p2p(op2, "t1", peer=2),
    ])

    assert work.wait() is True
    assert work.is_completed() is True


def test_batch_completed_only_when_all_children_done():
    from candle.distributed._p2p import batch_isend_irecv

    rec = _Recorder()
    work = batch_isend_irecv([
        _p2p(rec.make_op("isend", done=True), "t0", peer=1),
        _p2p(rec.make_op("irecv", done=False), "t1", peer=2),
    ])

    assert work.is_completed() is False
    work.wait()
    assert work.is_completed() is True


def test_batch_exception_propagates_from_child_wait():
    from candle.distributed._p2p import batch_isend_irecv

    rec = _Recorder()
    boom = RuntimeError("recv failed")
    work = batch_isend_irecv([
        _p2p(rec.make_op("isend"), "t0", peer=1),
        _p2p(rec.make_op("irecv", fail_exc=boom), "t1", peer=2),
    ])

    with pytest.raises(RuntimeError, match="recv failed"):
        work.wait()


def test_batch_future_resolves_after_wait():
    from candle.distributed._p2p import batch_isend_irecv

    rec = _Recorder()
    work = batch_isend_irecv([
        _p2p(rec.make_op("isend"), "t0", peer=1),
        _p2p(rec.make_op("irecv"), "t1", peer=2),
    ])
    fut = work.get_future()
    assert fut.done() is False
    work.wait()
    assert fut.done() is True


def test_batch_group_peer_routes_to_group_dst_for_send():
    from candle.distributed._p2p import batch_isend_irecv

    rec = _Recorder()
    batch_isend_irecv([
        _p2p(rec.make_op("isend"), "t0", peer=None, group_peer=7),
    ])

    assert rec.calls[0][2] is None
    assert rec.calls[0][3]["group_dst"] == 7


def test_batch_group_peer_routes_to_group_src_for_recv():
    from candle.distributed._p2p import batch_isend_irecv

    rec = _Recorder()
    batch_isend_irecv([
        _p2p(rec.make_op("irecv"), "t0", peer=None, group_peer=5),
    ])

    assert rec.calls[0][2] is None
    assert rec.calls[0][3]["group_src"] == 5
