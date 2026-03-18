"""Tests for the Cython-compiled candle.distributed._c10d module."""

import random
import pytest


# ---------------------------------------------------------------------------
# 1. test_cython_modules_importable
# ---------------------------------------------------------------------------

def test_cython_modules_importable():
    """All three Cython extension modules should be importable."""
    from candle.distributed import _c10d
    from candle.distributed import _c10d_gloo
    from candle.distributed import _c10d_hccl

    assert _c10d is not None
    assert _c10d_gloo is not None
    assert _c10d_hccl is not None


# ---------------------------------------------------------------------------
# 2. test_reduce_op_values
# ---------------------------------------------------------------------------

def test_reduce_op_values():
    """ReduceOp class attributes must match the expected integer values."""
    from candle.distributed._c10d import ReduceOp

    assert ReduceOp.SUM == 0
    assert ReduceOp.PRODUCT == 1
    assert ReduceOp.MAX == 2
    assert ReduceOp.MIN == 3
    assert ReduceOp.BAND == 4
    assert ReduceOp.BOR == 5
    assert ReduceOp.BXOR == 6
    assert ReduceOp.AVG == 7
    assert ReduceOp.PREMUL_SUM == 8
    assert ReduceOp.UNUSED == 9


# ---------------------------------------------------------------------------
# 3. test_reduce_op_equality
# ---------------------------------------------------------------------------

def test_reduce_op_equality():
    """ReduceOp equality works with ReduceOp instances and plain ints."""
    from candle.distributed._c10d import ReduceOp

    op_from_int = ReduceOp(0)
    op_from_attr = ReduceOp(ReduceOp.SUM)
    assert op_from_int == op_from_attr
    assert op_from_int == 0

    op_avg = ReduceOp(7)
    assert op_avg == ReduceOp(ReduceOp.AVG)
    assert op_avg == 7


# ---------------------------------------------------------------------------
# 4. test_reduce_op_hash
# ---------------------------------------------------------------------------

def test_reduce_op_hash():
    """ReduceOp instances with the same value produce the same hash."""
    from candle.distributed._c10d import ReduceOp

    a = ReduceOp(0)
    b = ReduceOp(ReduceOp.SUM)
    assert hash(a) == hash(b)

    # Different values should (in practice) have different hashes
    c = ReduceOp(7)
    assert hash(a) != hash(c)

    # Usable as dict keys
    d = {a: "sum"}
    assert d[b] == "sum"


# ---------------------------------------------------------------------------
# 5. test_reduce_op_repr
# ---------------------------------------------------------------------------

def test_reduce_op_repr():
    """ReduceOp repr contains the operation name."""
    from candle.distributed._c10d import ReduceOp

    r = repr(ReduceOp(0))
    assert "SUM" in r

    r = repr(ReduceOp(7))
    assert "AVG" in r


# ---------------------------------------------------------------------------
# 6. test_reduce_op_deprecated_alias
# ---------------------------------------------------------------------------

def test_reduce_op_deprecated_alias():
    """The deprecated reduce_op class aliases must match ReduceOp values."""
    from candle.distributed._c10d import ReduceOp, reduce_op

    assert reduce_op.SUM == ReduceOp.SUM
    assert reduce_op.PRODUCT == ReduceOp.PRODUCT
    assert reduce_op.MAX == ReduceOp.MAX
    assert reduce_op.MIN == ReduceOp.MIN


# ---------------------------------------------------------------------------
# 7. test_options_defaults
# ---------------------------------------------------------------------------

def test_options_defaults():
    """Options structs should have sensible default values."""
    from candle.distributed._c10d import AllreduceOptions, BroadcastOptions

    ar = AllreduceOptions()
    assert ar.reduceOp == 0
    assert ar.asyncOp is False

    bc = BroadcastOptions()
    assert bc.rootRank == 0
    assert bc.rootTensor == 0
    assert bc.asyncOp is False


# ---------------------------------------------------------------------------
# 8. test_options_custom
# ---------------------------------------------------------------------------

def test_options_custom():
    """Options structs accept custom values via constructor."""
    from candle.distributed._c10d import AllreduceOptions, BroadcastOptions

    bc = BroadcastOptions(rootRank=3)
    assert bc.rootRank == 3

    ar = AllreduceOptions(reduceOp=7)
    assert ar.reduceOp == 7


# ---------------------------------------------------------------------------
# 9. test_work_lifecycle
# ---------------------------------------------------------------------------

def test_work_lifecycle():
    """Work starts incomplete; wait() marks it completed and successful."""
    from candle.distributed._c10d import Work

    w = Work()
    assert not w.is_completed()

    w.wait()

    assert w.is_completed()
    assert w.is_success()


# ---------------------------------------------------------------------------
# 10. test_work_source_rank
# ---------------------------------------------------------------------------

def test_work_source_rank():
    """Work records the source_rank passed at construction."""
    from candle.distributed._c10d import Work

    w = Work(source_rank=5)
    assert w.source_rank() == 5

    w_default = Work()
    assert w_default.source_rank() == -1


# ---------------------------------------------------------------------------
# 11. test_work_result
# ---------------------------------------------------------------------------

def test_work_result():
    """Work.result() returns an empty list."""
    from candle.distributed._c10d import Work

    w = Work()
    assert w.result() == []


# ---------------------------------------------------------------------------
# 12. test_work_get_future
# ---------------------------------------------------------------------------

def test_work_get_future():
    """Work.get_future() returns a Future whose wait() returns []."""
    from candle.distributed._c10d import Work
    from candle.futures import Future

    w = Work()
    fut = w.get_future()
    assert isinstance(fut, Future)
    assert fut.wait() == []


# ---------------------------------------------------------------------------
# 13. test_hash_store_basic
# ---------------------------------------------------------------------------

def test_hash_store_basic():
    """HashStore set/get round-trip works."""
    from candle.distributed._c10d import HashStore

    store = HashStore()
    store.set("key1", b"value1")
    assert store.get("key1") == b"value1"


# ---------------------------------------------------------------------------
# 14. test_hash_store_overwrite
# ---------------------------------------------------------------------------

def test_hash_store_overwrite():
    """HashStore overwrites: the latest set() wins on get()."""
    from candle.distributed._c10d import HashStore

    store = HashStore()
    store.set("k", b"old")
    store.set("k", b"new")
    assert store.get("k") == b"new"


# ---------------------------------------------------------------------------
# 15. test_prefix_store
# ---------------------------------------------------------------------------

def test_prefix_store():
    """PrefixStore delegates to the underlying store with the prefix prepended."""
    from candle.distributed._c10d import HashStore, PrefixStore

    base = HashStore()
    ps = PrefixStore("myprefix", base)

    ps.set("key1", b"val1")
    # The underlying store should contain the prefixed key
    assert base.get("myprefix/key1") == b"val1"
    # And the PrefixStore should retrieve it transparently
    assert ps.get("key1") == b"val1"


# ---------------------------------------------------------------------------
# 16. test_tcp_store_roundtrip
# ---------------------------------------------------------------------------

def test_tcp_store_roundtrip():
    """TCPStore as master: set, get, and wait should work over TCP."""
    from candle.distributed._c10d import TCPStore

    port = random.randint(20000, 60000)
    store = TCPStore("127.0.0.1", port, 1, True, timeout=5.0)
    try:
        store.set("hello", b"world")
        assert store.get("hello") == b"world"

        store.set("another", b"value")
        # wait should return without error since both keys exist
        store.wait(["hello", "another"], timeout=2.0)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 17. test_process_group_basic
# ---------------------------------------------------------------------------

def test_process_group_basic():
    """ProcessGroup exposes rank, size, name, group_name, group_desc, bound_device_id."""
    from candle.distributed._c10d import ProcessGroup

    pg = ProcessGroup(rank=2, size=8)

    assert pg.rank() == 2
    assert pg.size() == 8
    assert pg.name() == ""
    assert pg.group_name == ""
    assert pg.group_desc == ""
    assert pg.bound_device_id is None
