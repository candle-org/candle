"""Tests for distributed tensor placement types."""
from candle.distributed.tensor.placement import Placement, Shard, Replicate, Partial


def test_shard_default_dim():
    s = Shard()
    assert s.dim == 0


def test_shard_custom_dim():
    s = Shard(1)
    assert s.dim == 1


def test_shard_is_placement():
    assert isinstance(Shard(0), Placement)


def test_replicate_is_placement():
    assert isinstance(Replicate(), Placement)


def test_partial_default_op():
    p = Partial()
    assert p.reduce_op == "sum"


def test_partial_is_placement():
    assert isinstance(Partial(), Placement)


def test_shard_equality():
    assert Shard(0) == Shard(0)
    assert Shard(0) != Shard(1)


def test_replicate_equality():
    assert Replicate() == Replicate()


def test_shard_repr():
    assert "Shard" in repr(Shard(0))
    assert "0" in repr(Shard(0))
