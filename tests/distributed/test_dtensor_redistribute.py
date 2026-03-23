"""Tests for DTensor.redistribute() and DTensor.full_tensor().

These tests are single-rank (no real collectives).  They verify the
coordinate / placement book-keeping and the redistribute() interface
behave correctly before involving real multi-rank communication.
"""
import sys
import types
import pytest


# ---------------------------------------------------------------------------
# Minimal stub infrastructure so we can import candle.distributed.tensor
# without a live distributed environment.
# ---------------------------------------------------------------------------

def _patch_dist_internals(monkeypatch):
    """Patch only the internal helpers that distributed code calls."""
    import candle.distributed as dist
    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0, raising=False)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1, raising=False)

    def _all_gather_into_tensor(out, inp, group=None, async_op=False):
        import numpy as np
        # Single rank: just copy inp into out (world_size==1 means no-op gather)
        np.copyto(
            out._storage,
            inp._storage.reshape(out._storage.shape)
        )
    def _reduce_scatter_tensor(out, inp, op="sum", group=None, async_op=False):
        import numpy as np
        np.copyto(out._storage, inp._storage[:out._storage.size].reshape(out._storage.shape))
    def _all_reduce(tensor, op="sum", group=None, async_op=False):
        pass

    monkeypatch.setattr(dist, "all_gather_into_tensor", _all_gather_into_tensor, raising=False)
    monkeypatch.setattr(dist, "reduce_scatter_tensor", _reduce_scatter_tensor, raising=False)
    monkeypatch.setattr(dist, "all_reduce", _all_reduce, raising=False)


@pytest.fixture(autouse=True)
def patch_dist(monkeypatch):
    _patch_dist_internals(monkeypatch)
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mesh(size=1):
    """Return a DeviceMesh with a single dim of *size* (no real PG init)."""
    from candle.distributed.device_mesh import DeviceMesh
    mesh = object.__new__(DeviceMesh)
    mesh.device_type = "npu"
    mesh._mesh_shape = (size,)
    mesh.mesh_dim_names = ("tp",)
    mesh._dim_groups = []
    return mesh


def _candle_tensor(data):
    import candle
    import numpy as np
    return candle.tensor(np.array(data, dtype="float32"))


# ---------------------------------------------------------------------------
# DeviceMesh coordinate + slicing tests
# ---------------------------------------------------------------------------

class TestDeviceMeshCoordinate:
    def test_get_coordinate_returns_list(self):
        """mesh.get_coordinate() must return a list of ints, one per mesh dim."""
        mesh = _make_mesh(size=1)
        coord = mesh.get_coordinate()
        assert isinstance(coord, list)
        assert len(coord) == mesh.ndim

    def test_get_coordinate_value_rank0(self):
        """Rank 0 in a size-1 mesh sits at coordinate [0]."""
        mesh = _make_mesh(size=1)
        coord = mesh.get_coordinate()
        assert coord == [0]

    def test_getitem_by_dim_name(self):
        """mesh['tp'] should return a sub-mesh for that dimension."""
        mesh = _make_mesh(size=1)
        sub = mesh["tp"]
        assert sub.ndim == 1
        assert sub.size(0) == 1

    def test_getitem_by_int(self):
        """mesh[0] should return a sub-mesh for dimension 0."""
        mesh = _make_mesh(size=1)
        sub = mesh[0]
        assert sub.ndim == 1


# ---------------------------------------------------------------------------
# DTensor.redistribute() tests
# ---------------------------------------------------------------------------

class TestDTensorRedistribute:
    """Verify redistribute() changes placements correctly."""

    def _shard_dtensor(self):
        """Return a Shard(0) DTensor on a size-1 mesh."""
        from candle.distributed.tensor.dtensor import DTensor
        from candle.distributed.tensor.placement import Shard
        mesh = _make_mesh(size=1)
        local = _candle_tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
        return DTensor.from_local(local, mesh, [Shard(0)])

    def test_redistribute_shard_to_replicate(self):
        """Redistributing Shard(0) -> Replicate should return a DTensor."""
        from candle.distributed.tensor.dtensor import DTensor
        from candle.distributed.tensor.placement import Replicate
        dt = self._shard_dtensor()
        result = dt.redistribute(dt.device_mesh, [Replicate()])
        assert isinstance(result, DTensor)

    def test_redistribute_replicate_to_shard(self):
        """Redistributing Replicate -> Shard(0) should return a DTensor."""
        from candle.distributed.tensor.dtensor import DTensor
        from candle.distributed.tensor.placement import Replicate, Shard
        mesh = _make_mesh(size=1)
        local = _candle_tensor([[1.0, 2.0], [3.0, 4.0]])
        dt = DTensor.from_local(local, mesh, [Replicate()])
        result = dt.redistribute(mesh, [Shard(0)])
        assert isinstance(result, DTensor)

    def test_redistribute_noop(self):
        """Redistributing to the same placement is a no-op."""
        from candle.distributed.tensor.dtensor import DTensor
        from candle.distributed.tensor.placement import Replicate
        mesh = _make_mesh(size=1)
        local = _candle_tensor([[1.0, 2.0]])
        dt = DTensor.from_local(local, mesh, [Replicate()])
        result = dt.redistribute(mesh, [Replicate()])
        assert isinstance(result, DTensor)

    def test_redistribute_preserves_global_shape(self):
        """Redistributed DTensor must preserve the global (unsharded) shape."""
        from candle.distributed.tensor.placement import Shard, Replicate
        dt = self._shard_dtensor()
        global_shape_before = dt._spec.tensor_meta.shape
        result = dt.redistribute(dt.device_mesh, [Replicate()])
        assert result._spec.tensor_meta.shape == global_shape_before


# ---------------------------------------------------------------------------
# DTensor.full_tensor() tests
# ---------------------------------------------------------------------------

class TestDTensorFullTensor:
    def test_full_tensor_replicated(self):
        """full_tensor() on a Replicated DTensor returns a plain Tensor."""
        from candle.distributed.tensor.dtensor import DTensor
        from candle.distributed.tensor.placement import Replicate
        mesh = _make_mesh(size=1)
        local = _candle_tensor([[1.0, 2.0], [3.0, 4.0]])
        dt = DTensor.from_local(local, mesh, [Replicate()])
        ft = dt.full_tensor()
        assert not isinstance(ft, DTensor)
        assert ft.shape == (2, 2)

    def test_full_tensor_sharded(self):
        """full_tensor() on a Shard(0) DTensor returns a plain Tensor."""
        from candle.distributed.tensor.dtensor import DTensor
        from candle.distributed.tensor.placement import Shard
        mesh = _make_mesh(size=1)
        local = _candle_tensor([[1.0, 2.0], [3.0, 4.0]])
        dt = DTensor.from_local(local, mesh, [Shard(0)])
        ft = dt.full_tensor()
        assert not isinstance(ft, DTensor)
        assert ft.shape == (2, 2)
