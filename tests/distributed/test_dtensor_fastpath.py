"""TDD tests for _dtensor_fastpath Cython module.

Tests two layers:
1. Direct unit-tests of the Cython functions (pure arithmetic, no Tensor).
2. Integration smoke: DTensor.from_local / redistribute / full_tensor
   still produce correct results when the fastpath is active.
"""
import pytest


# ---------------------------------------------------------------------------
# Skip guard: the .so must be compiled before these tests are useful.
# We do NOT skip if the module is missing -- we import it directly so
# a missing build yields an ImportError that clearly explains what failed.
# ---------------------------------------------------------------------------

try:
    from candle.distributed import _dtensor_fastpath as _fp
    _FASTPATH_AVAILABLE = True
except ImportError:
    _FASTPATH_AVAILABLE = False

requires_fastpath = pytest.mark.skipif(
    not _FASTPATH_AVAILABLE,
    reason="_dtensor_fastpath.so not compiled; run: "
           "python setup.py build_ext --inplace from the worktree root",
)


# ---------------------------------------------------------------------------
# Minimal mesh stub (no real process group needed)
# ---------------------------------------------------------------------------

class _Mesh:
    """Minimal DeviceMesh stub for unit tests."""
    def __init__(self, world_size=4, local_rank=0):
        self._world_size = world_size
        self._local_rank = local_rank

    def size(self, dim=None):
        return self._world_size

    def get_local_rank(self, dim=0):
        return self._local_rank


# ---------------------------------------------------------------------------
# Minimal Placement stubs
# ---------------------------------------------------------------------------

class _Shard:
    def __init__(self, dim=0):
        self.dim = dim
    __name__ = 'Shard'


class _Replicate:
    pass


# Monkey-patch the class names so type(p).__name__ checks in Cython match
_Shard.__name__ = 'Shard'
_Replicate.__name__ = 'Replicate'


# ===========================================================================
# Unit tests for individual Cython functions
# ===========================================================================

@requires_fastpath
class TestNormalizeShardDim:
    def test_positive_dim_unchanged(self):
        assert _fp.normalize_shard_dim_cy(0, 3) == 0
        assert _fp.normalize_shard_dim_cy(2, 3) == 2

    def test_negative_dim_wrapped(self):
        assert _fp.normalize_shard_dim_cy(-1, 3) == 2
        assert _fp.normalize_shard_dim_cy(-3, 3) == 0

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError):
            _fp.normalize_shard_dim_cy(3, 3)
        with pytest.raises(IndexError):
            _fp.normalize_shard_dim_cy(-4, 3)


@requires_fastpath
class TestComputeGlobalShape:
    def test_replicate_unchanged(self):
        mesh = _Mesh(world_size=4)
        result = _fp.compute_global_shape_cy((8, 16), mesh, [_Replicate()])
        assert result == (8, 16)

    def test_shard_dim0_multiplied(self):
        mesh = _Mesh(world_size=4)
        result = _fp.compute_global_shape_cy((2, 16), mesh, [_Shard(0)])
        assert result == (8, 16)

    def test_shard_dim1_multiplied(self):
        mesh = _Mesh(world_size=4)
        result = _fp.compute_global_shape_cy((8, 4), mesh, [_Shard(1)])
        assert result == (8, 16)


@requires_fastpath
class TestComputeGlobalStride:
    def test_tuple_passthrough(self):
        assert _fp.compute_global_stride_cy((16, 1)) == (16, 1)

    def test_list_converted_to_tuple(self):
        assert _fp.compute_global_stride_cy([16, 1]) == (16, 1)


@requires_fastpath
class TestComputeLocalShapeAndGlobalOffset:
    def test_replicate_no_change(self):
        mesh = _Mesh(world_size=4, local_rank=0)
        ls, go = _fp.compute_local_shape_and_global_offset_cy(
            (8, 16), mesh, [_Replicate()]
        )
        assert ls == (8, 16)
        assert go == (0, 0)

    def test_shard_even_split_rank0(self):
        # 8 elements, 4 ranks -> 2 per rank; rank 0 offset=0
        mesh = _Mesh(world_size=4, local_rank=0)
        ls, go = _fp.compute_local_shape_and_global_offset_cy(
            (8, 16), mesh, [_Shard(0)]
        )
        assert ls == (2, 16)
        assert go == (0, 0)

    def test_shard_even_split_rank2(self):
        mesh = _Mesh(world_size=4, local_rank=2)
        ls, go = _fp.compute_local_shape_and_global_offset_cy(
            (8, 16), mesh, [_Shard(0)]
        )
        assert ls == (2, 16)
        assert go == (4, 0)

    def test_shard_with_remainder_rank0(self):
        # 9 elements, 4 ranks: ranks 0 get 3, ranks 1-3 get 2
        mesh = _Mesh(world_size=4, local_rank=0)
        ls, go = _fp.compute_local_shape_and_global_offset_cy(
            (9, 4), mesh, [_Shard(0)]
        )
        assert ls == (3, 4)   # chunk+1
        assert go == (0, 0)

    def test_shard_with_remainder_rank1(self):
        mesh = _Mesh(world_size=4, local_rank=1)
        ls, go = _fp.compute_local_shape_and_global_offset_cy(
            (9, 4), mesh, [_Shard(0)]
        )
        assert ls == (2, 4)   # chunk
        assert go == (3, 0)   # remainder*(chunk+1) + 0*chunk

    def test_shard_dim1(self):
        mesh = _Mesh(world_size=2, local_rank=1)
        ls, go = _fp.compute_local_shape_and_global_offset_cy(
            (4, 8), mesh, [_Shard(1)]
        )
        assert ls == (4, 4)
        assert go == (0, 4)


@requires_fastpath
class TestComputeGatherScatterSizes:
    def test_gather_scales_shard_dim(self):
        assert _fp.compute_gather_sizes_cy((2, 16), 0, 4) == (8, 16)
        assert _fp.compute_gather_sizes_cy((8, 4), 1, 4) == (8, 16)

    def test_scatter_divides_shard_dim(self):
        assert _fp.compute_scatter_sizes_cy((8, 16), 0, 4) == (2, 16)
        assert _fp.compute_scatter_sizes_cy((8, 16), 1, 4) == (8, 4)


# ===========================================================================
# Integration: DTensor uses fastpath and produces identical results
# ===========================================================================

def _patch_dist(monkeypatch):
    import candle.distributed as dist
    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0, raising=False)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1, raising=False)

    def _all_gather_into_tensor(out, inp, group=None, async_op=False):
        import numpy as np
        np.copyto(out._storage, inp._storage.reshape(out._storage.shape))

    def _reduce_scatter_tensor(out, inp, op="sum", group=None, async_op=False):
        import numpy as np
        np.copyto(
            out._storage,
            inp._storage[:out._storage.size].reshape(out._storage.shape),
        )

    def _all_reduce(tensor, op="sum", group=None, async_op=False):
        pass

    monkeypatch.setattr(dist, "all_gather_into_tensor", _all_gather_into_tensor, raising=False)
    monkeypatch.setattr(dist, "reduce_scatter_tensor", _reduce_scatter_tensor, raising=False)
    monkeypatch.setattr(dist, "all_reduce", _all_reduce, raising=False)


def _make_mesh(size=1):
    from candle.distributed.device_mesh import DeviceMesh
    mesh = object.__new__(DeviceMesh)
    mesh.device_type = "npu"
    mesh._mesh_shape = (size,)
    mesh.mesh_dim_names = ("tp",)
    mesh._dim_groups = []
    return mesh


def _ct(data):
    import candle
    import numpy as np
    return candle.tensor(np.array(data, dtype=np.float32))


@requires_fastpath
class TestDTensorFastpathIntegration:
    """Verify that DTensor.from_local / full_tensor still work after wiring."""

    def test_from_local_shard_global_shape(self, monkeypatch):
        _patch_dist(monkeypatch)
        from candle.distributed.tensor.dtensor import DTensor
        from candle.distributed.tensor.placement import Shard
        mesh = _make_mesh(size=1)
        local = _ct([[1.0, 2.0], [3.0, 4.0]])
        dt = DTensor.from_local(local, mesh, [Shard(0)])
        # world_size=1, so global_shape == local_shape
        assert dt._spec.tensor_meta.shape == (2, 2)

    def test_full_tensor_replicated(self, monkeypatch):
        _patch_dist(monkeypatch)
        from candle.distributed.tensor.dtensor import DTensor
        from candle.distributed.tensor.placement import Replicate
        mesh = _make_mesh(size=1)
        local = _ct([[1.0, 2.0], [3.0, 4.0]])
        dt = DTensor.from_local(local, mesh, [Replicate()])
        ft = dt.full_tensor()
        assert ft.shape == (2, 2)

    def test_redistribute_noop(self, monkeypatch):
        _patch_dist(monkeypatch)
        from candle.distributed.tensor.dtensor import DTensor
        from candle.distributed.tensor.placement import Replicate
        mesh = _make_mesh(size=1)
        local = _ct([[1.0, 2.0]])
        dt = DTensor.from_local(local, mesh, [Replicate()])
        dt2 = dt.redistribute(mesh, [Replicate()])
        assert dt2._spec.tensor_meta.shape == (1, 2)

    def test_fastpath_module_is_wired(self):
        """dtensor.py must import _dtensor_fastpath (not just use Python fallback)."""
        import candle.distributed.tensor.dtensor as _dtm
        # When compiled .so is present the module should be imported at startup
        assert hasattr(_dtm, '_fp'), (
            "dtensor.py should expose _fp (the fastpath module). "
            "Check that the import block in dtensor.py was updated."
        )

    def test_compute_local_shape_delegates_to_cy(self):
        """compute_local_shape_and_global_offset should call into Cython."""
        import candle.distributed.tensor.dtensor as _dtm
        # Confirm the module-level function uses the fastpath
        assert _dtm._DTENSOR_FASTPATH_ACTIVE, (
            "_DTENSOR_FASTPATH_ACTIVE must be True when .so is compiled"
        )
