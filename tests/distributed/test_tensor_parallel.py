"""Tests for minimal tensor parallel: ColwiseParallel, RowwiseParallel,
parallelize_module() for nn.Linear only.

Single-rank only -- no real collectives required.
"""
import sys
import types
import pytest


def _patch_dist_internals(monkeypatch):
    import candle.distributed as dist
    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0, raising=False)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1, raising=False)
    def _all_gather_into_tensor(out, inp, group=None, async_op=False):
        import numpy as np
        np.copyto(out._storage, inp._storage.reshape(out._storage.shape))
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


def _make_mesh(size=1):
    from candle.distributed.device_mesh import DeviceMesh
    mesh = object.__new__(DeviceMesh)
    mesh.device_type = "npu"
    mesh._mesh_shape = (size,)
    mesh.mesh_dim_names = ("tp",)
    mesh._dim_groups = []
    return mesh


def _linear(in_f=8, out_f=4, bias=True):
    import candle.nn as nn
    return nn.Linear(in_f, out_f, bias=bias)


# ---------------------------------------------------------------------------
# parallelize_module import smoke test
# ---------------------------------------------------------------------------

class TestParallelizeModuleImport:
    def test_import_parallelize_module(self):
        from candle.distributed.tensor.parallel import parallelize_module
        assert callable(parallelize_module)

    def test_import_colwise(self):
        from candle.distributed.tensor.parallel import ColwiseParallel
        assert ColwiseParallel is not None

    def test_import_rowwise(self):
        from candle.distributed.tensor.parallel import RowwiseParallel
        assert RowwiseParallel is not None


# ---------------------------------------------------------------------------
# ColwiseParallel
# ---------------------------------------------------------------------------

class TestColwiseParallel:
    def test_parallelize_returns_module(self):
        from candle.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        import candle.nn as nn
        mesh = _make_mesh()
        model = _linear(8, 4)
        result = parallelize_module(model, mesh, ColwiseParallel())
        assert isinstance(result, nn.Module)

    def test_colwise_weight_is_dtensor(self):
        from candle.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        from candle.distributed.tensor.dtensor import DTensor
        mesh = _make_mesh()
        model = _linear(8, 4)
        parallelize_module(model, mesh, ColwiseParallel())
        assert isinstance(model.weight, DTensor)

    def test_colwise_weight_shard_placement(self):
        from candle.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        from candle.distributed.tensor.dtensor import DTensor
        from candle.distributed.tensor.placement import Shard
        mesh = _make_mesh()
        model = _linear(8, 4)
        parallelize_module(model, mesh, ColwiseParallel())
        assert isinstance(model.weight, DTensor)
        assert any(isinstance(p, Shard) and p.dim == 0
                   for p in model.weight.placements)

    def test_colwise_bias_replicated(self):
        from candle.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        from candle.distributed.tensor.dtensor import DTensor
        mesh = _make_mesh()
        model = _linear(8, 4, bias=True)
        parallelize_module(model, mesh, ColwiseParallel())
        if model.bias is not None:
            assert isinstance(model.bias, DTensor)

    def test_colwise_no_bias(self):
        from candle.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        mesh = _make_mesh()
        model = _linear(8, 4, bias=False)
        parallelize_module(model, mesh, ColwiseParallel())


# ---------------------------------------------------------------------------
# RowwiseParallel
# ---------------------------------------------------------------------------

class TestRowwiseParallel:
    def test_parallelize_returns_module(self):
        from candle.distributed.tensor.parallel import parallelize_module, RowwiseParallel
        import candle.nn as nn
        mesh = _make_mesh()
        model = _linear(8, 4)
        result = parallelize_module(model, mesh, RowwiseParallel())
        assert isinstance(result, nn.Module)

    def test_rowwise_weight_is_dtensor(self):
        from candle.distributed.tensor.parallel import parallelize_module, RowwiseParallel
        from candle.distributed.tensor.dtensor import DTensor
        mesh = _make_mesh()
        model = _linear(8, 4)
        parallelize_module(model, mesh, RowwiseParallel())
        assert isinstance(model.weight, DTensor)

    def test_rowwise_weight_shard_placement(self):
        from candle.distributed.tensor.parallel import parallelize_module, RowwiseParallel
        from candle.distributed.tensor.dtensor import DTensor
        from candle.distributed.tensor.placement import Shard
        mesh = _make_mesh()
        model = _linear(8, 4)
        parallelize_module(model, mesh, RowwiseParallel())
        assert any(isinstance(p, Shard) and p.dim == 1
                   for p in model.weight.placements)

    def test_rowwise_no_bias(self):
        from candle.distributed.tensor.parallel import parallelize_module, RowwiseParallel
        mesh = _make_mesh()
        model = _linear(8, 4, bias=False)
        parallelize_module(model, mesh, RowwiseParallel())


# ---------------------------------------------------------------------------
# Non-Linear module raises NotImplementedError
# ---------------------------------------------------------------------------

class TestParallelizeModuleRaisesForNonLinear:
    def test_non_linear_raises(self):
        from candle.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        import candle.nn as nn
        mesh = _make_mesh()
        model = nn.Sequential(nn.Linear(4, 4))
        with pytest.raises(NotImplementedError):
            parallelize_module(model, mesh, ColwiseParallel())


# ---------------------------------------------------------------------------
# SequenceParallel is deferred
# ---------------------------------------------------------------------------

class TestSequenceParallelDeferred:
    def test_sequence_parallel_exists(self):
        from candle.distributed.tensor.parallel import SequenceParallel
        assert SequenceParallel is not None

    def test_sequence_parallel_raises_on_use(self):
        from candle.distributed.tensor.parallel import parallelize_module, SequenceParallel
        import candle.nn as nn
        mesh = _make_mesh()
        model = nn.Linear(4, 4)
        with pytest.raises(NotImplementedError):
            parallelize_module(model, mesh, SequenceParallel())


# ---------------------------------------------------------------------------
# candle.distributed.tensor top-level re-exports (Task 16 compatibility check)
# ---------------------------------------------------------------------------

class TestDistributedTensorTopLevelExports:
    """Verify TP symbols are re-exported from candle.distributed.tensor.

    These imports mirror the torch.distributed.tensor public surface that
    user code is expected to use.
    """

    def test_parallelize_module_via_tensor_namespace(self):
        from candle.distributed.tensor import parallelize_module
        assert callable(parallelize_module)

    def test_colwise_parallel_via_tensor_namespace(self):
        from candle.distributed.tensor import ColwiseParallel
        from candle.distributed.tensor.parallel import ColwiseParallel as _Direct
        assert ColwiseParallel is _Direct

    def test_rowwise_parallel_via_tensor_namespace(self):
        from candle.distributed.tensor import RowwiseParallel
        from candle.distributed.tensor.parallel import RowwiseParallel as _Direct
        assert RowwiseParallel is _Direct

    def test_sequence_parallel_via_tensor_namespace(self):
        from candle.distributed.tensor import SequenceParallel
        assert SequenceParallel is not None

    def test_all_tp_symbols_in_dunder_all(self):
        import candle.distributed.tensor as cdt
        for name in ("parallelize_module", "ColwiseParallel", "RowwiseParallel",
                     "SequenceParallel"):
            assert name in cdt.__all__, f"{name!r} missing from candle.distributed.tensor.__all__"
