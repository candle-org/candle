"""Public FSDP API tests.

These tests lock the public `candle.distributed.fsdp` surface while keeping
execution single-process / CPU-friendly.  They intentionally exercise the
public namespace rather than the internal `_composable.fsdp` path.
"""

import candle as torch
import candle.nn as nn
from candle.distributed.tensor.dtensor import DTensor


class MockMesh:
    """Single-process mock mesh for public FSDP API tests."""
    def __init__(self):
        self.device_type = "cpu"
        self._mesh_shape = (1,)
        self.mesh_dim_names = ("shard",)
        self._dim_groups = [None]

    def get_group(self, dim=0):
        return None

    def size(self, dim=0):
        return 1

    @property
    def ndim(self):
        return 1


class TinyMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def test_public_fully_shard_import_and_basic_use():
    """Public fully_shard import should work and shard module params."""
    from candle.distributed.fsdp import fully_shard

    model = TinyMLP(8)
    mesh = MockMesh()

    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    assert isinstance(model.fc1.weight, DTensor)
    assert isinstance(model.fc2.weight, DTensor)
    assert hasattr(model, "_fsdp_state")


def test_public_fully_shard_matches_internal_callable():
    """Public fully_shard should re-export the composable implementation."""
    from candle.distributed.fsdp import fully_shard as public_fully_shard
    from candle.distributed._composable.fsdp import fully_shard as internal_fully_shard

    assert public_fully_shard is internal_fully_shard


def test_public_fully_shard_forward_works():
    """Forward should work when using the public fsdp namespace."""
    from candle.distributed.fsdp import fully_shard

    model = TinyMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    x = torch.randn(2, 8)
    out = model(x)
    assert out.shape == (2, 8)


def test_public_fully_sharded_data_parallel_still_raises():
    """Legacy FSDP wrapper remains unavailable until explicitly implemented."""
    from candle.distributed.fsdp import FullyShardedDataParallel

    model = TinyMLP(8)
    try:
        FullyShardedDataParallel(model)
        assert False, "expected RuntimeError for legacy FSDP wrapper"
    except RuntimeError as exc:
        assert "FSDP is not available" in str(exc)
