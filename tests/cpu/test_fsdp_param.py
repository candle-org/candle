"""Tests for FSDPParam and FSDPParamGroup (single-process, mock comm)."""
import candle as torch
import candle.nn as nn
from candle.distributed.tensor.placement import Shard
from candle.distributed.tensor.dtensor import DTensor


class MockMeshInfo:
    """Mock FSDPMeshInfo for single-process testing."""
    def __init__(self, rank=0, world_size=1):
        self.shard_mesh_rank = rank
        self.shard_mesh_size = world_size
        self.shard_process_group = None
        class _MockMesh:
            def __init__(self, ws):
                self._ws = ws
            def size(self, dim=0):
                return self._ws
            @property
            def ndim(self):
                return 1
        self.mesh = _MockMesh(world_size)


def test_fsdp_param_init_shards_parameter():
    """FSDPParam should shard a parameter and replace it with DTensor."""
    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    module = nn.Linear(8, 4)
    orig_shape = module.weight.shape
    mesh_info = MockMeshInfo(rank=0, world_size=1)
    param = FSDPParam(module.weight, module, "weight", mesh_info)
    assert isinstance(module.weight, DTensor)
    assert module.weight.placements == (Shard(0),)


def test_fsdp_param_unshard_reshard():
    """unshard should replace DTensor with plain Tensor, reshard should restore."""
    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam, ShardedState
    module = nn.Linear(8, 4)
    mesh_info = MockMeshInfo(rank=0, world_size=1)
    param = FSDPParam(module.weight, module, "weight", mesh_info)
    assert param._sharded_state == ShardedState.SHARDED
    assert isinstance(module.weight, DTensor)

    param._unshard_single_rank()
    assert param._sharded_state == ShardedState.UNSHARDED
    assert not isinstance(module.weight, DTensor)

    param.reshard()
    assert param._sharded_state == ShardedState.SHARDED
    assert isinstance(module.weight, DTensor)


def test_fsdp_param_padding_non_divisible():
    """FSDPParam should pad parameters whose dim-0 is not divisible by world_size."""
    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    # Simulate rank 0 of world_size=2 with dim-0 size 5 (not divisible by 2)
    module = nn.Linear(3, 5, bias=False)  # weight shape: (5, 3)
    orig_weight = module.weight.detach().clone()
    mesh_info = MockMeshInfo(rank=0, world_size=2)
    fp = FSDPParam(module.weight, module, "weight", mesh_info)
    # Shard should have padded size: ceil(5/2) = 3
    local_shard = module.weight.to_local()
    assert local_shard.shape[0] == 3, f"Expected shard dim-0=3, got {local_shard.shape[0]}"
    # Rank 0 gets first 3 rows of original
    for i in range(3):
        for j in range(3):
            assert abs(float(local_shard[i, j]) - float(orig_weight[i, j])) < 1e-6
    # Padding metadata should be stored
    assert fp._padded_dim_size == 1  # 6 - 5 = 1


def test_fsdp_param_padding_rank1():
    """Rank 1 of world_size=2 with non-divisible dim should get correct shard."""
    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    module = nn.Linear(3, 5, bias=False)  # weight shape: (5, 3)
    orig_weight = module.weight.detach().clone()
    mesh_info = MockMeshInfo(rank=1, world_size=2)
    fp = FSDPParam(module.weight, module, "weight", mesh_info)
    local_shard = module.weight.to_local()
    # Rank 1 gets chunk [3:6] of padded tensor, so shard dim-0 = 3
    assert local_shard.shape[0] == 3, f"Expected shard dim-0=3, got {local_shard.shape[0]}"
    # First 2 rows should be orig_weight[3:5], last row should be zeros (padding)
    for j in range(3):
        assert abs(float(local_shard[0, j]) - float(orig_weight[3, j])) < 1e-6
        assert abs(float(local_shard[1, j]) - float(orig_weight[4, j])) < 1e-6
        assert abs(float(local_shard[2, j])) < 1e-6  # padding zeros


def test_fsdp_param_group_lifecycle():
    """FSDPParamGroup should unshard/reshard all params together."""
    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    from candle.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
    module = nn.Linear(8, 4)
    mesh_info = MockMeshInfo(rank=0, world_size=1)
    fp_weight = FSDPParam(module.weight, module, "weight", mesh_info)
    fp_bias = FSDPParam(module.bias, module, "bias", mesh_info)
    group = FSDPParamGroup([fp_weight, fp_bias], module, mesh_info)
    assert not group._is_unsharded

    group.unshard()
    assert group._is_unsharded
    assert not isinstance(module.weight, DTensor)
    assert not isinstance(module.bias, DTensor)

    group.reshard()
    assert not group._is_unsharded
    assert isinstance(module.weight, DTensor)
    assert isinstance(module.bias, DTensor)
