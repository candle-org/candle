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
