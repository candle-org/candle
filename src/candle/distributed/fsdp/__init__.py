"""torch.distributed.fsdp public API for candle.

The legacy ``FullyShardedDataParallel`` wrapper remains unavailable, while the
composable FSDP2 Python surface is re-exported for PyTorch compatibility.
"""

from .._composable.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    UnshardHandle,
    fully_shard,
    register_fsdp_forward_method,
    share_comm_ctx,
)


class FullyShardedDataParallel:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("FSDP is not available in candle.")


class FullStateDictConfig:
    pass


class FullOptimStateDictConfig:
    pass


class StateDictType:
    FULL_STATE_DICT = 0


class ShardingStrategy:
    FULL_SHARD = 0
    SHARD_GRAD_OP = 1
    NO_SHARD = 2


__all__ = [
    "CPUOffloadPolicy",
    "FSDPModule",
    "FullOptimStateDictConfig",
    "FullStateDictConfig",
    "FullyShardedDataParallel",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "ShardingStrategy",
    "StateDictType",
    "UnshardHandle",
    "fully_shard",
    "register_fsdp_forward_method",
    "share_comm_ctx",
]


def __getattr__(name):
    raise AttributeError(
        f"module 'torch.distributed.fsdp' has no attribute '{name}'. "
        "FSDP is not available in candle."
    )
