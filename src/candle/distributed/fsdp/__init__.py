"""torch.distributed.fsdp public API for candle.

Current public surface exposes composable FSDP2's fully_shard path while the
legacy FullyShardedDataParallel wrapper remains unavailable.
"""

from .._composable.fsdp import fully_shard


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


def __getattr__(name):
    raise AttributeError(
        f"module 'torch.distributed.fsdp' has no attribute '{name}'. "
        "FSDP is not available in candle."
    )
