"""Common utilities for FSDP2."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MixedPrecisionPolicy:
    """Mixed precision configuration for FSDP2.

    Matches ``torch.distributed.fsdp.MixedPrecisionPolicy``.

    Attributes:
        param_dtype: Cast parameters to this dtype during forward/backward
            (e.g. bfloat16). ``None`` means no casting.
        reduce_dtype: Cast gradients to this dtype before reduce-scatter
            (e.g. float32). ``None`` means use the gradient's existing dtype.
        output_dtype: Cast forward output to this dtype. ``None`` means
            no casting.
    """
    param_dtype: Optional[object] = None   # candle.dtype
    reduce_dtype: Optional[object] = None
    output_dtype: Optional[object] = None


class FSDPMeshInfo:
    """Mesh information for FSDP parameter groups."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.shard_mesh_size = mesh.size(0)
        pg = mesh.get_group(0)
        self.shard_process_group = pg
        self.shard_mesh_rank = pg.rank() if pg is not None else 0
