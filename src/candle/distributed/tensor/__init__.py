"""Distributed tensor module."""
from .placement import Placement, Shard, Replicate, Partial
from .dtensor import DTensor, DTensorSpec, TensorMeta, compute_local_shape_and_global_offset
from .parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)

__all__ = [
    "Placement",
    "Shard",
    "Replicate",
    "Partial",
    "DTensor",
    "DTensorSpec",
    "TensorMeta",
    "compute_local_shape_and_global_offset",
    # Tensor Parallel
    "parallelize_module",
    "ColwiseParallel",
    "RowwiseParallel",
    "SequenceParallel",
]
