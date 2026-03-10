"""Distributed tensor module."""
from .placement import Placement, Shard, Replicate, Partial
from .dtensor import DTensor, DTensorSpec, TensorMeta

__all__ = [
    "Placement",
    "Shard",
    "Replicate",
    "Partial",
    "DTensor",
    "DTensorSpec",
    "TensorMeta",
]
