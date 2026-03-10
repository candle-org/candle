"""Distributed tensor module."""
from .placement import Placement, Shard, Replicate, Partial

__all__ = ["Placement", "Shard", "Replicate", "Partial"]
