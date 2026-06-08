"""Common utilities and public policy classes for FSDP2."""
from abc import ABC
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """Mixed precision configuration for FSDP2.

    Mirrors ``torch.distributed.fsdp.MixedPrecisionPolicy`` at the Python API
    layer while using Candle dtypes at runtime.
    """
    param_dtype: Optional[object] = None
    reduce_dtype: Optional[object] = None
    output_dtype: Optional[object] = None
    cast_forward_inputs: bool = True


@dataclass
class OffloadPolicy:
    """Base FSDP2 offload policy.

    The base policy means no offload, matching PyTorch FSDP2's default.  Candle
    accepts it for API compatibility; concrete offload policies are validated by
    ``fully_shard``.
    """


@dataclass
class CPUOffloadPolicy(OffloadPolicy):
    """CPU offload policy matching PyTorch FSDP2's public dataclass."""
    pin_memory: bool = True


class Comm(ABC):
    """Base custom communication interface matching PyTorch FSDP2."""

    def allocate(self, size, *, dtype, device):
        """Allocate a tensor for a custom communication implementation."""
        raise NotImplementedError


class AllGather(Comm):
    """Custom all-gather communication interface marker."""


class ReduceScatter(Comm):
    """Custom reduce-scatter communication interface marker."""


class FSDPMeshInfo:
    """Mesh information for FSDP parameter groups."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.shard_mesh_size = mesh.size(0)
        pg = mesh.get_group(0)
        self.shard_process_group = pg
        self.shard_mesh_rank = pg.rank() if pg is not None else 0


# Present the public API as torch.distributed.fsdp-compatible for code that
# introspects classes imported through ``candle.distributed.fsdp``.
_PUBLIC_MODULE = "torch.distributed.fsdp"
MixedPrecisionPolicy.__module__ = _PUBLIC_MODULE
OffloadPolicy.__module__ = _PUBLIC_MODULE
CPUOffloadPolicy.__module__ = _PUBLIC_MODULE
Comm.__module__ = _PUBLIC_MODULE
AllGather.__module__ = _PUBLIC_MODULE
ReduceScatter.__module__ = _PUBLIC_MODULE
