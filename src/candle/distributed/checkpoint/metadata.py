"""Metadata dataclasses for ``torch.distributed.checkpoint`` compatibility.

These mirror the classes in ``torch.distributed.checkpoint.metadata`` so that
pickled ``.metadata`` files produced by either candle or PyTorch can be loaded
by the other.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Chunk / tensor property descriptors
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ChunkStorageMetadata:
    """Describes one contiguous chunk inside a sharded tensor."""
    offsets: Tuple[int, ...]
    sizes: Tuple[int, ...]


@dataclasses.dataclass
class TensorProperties:
    """Scalar properties that describe a stored tensor."""
    dtype: Any  # candle DType or torch.dtype (kept opaque for pickle compat)
    layout: str = "torch.strided"
    requires_grad: bool = False
    memory_format: str = "torch.contiguous_format"
    pin_memory: bool = False

    def __getstate__(self):
        """Emit tuple state matching torch's TensorProperties pickle format."""
        return (self.dtype, self.layout, self.requires_grad,
                self.memory_format, self.pin_memory)

    def __setstate__(self, state):
        """Handle both dict state and tuple state (torch)."""
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple):
            fields = [f.name for f in dataclasses.fields(self)]
            for name, val in zip(fields, state):
                object.__setattr__(self, name, val)
        else:
            raise TypeError(f"unexpected state type: {type(state)}")


@dataclasses.dataclass
class TensorStorageMetadata:
    """Metadata for a single (possibly sharded) tensor in the checkpoint."""
    properties: TensorProperties
    size: Tuple[int, ...]
    chunks: List[ChunkStorageMetadata] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class BytesStorageMetadata:
    """Metadata for a non-tensor (bytes) entry in the checkpoint."""


STORAGE_TYPES = Union[TensorStorageMetadata, BytesStorageMetadata]


# ---------------------------------------------------------------------------
# Top-level metadata written to ``.metadata``
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StorageMeta:
    """Optional provenance information attached to a checkpoint."""
    checkpoint_id: Optional[str] = None
    save_id: Optional[str] = None
    load_id: Optional[str] = None
    modules: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class Metadata:
    """Root object persisted as ``.metadata`` in a DCP checkpoint directory."""
    state_dict_metadata: Dict[str, STORAGE_TYPES] = dataclasses.field(default_factory=dict)
    planner_data: Optional[Any] = None
    storage_data: Optional[Dict[Any, Any]] = dataclasses.field(default_factory=dict)
    storage_meta: Optional[StorageMeta] = None
    version: str = "1"


# ---------------------------------------------------------------------------
# Index types used as keys in ``storage_data``
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MetadataIndex:
    """Hashable key that identifies a tensor chunk in ``storage_data``."""
    fqn: str
    offset: Optional[Tuple[int, ...]] = None
    index: Optional[int] = None

    def __hash__(self):
        return hash((self.fqn, self.offset, self.index))

    def __eq__(self, other):
        if not isinstance(other, MetadataIndex):
            return NotImplemented
        return (self.fqn, self.offset, self.index) == (other.fqn, other.offset, other.index)


@dataclasses.dataclass
class _StorageInfo:
    """Locates a tensor chunk inside a ``.distcp`` file."""
    relative_path: str
    offset: int
    length: int
    transform_descriptors: Optional[Any] = None


# ---------------------------------------------------------------------------
# Pickle compatibility: allow torch to unpickle candle-written .metadata
# ---------------------------------------------------------------------------
# When torch unpickles .metadata it looks for these classes under
# ``torch.distributed.checkpoint.metadata``.  We register our classes so
# that pickle resolves them under that module path as well.

_COMPAT_CLASSES = (
    ChunkStorageMetadata,
    TensorProperties,
    TensorStorageMetadata,
    BytesStorageMetadata,
    StorageMeta,
    Metadata,
    MetadataIndex,
    _StorageInfo,
)

# Build a registry that _MetadataUnpickler can use to resolve torch paths
# back to candle classes.
_TORCH_TO_CANDLE = {}
for _cls in _COMPAT_CLASSES:
    _torch_path = f"torch.distributed.checkpoint.metadata.{_cls.__name__}"
    _TORCH_TO_CANDLE[("torch.distributed.checkpoint.metadata", _cls.__name__)] = _cls
