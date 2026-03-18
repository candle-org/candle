"""Planner interfaces and default implementations for DCP save/load.

Mirrors ``torch.distributed.checkpoint.planner``.

The default planners support the Checkpointable protocol: if a tensor has
``__create_write_items__`` / ``__create_chunk_list__`` dunder methods
(e.g. DTensor), the planner delegates to them.
"""

import dataclasses
import enum
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple

from .metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
    BytesStorageMetadata,
    Metadata,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WriteItemType(enum.Enum):
    TENSOR = 0
    SHARD = 1
    BYTE_IO = 2


class LoadItemType(enum.Enum):
    TENSOR = 0
    BYTE_IO = 1


# ---------------------------------------------------------------------------
# Write-side data descriptors
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TensorWriteData:
    chunk: ChunkStorageMetadata
    properties: TensorProperties
    size: Tuple[int, ...]


@dataclasses.dataclass
class BytesIOWriteData:
    nbytes: int


# ---------------------------------------------------------------------------
# Items that flow through the planner pipeline
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class WriteItem:
    index: MetadataIndex
    type: WriteItemType
    tensor_data: Optional[TensorWriteData] = None
    bytes_io_data: Optional[BytesIOWriteData] = None


@dataclasses.dataclass
class ReadItem:
    type: LoadItemType
    dest_index: MetadataIndex
    dest_offsets: Tuple[int, ...]
    storage_index: MetadataIndex
    storage_offsets: Tuple[int, ...]
    lengths: Tuple[int, ...]


# ---------------------------------------------------------------------------
# Plans exchanged between planner <-> storage
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SavePlan:
    items: List[WriteItem] = dataclasses.field(default_factory=list)
    storage_data: Optional[Any] = None
    planner_data: Optional[Any] = None


@dataclasses.dataclass
class LoadPlan:
    items: List[ReadItem] = dataclasses.field(default_factory=list)
    storage_data: Optional[Any] = None
    planner_data: Optional[Any] = None


# ---------------------------------------------------------------------------
# SavePlanner ABC + default
# ---------------------------------------------------------------------------

class SavePlanner(ABC):
    @abstractmethod
    def set_up_planner(self, state_dict, is_coordinator):
        ...

    @abstractmethod
    def create_local_plan(self):
        ...

    @abstractmethod
    def create_global_plan(self, all_plans):
        ...

    @abstractmethod
    def finish_plan(self, new_plan):
        ...

    @abstractmethod
    def resolve_data(self, write_item):
        ...


class DefaultSavePlanner(SavePlanner):
    """One WriteItem per tensor key, resolves to contiguous tensor data.

    Supports the Checkpointable protocol: tensors with
    ``__create_write_items__`` (e.g. DTensor) produce SHARD WriteItems.
    Regular tensors produce a single full-tensor WriteItem.
    """

    def __init__(self):
        self.state_dict = None
        self.is_coordinator = False

    def set_up_planner(self, state_dict, is_coordinator=False):
        self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def create_local_plan(self):
        items = []
        for fqn, tensor in self.state_dict.items():
            if not hasattr(tensor, "shape"):
                continue
            # Checkpointable protocol: delegate to DTensor
            if hasattr(tensor, "__create_write_items__"):
                items.extend(tensor.__create_write_items__(fqn, tensor))
                continue
            # Regular tensor: full chunk
            shape = tuple(int(s) for s in tensor.shape)
            chunk = ChunkStorageMetadata(
                offsets=tuple(0 for _ in shape),
                sizes=shape,
            )
            props = TensorProperties(
                dtype=tensor.dtype,
                requires_grad=bool(tensor.requires_grad),
            )
            index = MetadataIndex(fqn, offset=chunk.offsets)
            items.append(WriteItem(
                index=index,
                type=WriteItemType.TENSOR,
                tensor_data=TensorWriteData(chunk=chunk, properties=props, size=shape),
            ))
        return SavePlan(items=items)

    def create_global_plan(self, all_plans):
        # Build metadata from all plans
        merged_metadata = {}
        for plan in all_plans:
            for item in plan.items:
                if item.tensor_data is not None:
                    fqn = item.index.fqn
                    td = item.tensor_data
                    if fqn not in merged_metadata:
                        merged_metadata[fqn] = TensorStorageMetadata(
                            properties=td.properties,
                            size=td.size,
                            chunks=[td.chunk],
                        )
                    else:
                        merged_metadata[fqn].chunks.append(td.chunk)
        metadata = Metadata(state_dict_metadata=merged_metadata)
        return all_plans, metadata

    def finish_plan(self, new_plan):
        return new_plan

    def resolve_data(self, write_item):
        tensor = self.state_dict[write_item.index.fqn]
        # DTensor: return local shard
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if hasattr(tensor, "contiguous"):
            tensor = tensor.contiguous()
        return tensor


# ---------------------------------------------------------------------------
# LoadPlanner ABC + default
# ---------------------------------------------------------------------------

class LoadPlanner(ABC):
    @abstractmethod
    def set_up_planner(self, state_dict, metadata, is_coordinator):
        ...

    @abstractmethod
    def create_local_plan(self):
        ...

    @abstractmethod
    def create_global_plan(self, all_plans):
        ...

    @abstractmethod
    def finish_plan(self, new_plan):
        ...

    @abstractmethod
    def load_bytes(self, read_item, value):
        ...

    @abstractmethod
    def resolve_tensor(self, read_item):
        ...

    @abstractmethod
    def commit_tensor(self, read_item, tensor):
        ...


class DefaultLoadPlanner(LoadPlanner):
    """One ReadItem per tensor key, resolves to pre-allocated tensor in state_dict.

    Supports the Checkpointable protocol: tensors with
    ``__create_chunk_list__`` (e.g. DTensor) declare which local chunks
    they need.  The planner matches those against saved chunks, supporting
    resharding (save N ranks -> load M ranks).
    """

    def __init__(self):
        self.state_dict = None
        self.metadata = None
        self.is_coordinator = False

    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self):
        items = []
        if self.metadata is None:
            return LoadPlan(items=items)
        for fqn, tensor_meta in self.metadata.state_dict_metadata.items():
            if fqn not in self.state_dict:
                continue
            if not isinstance(tensor_meta, TensorStorageMetadata):
                continue
            obj = self.state_dict[fqn]
            # Checkpointable protocol: get local chunks from DTensor
            if hasattr(obj, "__create_chunk_list__"):
                local_chunks = obj.__create_chunk_list__()
            else:
                # Full tensor: use all saved chunks
                local_chunks = tensor_meta.chunks
            items.extend(_create_read_items(fqn, tensor_meta, local_chunks))
        return LoadPlan(items=items)

    def create_global_plan(self, all_plans):
        return all_plans

    def finish_plan(self, new_plan):
        return new_plan

    def load_bytes(self, read_item, value):
        self.state_dict[read_item.dest_index.fqn] = value

    def resolve_tensor(self, read_item):
        obj = self.state_dict.get(read_item.dest_index.fqn)
        if obj is None:
            return None
        # DTensor: resolve to local shard
        if hasattr(obj, "to_local"):
            return obj.to_local()
        return obj

    def commit_tensor(self, read_item, tensor):
        fqn = read_item.dest_index.fqn
        obj = self.state_dict.get(fqn)
        # DTensor: data was written into local shard, don't replace
        if hasattr(obj, "to_local"):
            return
        self.state_dict[fqn] = tensor


# ---------------------------------------------------------------------------
# Resharding helpers
# ---------------------------------------------------------------------------

def _create_read_items(fqn, tensor_meta, local_chunks):
    """Match local chunks against saved chunks, creating ReadItems.

    Supports resharding: if a local chunk spans multiple saved chunks,
    creates multiple ReadItems for the overlapping regions.
    """
    items = []
    for local_chunk in local_chunks:
        for saved_chunk in tensor_meta.chunks:
            overlap = _chunk_overlap(local_chunk, saved_chunk)
            if overlap is None:
                continue
            dest_offsets, storage_offsets, lengths = overlap
            items.append(ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex(
                    fqn, offset=tuple(local_chunk.offsets),
                ),
                dest_offsets=dest_offsets,
                storage_index=MetadataIndex(
                    fqn, offset=tuple(saved_chunk.offsets),
                ),
                storage_offsets=storage_offsets,
                lengths=lengths,
            ))
    return items


def _chunk_overlap(local_chunk, saved_chunk):
    """Compute the overlap region between two chunks.

    Returns ``(dest_offsets, storage_offsets, lengths)`` or ``None``
    if there is no overlap.
    """
    ndim = len(local_chunk.offsets)
    dest_offsets = []
    storage_offsets = []
    lengths = []
    for dim in range(ndim):
        local_start = local_chunk.offsets[dim]
        local_end = local_start + local_chunk.sizes[dim]
        saved_start = saved_chunk.offsets[dim]
        saved_end = saved_start + saved_chunk.sizes[dim]

        overlap_start = max(local_start, saved_start)
        overlap_end = min(local_end, saved_end)
        if overlap_start >= overlap_end:
            return None

        dest_offsets.append(overlap_start - local_start)
        storage_offsets.append(overlap_start - saved_start)
        lengths.append(overlap_end - overlap_start)

    return tuple(dest_offsets), tuple(storage_offsets), tuple(lengths)
