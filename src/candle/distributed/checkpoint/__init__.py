"""Distributed checkpoint APIs available in candle.

Provides both the legacy model-level helpers (``get_state_dict``,
``set_state_dict``) and the full ``torch.distributed.checkpoint`` DCP API
(``save``, ``load``, planners, storage backends, metadata types).
"""

# Legacy single-file helpers (kept for backward compat)
from .state_dict import (
    get_state_dict,
    set_state_dict,
    save as _legacy_save,
    load as _legacy_load,
)

# DCP save / load entry points
from .state_dict_saver import save
from .state_dict_loader import load

# Metadata types
from .metadata import (
    ChunkStorageMetadata,
    TensorProperties,
    TensorStorageMetadata,
    BytesStorageMetadata,
    StorageMeta,
    Metadata,
    MetadataIndex,
)

# Storage ABCs + result
from .storage import StorageReader, StorageWriter, WriteResult

# FileSystem backend
from .filesystem import FileSystemReader, FileSystemWriter

# Planner interfaces + defaults
from .planner import (
    SavePlanner,
    LoadPlanner,
    DefaultSavePlanner,
    DefaultLoadPlanner,
    SavePlan,
    LoadPlan,
    ReadItem,
    WriteItem,
    WriteItemType,
    LoadItemType,
    TensorWriteData,
)

__all__ = [
    # Legacy helpers
    "get_state_dict",
    "set_state_dict",
    # DCP entry points
    "save",
    "load",
    # Metadata
    "ChunkStorageMetadata",
    "TensorProperties",
    "TensorStorageMetadata",
    "BytesStorageMetadata",
    "StorageMeta",
    "Metadata",
    "MetadataIndex",
    # Storage
    "StorageReader",
    "StorageWriter",
    "WriteResult",
    "FileSystemReader",
    "FileSystemWriter",
    # Planners
    "SavePlanner",
    "LoadPlanner",
    "DefaultSavePlanner",
    "DefaultLoadPlanner",
    "SavePlan",
    "LoadPlan",
    "ReadItem",
    "WriteItem",
    "WriteItemType",
    "LoadItemType",
    "TensorWriteData",
]
