"""FileSystem storage backend for DCP save/load.

Mirrors ``torch.distributed.checkpoint.filesystem``.
Each rank writes a single ``.distcp`` file containing concatenated mini-zip
archives (one per tensor chunk).  The coordinator writes ``.metadata``.
"""

import io
import os
import pickle
import sys

import numpy as np

from ..._stream import PyTorchStreamReader, PyTorchStreamWriter
from ..._tensor import Tensor as MindTensor
from ..._C import typed_storage_from_numpy
from ..._dtype import (
    bool as mt_bool,
    bfloat16,
    complex64,
    complex128,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    to_numpy_dtype,
)

from .metadata import (
    Metadata,
    MetadataIndex,
    _StorageInfo,
    TensorStorageMetadata,
    _TORCH_TO_CANDLE,
)
from .storage import StorageWriter, StorageReader, WriteResult, _Future
from .planner import WriteItemType, LoadItemType

DEFAULT_SUFFIX = ".distcp"
_METADATA_FN = ".metadata"

# Map torch dtype strings to candle dtypes (for reading torch-saved checkpoints)
_TORCH_DTYPE_STR_TO_CANDLE = {
    "torch.float16": float16,
    "torch.float32": float32,
    "torch.float64": float64,
    "torch.bfloat16": bfloat16,
    "torch.int8": int8,
    "torch.int16": int16,
    "torch.int32": int32,
    "torch.int64": int64,
    "torch.uint8": uint8,
    "torch.bool": mt_bool,
    "torch.complex64": complex64,
    "torch.complex128": complex128,
}


# ---------------------------------------------------------------------------
# Unpickler that resolves torch.distributed.checkpoint.metadata classes
# ---------------------------------------------------------------------------

class _MetadataUnpickler(pickle.Unpickler):
    """Unpickler that maps torch DCP metadata classes to candle equivalents."""

    def find_class(self, module, name):
        key = (module, name)
        if key in _TORCH_TO_CANDLE:
            return _TORCH_TO_CANDLE[key]
        # torch.Size -> tuple
        if module == "torch" and name == "Size":
            return tuple
        # torch dtype objects: return candle dtype singletons
        _TORCH_DTYPE_SINGLETONS = {
            "float16": float16, "float32": float32, "float64": float64,
            "bfloat16": bfloat16, "int8": int8, "int16": int16,
            "int32": int32, "int64": int64, "uint8": uint8,
            "bool": mt_bool, "complex64": complex64, "complex128": complex128,
        }
        if module == "torch" and name in _TORCH_DTYPE_SINGLETONS:
            return _TORCH_DTYPE_SINGLETONS[name]
        # torch.serialization._get_layout -> return string representation
        if module == "torch.serialization" and name == "_get_layout":
            return _get_layout_stub
        # torch.distributed.checkpoint.metadata._MEM_FORMAT_ENCODING
        if name == "_MEM_FORMAT_ENCODING":
            return _MemFormatEncoding
        # collections.OrderedDict
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict  # pylint: disable=import-outside-toplevel
            return OrderedDict
        return super().find_class(module, name)


# Stubs for torch pickle compatibility
def _get_layout_stub(name):
    """Stand-in for torch.serialization._get_layout — returns the string name."""
    return name


import enum  # pylint: disable=wrong-import-position

class _MemFormatEncoding(enum.IntEnum):
    """Stand-in for torch.distributed.checkpoint.metadata._MEM_FORMAT_ENCODING."""
    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2


def _contiguous_strides(shape):
    """Compute C-contiguous strides for *shape*."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


# ---------------------------------------------------------------------------
# Helper: serialize a single tensor chunk into a mini-zip in memory
# ---------------------------------------------------------------------------

def _tensor_to_zip_bytes(tensor):
    """Serialize *tensor* as a mini PyTorch zip archive in a BytesIO buffer.

    If torch is available, uses ``torch.save`` to produce a fully compatible
    mini-zip.  Otherwise falls back to a raw-data-only archive that candle
    can read but torch cannot.

    Returns ``(bytes_data, nbytes)`` where *nbytes* is the raw tensor size.
    """
    if hasattr(tensor, "numpy"):
        arr = np.ascontiguousarray(tensor.detach().numpy())
    elif isinstance(tensor, np.ndarray):
        arr = np.ascontiguousarray(tensor)
    else:
        arr = np.ascontiguousarray(np.array(tensor))
    raw = arr.tobytes()

    try:
        return _tensor_to_zip_bytes_torch(arr), len(raw)
    except ImportError:
        pass

    # Fallback: raw-data archive (readable by candle, not by torch)
    buf = io.BytesIO()

    def _writer_func(data, n):
        if data is None:
            buf.seek(n, os.SEEK_CUR)
        else:
            buf.write(data)
        return n

    writer = PyTorchStreamWriter(_writer_func)
    writer.writeRecord("data/0", raw)
    writer.writeRecord("byteorder", sys.byteorder.encode("ascii"))
    writer.writeEndOfFile()
    return buf.getvalue(), len(raw)


def _tensor_to_zip_bytes_torch(arr):
    """Use torch.save to produce a torch-compatible mini-zip from a numpy array."""
    import torch as _torch  # pylint: disable=import-outside-toplevel
    t = _torch.from_numpy(arr)
    buf = io.BytesIO()
    _torch.save(t, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# FileSystemWriter
# ---------------------------------------------------------------------------

class FileSystemWriter(StorageWriter):
    """Writes DCP checkpoints to a filesystem directory."""

    def __init__(
        self,
        path,
        single_file_per_rank=True,
        sync_files=True,
        thread_count=1,
        overwrite=True,
    ):
        self.path = str(path)
        self.single_file_per_rank = single_file_per_rank
        self.sync_files = sync_files
        self.thread_count = thread_count
        self.overwrite = overwrite
        self._rank = 0
        self._checkpoint_id = None

    def reset(self, checkpoint_id=None):
        if checkpoint_id is not None:
            self.path = str(checkpoint_id)
        self._checkpoint_id = checkpoint_id

    def set_up_storage_writer(self, is_coordinator):
        os.makedirs(self.path, exist_ok=True)

    def prepare_local_plan(self, plan):
        return plan

    def prepare_global_plan(self, global_plan):
        return global_plan

    def set_rank(self, rank):
        """Set the rank for this writer (called by save())."""
        self._rank = rank

    def write_data(self, plan, planner):
        """Write all items in *plan* into a single .distcp file for this rank."""
        filename = f"__{self._rank}_0{DEFAULT_SUFFIX}"
        filepath = os.path.join(self.path, filename)
        results = []
        # Concatenate mini-zips into a single file, tracking offsets
        with open(filepath, "wb") as fh:
            for item in plan.items:
                if item.type in (WriteItemType.TENSOR, WriteItemType.SHARD):
                    tensor = planner.resolve_data(item)
                    zip_bytes, raw_size = _tensor_to_zip_bytes(tensor)
                    offset = fh.tell()
                    fh.write(zip_bytes)
                    storage_info = _StorageInfo(
                        relative_path=filename,
                        offset=offset,
                        length=len(zip_bytes),
                    )
                    results.append(WriteResult(
                        index=item.index,
                        size_in_bytes=raw_size,
                        storage_data=storage_info,
                    ))
        return _Future(results)

    def finish(self, metadata, results):
        """Write .metadata pickle. Coordinator only."""
        # Populate storage_data from write results
        if metadata.storage_data is None:
            metadata.storage_data = {}
        for rank_results in results:
            for wr in rank_results:
                metadata.storage_data[wr.index] = wr.storage_data
        meta_path = os.path.join(self.path, _METADATA_FN)
        meta_bytes = _patch_metadata_for_torch(metadata)
        with open(meta_path, "wb") as f:
            f.write(meta_bytes)


def _patch_metadata_for_torch(metadata):
    """Pickle *metadata* so that both torch and candle can unpickle it.

    If torch is available, we build the metadata using torch's actual classes
    so pickle emits the correct module paths and types natively.
    Otherwise, fall back to simple module-path byte replacement.
    """
    try:
        return _build_torch_native_metadata(metadata)
    except ImportError:
        return _build_fallback_metadata(metadata)


def _build_torch_native_metadata(metadata):
    """Build metadata using torch's actual DCP classes and pickle it."""
    import torch  # pylint: disable=import-outside-toplevel
    from torch.distributed.checkpoint.metadata import (  # pylint: disable=import-outside-toplevel
        Metadata as TorchMetadata,
        TensorStorageMetadata as TorchTSM,
        TensorProperties as TorchTP,
        ChunkStorageMetadata as TorchCSM,
        MetadataIndex as TorchMI,
    )
    from torch.distributed.checkpoint.filesystem import (  # pylint: disable=import-outside-toplevel
        _StorageInfo as TorchSI,
    )
    from ..._dtype import DType as _DType

    _dtype_map = _get_torch_dtype_map()

    # Convert state_dict_metadata
    sd_meta = {}
    for fqn, sm in metadata.state_dict_metadata.items():
        if not hasattr(sm, 'properties'):
            continue
        dtype = sm.properties.dtype
        if isinstance(dtype, _DType):
            dtype = _dtype_map.get(dtype.name, dtype)
        tp = TorchTP(
            dtype=dtype,
            layout=torch.strided,
            requires_grad=sm.properties.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=sm.properties.pin_memory,
        )
        chunks = [
            TorchCSM(offsets=torch.Size(c.offsets), sizes=torch.Size(c.sizes))
            for c in sm.chunks
        ]
        sd_meta[fqn] = TorchTSM(
            properties=tp,
            size=torch.Size(sm.size),
            chunks=chunks,
        )

    # Convert storage_data
    storage_data = {}
    for idx, info in (metadata.storage_data or {}).items():
        torch_idx = TorchMI(
            fqn=idx.fqn,
            offset=torch.Size(idx.offset) if idx.offset is not None else None,
            index=idx.index,
        )
        torch_info = TorchSI(
            relative_path=info.relative_path,
            offset=info.offset,
            length=info.length,
        )
        storage_data[torch_idx] = torch_info

    torch_meta = TorchMetadata(
        state_dict_metadata=sd_meta,
        planner_data=metadata.planner_data,
        storage_data=storage_data,
    )

    return pickle.dumps(torch_meta, protocol=2)


def _build_fallback_metadata(metadata):
    """Fallback: pickle with candle classes and replace module paths in bytes."""
    raw = pickle.dumps(metadata, protocol=2)
    local_mod = b"candle.distributed.checkpoint.metadata"
    torch_mod = b"torch.distributed.checkpoint.metadata"
    return raw.replace(local_mod, torch_mod)


def _get_torch_dtype_map():
    """Return a mapping of dtype name -> torch dtype object, or empty dict."""
    try:
        import torch  # pylint: disable=import-outside-toplevel
        return {
            "float16": torch.float16, "float32": torch.float32,
            "float64": torch.float64, "bfloat16": torch.bfloat16,
            "int8": torch.int8, "int16": torch.int16,
            "int32": torch.int32, "int64": torch.int64,
            "uint8": torch.uint8, "bool": torch.bool,
            "complex64": torch.complex64, "complex128": torch.complex128,
        }
    except ImportError:
        return {}


def _get_torch_layout(layout_str):
    """Convert layout string to torch.layout if torch is available."""
    try:
        import torch  # pylint: disable=import-outside-toplevel
        if layout_str == "torch.strided" or layout_str is torch.strided:
            return torch.strided
    except ImportError:
        pass
    return layout_str


def _get_torch_memory_format(fmt):
    """Convert memory_format string to torch _MEM_FORMAT_ENCODING if torch is available."""
    try:
        from torch.distributed.checkpoint.metadata import _MEM_FORMAT_ENCODING  # pylint: disable=import-outside-toplevel
        if fmt == "torch.contiguous_format" or str(fmt) == "torch.contiguous_format":
            return _MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT
        if fmt == "torch.channels_last" or str(fmt) == "torch.channels_last":
            return _MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST
        if fmt == "torch.preserve_format" or str(fmt) == "torch.preserve_format":
            return _MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT
    except ImportError:
        pass
    return fmt


# ---------------------------------------------------------------------------
# FileSystemReader
# ---------------------------------------------------------------------------

class FileSystemReader(StorageReader):
    """Reads DCP checkpoints from a filesystem directory."""

    def __init__(self, path):
        self.path = str(path)
        self._metadata = None
        self._file_cache = {}

    def reset(self, checkpoint_id=None):
        if checkpoint_id is not None:
            self.path = str(checkpoint_id)
        self._metadata = None
        # Close any cached file handles
        for fh in self._file_cache.values():
            fh.close()
        self._file_cache = {}

    def read_metadata(self):
        meta_path = os.path.join(self.path, _METADATA_FN)
        with open(meta_path, "rb") as f:
            self._metadata = _MetadataUnpickler(f).load()
        return self._metadata

    def set_up_storage_reader(self, metadata, is_coordinator):
        self._metadata = metadata

    def prepare_local_plan(self, plan):
        return plan

    def prepare_global_plan(self, global_plan):
        return global_plan

    def read_data(self, plan, planner):
        """Read tensor data for each ReadItem in *plan*."""
        for item in plan.items:
            if item.type == LoadItemType.TENSOR:
                self._read_tensor_item(item, planner)
        # Close cached handles
        for fh in self._file_cache.values():
            fh.close()
        self._file_cache = {}
        return _Future(None)

    def _read_tensor_item(self, item, planner):
        """Read a single tensor chunk from the .distcp file."""
        # Look up storage info from metadata — try exact match first,
        # then fall back to matching by (fqn, offset) ignoring index
        storage_info = self._metadata.storage_data.get(item.storage_index)
        if storage_info is None:
            for key, val in self._metadata.storage_data.items():
                if key.fqn == item.storage_index.fqn and key.offset == item.storage_index.offset:
                    storage_info = val
                    break
        if storage_info is None:
            raise RuntimeError(
                f"no storage_data entry for {item.storage_index.fqn!r} "
                f"offset={item.storage_index.offset}"
            )

        # Get tensor metadata for dtype
        tensor_meta = self._metadata.state_dict_metadata.get(item.storage_index.fqn)
        if tensor_meta is None:
            raise RuntimeError(
                f"no state_dict_metadata for {item.storage_index.fqn!r}"
            )

        dtype_str = str(tensor_meta.properties.dtype)
        candle_dtype = _TORCH_DTYPE_STR_TO_CANDLE.get(dtype_str)
        if candle_dtype is None:
            # Try direct lookup (candle dtype objects stringify differently)
            candle_dtype = _resolve_candle_dtype(tensor_meta.properties.dtype)
        if candle_dtype is None:
            raise TypeError(
                f"unsupported dtype in distcp checkpoint: {dtype_str}"
            )
        np_dtype = to_numpy_dtype(candle_dtype)

        # Read chunk bytes from .distcp file
        rel_path = storage_info.relative_path
        if rel_path not in self._file_cache:
            self._file_cache[rel_path] = open(
                os.path.join(self.path, rel_path), "rb",
            )
        fh = self._file_cache[rel_path]
        fh.seek(storage_info.offset)
        chunk_bytes = fh.read(storage_info.length)

        reader = PyTorchStreamReader(io.BytesIO(chunk_bytes))
        arr = None
        for rec in reader.getAllRecords():
            if rec.startswith("data/"):
                payload, _ = reader.getRecord(rec)
                arr = np.frombuffer(payload, dtype=np_dtype).copy()
                break
        if arr is None:
            raise RuntimeError(
                f"no data/ record in distcp chunk for {item.storage_index.fqn!r}"
            )

        # Reshape to chunk shape
        arr = arr.reshape(tuple(item.lengths))

        # Build a candle tensor from the numpy array
        storage = typed_storage_from_numpy(
            arr.ravel(), dtype=candle_dtype, device="cpu",
        )
        chunk_tensor = MindTensor(
            storage,
            tuple(item.lengths),
            _contiguous_strides(tuple(item.lengths)),
            requires_grad=bool(tensor_meta.properties.requires_grad),
        )

        # Commit: if chunk covers the full tensor, just commit directly.
        # Otherwise the planner handles slicing.
        target = planner.resolve_tensor(item)
        if target is not None and hasattr(target, "shape"):
            # Check if this is a sub-chunk
            offsets = item.dest_offsets
            if all(o == 0 for o in offsets) and tuple(item.lengths) == tuple(target.shape):
                planner.commit_tensor(item, chunk_tensor)
            else:
                # Multi-chunk: copy into the right slice of the target
                slices = tuple(
                    slice(o, o + s) for o, s in zip(offsets, item.lengths)
                )
                target_np = target.detach().numpy()
                target_np[slices] = arr
                planner.commit_tensor(item, target)
        else:
            planner.commit_tensor(item, chunk_tensor)


def _resolve_candle_dtype(dtype_obj):
    """Try to resolve a dtype object to a candle dtype."""
    # If it's already a candle DType, return it
    dtype_name = getattr(dtype_obj, "name", None) or str(dtype_obj)
    mapping = {
        "float16": "float16", "float32": "float32", "float64": "float64",
        "bfloat16": "bfloat16", "int8": "int8", "int16": "int16",
        "int32": "int32", "int64": "int64", "uint8": "uint8",
        "bool": "bool", "complex64": "complex64", "complex128": "complex128",
    }
    if dtype_name in mapping:
        from ..._dtype import DType as _DType  # pylint: disable=import-outside-toplevel
        return _DType(mapping[dtype_name])
    return None
