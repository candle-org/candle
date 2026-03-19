"""Torch-compatible serialization without runtime torch dependency.

This module implements a subset of the PyTorch zip checkpoint format that is
sufficient for common checkpoint paths (state_dict, optimizer state, nested
containers) while keeping candle save/load independent from torch
runtime imports.
"""

import io
import os
import pickle
import sys
from collections import OrderedDict

import numpy as np

from ._dtype import (
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
from ._storage import TypedStorage, _CPUUntypedStorage, typed_storage_from_numpy
from ._C import PyTorchFileReader, PyTorchFileWriter
from ._stream import PyTorchStreamReader, PyTorchStreamWriter
from ._tensor import Tensor as MindTensor


def _check_filelike_for_read(f):
    if isinstance(f, (str, os.PathLike)):
        return
    if not hasattr(f, "read"):
        raise AttributeError(
            "expected 'f' to be string, path, or a file-like object with a 'read' attribute"
        )


def _check_filelike_for_write(f):
    if isinstance(f, (str, os.PathLike)):
        return
    if not hasattr(f, "write"):
        raise AttributeError(
            "expected 'f' to be string, path, or a file-like object with a 'write' attribute"
        )


def _is_pathlike(f):
    return isinstance(f, (str, os.PathLike))


def _maybe_decode_ascii(value):
    if isinstance(value, bytes):
        return value.decode("ascii")
    return value




_ALLOWED_WEIGHTS_GLOBALS = {
    ("collections", "OrderedDict"),
    ("__builtin__", "set"),
    ("builtins", "set"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_tensor"),
    ("torch", "FloatStorage"),
    ("torch", "DoubleStorage"),
    ("torch", "HalfStorage"),
    ("torch", "BFloat16Storage"),
    ("torch", "LongStorage"),
    ("torch", "IntStorage"),
    ("torch", "ShortStorage"),
    ("torch", "CharStorage"),
    ("torch", "ByteStorage"),
    ("torch", "BoolStorage"),
    ("torch", "ComplexFloatStorage"),
    ("torch", "ComplexDoubleStorage"),
}


class _WeightsOnlyUnpickler(pickle.Unpickler):
    def find_class(self, mod_name, name):
        key = (mod_name, name)
        if key not in _ALLOWED_WEIGHTS_GLOBALS:
            raise pickle.UnpicklingError(
                f"weights_only reject global {mod_name}.{name}"
            )
        if mod_name == "collections" and name == "OrderedDict":
            return OrderedDict
        if name == "set" and mod_name in {"__builtin__", "builtins"}:
            return set
        if mod_name == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return _rebuild_tensor_v2
        if mod_name == "torch" and name in _STORAGE_NAME_TO_DTYPE:
            return globals()[name]
        raise pickle.UnpicklingError(f"weights_only unsupported global {mod_name}.{name}")


class _LegacyBridgeUnpickler(pickle.Unpickler):
    def find_class(self, mod_name, name):
        if mod_name == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return _rebuild_tensor_v2
        if mod_name == "torch" and name in _STORAGE_NAME_TO_DTYPE:
            return globals()[name]
        if mod_name == "collections" and name == "OrderedDict":
            return OrderedDict
        return super().find_class(mod_name, name)


def _make_unpickler_class(base_cls, pickle_module):
    # Build a dynamic subclass so custom pickle_module.Unpickler is honored.
    class _Unpickler(base_cls, pickle_module.Unpickler):
        pass

    return _Unpickler


def _build_zip_unpickler(data, *, pickle_module=pickle, weights_only=False, **pickle_load_args):
    base_cls = _WeightsOnlyUnpickler if weights_only else _TorchCompatUnpickler
    cls = _make_unpickler_class(base_cls, pickle_module)
    return cls(io.BytesIO(data), **pickle_load_args)
# Storage type proxies that get rewritten to torch globals in data.pkl.
class FloatStorage:  # pragma: no cover - marker type used by pickle
    pass


class DoubleStorage:  # pragma: no cover - marker type used by pickle
    pass


class HalfStorage:  # pragma: no cover - marker type used by pickle
    pass


class BFloat16Storage:  # pragma: no cover - marker type used by pickle
    pass


class LongStorage:  # pragma: no cover - marker type used by pickle
    pass


class IntStorage:  # pragma: no cover - marker type used by pickle
    pass


class ShortStorage:  # pragma: no cover - marker type used by pickle
    pass


class CharStorage:  # pragma: no cover - marker type used by pickle
    pass


class ByteStorage:  # pragma: no cover - marker type used by pickle
    pass


class BoolStorage:  # pragma: no cover - marker type used by pickle
    pass


class ComplexFloatStorage:  # pragma: no cover - marker type used by pickle
    pass


class ComplexDoubleStorage:  # pragma: no cover - marker type used by pickle
    pass


# Force global names emitted in pickle to be stable and patchable.
for _cls in (
    FloatStorage,
    DoubleStorage,
    HalfStorage,
    BFloat16Storage,
    LongStorage,
    IntStorage,
    ShortStorage,
    CharStorage,
    ByteStorage,
    BoolStorage,
    ComplexFloatStorage,
    ComplexDoubleStorage,
):
    _cls.__module__ = __name__


_DTYPE_NAME_TO_STORAGE = {
    "float32": FloatStorage,
    "float64": DoubleStorage,
    "float16": HalfStorage,
    "bfloat16": BFloat16Storage,
    "int64": LongStorage,
    "int32": IntStorage,
    "int16": ShortStorage,
    "int8": CharStorage,
    "uint8": ByteStorage,
    "bool": BoolStorage,
    "complex64": ComplexFloatStorage,
    "complex128": ComplexDoubleStorage,
}

_STORAGE_NAME_TO_DTYPE = {
    "FloatStorage": float32,
    "DoubleStorage": float64,
    "HalfStorage": float16,
    "BFloat16Storage": bfloat16,
    "LongStorage": int64,
    "IntStorage": int32,
    "ShortStorage": int16,
    "CharStorage": int8,
    "ByteStorage": uint8,
    "BoolStorage": mt_bool,
    "ComplexFloatStorage": complex64,
    "ComplexDoubleStorage": complex128,
}


class _StorageRef:
    __slots__ = ("storage_type", "key", "location", "numel", "raw_bytes")

    def __init__(self, storage_type, key, location, numel, raw_bytes):
        self.storage_type = storage_type
        self.key = str(key)
        self.location = location
        self.numel = int(numel)
        self.raw_bytes = raw_bytes


class _LegacyStorageView:
    __slots__ = ("storage", "base_offset")

    def __init__(self, storage, base_offset):
        self.storage = storage
        self.base_offset = int(base_offset)


class _TensorReduceProxy:
    __slots__ = ("storage_ref", "storage_offset", "size", "stride", "requires_grad")

    def __init__(self, storage_ref, storage_offset, size, stride, requires_grad):
        self.storage_ref = storage_ref
        self.storage_offset = int(storage_offset)
        self.size = tuple(size)
        self.stride = tuple(stride)
        self.requires_grad = bool(requires_grad)

    def __reduce_ex__(self, _protocol):
        # Match torch tensor pickle path: _rebuild_tensor_v2(storage, offset, size, stride, ...)
        return (
            _rebuild_tensor_v2,
            (
                self.storage_ref,
                self.storage_offset,
                self.size,
                self.stride,
                self.requires_grad,
                OrderedDict(),
            ),
        )


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, _backward_hooks, _metadata=None):
    offset = int(storage_offset)
    if isinstance(storage, _LegacyStorageView):
        offset += storage.base_offset
        storage = storage.storage

    tensor = MindTensor(
        storage,
        tuple(size),
        tuple(stride),
        offset=offset,
        requires_grad=bool(requires_grad),
    )
    if not requires_grad:
        tensor.grad_fn = None
    return tensor


def _tensor_to_proxy(tensor, storage_refs_by_id):
    source_location = str(tensor.device)
    cpu_tensor = tensor.detach().to("cpu") if tensor.device.type != "cpu" else tensor.detach()
    storage = cpu_tensor.storage()
    untyped = storage.untyped_storage()
    storage_id = id(untyped)
    storage_ref = storage_refs_by_id.get(storage_id)
    if storage_ref is None:
        storage_type = _DTYPE_NAME_TO_STORAGE.get(cpu_tensor.dtype.name)
        if storage_type is None:
            raise TypeError(f"unsupported dtype for serialization: {cpu_tensor.dtype}")
        raw = np.ascontiguousarray(storage.data).tobytes()
        storage_ref = _StorageRef(
            storage_type=storage_type,
            key=str(len(storage_refs_by_id)),
            location=source_location,
            numel=int(storage.size()),
            raw_bytes=raw,
        )
        storage_refs_by_id[storage_id] = storage_ref
    return _TensorReduceProxy(
        storage_ref=storage_ref,
        storage_offset=int(cpu_tensor.offset),
        size=tuple(int(s) for s in cpu_tensor.shape),
        stride=tuple(int(s) for s in cpu_tensor.stride),
        requires_grad=bool(tensor.requires_grad),
    )


def _prepare_for_pickle(obj, storage_refs_by_id):
    if isinstance(obj, MindTensor):
        return _tensor_to_proxy(obj, storage_refs_by_id)
    if isinstance(obj, OrderedDict):
        return OrderedDict((k, _prepare_for_pickle(v, storage_refs_by_id)) for k, v in obj.items())
    if isinstance(obj, dict):
        return {k: _prepare_for_pickle(v, storage_refs_by_id) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_prepare_for_pickle(v, storage_refs_by_id) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_prepare_for_pickle(v, storage_refs_by_id) for v in obj)
    if isinstance(obj, set):
        return {_prepare_for_pickle(v, storage_refs_by_id) for v in obj}
    return obj


def _persistent_id(obj):
    if isinstance(obj, _StorageRef):
        return ("storage", obj.storage_type, obj.key, obj.location, obj.numel)
    return None


def _patch_pickle_globals_for_torch(data_bytes):
    patched = data_bytes
    local_mod = __name__

    # Rebuild op must resolve to torch._utils on torch.load.
    patched = patched.replace(
        f"{local_mod}\n_rebuild_tensor_v2\n".encode("utf-8"),
        b"torch._utils\n_rebuild_tensor_v2\n",
    )

    # Storage classes must resolve to torch.*Storage.
    for storage_name in _STORAGE_NAME_TO_DTYPE.keys():
        patched = patched.replace(
            f"{local_mod}\n{storage_name}\n".encode("utf-8"),
            f"torch\n{storage_name}\n".encode("utf-8"),
        )

    return patched


def _write_zip_checkpoint(obj, f, pickle_module, pickle_protocol):
    storage_refs_by_id = {}
    prepared = _prepare_for_pickle(obj, storage_refs_by_id)

    data_buf = io.BytesIO()
    pickler = pickle_module.Pickler(data_buf, protocol=pickle_protocol)
    pickler.persistent_id = _persistent_id
    pickler.dump(prepared)
    data_pkl = _patch_pickle_globals_for_torch(data_buf.getvalue())

    if isinstance(f, (str, os.PathLike)):
        writer = PyTorchFileWriter(str(f))
        write_record = writer.write_record
        write_end_of_file = writer.write_end_of_file
    else:
        # file-like: wrap in a writer callable
        def _writer_func(buf, n):
            if buf is None:
                f.seek(n, os.SEEK_CUR)
            else:
                f.write(buf)
            return n
        writer = PyTorchStreamWriter(_writer_func)
        write_record = writer.writeRecord
        write_end_of_file = writer.writeEndOfFile

    write_record("data.pkl", data_pkl)
    write_record("byteorder", sys.byteorder.encode("ascii"))

    refs = sorted(storage_refs_by_id.values(), key=lambda r: int(r.key))
    for ref in refs:
        write_record(f"data/{ref.key}", ref.raw_bytes)

    write_end_of_file()


def _storage_dtype_from_type(storage_type):
    name = getattr(storage_type, "__name__", None)
    if name is None:
        name = str(storage_type)
    dtype = _STORAGE_NAME_TO_DTYPE.get(name)
    if dtype is None:
        raise TypeError(f"unsupported storage type in checkpoint: {storage_type}")
    return dtype


class _TorchCompatUnpickler(pickle.Unpickler):
    def find_class(self, mod_name, name):
        if mod_name == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return _rebuild_tensor_v2
        if mod_name == "torch" and name in _STORAGE_NAME_TO_DTYPE:
            return globals()[name]
        if mod_name == "collections" and name == "OrderedDict":
            return OrderedDict
        return super().find_class(mod_name, name)




def _apply_map_location(storage, location, map_location):
    if map_location is None:
        return storage

    if isinstance(map_location, str):
        return storage

    if hasattr(map_location, "type"):
        return storage

    if isinstance(map_location, dict):
        mapped = map_location.get(location, location)
        if mapped not in ("cpu", None):
            raise RuntimeError(
                f"unsupported remapped location: {mapped}; target device is not available"
            )
        return storage

    if callable(map_location):
        remapped = map_location(storage, location)
        if remapped is None:
            return storage
        return remapped

    raise NotImplementedError(
        "candle.load supports map_location=None, string/device, dict, or callable for torch zip checkpoints"
    )



def _resolve_storage_location(location, map_location, storage=None):
    if map_location is None:
        return location
    if isinstance(map_location, str):
        return map_location
    if hasattr(map_location, "type"):
        if getattr(map_location, "index", None) is None:
            return str(map_location.type)
        return f"{map_location.type}:{map_location.index}"
    if isinstance(map_location, dict):
        return map_location.get(location, location)
    if callable(map_location):
        remapped = map_location(storage, location)
        if remapped is None:
            return location
        if isinstance(remapped, str):
            return remapped
        remap_device = getattr(remapped, "device", None)
        if remap_device is not None:
            return str(remap_device)
        return location
    return location


def _validate_resolved_location(resolved_location):
    if resolved_location in ("cpu", None):
        return
    raise RuntimeError(
        f"unsupported checkpoint storage location: {resolved_location}; target device is not available"
    )


def _validate_map_location(map_location):
    if map_location is None:
        return
    if isinstance(map_location, str):
        return
    if hasattr(map_location, "type"):
        return
    if isinstance(map_location, dict):
        return
    if callable(map_location):
        return
    raise NotImplementedError(
        "candle.load supports map_location=None, string/device, dict, or callable for torch zip checkpoints"
    )




def _load_zip_checkpoint(
    file_obj,
    map_location=None,
    pickle_module=pickle,
    weights_only=False,
    mmap=False,
    mmap_path=None,
    **pickle_load_args,
):
    _validate_map_location(map_location)

    loaded_storages = {}
    if isinstance(file_obj, (str, os.PathLike)):
        reader = PyTorchFileReader(str(file_obj))
    else:
        reader = PyTorchStreamReader(file_obj)

    if not reader.hasRecord("data.pkl"):
        raise RuntimeError("checkpoint missing data.pkl record")
    data_pkl, _ = reader.getRecord("data.pkl")

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        if typename != "storage":
            raise RuntimeError(
                f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
            )

        storage_type, key, location, numel = saved_id[1:]
        key = _maybe_decode_ascii(key)
        location = _maybe_decode_ascii(location)

        if key in loaded_storages:
            return loaded_storages[key]

        dtype = _storage_dtype_from_type(storage_type)
        np_dtype = to_numpy_dtype(dtype)
        record_name = f"data/{key}"
        if mmap and mmap_path is not None:
            if reader.isRecordCompressed(record_name):
                raise RuntimeError(
                    "mmap requires uncompressed zip storage records"
                )
            byte_offset = reader.getRecordOffset(record_name)
            nbytes = int(numel) * np.dtype(np_dtype).itemsize
            raw = np.memmap(
                mmap_path,
                mode="r",
                dtype=np.uint8,
                offset=int(byte_offset),
                shape=(nbytes,),
            )
            arr = raw.view(np_dtype)
            untyped = _CPUUntypedStorage(raw, device="cpu")
            storage = TypedStorage(untyped, dtype=dtype, size=int(numel), data=arr)
        else:
            payload, _ = reader.getRecord(record_name)
            arr = np.frombuffer(payload, dtype=np_dtype, count=int(numel)).copy()
            storage = typed_storage_from_numpy(arr, dtype=dtype, device="cpu")

        resolved_location = _resolve_storage_location(location, map_location)
        _validate_resolved_location(resolved_location)

        storage = _apply_map_location(storage, location, map_location)
        loaded_storages[key] = storage
        return storage

    unpickler = _build_zip_unpickler(
        data_pkl,
        pickle_module=pickle_module,
        weights_only=weights_only,
        **pickle_load_args,
    )
    unpickler.persistent_load = persistent_load
    return unpickler.load()




def _legacy_element_size(dtype):
    return int(np.dtype(to_numpy_dtype(dtype)).itemsize)


def _load_legacy_checkpoint(
    file_obj,
    map_location=None,
    pickle_module=pickle,
    weights_only=False,
    **pickle_load_args,
):
    if map_location not in (None, "cpu"):
        raise NotImplementedError(
            "candle.load currently supports map_location=None or 'cpu' for legacy checkpoints"
        )

    deserialized_objects = {}

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        if typename == "module":
            return data[0]

        if typename == "storage":
            storage_type, root_key, location, numel, view_metadata = data
            root_key = _maybe_decode_ascii(root_key)
            location = _maybe_decode_ascii(location)
            resolved_location = _resolve_storage_location(location, map_location)
            _validate_resolved_location(resolved_location)

            dtype = _storage_dtype_from_type(storage_type)
            if root_key not in deserialized_objects:
                arr = np.empty(int(numel), dtype=to_numpy_dtype(dtype))
                deserialized_objects[root_key] = typed_storage_from_numpy(arr, dtype=dtype, device="cpu")
            root_storage = deserialized_objects[root_key]

            if view_metadata is not None:
                view_key, offset, _view_size = view_metadata
                view_key = _maybe_decode_ascii(view_key)
                if view_key not in deserialized_objects:
                    deserialized_objects[view_key] = _LegacyStorageView(root_storage, int(offset))
                return deserialized_objects[view_key]
            return root_storage

        raise RuntimeError(f"Unknown saved id type: {saved_id[0]}")

    magic_number = pickle_module.load(file_obj, **pickle_load_args)
    if magic_number != 0x1950A86A20F9469CFC6C:
        raise RuntimeError("Invalid magic number; corrupt file?")

    protocol_version = pickle_module.load(file_obj, **pickle_load_args)
    if protocol_version != 1001:
        raise RuntimeError(f"Invalid protocol version: {protocol_version}")

    _ = pickle_module.load(file_obj, **pickle_load_args)

    base_unpickler_cls = _WeightsOnlyUnpickler if weights_only else _LegacyBridgeUnpickler
    unpickler_cls = _make_unpickler_class(base_unpickler_cls, pickle_module)
    unpickler = unpickler_cls(file_obj, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    deserialized_storage_keys = pickle_module.load(file_obj, **pickle_load_args)

    for key in deserialized_storage_keys:
        key = _maybe_decode_ascii(key)
        storage = deserialized_objects[key]
        if isinstance(storage, _LegacyStorageView):
            storage = storage.storage

        # Legacy stream stores an 8-byte record header before each raw storage payload.
        header = file_obj.read(8)
        if len(header) != 8:
            raise RuntimeError("corrupt legacy checkpoint: missing storage record header")

        nbytes = storage.nbytes()
        payload = file_obj.read(nbytes)
        if len(payload) != nbytes:
            raise RuntimeError("corrupt legacy checkpoint: truncated storage payload")
        arr = np.frombuffer(payload, dtype=storage.data.dtype, count=storage.size()).copy()
        storage.data[:] = arr

    return result


# ── torch.distributed.checkpoint (.distcp) support ───────────────────

# Map torch dtype strings to candle dtypes
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


def _is_distcp_dir(path):
    """Return True if *path* is a DCP checkpoint directory."""
    if not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, ".metadata"))


def _load_distcp_checkpoint(checkpoint_dir, map_location=None):
    """Load a ``torch.distributed.checkpoint`` directory into a state_dict.

    Each ``.distcp`` file contains concatenated zip archives (one per tensor
    chunk).  The ``.metadata`` pickle describes the layout.
    """
    meta_path = os.path.join(checkpoint_dir, ".metadata")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Build fqn -> (storage_info, tensor_metadata) mapping
    fqn_to_info = {}
    for idx, storage_info in meta.storage_data.items():
        fqn_to_info.setdefault(idx.fqn, []).append((idx, storage_info))

    state_dict = {}
    # Cache open file handles for .distcp files
    file_cache = {}

    try:
        for fqn, tensor_meta in meta.state_dict_metadata.items():
            dtype_str = str(tensor_meta.properties.dtype)
            candle_dtype = _TORCH_DTYPE_STR_TO_CANDLE.get(dtype_str)
            if candle_dtype is None:
                raise TypeError(f"unsupported dtype in distcp checkpoint: {dtype_str}")
            np_dtype = to_numpy_dtype(candle_dtype)
            shape = tuple(tensor_meta.size)

            if fqn not in fqn_to_info:
                raise RuntimeError(f"no storage_data entry for {fqn!r}")

            # For single-chunk tensors (most common case)
            chunks = fqn_to_info[fqn]
            if len(chunks) == 1:
                _idx, info = chunks[0]
                arr = _read_distcp_chunk(
                    checkpoint_dir, info, np_dtype, file_cache,
                )
                arr = arr.reshape(shape)
            else:
                # Multi-chunk: allocate full tensor, fill each chunk
                arr = np.empty(shape, dtype=np_dtype)
                for _idx, info in chunks:
                    chunk_offsets = tuple(_idx.offset)
                    chunk_sizes = tuple(
                        next(
                            c.sizes
                            for c in tensor_meta.chunks
                            if tuple(c.offsets) == chunk_offsets
                        )
                    )
                    chunk_data = _read_distcp_chunk(
                        checkpoint_dir, info, np_dtype, file_cache,
                    )
                    chunk_data = chunk_data.reshape(tuple(chunk_sizes))
                    slices = tuple(
                        slice(o, o + s) for o, s in zip(chunk_offsets, chunk_sizes)
                    )
                    arr[slices] = chunk_data

            storage = typed_storage_from_numpy(
                arr.ravel(), dtype=candle_dtype, device="cpu",
            )
            tensor = MindTensor(
                storage, shape, _contiguous_strides(shape),
                requires_grad=bool(tensor_meta.properties.requires_grad),
            )
            state_dict[fqn] = tensor
    finally:
        for fh in file_cache.values():
            fh.close()

    return state_dict


def _read_distcp_chunk(checkpoint_dir, storage_info, np_dtype, file_cache):
    """Read one tensor chunk from a .distcp file."""
    rel_path = storage_info.relative_path
    if rel_path not in file_cache:
        file_cache[rel_path] = open(
            os.path.join(checkpoint_dir, rel_path), "rb",
        )
    fh = file_cache[rel_path]
    fh.seek(storage_info.offset)
    chunk_bytes = fh.read(storage_info.length)

    reader = PyTorchStreamReader(io.BytesIO(chunk_bytes))
    # Each chunk zip has a single storage in data/0
    for rec in reader.getAllRecords():
        if rec.startswith("data/"):
            payload, _ = reader.getRecord(rec)
            return np.frombuffer(payload, dtype=np_dtype).copy()

    raise RuntimeError(
        f"no data/ record found in distcp chunk at offset {storage_info.offset}"
    )


def _contiguous_strides(shape):
    """Compute C-contiguous strides for *shape*."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _is_zip_checkpoint(file_obj):
    """Check for ZIP local-file-header magic (PK\\x03\\x04) without importing zipfile."""
    try:
        cur = file_obj.tell()
    except Exception:
        cur = None

    try:
        magic = file_obj.read(4)
        return magic == b'PK\x03\x04'
    except Exception:
        return False
    finally:
        if cur is not None:
            try:
                file_obj.seek(cur)
            except Exception:
                pass


def _coerce_map_location_arg(map_location):
    if map_location in (None, "cpu"):
        return map_location
    if isinstance(map_location, dict) or callable(map_location):
        return map_location

    # Accept torch.device(cpu) style objects without importing torch.
    device_type = getattr(map_location, "type", None)
    if device_type is not None and str(device_type) == "cpu":
        return "cpu"

    return map_location


def save(obj, f, pickle_module=pickle, pickle_protocol=2, **kwargs):
    """Save object in torch-compatible zip checkpoint format without torch import."""
    _check_filelike_for_write(f)
    use_new_zipfile = kwargs.pop("_use_new_zipfile_serialization", True)
    if use_new_zipfile is False:
        raise NotImplementedError(
            "candle.save does not support _use_new_zipfile_serialization=False"
        )
    _ = kwargs

    if _is_pathlike(f):
        _write_zip_checkpoint(obj, str(f), pickle_module, pickle_protocol)
        return

    _write_zip_checkpoint(obj, f, pickle_module, pickle_protocol)


def load(f, map_location=None, pickle_module=pickle, *, weights_only=False, mmap=None, **kwargs):
    """Load checkpoint without torch runtime dependency.

    Supports torch zip checkpoints, the zip checkpoints produced by
    :func:`save`, ``torch.distributed.checkpoint`` directories (``.distcp``),
    and legacy pickle checkpoints.
    """
    _check_filelike_for_read(f)
    if pickle_module is None:
        pickle_module = pickle
    if mmap is None:
        mmap = False
    map_location = _coerce_map_location_arg(map_location)
    _ = pickle_module, weights_only, kwargs

    # DCP directory checkpoint
    if _is_pathlike(f) and _is_distcp_dir(str(f)):
        return _load_distcp_checkpoint(str(f), map_location=map_location)

    if mmap and not _is_pathlike(f):
        raise ValueError("f must be a string filename in order to use mmap argument")

    if _is_pathlike(f):
        with open(f, "rb") as fh:
            if _is_zip_checkpoint(fh):
                return _load_zip_checkpoint(
                    fh,
                    map_location=map_location,
                    pickle_module=pickle_module,
                    weights_only=weights_only,
                    mmap=bool(mmap),
                    mmap_path=str(f),
                    encoding="utf-8",
                )
            if mmap:
                raise RuntimeError(
                    "mmap can only be used with files saved with torch zip serialization"
                )
            return _load_legacy_checkpoint(
                fh,
                map_location=map_location,
                pickle_module=pickle_module,
                weights_only=weights_only,
                encoding="utf-8",
            )

    if _is_zip_checkpoint(f):
        return _load_zip_checkpoint(
            f,
            map_location=map_location,
            pickle_module=pickle_module,
            weights_only=weights_only,
            mmap=bool(mmap),
            encoding="utf-8",
        )
    return _load_legacy_checkpoint(
        f,
        map_location=map_location,
        pickle_module=pickle_module,
        weights_only=weights_only,
        encoding="utf-8",
    )
