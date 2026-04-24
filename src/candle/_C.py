# pylint: disable=import-error,no-name-in-module,possibly-unused-variable
import abc


def _get_PyTorchFileReader():
    from ._stream import PyTorchFileReader
    return PyTorchFileReader


def _get_PyTorchFileWriter():
    from ._stream import PyTorchFileWriter
    return PyTorchFileWriter


def _get_privateuse1_backend_name():
    return "npu"


class StorageBase(metaclass=abc.ABCMeta):
    """Minimal replacement for torch._C.StorageBase."""

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls.__name__ in ('StorageBase', 'UntypedStorage'):
            if hasattr(subclass, 'nbytes') and hasattr(subclass, 'data_ptr'):
                return True
        return NotImplemented

    def __getitem__(self, *args, **kwargs):
        raise NotImplementedError

    def __setitem__(self, *args, **kwargs):
        raise NotImplementedError

    def nbytes(self):
        raise NotImplementedError

    def data_ptr(self):
        raise NotImplementedError

    def copy_(self, source, non_blocking=None):
        raise NotImplementedError

    def resize_(self, size):
        raise NotImplementedError

    def share_memory_(self):
        raise NotImplementedError

    def _share_fd_cpu_(self, *args, **kwargs):
        raise NotImplementedError

    def _share_filename_cpu_(self, *args, **kwargs):
        raise NotImplementedError

    def is_shared(self):
        raise NotImplementedError

    def is_pinned(self, device="cuda"):
        raise NotImplementedError

    def _get_filename(self):
        raise NotImplementedError

    def _write_file(self, *args, **kwargs):
        raise NotImplementedError

    def _set_from_file(self, *args, **kwargs):
        raise NotImplementedError

    def _set_cdata(self, *args, **kwargs):
        raise NotImplementedError

    def _share_cuda_(self, *args, **kwargs):
        raise NotImplementedError

    def _shared_decref(self):
        raise NotImplementedError

    def _shared_incref(self, *args, **kwargs):
        raise NotImplementedError

    def _byteswap(self, *args, **kwargs):
        raise NotImplementedError

    def resizable(self):
        raise NotImplementedError

    @classmethod
    def _new_shared_cuda(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _new_shared_filename_cpu(cls, manager, obj, size, *, device=None, dtype=None):
        raise NotImplementedError

    @classmethod
    def _new_using_filename_cpu(cls, size):
        raise NotImplementedError

    @classmethod
    def _new_using_fd_cpu(cls, size):
        raise NotImplementedError

    @classmethod
    def _free_weak_ref(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _expired(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _release_ipc_counter_cuda(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_buffer(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_file(cls, filename, shared=False, nbytes=0):
        from ._cython._storage import CyCPUUntypedStorage
        return CyCPUUntypedStorage.from_file(filename, shared=shared)

    @classmethod
    def _new_shared(cls, size, *, device="cpu"):
        raise NotImplementedError

    def _weak_ref(self, *args, **kwargs):
        raise NotImplementedError

    def untyped(self):
        return self

    @property
    def is_cuda(self):
        return getattr(self, 'device', None) and self.device.type == "cuda"

    @property
    def is_hpu(self):
        return getattr(self, 'device', None) and self.device.type == "hpu"


def _npu_probe_model_dirs():
    from ._backends.npu import runtime as _npu_runtime
    return _npu_runtime._probe_model_dirs()


def _npu_model_dir():
    from ._backends.npu import runtime as _npu_runtime
    return _npu_runtime._model_dir()


def _npu_aclnn_available():
    from ._backends.npu import aclnn as _aclnn
    return _aclnn.is_available()


def _npu_aclnn_symbols_ok():
    from ._backends.npu import aclnn as _aclnn
    return _aclnn.symbols_ok()


def _npu_aclnn_ones_zero_ok():
    from ._backends.npu import aclnn as _aclnn
    return _aclnn.ones_zero_symbols_ok()


def _npu_device_count():
    from ._backends.npu import runtime as _npu_runtime
    return _npu_runtime.device_count()


# =============================================================================
# Storage factory functions and backend classes (candle-specific)
# These will eventually move to Cython to match torch's C++ backend.
# =============================================================================

import ctypes as _ctypes
import weakref as _weakref

import numpy as _np


def _get_storage_classes():
    from .storage import TypedStorage, UntypedStorage, _LegacyStorage
    return TypedStorage, UntypedStorage, _LegacyStorage


# -- Backend untyped storage classes --

class _NPUUntypedStorage:
    _npu_allocator_mod = None

    def __init__(self, device_ptr, nbytes, device=None):
        from ._device import device as _Device
        if isinstance(device, str):
            device = _Device(device)
        self.device = device or _Device("npu")
        self._device_ptr = int(device_ptr)
        self._nbytes = int(nbytes)
        if _NPUUntypedStorage._npu_allocator_mod is None:
            from ._backends.npu import allocator as _npu_alloc
            _NPUUntypedStorage._npu_allocator_mod = _npu_alloc
        alloc = _NPUUntypedStorage._npu_allocator_mod.get_allocator(self.device.index or 0)
        self._finalizer = _weakref.finalize(self, alloc.free, self._device_ptr, None)

    def nbytes(self): return self._nbytes
    def data_ptr(self): return self._device_ptr
    def is_pinned(self, device="cuda"): return False

    def buffer(self):
        raise RuntimeError("Cannot get buffer of NPU storage on CPU")

    def resize_(self, new_nbytes):
        new_nbytes = int(new_nbytes)
        if new_nbytes == self._nbytes:
            return self
        from ._backends.npu import allocator as _npu_allocator, runtime as _npu_runtime, state as _npu_state
        device_id = self.device.index or 0
        runtime = _npu_runtime.get_runtime(device_id)
        stream = _npu_state.current_stream(device_id).stream
        alloc = _npu_allocator.get_allocator(device_id)
        runtime.activate()
        new_ptr = alloc.malloc(new_nbytes, stream=stream)
        if self._device_ptr:
            copy_bytes = min(self._nbytes, new_nbytes)
            if copy_bytes:
                _npu_runtime.memcpy_d2d(new_ptr, copy_bytes, self._device_ptr, runtime=runtime, stream=stream)
        alloc.free(self._device_ptr, stream=stream)
        self._device_ptr = int(new_ptr)
        self._nbytes = new_nbytes
        self._finalizer.detach()
        self._finalizer = _weakref.finalize(self, alloc.free, self._device_ptr, None)
        return self


class _MPSUntypedStorage:
    def __init__(self, metal_buffer, nbytes, device=None):
        from ._device import device as _Device
        if isinstance(device, str):
            device = _Device(device)
        self.device = device or _Device("mps")
        self._metal_buffer = metal_buffer
        self._nbytes = int(nbytes)
        from ._backends.mps.runtime import buffer_contents
        self._contents_ptr = buffer_contents(metal_buffer)

    def nbytes(self): return self._nbytes
    def data_ptr(self): return self._contents_ptr
    def is_pinned(self, device="cuda"): return False

    def buffer(self):
        return _np.ctypeslib.as_array(
            (_ctypes.c_uint8 * self._nbytes).from_address(self._contents_ptr)
        )

    def resize_(self, new_nbytes):
        new_nbytes = int(new_nbytes)
        if new_nbytes == self._nbytes:
            return self
        from ._backends.mps.runtime import get_runtime, buffer_contents
        rt = get_runtime()
        new_buf = rt.create_buffer(new_nbytes)
        new_ptr = buffer_contents(new_buf)
        copy_bytes = min(self._nbytes, new_nbytes)
        if copy_bytes > 0:
            _ctypes.memmove(new_ptr, self._contents_ptr, copy_bytes)
        self._metal_buffer = new_buf
        self._contents_ptr = new_ptr
        self._nbytes = new_nbytes
        return self


class _MetaUntypedStorage:
    def __init__(self, nbytes, device=None):
        from ._device import device as _Device
        if isinstance(device, str):
            device = _Device(device)
        self.device = device or _Device("meta")
        self._nbytes = int(nbytes)

    def nbytes(self): return self._nbytes
    def data_ptr(self): raise RuntimeError("meta tensor has no data")
    def is_pinned(self, device="cuda"): return False

    def resize_(self, new_nbytes):
        self._nbytes = int(new_nbytes)
        return self


# -- Factory functions --

def typed_storage_from_numpy(arr, dtype, device=None):
    from ._dtype import to_numpy_dtype
    from ._cython._storage import CyCPUUntypedStorage
    arr = _np.ascontiguousarray(arr, dtype=to_numpy_dtype(dtype))
    untyped = CyCPUUntypedStorage(arr.view(_np.uint8), device=device)
    TypedStorage, _u, _l = _get_storage_classes()
    return TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)


def empty_cpu_typed_storage(shape, dtype, device=None):
    from ._dtype import to_numpy_dtype
    arr = _np.empty(shape, dtype=to_numpy_dtype(dtype))
    return typed_storage_from_numpy(arr, dtype, device=device)


def meta_typed_storage_from_shape(shape, dtype, device=None):
    size = int(_np.prod(shape))
    return meta_typed_storage_from_size(size, dtype, device=device)


def meta_typed_storage_from_size(size, dtype, device=None):
    from ._dtype import to_numpy_dtype
    itemsize = _np.dtype(to_numpy_dtype(dtype)).itemsize
    untyped = _MetaUntypedStorage(int(size) * itemsize, device=device)
    TypedStorage, _u, _l = _get_storage_classes()
    return TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)


def npu_typed_storage_from_ptr(device_ptr, size, dtype, device=None):
    from ._cython._storage import cy_npu_storage_from_ptr
    return cy_npu_storage_from_ptr(device_ptr, size, dtype, device=device)


def mps_typed_storage_from_numpy(arr, dtype, device=None):
    from ._dtype import to_numpy_dtype
    from ._backends.mps.runtime import get_runtime, buffer_contents
    arr = _np.ascontiguousarray(arr, dtype=to_numpy_dtype(dtype))
    rt = get_runtime()
    nbytes = int(arr.nbytes)
    metal_buf = rt.create_buffer(max(nbytes, 1))
    ptr = buffer_contents(metal_buf)
    if nbytes > 0:
        _ctypes.memmove(ptr, arr.ctypes.data, nbytes)
    untyped = _MPSUntypedStorage(metal_buf, nbytes, device=device)
    TypedStorage, _u, _l = _get_storage_classes()
    return TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)


def mps_typed_storage_from_ptr(metal_buffer, size, dtype, device=None):
    from ._dtype import to_numpy_dtype
    from ._backends.mps.runtime import buffer_contents
    itemsize = _np.dtype(to_numpy_dtype(dtype)).itemsize
    nbytes = int(size) * itemsize
    untyped = _MPSUntypedStorage(metal_buffer, nbytes, device=device)
    TypedStorage, _u, _l = _get_storage_classes()
    return TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)


def cuda_typed_storage_from_numpy(arr, dtype, device=None, stream=None):
    from ._dtype import to_numpy_dtype
    from ._backends.cuda import storage as _cuda_storage
    arr = _np.ascontiguousarray(arr, dtype=to_numpy_dtype(dtype))
    untyped = _cuda_storage.untyped_from_numpy(arr, device=device, stream=stream)
    TypedStorage, _u, _l = _get_storage_classes()
    return TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)


def empty_cuda_typed_storage(shape, dtype, device=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    from ._backends.cuda import storage as _cuda_storage
    untyped = _cuda_storage.empty_untyped(shape, dtype, device=device)
    TypedStorage, _u, _l = _get_storage_classes()
    return TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)


def cuda_typed_storage_to_numpy(storage, shape, dtype, stream=None):
    from ._backends.cuda import storage as _cuda_storage
    return _cuda_storage.to_numpy(storage.untyped_storage(), dtype, shape=shape, stream=stream)


def pinned_cpu_typed_storage_from_numpy(arr, dtype, device=None):
    from ._dtype import to_numpy_dtype
    from ._backends.npu import runtime as _npu_runtime
    from ._cython._storage import CyPinnedCPUUntypedStorage
    arr = _np.ascontiguousarray(arr, dtype=to_numpy_dtype(dtype))
    size = int(arr.nbytes)
    host_ptr = _npu_runtime.alloc_host(size)
    buf = _np.ctypeslib.as_array((_ctypes.c_uint8 * size).from_address(int(host_ptr)))
    buf[:] = arr.view(_np.uint8).reshape(-1)
    raw = _np.frombuffer(buf, dtype=_np.uint8)
    untyped = CyPinnedCPUUntypedStorage(raw, host_ptr, device=device)
    TypedStorage, _u, _l = _get_storage_classes()
    return TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)


class PendingStorage:
    def __init__(self, shape, dtype, device):
        from ._device import device as _Device
        from ._dtype import to_numpy_dtype
        if isinstance(device, str):
            device = _Device(device)
        self._shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        size = 1
        for d in self._shape:
            size *= d
        self._size = int(size)

    def size(self): return self._size

    def nbytes(self):
        from ._dtype import to_numpy_dtype
        itemsize = _np.dtype(to_numpy_dtype(self.dtype)).itemsize
        return int(self._size * itemsize)

    def data_ptr(self): raise RuntimeError("pending tensor has no data")

    @property
    def data(self):
        raise RuntimeError(
            "PendingStorage has no data. Call flush() on the pipeline context "
            "to materialize the storage, or move the tensor to a device."
        )

    def untyped_storage(self): return self
    def is_shared(self): return False
    def is_pinned(self): return False


# -- TypedStorage compat (installed once after storage.py is loaded) --

def _install_typed_storage_compat():
    """Install candle-specific methods on TypedStorage from storage.py."""
    from .storage import TypedStorage, UntypedStorage
    from ._dtype import to_numpy_dtype
    from ._cython._storage import CyCPUUntypedStorage as _CyCPU

    def _untyped_storage(self):
        return self.untyped()

    TypedStorage.untyped_storage = _untyped_storage

    @property
    def _data(self):
        untyped = self._untyped_storage
        if hasattr(untyped, 'buffer'):
            raw = untyped.buffer()
            return _np.frombuffer(raw, dtype=to_numpy_dtype(self.dtype), count=self._size())
        raise RuntimeError(f"Cannot get data from {type(untyped).__name__}")

    TypedStorage.data = _data

    def _reinterpret(self, dtype):
        if dtype == self.dtype:
            return self
        return TypedStorage(wrap_storage=self._untyped_storage, dtype=dtype, _internal=True)

    TypedStorage._reinterpret = _reinterpret

    def _is_pinned(self, device="cuda"):
        untyped = self._untyped_storage
        try:
            return untyped.is_pinned(device)
        except TypeError:
            return untyped.is_pinned()

    TypedStorage.is_pinned = _is_pinned

    # Override UntypedStorage.from_file to use Cython CPU implementation
    @classmethod
    def _from_file(cls, filename, shared=False, nbytes=0):
        return _CyCPU.from_file(filename, shared=shared)

    StorageBase.from_file = _from_file

    # Also override TypedStorage.from_file if it exists (uses 'size' param name)
    if hasattr(TypedStorage, 'from_file'):
        @classmethod
        def _typed_from_file(cls, filename, shared=False, size=0, nbytes=0):
            from ._dtype import float32
            cpu_storage = _CyCPU.from_file(filename, shared=shared)
            dtype = getattr(cls, 'dtype', float32)
            return TypedStorage(wrap_storage=cpu_storage, dtype=dtype, _internal=True)
        TypedStorage.from_file = _typed_from_file

    # Register Cython classes as virtual subclasses of UntypedStorage
    UntypedStorage.register(_CyCPU)
    from ._cython._storage import CyPinnedCPUUntypedStorage as _CyPinned
    UntypedStorage.register(_CyPinned)


# -- Legacy storage aliases (created lazily) --

_FloatStorage = None
_DoubleStorage = None
_HalfStorage = None
_LongStorage = None
_IntStorage = None
_ShortStorage = None
_ByteStorage = None
_BoolStorage = None
_BFloat16Storage = None
_ComplexFloatStorage = None
_ComplexDoubleStorage = None
_Storage = None


def _make_legacy_classes():
    global _FloatStorage, _DoubleStorage, _HalfStorage, _LongStorage
    global _IntStorage, _ShortStorage, _ByteStorage, _BoolStorage
    global _BFloat16Storage, _ComplexFloatStorage, _ComplexDoubleStorage, _Storage

    from ._dtype import (
        bool as dtype_bool, float16, float32, float64, bfloat16,
        int16, int32, int64, uint8, complex64, complex128,
    )
    TypedStorage, _u, _LegacyStorage = _get_storage_classes()

    class FloatStorage(_LegacyStorage): dtype = float32
    class DoubleStorage(_LegacyStorage): dtype = float64
    class HalfStorage(_LegacyStorage): dtype = float16
    class LongStorage(_LegacyStorage): dtype = int64
    class IntStorage(_LegacyStorage): dtype = int32
    class ShortStorage(_LegacyStorage): dtype = int16
    class ByteStorage(_LegacyStorage): dtype = uint8
    class BoolStorage(_LegacyStorage): dtype = dtype_bool
    class BFloat16Storage(_LegacyStorage): dtype = bfloat16
    class ComplexFloatStorage(_LegacyStorage): dtype = complex64
    class ComplexDoubleStorage(_LegacyStorage): dtype = complex128

    _FloatStorage = FloatStorage
    _DoubleStorage = DoubleStorage
    _HalfStorage = HalfStorage
    _LongStorage = LongStorage
    _IntStorage = IntStorage
    _ShortStorage = ShortStorage
    _ByteStorage = ByteStorage
    _BoolStorage = BoolStorage
    _BFloat16Storage = BFloat16Storage
    _ComplexFloatStorage = ComplexFloatStorage
    _ComplexDoubleStorage = ComplexDoubleStorage
    _Storage = TypedStorage


def __getattr__(name):
    if name in ("PyTorchFileReader", "PyTorchFileWriter"):
        from ._stream import PyTorchFileReader, PyTorchFileWriter
        return locals()[name]
    if name in ("TypedStorage", "UntypedStorage"):
        from .storage import TypedStorage, UntypedStorage
        return locals()[name]
    _map = {
        "FloatStorage": "_FloatStorage", "DoubleStorage": "_DoubleStorage",
        "HalfStorage": "_HalfStorage", "LongStorage": "_LongStorage",
        "IntStorage": "_IntStorage", "ShortStorage": "_ShortStorage",
        "ByteStorage": "_ByteStorage", "BoolStorage": "_BoolStorage",
        "BFloat16Storage": "_BFloat16Storage",
        "ComplexFloatStorage": "_ComplexFloatStorage",
        "ComplexDoubleStorage": "_ComplexDoubleStorage",
        "Storage": "_Storage",
    }
    if name in _map:
        _make_legacy_classes()
        return globals()[_map[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
