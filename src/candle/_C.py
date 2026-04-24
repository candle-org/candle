# pylint: disable=import-error,no-name-in-module,possibly-unused-variable
import abc


def _add_docstr(obj, docstr):
    """Minimal torch._C._add_docstr stub."""
    obj.__doc__ = docstr
    return obj


class _disabled_torch_dispatch_impl:
    """Minimal torch._C._disabled_torch_dispatch_impl context manager."""
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


_torch_function_enabled = True


class DisableTorchFunctionSubclass:
    """Minimal torch._C.DisableTorchFunctionSubclass context manager."""
    def __init__(self): pass
    def __enter__(self):
        global _torch_function_enabled
        self._prev = _torch_function_enabled
        _torch_function_enabled = False
        return self
    def __exit__(self, *args):
        global _torch_function_enabled
        _torch_function_enabled = self._prev


def _has_storage(tensor):
    """Minimal torch._C._has_storage stub."""
    return hasattr(tensor, '_storage') and tensor._storage is not None


def _get_tracing_state():
    """Minimal torch._C._get_tracing_state stub."""
    return None


def _get_privateuse():
    """Minimal torch._C._get_privateuse stub."""
    return "npu"


class _dlpack_exchange_api:
    """Minimal torch._C._dlpack_exchange_api stub."""
    @staticmethod
    def to_dlpack(tensor): raise NotImplementedError("DLPack not supported")
    @staticmethod
    def from_dlpack(dlpack): raise NotImplementedError("DLPack not supported")


def _to_dlpack(tensor):
    raise NotImplementedError("DLPack not supported")


def _to_dlpack_versioned(tensor, version):
    raise NotImplementedError("DLPack not supported")


class _VariableFunctions:
    """Minimal torch._C._VariableFunctions stub."""
    @staticmethod
    def rsub(tensor, other):
        import numpy as np
        if isinstance(other, (int, float, bool, complex, np.integer, np.floating)):
            from ._functional import mul as _mul, sub as _sub
            result = _mul(tensor, -1)
            return result + other
        from ._functional import sub as _sub
        return _sub(other, tensor)


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


# =============================================================================
# TensorBase — torch._C.TensorBase equivalent for candle/tensor.py
# Must be defined AFTER all storage factories so _tensor_impl can import from _C.
# =============================================================================

from ._cython._tensor_impl import TensorImpl  # pylint: disable=import-error,no-name-in-module,wrong-import-position
from ._tensor_helpers import _StrideTuple  # noqa: E402


class TensorBase(TensorImpl):
    """torch._C.TensorBase equivalent for candle.

    Inherits directly from the Cython TensorImpl which provides the runtime backing.
    torch/tensor.py's Tensor class inherits from this.
    """

    _DEVICE_MAP = {"cpu": 0, "npu": 1, "cuda": 2, "mps": 3, "meta": 4}
    _DK_CPU  = 1 << 15
    _DK_NPU  = 1 << 13
    _DK_CUDA = 1 << 14
    _DK_MPS  = 1 << 21
    _DK_META = 1 << 12
    _DK_ADINPLACEORVIEW   = 1 << 4
    _DK_AUTOGRAD          = 1 << 11
    _DK_AUTOGRAD_CPU      = 1 << 6
    _DK_AUTOGRAD_NPU      = 1 << 7
    _DK_AUTOGRAD_CUDA     = 1 << 8
    _DK_AUTOGRAD_MPS      = 1 << 22
    _DK_AUTOGRAD_META     = 1 << 10

    def __init__(self, storage, shape, stride, offset=0, requires_grad=False):
        from ._cython._tensor_impl import cy_init_tensor_fields
        cy_init_tensor_fields(
            self, storage, tuple(shape), _StrideTuple(stride),
            int(offset), bool(requires_grad),
            None, None, None, None, False, False, None, 0, None,
        )

    def _set_device_from_storage(self, dev):
        self._set_device_from_obj(dev)

    def _set_dtype_from_storage(self, dtype):
        self._set_dtype_from_obj(dtype)

    def __delattr__(self, name):
        if name == "grad":
            object.__setattr__(self, "grad", None)
            return
        if name in {"data", "requires_grad", "_grad_fn", "grad_fn", "_backward_hooks"}:
            raise RuntimeError(f"cannot delete {name}")
        object.__delattr__(self, name)

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, new_data):
        if not isinstance(new_data, TensorBase):
            raise TypeError(f"data must be a Tensor, got {type(new_data).__name__}")
        if new_data.shape != self.shape:
            raise RuntimeError(f"shape mismatch: expected {self.shape}, got {new_data.shape}")
        if new_data.dtype != self.dtype:
            raise RuntimeError(f"dtype mismatch: expected {self.dtype}, got {new_data.dtype}")
        self.cy_set_data_runtime_truth_from(new_data)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return NotImplemented

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return NotImplemented

    def _fw_get(self, level):
        tangents = getattr(self, "_fw_tangents", None)
        if not tangents:
            return None
        return tangents.get(level)

    def _fw_set(self, level, tangent):
        tangents = getattr(self, "_fw_tangents", None)
        if tangents is None:
            tangents = {}
            self._fw_tangents = tangents
        tangents[level] = tangent

    def _fw_clear(self, level):
        tangents = getattr(self, "_fw_tangents", None)
        if not tangents:
            return
        tangents.pop(level, None)
        if not tangents:
            self._fw_tangents = {}

    def _fw_has(self, level):
        tangents = getattr(self, "_fw_tangents", None)
        return bool(tangents) and level in tangents

    def untyped_storage(self):
        return self._storage.untyped_storage()

    def _typed_storage(self):
        return self._storage

    def storage(self):
        from .storage import _warn_typed_storage_removal
        _warn_typed_storage_removal(stacklevel=2)
        return self._storage

    def data_ptr(self):
        storage = self._storage.untyped_storage()
        base = storage.data_ptr()
        return base + self.offset * self.dtype.itemsize

    @property
    def ndim(self):
        return self._ndim

    def is_floating_point(self):
        return self.dtype.is_floating_point

    def is_complex(self):
        return self.dtype.is_complex

    def detach(self):
        return self.cy_detach()

    def detach_(self):
        return self

    def pow(self, exponent):
        from ._functional import pow as _pow
        return _pow(self, exponent)

    def pow_(self, exponent):
        from ._functional import pow as _pow
        result = _pow(self, exponent)
        self.copy_(result)
        return self

    def positive(self):
        return self

    def neg(self):
        from ._functional import neg as _neg
        return _neg(self)

    def abs(self):
        from ._functional import abs as _abs
        return _abs(self)

    def __idiv__(self, other):
        from ._functional import div as _div
        result = _div(self, other)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def as_subclass(self, cls):
        return self

    def is_contiguous(self, memory_format=None):
        from ._tensor_helpers import _compute_strides
        expected = _compute_strides(self.shape)
        return self.stride == expected

    def contiguous(self, memory_format=None):
        if self.is_contiguous():
            return self
        from ._dispatch import dispatch
        return dispatch("contiguous", self.device.type, self)

    def _numpy_view(self):
        import numpy as np
        if self.device.type == "meta":
            raise RuntimeError("meta tensor has no data")
        if self.device.type != "cpu":
            return self.to("cpu")._numpy_view()
        base = self._storage.data.ravel()
        itemsize = base.itemsize
        strides = tuple(s * itemsize for s in self.stride)
        return np.lib.stride_tricks.as_strided(
            base[self.offset:], shape=self.shape, strides=strides
        )

    def reshape(self, *shape):
        if not shape:
            raise TypeError("reshape() missing shape arguments")
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if not self.requires_grad:
            from ._functional import reshape as reshape_dispatch
            return reshape_dispatch(self, shape)
        from ._dispatch import dispatch
        return dispatch("reshape", self.device.type, self, shape)

    def view(self, *shape):
        if not shape:
            raise TypeError(
                "view() received an invalid combination of arguments - got (), but expected one of:\n"
                " * (torch.dtype dtype)\n"
                " * (tuple of ints size)\n"
            )
        if len(shape) == 1:
            if isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            else:
                shape = (shape[0],)
        else:
            shape = tuple(shape)

        if not self.is_contiguous():
            raise RuntimeError(
                "view size is not compatible with input tensor's size and stride "
                "(at least one dimension spans across two contiguous subspaces). "
                "Use .reshape(...) instead."
            )

        size = 1
        for dim in self.shape:
            size *= dim

        infer_idx = None
        known_size = 1
        shape_list = list(shape)
        for idx, dim in enumerate(shape_list):
            if dim == -1:
                if infer_idx is not None:
                    raise RuntimeError("only one dimension can be inferred")
                infer_idx = idx
                continue
            known_size *= dim

        if infer_idx is not None:
            if known_size == 0 or size % known_size != 0:
                raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {size}")
            shape_list[infer_idx] = size // known_size

        shape = tuple(shape_list)
        new_size = 1
        for dim in shape:
            new_size *= dim
        if size != new_size:
            raise ValueError("view size mismatch")

        if self.requires_grad:
            from ._dispatch import dispatch
            return dispatch("view", self.device.type, self, shape)

        view = self.cy_view(shape)
        source_view_meta = getattr(self, "_view_meta", None) or {}
        from candle.autograd.grad_mode import current_creation_mode
        creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
        creation_kind = source_view_meta.get("creation_kind")
        if creation_mode is not None:
            if self._is_view():
                creation_kind = "view_of_view"
            else:
                creation_kind = "view"
        view._view_meta = {
            "op": "view",
            "shape": tuple(view.shape),
            "stride": tuple(view.stride),
            "offset": int(view.offset),
            "creation_mode": creation_mode,
            "creation_kind": creation_kind,
        }
        from candle.autograd import forward_ad
        level = forward_ad._current_level()
        if level >= 0:
            tangent = forward_ad.get_tangent(self, level)
            if tangent is not None:
                view._fw_set(level, tangent.view(shape))
        return view

    def flatten(self, start_dim=0, end_dim=-1):
        if self.requires_grad:
            from ._dispatch import dispatch
            return dispatch("flatten", self.device.type, self, start_dim, end_dim)
        from ._functional import flatten as flatten_dispatch
        return flatten_dispatch(self, start_dim, end_dim)

    def _transpose_view(self, dim0, dim1):
        return self.cy_transpose(dim0, dim1)

    def transpose(self, dim0, dim1):
        if self.requires_grad:
            from ._dispatch import dispatch
            return dispatch("transpose", self.device.type, self, dim0, dim1)
        from ._functional import transpose as transpose_dispatch
        return transpose_dispatch(self, dim0, dim1)

    def transpose_(self, dim0, dim1):
        result = self.transpose(dim0, dim1)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def t(self):
        if self.ndim < 2:
            return self
        return self.transpose(0, 1)

    def t_(self):
        if self.ndim >= 2:
            result = self.transpose(0, 1)
            self.cy_set_data_runtime_truth_from(result)
        return self

    @property
    def T(self):
        if self.ndim < 2:
            return self
        return self.transpose(0, 1)

    def view_as(self, other):
        return self.view(other.shape)

    def set_(self, typed_storage, storage_offset=None, size=None, stride=None):
        from .storage import TypedStorage
        from ._tensor_helpers import _compute_strides
        if not isinstance(typed_storage, TypedStorage):
            raise TypeError("set_() currently only supports TypedStorage input")
        if storage_offset is None:
            storage_offset = 0
        if size is None:
            total = typed_storage._size()
            if total == 0:
                size = ()
            else:
                size = (total,)
        if stride is None:
            stride = _compute_strides(size)
        self.cy_set_runtime_truth(typed_storage, size, stride, storage_offset)
        return self

    def as_strided(self, size, stride, storage_offset=None):
        if storage_offset is None:
            storage_offset = self.offset
        return self.cy_as_strided(size, stride, storage_offset)

    def _ones_like(self):
        from ._functional import ones_like
        return ones_like(self)

    def record_stream(self, stream):
        pass

    def numpy(self):
        arr = self._numpy_view()
        if self.dtype == bfloat16:
            from ._tensor_helpers import _bf16_to_f32
            arr = _bf16_to_f32(arr)
        return arr

    def backward(self, gradient=None, retain_graph=False, create_graph=False, inputs=None):
        from .autograd.engine import backward as _backward
        _backward(self, gradient, retain_graph, create_graph, inputs=inputs)

    def pin_memory(self):
        storage = pinned_cpu_typed_storage_from_numpy(self._numpy_view(), self.dtype, device=self.device)
        from ._tensor_helpers import _compute_strides
        return type(self)(storage, self.shape, _compute_strides(self.shape), 0, self.requires_grad)

    def is_pinned(self):
        return getattr(self._storage.untyped_storage(), 'is_pinned', lambda: False)()

    def retain_grad(self):
        if not self.requires_grad:
            raise RuntimeError("can't retain_grad on Tensor that has requires_grad=False")
        self._retain_grad = True

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        return self

    def register_hook(self, hook):
        from collections import OrderedDict
        if not self.requires_grad:
            raise RuntimeError("cannot register a hook on a tensor that doesn't require gradient")
        if self._backward_hooks is None:  # pylint: disable=access-member-before-definition
            self._backward_hooks = OrderedDict()
            if self.grad_fn is not None and hasattr(self.grad_fn, '_register_hook_dict'):
                self.grad_fn._register_hook_dict(self)
        from ._tensor_helpers import _HookHandle
        handle = _HookHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def _is_view(self):
        return self._base is not None

    def _check_inplace(self, other):
        pass

    def add_(self, other, *, alpha=1):
        from ._functional import add as add_dispatch
        result = add_dispatch(self, other, alpha=alpha)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def mul_(self, other):
        from ._functional import mul as mul_dispatch
        result = mul_dispatch(self, other)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def relu_(self):
        from ._functional import relu as relu_dispatch
        result = relu_dispatch(self)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def zero_(self):
        from ._functional import zeros_like
        result = zeros_like(self)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def fill_(self, value):
        from ._functional import full_like
        result = full_like(self, value)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def new_empty(self, size, *, dtype=None, device=None, requires_grad=False):
        from ._functional import empty
        return empty(size, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_tensor(self, data, *, dtype=None, device=None, requires_grad=False):
        from ._functional import tensor
        return tensor(data, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_empty_strided(self, size, stride, *, dtype=None, device=None, requires_grad=False):
        from ._functional import empty_strided
        return empty_strided(size, stride, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_ones(self, size, *, dtype=None, device=None, requires_grad=False):
        from ._functional import ones
        return ones(size, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_zeros(self, size, *, dtype=None, device=None, requires_grad=False):
        from ._functional import zeros
        return zeros(size, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_full(self, size, fill_value, *, dtype=None, device=None, requires_grad=False):
        from ._functional import full
        return full(size, fill_value, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def var_mean(self, dim=None, keepdim=False, unbiased=True):
        from ._functional import var_mean as var_mean_dispatch
        return var_mean_dispatch(self, dim, keepdim=keepdim, unbiased=unbiased)

    def __rsub__(self, other):
        from ._functional import sub as sub_dispatch
        return sub_dispatch(other, self)

    def __getitem__(self, index):
        from ._functional import getitem as getitem_dispatch
        return getitem_dispatch(self, index)

    def __setitem__(self, index, value):
        from ._functional import setitem as setitem_dispatch
        return setitem_dispatch(self, index, value)

    def __iadd__(self, other):
        return self.add_(other)

    def __isub__(self, other):
        from ._functional import sub as sub_dispatch
        result = sub_dispatch(self, other)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def __imul__(self, other):
        return self.mul_(other)

    def __itruediv__(self, other):
        from ._functional import true_divide as true_divide_dispatch
        result = true_divide_dispatch(self, other)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def __neg__(self):
        from ._functional import neg as neg_dispatch
        return neg_dispatch(self)

    def clone(self):
        from ._functional import clone as clone_dispatch
        return clone_dispatch(self)

    def to(self, *args, **kwargs):
        from ._functional import to as to_dispatch
        return to_dispatch(self, *args, **kwargs)

    def _to_dtype(self, dtype):
        from ._functional import to_dtype
        return to_dtype(self, dtype)

    def cpu(self):
        return self.to("cpu")

    def npu(self):
        return self.to("npu")

    def mps(self):
        return self.to("mps")

    def cuda(self):
        return self.to("cuda")

    def __repr__(self):
        from ._tensor_str import _str
        return _str(self)

    def __str__(self):
        from ._tensor_str import _str
        return _str(self)

    def __len__(self):
        if self._ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __iter__(self):
        if self._ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(self.shape[0]):
            yield self[i]

    def __hash__(self):
        return id(self)


_TensorBase = TensorBase


def _install_tensor_api():
    """Install Cython tensor API methods on TensorBase (called after all modules loaded)."""
    from . import _cython as _cython_mod

    if not getattr(_cython_mod, "_HAS_CYTHON_TENSOR_API", False):
        return
    TensorBase._set_device_from_storage = _cython_mod.tensor_set_device_from_storage
    TensorBase._set_dtype_from_storage = _cython_mod.tensor_set_dtype_from_storage
    TensorBase.data = property(TensorBase.data.fget, _cython_mod.tensor_set_data)
    TensorBase.__delattr__ = _cython_mod.tensor_delattr
    TensorBase._fw_get = _cython_mod.tensor_fw_get
    TensorBase._fw_set = _cython_mod.tensor_fw_set
    TensorBase._fw_clear = _cython_mod.tensor_fw_clear
    TensorBase._fw_has = _cython_mod.tensor_fw_has
    TensorBase.untyped_storage = _cython_mod.tensor_untyped_storage
    TensorBase.record_stream = _cython_mod.tensor_record_stream
    TensorBase.is_pinned = _cython_mod.tensor_is_pinned

    TensorBase.__add__ = _cython_mod.tensor_add
    TensorBase.__sub__ = _cython_mod.tensor_sub
    TensorBase.__mul__ = _cython_mod.tensor_mul
    TensorBase.__matmul__ = _cython_mod.tensor_matmul
    TensorBase.__getitem__ = _cython_mod.tensor_getitem
    TensorBase.__setitem__ = _cython_mod.tensor_setitem
    TensorBase.__iadd__ = _cython_mod.tensor_iadd
    TensorBase.__isub__ = _cython_mod.tensor_isub
    TensorBase.__imul__ = _cython_mod.tensor_imul
    TensorBase.__itruediv__ = _cython_mod.tensor_itruediv
    TensorBase.__neg__ = _cython_mod.tensor_neg
    TensorBase.neg = _cython_mod.tensor_neg
    
    TensorBase.clone = _cython_mod.tensor_clone
    TensorBase.detach = _cython_mod.tensor_detach
    TensorBase.detach_ = _cython_mod.tensor_detach_
    TensorBase.to = _cython_mod.tensor_to
    TensorBase._to_dtype = _cython_mod.tensor_to_dtype
    TensorBase.cpu = _cython_mod.tensor_cpu
    TensorBase.npu = _cython_mod.tensor_npu
    TensorBase.mps = _cython_mod.tensor_mps
    TensorBase.cuda = _cython_mod.tensor_cuda
    TensorBase.backward = _cython_mod.tensor_backward
    TensorBase.relu = _cython_mod.tensor_relu
    TensorBase.is_contiguous = _cython_mod.tensor_is_contiguous
    TensorBase.contiguous = _cython_mod.tensor_contiguous
    TensorBase.reshape = _cython_mod.tensor_reshape
    TensorBase.transpose = _cython_mod.tensor_transpose
    TensorBase.view = _cython_mod.tensor_view
    TensorBase.flatten = _cython_mod.tensor_flatten
    TensorBase.t = _cython_mod.tensor_t
    TensorBase.as_strided = _cython_mod.tensor_as_strided
    TensorBase.size = _cython_mod.tensor_size
    TensorBase.dim = _cython_mod.tensor_dim
    
    TensorBase.retain_grad = _cython_mod.tensor_retain_grad
    TensorBase.requires_grad_ = _cython_mod.tensor_requires_grad_
    TensorBase.register_hook = _cython_mod.tensor_register_hook
    TensorBase._is_view = _cython_mod.tensor_is_view
    TensorBase._check_inplace = _cython_mod.tensor_check_inplace
    
    TensorBase.add_ = _cython_mod.tensor_add_
    TensorBase.mul_ = _cython_mod.tensor_mul_
    TensorBase.relu_ = _cython_mod.tensor_relu_
    TensorBase.zero_ = _cython_mod.tensor_zero_
    TensorBase.fill_ = _cython_mod.tensor_fill_
    TensorBase.copy_ = _cython_mod.tensor_copy_
    
    TensorBase.abs_ = _cython_mod.tensor_abs_
    TensorBase.neg_ = _cython_mod.tensor_neg_
    TensorBase.exp_ = _cython_mod.tensor_exp_
    TensorBase.log_ = _cython_mod.tensor_log_
    TensorBase.log2_ = _cython_mod.tensor_log2_
    TensorBase.log10_ = _cython_mod.tensor_log10_
    TensorBase.sqrt_ = _cython_mod.tensor_sqrt_
    TensorBase.sin_ = _cython_mod.tensor_sin_
    TensorBase.cos_ = _cython_mod.tensor_cos_
    TensorBase.tan_ = _cython_mod.tensor_tan_
    TensorBase.tanh_ = _cython_mod.tensor_tanh_
    TensorBase.sigmoid_ = _cython_mod.tensor_sigmoid_
    TensorBase.floor_ = _cython_mod.tensor_floor_
    TensorBase.ceil_ = _cython_mod.tensor_ceil_
    TensorBase.round_ = _cython_mod.tensor_round_
    TensorBase.trunc_ = _cython_mod.tensor_trunc_
    TensorBase.pow_ = _cython_mod.tensor_pow_
    TensorBase.reciprocal_ = _cython_mod.tensor_reciprocal_
    TensorBase.erfinv_ = _cython_mod.tensor_erfinv_
    
    TensorBase.sub_ = _cython_mod.tensor_sub_
    TensorBase.clamp_ = _cython_mod.tensor_clamp_
    TensorBase.uniform_ = _cython_mod.tensor_uniform_
    TensorBase.normal_ = _cython_mod.tensor_normal_
    TensorBase.random_ = _cython_mod.tensor_random_
    TensorBase.randint_ = _cython_mod.tensor_randint_
    TensorBase.bernoulli_ = _cython_mod.tensor_bernoulli_
    TensorBase.exponential_ = _cython_mod.tensor_exponential_
    TensorBase.log_normal_ = _cython_mod.tensor_log_normal_
    TensorBase.cauchy_ = _cython_mod.tensor_cauchy_
    TensorBase.geometric_ = _cython_mod.tensor_geometric_
    
    TensorBase.transpose_ = _cython_mod.tensor_transpose_
    TensorBase.t_ = _cython_mod.tensor_t_
    TensorBase.squeeze_ = _cython_mod.tensor_squeeze_
    TensorBase.unsqueeze_ = _cython_mod.tensor_unsqueeze_
    TensorBase.as_strided_ = _cython_mod.tensor_as_strided_
    TensorBase.swapdims_ = _cython_mod.tensor_swapdims_
    TensorBase.swapaxes_ = _cython_mod.tensor_swapaxes_
    
    TensorBase.scatter_add = _cython_mod.tensor_scatter_add
    TensorBase.index_fill = _cython_mod.tensor_index_fill
    TensorBase.index_copy = _cython_mod.tensor_index_copy
    TensorBase.index_add = _cython_mod.tensor_index_add
    TensorBase.put_ = _cython_mod.tensor_put_
    TensorBase.scatter_ = _cython_mod.tensor_scatter_
    TensorBase.scatter_add_ = _cython_mod.tensor_scatter_add_
    TensorBase.masked_fill_ = _cython_mod.tensor_masked_fill_
    TensorBase.masked_scatter_ = _cython_mod.tensor_masked_scatter_
    TensorBase.index_put_ = _cython_mod.tensor_index_put_
    TensorBase.index_copy_ = _cython_mod.tensor_index_copy_
    TensorBase.index_fill_ = _cython_mod.tensor_index_fill_
    TensorBase.index_add_ = _cython_mod.tensor_index_add_
    
    TensorBase.new_empty = _cython_mod.tensor_new_empty
    TensorBase.new_tensor = _cython_mod.tensor_new_tensor
    TensorBase.new_empty_strided = _cython_mod.tensor_new_empty_strided
    TensorBase._ones_like = _cython_mod.tensor_ones_like
    TensorBase.new_ones = _cython_mod.tensor_new_ones
    TensorBase.new_zeros = _cython_mod.tensor_new_zeros
    TensorBase.new_full = _cython_mod.tensor_new_full
    TensorBase.div_ = _cython_mod.tensor_div_
    TensorBase.unflatten = _cython_mod.tensor_unflatten
    TensorBase.bitwise_and_ = _cython_mod.tensor_bitwise_and_
    TensorBase.bitwise_or_ = _cython_mod.tensor_bitwise_or_
    TensorBase.bitwise_xor_ = _cython_mod.tensor_bitwise_xor_
    TensorBase.type = _cython_mod.tensor_type
    TensorBase.type_as = _cython_mod.tensor_type_as
    TensorBase.reshape_as = _cython_mod.tensor_reshape_as
    TensorBase.permute = _cython_mod.tensor_permute
    TensorBase.mean = _cython_mod.tensor_mean
    TensorBase.std = _cython_mod.tensor_std
    TensorBase.repeat = _cython_mod.tensor_repeat
    TensorBase.tile = _cython_mod.tensor_tile
    TensorBase.flip = _cython_mod.tensor_flip
    TensorBase.logsumexp = _cython_mod.tensor_logsumexp
    TensorBase.trace = _cython_mod.tensor_trace
    TensorBase.det = _cython_mod.tensor_det
    TensorBase.matrix_power = _cython_mod.tensor_matrix_power
    TensorBase.dist = _cython_mod.tensor_dist
    TensorBase.renorm = _cython_mod.tensor_renorm
    TensorBase.nansum = _cython_mod.tensor_nansum
    TensorBase.nanmean = _cython_mod.tensor_nanmean
    TensorBase.argwhere = _cython_mod.tensor_argwhere
    TensorBase.baddbmm = _cython_mod.tensor_baddbmm
    TensorBase.vsplit = _cython_mod.tensor_vsplit
    TensorBase.hsplit = _cython_mod.tensor_hsplit
    TensorBase.dsplit = _cython_mod.tensor_dsplit
    TensorBase.take_along_dim = _cython_mod.tensor_take_along_dim
    TensorBase.cummin = _cython_mod.tensor_cummin
    TensorBase.log1p = _cython_mod.tensor_log1p
    TensorBase.expm1 = _cython_mod.tensor_expm1
    TensorBase.lt = _cython_mod.tensor_lt
    TensorBase.le = _cython_mod.tensor_le
    TensorBase.gt = _cython_mod.tensor_gt
    TensorBase.ge = _cython_mod.tensor_ge
    TensorBase.abs = _cython_mod.tensor_abs
    TensorBase.exp = _cython_mod.tensor_exp
    TensorBase.log = _cython_mod.tensor_log
    TensorBase.sqrt = _cython_mod.tensor_sqrt
    TensorBase.sin = _cython_mod.tensor_sin
    TensorBase.cos = _cython_mod.tensor_cos
    TensorBase.tan = _cython_mod.tensor_tan
    TensorBase.tanh = _cython_mod.tensor_tanh
    TensorBase.sigmoid = _cython_mod.tensor_sigmoid
    TensorBase.floor = _cython_mod.tensor_floor
    TensorBase.ceil = _cython_mod.tensor_ceil
    TensorBase.round = _cython_mod.tensor_round
    TensorBase.trunc = _cython_mod.tensor_trunc
    TensorBase.frac = _cython_mod.tensor_frac
    TensorBase.log2 = _cython_mod.tensor_log2
    TensorBase.log10 = _cython_mod.tensor_log10
    TensorBase.exp2 = _cython_mod.tensor_exp2
    TensorBase.rsqrt = _cython_mod.tensor_rsqrt
    TensorBase.sign = _cython_mod.tensor_sign
    TensorBase.signbit = _cython_mod.tensor_signbit
    TensorBase.square = _cython_mod.tensor_square
    TensorBase.isnan = _cython_mod.tensor_isnan
    TensorBase.isinf = _cython_mod.tensor_isinf
    TensorBase.isfinite = _cython_mod.tensor_isfinite
    TensorBase.sinh = _cython_mod.tensor_sinh
    TensorBase.cosh = _cython_mod.tensor_cosh
    TensorBase.asinh = _cython_mod.tensor_asinh
    TensorBase.acosh = _cython_mod.tensor_acosh
    TensorBase.atanh = _cython_mod.tensor_atanh
    TensorBase.erf = _cython_mod.tensor_erf
    TensorBase.erfc = _cython_mod.tensor_erfc
    TensorBase.reciprocal = _cython_mod.tensor_reciprocal
    TensorBase.tril = _cython_mod.tensor_tril
    TensorBase.triu = _cython_mod.tensor_triu
    TensorBase.diag = _cython_mod.tensor_diag
    TensorBase.add = _cython_mod.tensor_add_method
    TensorBase.sub = _cython_mod.tensor_sub_method
    TensorBase.mul = _cython_mod.tensor_mul_method
    TensorBase.div = _cython_mod.tensor_div_method
    TensorBase.pow = _cython_mod.tensor_pow_method
    TensorBase.matmul = _cython_mod.tensor_matmul_method
    TensorBase.__rsub__ = _cython_mod.tensor_rsub
    TensorBase.__rmul__ = _cython_mod.tensor_rmul
    TensorBase.__truediv__ = _cython_mod.tensor_truediv
    TensorBase.__rtruediv__ = _cython_mod.tensor_rtruediv
    TensorBase.__pow__ = _cython_mod.tensor_pow_op
    TensorBase.__rpow__ = _cython_mod.tensor_rpow
    TensorBase.__floordiv__ = _cython_mod.tensor_floordiv
    TensorBase.__rfloordiv__ = _cython_mod.tensor_rfloordiv
    TensorBase.__mod__ = _cython_mod.tensor_mod
    TensorBase.__rmod__ = _cython_mod.tensor_rmod
    TensorBase.__rmatmul__ = _cython_mod.tensor_rmatmul
    TensorBase.__and__ = _cython_mod.tensor_and
    TensorBase.__or__ = _cython_mod.tensor_or
    TensorBase.__xor__ = _cython_mod.tensor_xor
    TensorBase.all = _cython_mod.tensor_all_method
    TensorBase.any = _cython_mod.tensor_any_method
    TensorBase.sum = _cython_mod.tensor_sum_method
    TensorBase.prod = _cython_mod.tensor_prod_method
    TensorBase.var = _cython_mod.tensor_var_method
    TensorBase.var_mean = _cython_mod.tensor_var_mean_method
    TensorBase.norm = _cython_mod.tensor_norm_method
    TensorBase.count_nonzero = _cython_mod.tensor_count_nonzero_method
    TensorBase.cumsum = _cython_mod.tensor_cumsum_method
    TensorBase.cumprod = _cython_mod.tensor_cumprod_method
    TensorBase.cummax = _cython_mod.tensor_cummax_method
    TensorBase.argsort = _cython_mod.tensor_argsort_method
    TensorBase.sort = _cython_mod.tensor_sort_method
    TensorBase.topk = _cython_mod.tensor_topk_method
    TensorBase.eq = _cython_mod.tensor_eq_method
    TensorBase.ne = _cython_mod.tensor_ne_method
    TensorBase.allclose = _cython_mod.tensor_allclose_method
    TensorBase.isclose = _cython_mod.tensor_isclose_method
    TensorBase.equal = _cython_mod.tensor_equal_method
    TensorBase.view_as = _cython_mod.tensor_view_as
    TensorBase.expand = _cython_mod.tensor_expand_method
    TensorBase.expand_as = _cython_mod.tensor_expand_as_method
    TensorBase.expand_copy = _cython_mod.tensor_expand_copy_method
    TensorBase.narrow = _cython_mod.tensor_narrow_method
    TensorBase.select = _cython_mod.tensor_select_method
    TensorBase.unfold = _cython_mod.tensor_unfold_method
    TensorBase.moveaxis = _cython_mod.tensor_moveaxis_method
    TensorBase.swapdims = _cython_mod.tensor_swapdims_method
    TensorBase.swapaxes = _cython_mod.tensor_swapaxes_method
    TensorBase.gather = _cython_mod.tensor_gather_method
    TensorBase.scatter = _cython_mod.tensor_scatter_method
    TensorBase.index_select = _cython_mod.tensor_index_select_method
    TensorBase.take = _cython_mod.tensor_take_method
    TensorBase.masked_fill = _cython_mod.tensor_masked_fill_method
    TensorBase.masked_select = _cython_mod.tensor_masked_select_method
    TensorBase.index_put = _cython_mod.tensor_index_put_method
    TensorBase.slice = _cython_mod.tensor_slice_method
    TensorBase.slice_copy = _cython_mod.tensor_slice_copy_method
    TensorBase.slice_scatter = _cython_mod.tensor_slice_scatter_method
    TensorBase.nonzero = _cython_mod.tensor_nonzero_method
    TensorBase.sum_to_size = _cython_mod.tensor_sum_to_size_method
    TensorBase.softplus = _cython_mod.tensor_softplus_method
    TensorBase.clamp = _cython_mod.tensor_clamp_method
    TensorBase.relu6 = _cython_mod.tensor_relu6_method
    TensorBase.hardtanh = _cython_mod.tensor_hardtanh_method
    TensorBase.min = _cython_mod.tensor_min_method
    TensorBase.max = _cython_mod.tensor_max_method
    TensorBase.amin = _cython_mod.tensor_amin_method
    TensorBase.amax = _cython_mod.tensor_amax_method
    TensorBase.addmm = _cython_mod.tensor_addmm_method
    TensorBase.bmm = _cython_mod.tensor_bmm_method
    TensorBase.mm = _cython_mod.tensor_mm_method
    TensorBase.chunk = _cython_mod.tensor_chunk_method
    TensorBase.split = _cython_mod.tensor_split_method
    TensorBase.roll = _cython_mod.tensor_roll_method
    TensorBase.rot90 = _cython_mod.tensor_rot90_method
    TensorBase.addcdiv = _cython_mod.tensor_addcdiv_method
    TensorBase.addcmul = _cython_mod.tensor_addcmul_method
    TensorBase.hypot = _cython_mod.tensor_hypot_method
    TensorBase.lerp = _cython_mod.tensor_lerp_method
    TensorBase.atan2 = _cython_mod.tensor_atan2_method
    TensorBase.asin = _cython_mod.tensor_asin_method
    TensorBase.acos = _cython_mod.tensor_acos_method
    TensorBase.atan = _cython_mod.tensor_atan_method
    TensorBase.as_strided_copy = _cython_mod.tensor_as_strided_copy_method
    TensorBase.as_strided_scatter = _cython_mod.tensor_as_strided_scatter_method
    TensorBase.multinomial = _cython_mod.tensor_multinomial_method
    TensorBase.ndim = property(_cython_mod.tensor_ndim_fget)
    TensorBase.T = property(_cython_mod.tensor_T_fget)
    TensorBase.is_floating_point = _cython_mod.tensor_is_floating_point
    TensorBase.is_complex = _cython_mod.tensor_is_complex
    TensorBase.clamp_min = _cython_mod.tensor_clamp_min_method
    TensorBase.clamp_max = _cython_mod.tensor_clamp_max_method
    TensorBase.fmin = _cython_mod.tensor_fmin_method
    TensorBase.fmax = _cython_mod.tensor_fmax_method
    TensorBase.where = _cython_mod.tensor_where_method
    TensorBase.logaddexp = _cython_mod.tensor_logaddexp_method
    TensorBase.logaddexp2 = _cython_mod.tensor_logaddexp2_method
    TensorBase.remainder = _cython_mod.tensor_remainder_method
    TensorBase.fmod = _cython_mod.tensor_fmod_method
    TensorBase.squeeze = _cython_mod.tensor_squeeze_method
    TensorBase.unsqueeze = _cython_mod.tensor_unsqueeze_method
    TensorBase.argmax = _cython_mod.tensor_argmax_method
    TensorBase.argmin = _cython_mod.tensor_argmin_method
    TensorBase.logical_and = _cython_mod.tensor_logical_and
    TensorBase.logical_or = _cython_mod.tensor_logical_or
    TensorBase.logical_xor = _cython_mod.tensor_logical_xor
    TensorBase.logical_not = _cython_mod.tensor_logical_not
    TensorBase.bitwise_and = _cython_mod.tensor_bitwise_and
    TensorBase.bitwise_or = _cython_mod.tensor_bitwise_or
    TensorBase.bitwise_xor = _cython_mod.tensor_bitwise_xor
    TensorBase.bitwise_not = _cython_mod.tensor_bitwise_not
    TensorBase.movedim = _cython_mod.tensor_movedim
    TensorBase.diagonal = _cython_mod.tensor_diagonal
    TensorBase.unbind = _cython_mod.tensor_unbind
    
    TensorBase.numpy = _cython_mod.tensor_numpy
    TensorBase._numpy_view = _cython_mod.tensor_numpy_view
    TensorBase.pin_memory = _cython_mod.tensor_pin_memory
