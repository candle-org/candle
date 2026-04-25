"""Candle's C extension layer — Cython accelerators for hot paths.

This package provides Cython implementations of performance-critical code
paths (dispatcher, allocator, storage creation, NPU ops, ACLNN FFI,
TensorImpl, dispatcher core, device, dtype, autograd node, autograd graph,
autograd function, autograd ops, functional wrappers, fast ops, tensor API).

Feature flags (set after import):
    _HAS_CYTHON_DISPATCH        — True if _dispatch.pyx compiled successfully
    _HAS_CYTHON_ALLOCATOR       — True if _allocator.pyx compiled successfully
    _HAS_CYTHON_STORAGE         — True if _storage.pyx compiled successfully
    _HAS_CYTHON_NPU_OPS         — True if _npu_ops.pyx compiled successfully
    _HAS_CYTHON_ACLNN_FFI       — True if _aclnn_ffi.pyx compiled successfully
    _HAS_CYTHON_DISPATCHER_CORE — True if _dispatcher_core.pyx compiled
    _HAS_CYTHON_DEVICE          — True if _device.pyx compiled successfully
    _HAS_CYTHON_DTYPE           — True if _dtype.pyx compiled successfully
    _HAS_CYTHON_AUTOGRAD_NODE   — always True (hard import, no fallback)
    _HAS_CYTHON_AUTOGRAD_GRAPH  — always True (hard import, no fallback)
    _HAS_CYTHON_AUTOGRAD_ENGINE — always True (hard import, no fallback)
    _HAS_CYTHON_AUTOGRAD_FUNCTION — always True (hard import, no fallback)
    _HAS_CYTHON_AUTOGRAD_OPS    — always True (hard import, no fallback)
    _HAS_CYTHON_FUNCTIONAL_OPS  — True if _functional_ops.pyx compiled successfully
    _HAS_CYTHON_FAST_OPS        — True if _fast_ops.pyx compiled successfully
    _HAS_CYTHON_TENSOR_API      — True if _tensor_api.pyx compiled successfully
    _HAS_CYTHON_STORAGE_IMPL    — True if _storage_impl.pyx compiled successfully
"""

_HAS_CYTHON_DISPATCH = False
_HAS_CYTHON_ALLOCATOR = False
_HAS_CYTHON_STORAGE = False
_HAS_CYTHON_NPU_OPS = False
_HAS_CYTHON_ACLNN_FFI = False
_HAS_CYTHON_DISPATCHER_CORE = False
_HAS_CYTHON_DEVICE = False
_HAS_CYTHON_DTYPE = False
# Autograd core modules are required (hard imports below -- no fallback).
_HAS_CYTHON_AUTOGRAD_NODE = False
_HAS_CYTHON_AUTOGRAD_GRAPH = False
_HAS_CYTHON_AUTOGRAD_ENGINE = False
_HAS_CYTHON_AUTOGRAD_FUNCTION = False
_HAS_CYTHON_AUTOGRAD_OPS = False
_HAS_CYTHON_FUNCTIONAL_OPS = False
_HAS_CYTHON_FAST_OPS = False
_HAS_CYTHON_TENSOR_API = False

try:
    from candle._dispatch import cy_dispatch, cy_dispatch_with_keyset  # noqa: F401
    _HAS_CYTHON_DISPATCH = True
except ImportError:
    pass

try:
    from ._allocator import FastNpuAllocator  # noqa: F401
    _HAS_CYTHON_ALLOCATOR = True
except ImportError:
    pass

try:
    from ._storage import cy_npu_storage_from_ptr  # noqa: F401
    _HAS_CYTHON_STORAGE = True
except ImportError:
    pass

try:
    from ._npu_ops import fast_binary_op  # noqa: F401
    _HAS_CYTHON_NPU_OPS = True
except ImportError:
    pass

try:
    from ._aclnn_ffi import (  # noqa: F401
        init as aclnn_ffi_init,
        create_tensor, destroy_tensor,
        create_scalar, destroy_scalar,
        create_int_array, destroy_int_array,
        destroy_executor, resolve_op, execute,
        binary_op_with_alpha, binary_op_no_alpha,
    )
    _HAS_CYTHON_ACLNN_FFI = True
except ImportError:
    pass

from ._tensor_impl import TensorImpl, _VersionCounterProxy  # noqa: F401  # pylint: disable=import-error,no-name-in-module

try:
    from ._dispatcher_core import cy_dispatch_with_keyset_fast  # noqa: F401
    _HAS_CYTHON_DISPATCHER_CORE = True
except ImportError:
    pass

try:
    from candle._device import FastDevice  # noqa: F401
    _HAS_CYTHON_DEVICE = True
except ImportError:
    pass

try:
    from candle._dtype import FastDType  # noqa: F401
    _HAS_CYTHON_DTYPE = True
except ImportError:
    pass

try:
    from ._autograd_node import (  # pylint: disable=import-error,no-name-in-module
        AccumulateGrad,  # noqa: F401
        FastNode,  # noqa: F401
        InputMetadata,  # noqa: F401
        Node,  # noqa: F401
        SavedTensor,  # noqa: F401
        _NodeHookHandle,  # noqa: F401
        _SavedValue,  # noqa: F401
    )
    _HAS_CYTHON_AUTOGRAD_NODE = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_NODE = False

try:
    from ._autograd_graph import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
        GradientEdge,
        current_saved_tensors_hooks,
        get_gradient_edge,
        saved_tensors_hooks,
    )
    _HAS_CYTHON_AUTOGRAD_GRAPH = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_GRAPH = False

try:
    from ._autograd_engine import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
        _GraphTask,
        _build_dependencies,
        _run_backward,
        backward,
        current_anomaly_parent,
        grad,
        is_anomaly_check_nan_enabled,
        is_anomaly_enabled,
        is_create_graph_enabled,
        pop_anomaly_config,
        pop_evaluating_node,
        push_anomaly_config,
        push_evaluating_node,
    )
    _HAS_CYTHON_AUTOGRAD_ENGINE = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_ENGINE = False

try:
    from ._autograd_function import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
        FunctionCtx,
        _function_apply,
    )
    _HAS_CYTHON_AUTOGRAD_FUNCTION = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_FUNCTION = False

try:
    from ._autograd_ops import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
        _strip_autograd_keys,
        _grad_context,
        _backward_dispatch_keyset,
        _autograd_unary_passthrough,
        _autograd_binary,
        _autograd_binary_args,
        _autograd_unary_args,
        _norm_extract_weight_bias,
        _autograd_norm,
    )
    _HAS_CYTHON_AUTOGRAD_OPS = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_OPS = False

try:
    from ._functional_ops import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
        _has_torch_function as cy_has_torch_function,
        _handle_torch_function as cy_handle_torch_function,
        add as functional_add,
        mul as functional_mul,
        matmul as functional_matmul,
        relu as functional_relu,
        transpose as functional_transpose,
        reshape as functional_reshape,
        neg as functional_neg,
    )
    _HAS_CYTHON_FUNCTIONAL_OPS = True
except ImportError:
    pass

try:
    import importlib
    _legacy_fast_ops = importlib.import_module(f"{__name__}._fast_ops")  # noqa: F401
    _HAS_CYTHON_FAST_OPS = True
except ImportError:
    pass

try:
    from ._TensorBase import TensorBase, _TensorBase  # noqa: F401  # pylint: disable=import-error,no-name-in-module
    _HAS_CYTHON_TENSOR_API = True
except ImportError:
    _HAS_CYTHON_TENSOR_API = False

_HAS_CYTHON_STORAGE_IMPL = False

try:
    from ._storage_impl import StorageImpl  # noqa: F401
    _HAS_CYTHON_STORAGE_IMPL = True
except ImportError:
    pass

# =============================================================================
# torch._C stubs and TensorBase (from _C_stubs.py)
# =============================================================================

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
            from candle._functional import mul as _mul, sub as _sub
            result = _mul(tensor, -1)
            return result + other
        from candle._functional import sub as _sub
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
        from ._storage import CyCPUUntypedStorage
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
    from candle._backends.npu import runtime as _npu_runtime
    return _npu_runtime._probe_model_dirs()


def _npu_model_dir():
    from candle._backends.npu import runtime as _npu_runtime
    return _npu_runtime._model_dir()


def _npu_aclnn_available():
    from candle._backends.npu import aclnn as _aclnn
    return _aclnn.is_available()


def _npu_aclnn_symbols_ok():
    from candle._backends.npu import aclnn as _aclnn
    return _aclnn.symbols_ok()


def _npu_aclnn_ones_zero_ok():
    from candle._backends.npu import aclnn as _aclnn
    return _aclnn.ones_zero_symbols_ok()


def _npu_device_count():
    from candle._backends.npu import runtime as _npu_runtime
    return _npu_runtime.device_count()


# =============================================================================
# Storage factory functions and backend classes (candle-specific)
# These will eventually move to Cython to match torch's C++ backend.
# =============================================================================

import ctypes as _ctypes
import weakref as _weakref

import numpy as _np


def _get_storage_classes():
    from candle.storage import TypedStorage, UntypedStorage, _LegacyStorage
    return TypedStorage, UntypedStorage, _LegacyStorage


# -- Backend untyped storage classes --

class _NPUUntypedStorage:
    _npu_allocator_mod = None

    def __init__(self, device_ptr, nbytes, device=None):
        from candle._device import device as _Device
        if isinstance(device, str):
            device = _Device(device)
        self.device = device or _Device("npu")
        self._device_ptr = int(device_ptr)
        self._nbytes = int(nbytes)
        if _NPUUntypedStorage._npu_allocator_mod is None:
            from candle._backends.npu import allocator as _npu_alloc
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
        from candle._backends.npu import allocator as _npu_allocator, runtime as _npu_runtime, state as _npu_state
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
        from candle._device import device as _Device
        if isinstance(device, str):
            device = _Device(device)
        self.device = device or _Device("mps")
        self._metal_buffer = metal_buffer
        self._nbytes = int(nbytes)
        from candle._backends.mps.runtime import buffer_contents
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
        from candle._backends.mps.runtime import get_runtime, buffer_contents
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
        from candle._device import device as _Device
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
    from candle._dtype import to_numpy_dtype
    from ._storage import CyCPUUntypedStorage
    arr = _np.ascontiguousarray(arr, dtype=to_numpy_dtype(dtype))
    untyped = CyCPUUntypedStorage(arr.view(_np.uint8), device=device)
    TypedStorage, _u, _l = _get_storage_classes()
    return TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)


def empty_cpu_typed_storage(shape, dtype, device=None):
    from candle._dtype import to_numpy_dtype
    arr = _np.empty(shape, dtype=to_numpy_dtype(dtype))
    return typed_storage_from_numpy(arr, dtype, device=device)


def meta_typed_storage_from_shape(shape, dtype, device=None):
    size = int(_np.prod(shape))
    return meta_typed_storage_from_size(size, dtype, device=device)


def meta_typed_storage_from_size(size, dtype, device=None):
    from candle._dtype import to_numpy_dtype
    itemsize = _np.dtype(to_numpy_dtype(dtype)).itemsize
    untyped = _MetaUntypedStorage(int(size) * itemsize, device=device)
    TypedStorage, _u, _l = _get_storage_classes()
    return TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)


def npu_typed_storage_from_ptr(device_ptr, size, dtype, device=None):
    from ._storage import cy_npu_storage_from_ptr
    return cy_npu_storage_from_ptr(device_ptr, size, dtype, device=device)


def mps_typed_storage_from_numpy(arr, dtype, device=None):
    from candle._dtype import to_numpy_dtype
    from candle._backends.mps.runtime import get_runtime, buffer_contents
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
    from candle._dtype import to_numpy_dtype
    from candle._backends.mps.runtime import buffer_contents
    itemsize = _np.dtype(to_numpy_dtype(dtype)).itemsize
    nbytes = int(size) * itemsize
    untyped = _MPSUntypedStorage(metal_buffer, nbytes, device=device)
    TypedStorage, _u, _l = _get_storage_classes()
    return TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)


def cuda_typed_storage_from_numpy(arr, dtype, device=None, stream=None):
    from candle._dtype import to_numpy_dtype
    from candle._backends.cuda import storage as _cuda_storage
    arr = _np.ascontiguousarray(arr, dtype=to_numpy_dtype(dtype))
    untyped = _cuda_storage.untyped_from_numpy(arr, device=device, stream=stream)
    TypedStorage, _u, _l = _get_storage_classes()
    return TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)


def empty_cuda_typed_storage(shape, dtype, device=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    from candle._backends.cuda import storage as _cuda_storage
    untyped = _cuda_storage.empty_untyped(shape, dtype, device=device)
    TypedStorage, _u, _l = _get_storage_classes()
    return TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)


def cuda_typed_storage_to_numpy(storage, shape, dtype, stream=None):
    from candle._backends.cuda import storage as _cuda_storage
    return _cuda_storage.to_numpy(storage.untyped_storage(), dtype, shape=shape, stream=stream)


def pinned_cpu_typed_storage_from_numpy(arr, dtype, device=None):
    from candle._dtype import to_numpy_dtype
    from candle._backends.npu import runtime as _npu_runtime
    from ._storage import CyPinnedCPUUntypedStorage
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
        from candle._device import device as _Device
        from candle._dtype import to_numpy_dtype
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
        from candle._dtype import to_numpy_dtype
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
    from candle.storage import TypedStorage, UntypedStorage
    from candle._dtype import to_numpy_dtype
    from ._storage import CyCPUUntypedStorage as _CyCPU

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
            from candle._dtype import float32
            cpu_storage = _CyCPU.from_file(filename, shared=shared)
            dtype = getattr(cls, 'dtype', float32)
            return TypedStorage(wrap_storage=cpu_storage, dtype=dtype, _internal=True)
        TypedStorage.from_file = _typed_from_file

    # Register Cython classes as virtual subclasses of UntypedStorage
    UntypedStorage.register(_CyCPU)
    from ._storage import CyPinnedCPUUntypedStorage as _CyPinned
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

    from candle._dtype import (
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
        from candle.storage import TypedStorage, UntypedStorage
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

from ._tensor_impl import TensorImpl, _StrideTuple

import numpy as _np


def _compute_strides(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return _StrideTuple(reversed(stride))


def _bf16_to_f32(arr):
    u32 = arr.astype(_np.uint32) << 16
    return u32.view(_np.float32)


def _f32_to_bf16(arr):
    u32 = arr.view(_np.uint32)
    rounding_bias = (u32 >> 16) & 1
    u32 = u32 + 0x7FFF + rounding_bias
    return (u32 >> 16).astype(_np.uint16)




def _install_tensor_api():
    """Install Cython tensor API methods on TensorBase (called after all modules loaded)."""
    if not _HAS_CYTHON_TENSOR_API:
        return
    from . import _tensor_api as _cython_mod
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

