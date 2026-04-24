# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython storage runtime/helper layer.

This module is part of Candle's storage runtime boundary, not just an
optimization layer. It centralizes storage-backed object construction for
runtime-owned device pointers and helpers that Python storage shells call into.

NPU path is intentionally Cython-only — no Python fallback.
"""

import ctypes
import mmap
import os
import tempfile
import threading
import weakref

import numpy as np

from candle._cython._storage_impl import StorageImpl  # pylint: disable=import-error,no-name-in-module

from libc.stdint cimport int64_t


# ---------------------------------------------------------------------------
# Shared-file bookkeeping helpers for Python storage shell
# ---------------------------------------------------------------------------

_SHARED_FILE_REGISTRY = set()
_SHARED_FILE_REGISTRY_LOCK = threading.Lock()


def cy_register_shared_file(path):
    if not path:
        return
    with _SHARED_FILE_REGISTRY_LOCK:
        _SHARED_FILE_REGISTRY.add(path)


def cy_unregister_shared_file(path):
    if not path:
        return
    with _SHARED_FILE_REGISTRY_LOCK:
        _SHARED_FILE_REGISTRY.discard(path)


def cy_cleanup_shared_files():
    with _SHARED_FILE_REGISTRY_LOCK:
        paths = list(_SHARED_FILE_REGISTRY)
    for path in paths:
        cy_cleanup_shared_resource(None, path)


def cy_cleanup_shared_resource(fd=None, path=None):
    if fd is not None:
        try:
            os.close(int(fd))
        except Exception:
            pass
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except Exception:
        return
    cy_unregister_shared_file(path)


def cy_shared_files_count():
    with _SHARED_FILE_REGISTRY_LOCK:
        return len(_SHARED_FILE_REGISTRY)


class CyPinnedCPUUntypedStorage:
    def __init__(self, array, ptr, filename=None, shared=False, device=None):
        from candle._device import device as Device
        from candle._backends.npu import runtime as npu_runtime

        if isinstance(device, str):
            device = Device(device)
        self.device = device or Device("cpu")
        self._array = array
        self._shared = shared
        self._filename = filename
        self._ptr = int(ptr)
        self._finalizer = weakref.finalize(self, npu_runtime.free_host, self._ptr)

    def __repr__(self):
        values = [f" {item}" for item in self.buffer().tolist()]
        body = "\n".join(values) if values else " "
        return (
            f"{body}\n"
            f"[torch.storage.UntypedStorage(device={self.device.type}) of size {self.nbytes()}]"
        )

    def __len__(self):
        return self.nbytes()

    def __iter__(self):
        return iter(self.buffer().tolist())

    def __getitem__(self, index):
        if isinstance(index, bool):
            raise TypeError("can't index a torch.UntypedStorage with bool")
        if isinstance(index, np.integer):
            index = int(index)
        elif isinstance(index, np.generic):
            raise TypeError(
                f"can't index a torch.UntypedStorage with numpy.{type(index).__name__}"
            )
        if isinstance(index, np.ndarray):
            raise TypeError("can't index a torch.UntypedStorage with numpy.ndarray")
        if isinstance(index, float):
            raise TypeError("can't index a torch.UntypedStorage with float")
        if isinstance(index, str):
            raise TypeError("can't index a torch.UntypedStorage with str")
        if isinstance(index, bytes):
            raise TypeError("can't index a torch.UntypedStorage with bytes")
        if isinstance(index, bytearray):
            raise TypeError("can't index a torch.UntypedStorage with bytearray")
        if isinstance(index, memoryview):
            raise TypeError("can't index a torch.UntypedStorage with memoryview")
        if isinstance(index, list):
            raise TypeError("can't index a torch.UntypedStorage with list")
        if isinstance(index, tuple):
            raise TypeError("can't index a torch.UntypedStorage with tuple")
        from candle._tensor import Tensor

        if isinstance(index, Tensor):
            raise TypeError("can't index a torch.UntypedStorage with Tensor")
        if index is None:
            raise TypeError("can't index a torch.UntypedStorage with NoneType")
        if index is Ellipsis:
            raise TypeError("can't index a torch.UntypedStorage with ellipsis")
        if isinstance(index, slice):
            return CyCPUUntypedStorage(
                self.buffer()[index],
                filename=None,
                shared=self._shared,
                device=self.device,
            )
        try:
            return self.buffer()[index].item()
        except IndexError as exc:
            err_index = index
            if isinstance(index, int) and index < 0:
                err_index = index + self.nbytes()
            raise IndexError(
                f"index {err_index} out of range for storage of size {self.nbytes()}"
            ) from exc

    def __setitem__(self, index, value):
        from candle._tensor import Tensor

        if isinstance(index, np.integer):
            index = int(index)
        elif (
            isinstance(
                index,
                (
                    bool,
                    list,
                    tuple,
                    np.generic,
                    np.ndarray,
                    Tensor,
                    float,
                    str,
                    bytes,
                    bytearray,
                    memoryview,
                ),
            )
            or index is None
            or index is Ellipsis
        ):
            raise SystemError("error return without exception set")
        try:
            self.buffer()[index] = value
        except IndexError as exc:
            raise RuntimeError("out of bounds") from exc

    def nbytes(self):
        return int(self._array.nbytes)

    def data_ptr(self):
        return int(self._array.ctypes.data)

    def buffer(self):
        return self._array

    def resize_(self, new_nbytes):
        raise NotImplementedError("pinned storage cannot resize")

    def share_memory_(self):
        self._shared = True
        return self

    def is_shared(self):
        return self._shared

    def is_pinned(self):
        return True

    @classmethod
    def from_file(cls, filename, shared=False):
        data = np.memmap(filename, mode="r+", dtype=np.uint8)
        return cls(data, int(data.ctypes.data), filename=filename, shared=shared)

    def filename(self):
        return self._filename


class CyCPUUntypedStorage:
    def __init__(
        self,
        array,
        filename=None,
        shared=False,
        device=None,
        mmap_obj=None,
        fd=None,
        tmp_file=None,
        sharing_mechanism=None,
        cleanup_finalizer=None,
    ):
        from candle._device import device as Device

        if isinstance(device, str):
            device = Device(device)
        self.device = device or Device("cpu")
        self._mmap = mmap_obj
        self._tmp_file = tmp_file
        if cleanup_finalizer is not None:
            finalizer = cleanup_finalizer
        elif shared and sharing_mechanism == "file_descriptor" and fd is not None:
            finalizer = weakref.finalize(
                self,
                cy_cleanup_shared_resource,
                int(fd),
                filename,
            )
        else:
            finalizer = None
        self._impl = StorageImpl.from_numpy(
            array,
            shared=shared,
            filename=filename,
            sharing_mechanism=sharing_mechanism,
            cleanup_finalizer=finalizer,
            fd=fd,
        )

    def __repr__(self):
        values = [f" {item}" for item in self.buffer().tolist()]
        body = "\n".join(values) if values else " "
        return (
            f"{body}\n"
            f"[torch.storage.UntypedStorage(device={self.device.type}) of size {self.nbytes()}]"
        )

    def __len__(self):
        return self.nbytes()

    def __iter__(self):
        return iter(self.buffer().tolist())

    def __getitem__(self, index):
        if isinstance(index, bool):
            raise TypeError("can't index a torch.UntypedStorage with bool")
        if isinstance(index, np.integer):
            index = int(index)
        elif isinstance(index, np.generic):
            raise TypeError(
                f"can't index a torch.UntypedStorage with numpy.{type(index).__name__}"
            )
        if isinstance(index, np.ndarray):
            raise TypeError("can't index a torch.UntypedStorage with numpy.ndarray")
        if isinstance(index, float):
            raise TypeError("can't index a torch.UntypedStorage with float")
        if isinstance(index, str):
            raise TypeError("can't index a torch.UntypedStorage with str")
        if isinstance(index, bytes):
            raise TypeError("can't index a torch.UntypedStorage with bytes")
        if isinstance(index, bytearray):
            raise TypeError("can't index a torch.UntypedStorage with bytearray")
        if isinstance(index, memoryview):
            raise TypeError("can't index a torch.UntypedStorage with memoryview")
        if isinstance(index, list):
            raise TypeError("can't index a torch.UntypedStorage with list")
        if isinstance(index, tuple):
            raise TypeError("can't index a torch.UntypedStorage with tuple")
        from candle._tensor import Tensor

        if isinstance(index, Tensor):
            raise TypeError("can't index a torch.UntypedStorage with Tensor")
        if index is None:
            raise TypeError("can't index a torch.UntypedStorage with NoneType")
        if index is Ellipsis:
            raise TypeError("can't index a torch.UntypedStorage with ellipsis")
        if isinstance(index, slice):
            return CyCPUUntypedStorage(
                self.buffer()[index],
                filename=None,
                shared=False,
                device=self.device,
                sharing_mechanism=None,
            )
        try:
            return self.buffer()[index].item()
        except IndexError as exc:
            err_index = index
            if isinstance(index, int) and index < 0:
                err_index = index + self.nbytes()
            raise IndexError(
                f"index {err_index} out of range for storage of size {self.nbytes()}"
            ) from exc

    def __setitem__(self, index, value):
        from candle._tensor import Tensor

        if isinstance(index, np.integer):
            index = int(index)
        elif (
            isinstance(
                index,
                (
                    bool,
                    list,
                    tuple,
                    np.generic,
                    np.ndarray,
                    Tensor,
                    float,
                    str,
                    bytes,
                    bytearray,
                    memoryview,
                ),
            )
            or index is None
            or index is Ellipsis
        ):
            raise SystemError("error return without exception set")
        try:
            self.buffer()[index] = value
        except IndexError as exc:
            raise RuntimeError("out of bounds") from exc

    def nbytes(self):
        return int(self._impl.nbytes())

    def data_ptr(self):
        return int(self._impl.data_ptr())

    def buffer(self):
        return self._impl.owner()

    def resize_(self, new_nbytes):
        if self._impl.filename() is not None or self._impl.is_shared():
            raise RuntimeError("Trying to resize storage that is not resizable")
        new_array = np.empty(int(new_nbytes), dtype=np.uint8)
        old_bytes = self.buffer().view(np.uint8)
        copy_bytes = min(old_bytes.size, new_array.size)
        new_array[:copy_bytes] = old_bytes[:copy_bytes]
        self._impl = StorageImpl.from_numpy(new_array)
        self._mmap = None
        self._tmp_file = None
        return self

    def share_memory_(self, strategy="file_descriptor"):
        if self._impl.is_shared():
            return self

        nbytes = int(self._impl.nbytes())
        if nbytes == 0:
            self._impl = StorageImpl.from_numpy(
                self.buffer(),
                shared=True,
                sharing_mechanism=strategy,
            )
            return self

        if strategy == "file_descriptor":
            fd, filename = tempfile.mkstemp(prefix="candle_fd_", suffix=".bin")
            try:
                os.ftruncate(fd, nbytes)
                mm = mmap.mmap(fd, nbytes)
            except Exception:
                os.close(fd)
                raise

            dst = np.frombuffer(mm, dtype=np.uint8, count=nbytes)
            dst[:] = self.buffer().view(np.uint8).reshape(-1)
            finalizer = weakref.finalize(
                self,
                cy_cleanup_shared_resource,
                int(fd),
                filename,
            )
            self._mmap = mm
            self._tmp_file = filename
            self._impl = StorageImpl.from_numpy(
                dst,
                shared=True,
                filename=filename,
                sharing_mechanism="file_descriptor",
                cleanup_finalizer=finalizer,
                fd=fd,
            )
            return self

        if strategy == "file_system":
            fd, filename = tempfile.mkstemp(prefix="candle_shm_", suffix=".bin")
            try:
                with open(fd, "wb", closefd=False) as f:
                    f.truncate(nbytes)
            finally:
                os.close(fd)

            mmap_arr = np.memmap(filename, mode="r+", dtype=np.uint8, shape=(nbytes,))
            mmap_arr[:] = self.buffer().view(np.uint8).reshape(-1)
            cy_register_shared_file(filename)
            self._mmap = None
            self._tmp_file = filename
            self._impl = StorageImpl.from_numpy(
                mmap_arr,
                shared=True,
                filename=filename,
                sharing_mechanism="file_system",
            )
            return self

        raise ValueError(f"unsupported sharing strategy: {strategy}")

    def is_shared(self):
        return bool(self._impl.is_shared())

    def is_pinned(self):
        return False

    def shared_memory_meta(self):
        mechanism = self._impl.sharing_mechanism()
        if mechanism == "file_descriptor":
            return {
                "mechanism": "file_descriptor",
                "fd": int(self._impl.shared_fd()),
                "filename": self._impl.filename(),
                "nbytes": int(self._impl.nbytes()),
            }
        if mechanism == "file_system":
            return {
                "mechanism": "file_system",
                "filename": self._impl.filename(),
                "nbytes": int(self._impl.nbytes()),
            }
        return None

    def typed_view(self, dtype, size):
        return self._impl.typed_view(dtype, size)

    @classmethod
    def from_shared_memory(cls, filename, nbytes):
        data = np.memmap(filename, mode="r+", dtype=np.uint8, shape=(int(nbytes),))
        cy_register_shared_file(filename)
        return cls(data, filename=filename, shared=True, sharing_mechanism="file_system")

    @classmethod
    def from_shared_fd(cls, fd, nbytes, filename=None):
        fd = int(fd)
        mm = mmap.mmap(fd, int(nbytes))
        arr = np.frombuffer(mm, dtype=np.uint8, count=int(nbytes))
        return cls(
            arr,
            filename=filename,
            shared=True,
            mmap_obj=mm,
            fd=fd,
            tmp_file=filename,
            sharing_mechanism="file_descriptor",
        )

    @classmethod
    def from_file(cls, filename, shared=False):
        data = np.memmap(filename, mode="r+", dtype=np.uint8)
        return cls(data, filename=filename, shared=shared)

    def filename(self):
        return self._impl.filename()


# ---------------------------------------------------------------------------
# C-level dtype itemsize (no numpy needed)
# ---------------------------------------------------------------------------

cdef int _c_dtype_itemsize(object dtype):
    """Return byte size from a candle dtype object — C switch, no dict."""
    cdef object size = getattr(dtype, "itemsize", None)
    if size is not None:
        return <int>size
    cdef str name = getattr(dtype, "name", None)
    if name is None:
        s = str(dtype)
        parts = s.split(".")
        name = parts[len(parts) - 1]
    if name == "float32" or name == "int32":
        return 4
    if name == "float64" or name == "int64":
        return 8
    if name == "float16" or name == "bfloat16" or name == "int16":
        return 2
    if name == "int8" or name == "uint8" or name == "bool":
        return 1
    return 4


# ---------------------------------------------------------------------------
# NPU storage creation — hard require Cython storage classes
# ---------------------------------------------------------------------------

cdef object _FastNPUStorage_cls = None
cdef object _FastTypedStorage_cls = None


cdef inline void _ensure_fast_storage():
    """Load FastNPUStorage/FastTypedStorage cdef classes.

    NPU path is Cython-only by design. If `_npu_storage` is unavailable,
    import should fail loudly instead of silently falling back to Python.
    """
    global _FastNPUStorage_cls, _FastTypedStorage_cls
    if _FastNPUStorage_cls is not None:
        return
    from candle._cython._npu_storage import FastNPUStorage, FastTypedStorage  # pylint: disable=import-error,no-name-in-module
    _FastNPUStorage_cls = FastNPUStorage
    _FastTypedStorage_cls = FastTypedStorage


def cy_npu_storage_from_ptr(int64_t device_ptr, int64_t size,
                            object dtype, object device=None):
    """Create typed NPU storage from a raw device pointer.

    NPU path is Cython-only: this function always constructs
    FastNPUStorage + FastTypedStorage.
    """
    _ensure_fast_storage()

    cdef int itemsize = _c_dtype_itemsize(dtype)
    cdef int64_t nbytes = size * itemsize

    if device is None:
        from candle._device import device as _Device
        device = _Device("npu")

    untyped = _FastNPUStorage_cls(device_ptr, nbytes, device)
    return _FastTypedStorage_cls(untyped, dtype, size)


# ---------------------------------------------------------------------------
# Stride tuple class cache — loaded once for cy_make_npu_tensor
# ---------------------------------------------------------------------------

cdef object _StrideTuple_cls = None


cdef inline void _ensure_tensor_cls():
    global _StrideTuple_cls
    if _StrideTuple_cls is not None:
        return
    from candle._tensor import _StrideTuple
    _StrideTuple_cls = _StrideTuple


def cy_make_npu_tensor(int64_t device_ptr, int64_t n_elements,
                       object dtype, object device,
                       tuple shape, object stride):
    """Construct an NPU Tensor entirely in Cython via the unified tensor factory.

    Equivalent to::

        storage = npu_typed_storage_from_ptr(device_ptr, n_elements, dtype, device)
        return Tensor(storage, shape, stride)

    Routes through cy_make_tensor_from_storage so all tensor births share a
    single initialisation path.
    """
    from candle._cython._tensor_impl import cy_make_tensor_from_storage

    _ensure_fast_storage()
    _ensure_tensor_cls()

    cdef int itemsize = _c_dtype_itemsize(dtype)
    cdef int64_t nbytes = n_elements * itemsize

    # 1. FastNPUStorage + FastTypedStorage (unchanged)
    untyped = _FastNPUStorage_cls(device_ptr, nbytes, device)
    typed = _FastTypedStorage_cls(untyped, dtype, n_elements)

    # 2. Delegate all field initialisation to the unified factory
    return cy_make_tensor_from_storage(
        typed,
        shape,
        _StrideTuple_cls(stride),
        0,
        False,
    )
