import abc
import atexit
import ctypes
import weakref
import numpy as np

from ._cython._storage import (  # pylint: disable=import-error,no-name-in-module
    CyCPUUntypedStorage,
    CyPinnedCPUUntypedStorage,
    cy_cleanup_shared_files as _cy_cleanup_shared_files,
)
from ._device import _default_device, device as Device
from ._dtype import float32, to_numpy_dtype
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


atexit.register(_cy_cleanup_shared_files)


class UntypedStorage(metaclass=abc.ABCMeta):
    """Public storage shell around runtime-owned backing memory.

    UntypedStorage is the user-facing storage shell exposed by Candle's Python API.
    The underlying pointer, allocation lifetime, and device placement are runtime
    truths supplied by backend/runtime storage objects; Python methods here wrap
    that runtime-owned backing data rather than owning it independently.
    """

    # A1 boundary note: this file remains the public storage API shell.
    # Future runtime-sensitive storage ownership should move into Cython helpers
    # instead of expanding Python-side owner state here.

    def __init__(self, device):
        if isinstance(device, str):
            device = Device(device)
        self.device = device
        self._shared = False

    def __repr__(self):
        if self.device.type == "cpu":
            values = [f" {item}" for item in self.buffer().tolist()]
            body = "\n".join(values) if values else " "
            return (
                f"{body}\n"
                f"[torch.storage.UntypedStorage(device={self.device.type}) of size {self.nbytes()}]"
            )
        return object.__repr__(self)

    def __len__(self):
        return self.nbytes()

    def __iter__(self):
        if self.device.type == "cpu":
            return iter(self.buffer().tolist())
        raise RuntimeError("storage has no CPU data")

    def __getitem__(self, index):
        if self.device.type == "cpu":
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
            from ._tensor import Tensor
            if isinstance(index, Tensor):
                raise TypeError("can't index a torch.UntypedStorage with Tensor")
            if index is None:
                raise TypeError("can't index a torch.UntypedStorage with NoneType")
            if index is Ellipsis:
                raise TypeError("can't index a torch.UntypedStorage with ellipsis")
            if isinstance(index, slice):
                return CyCPUUntypedStorage(self.buffer()[index], filename=None, shared=self._shared, device=self.device)
            try:
                return self.buffer()[index].item()
            except IndexError as exc:
                err_index = index
                if isinstance(index, int) and index < 0:
                    err_index = index + self.nbytes()
                raise IndexError(f"index {err_index} out of range for storage of size {self.nbytes()}") from exc
        raise RuntimeError("storage has no CPU data")

    def __setitem__(self, index, value):
        if self.device.type == "cpu":
            from ._tensor import Tensor

            if isinstance(index, np.integer):
                index = int(index)
            elif (
                isinstance(index, (bool, list, tuple, np.generic, np.ndarray, Tensor, float, str, bytes, bytearray, memoryview))
                or index is None
                or index is Ellipsis
            ):
                raise SystemError("error return without exception set")
            try:
                self.buffer()[index] = value
            except IndexError as exc:
                raise RuntimeError("out of bounds") from exc
            return
        raise RuntimeError("storage has no CPU data")

    def nbytes(self):
        raise NotImplementedError

    def data_ptr(self):
        raise NotImplementedError

    def resize_(self, new_nbytes):
        raise NotImplementedError

    def share_memory_(self):
        raise NotImplementedError

    def buffer(self):
        raise NotImplementedError("Subclass must implement buffer()")

    def is_shared(self):
        return False

    def is_pinned(self):
        return False

    @classmethod
    def from_file(cls, filename, shared=False):
        return CyCPUUntypedStorage.from_file(filename, shared=shared)

    def filename(self):
        return None



_npu_allocator_mod = None  # cached after first NPU storage creation


class _NPUUntypedStorage(UntypedStorage):
    def __init__(self, device_ptr, nbytes, device=None):
        super().__init__(device or Device("npu"))
        self._device_ptr = int(device_ptr)
        self._nbytes = int(nbytes)
        global _npu_allocator_mod
        if _npu_allocator_mod is None:
            from ._backends.npu import allocator as _npu_alloc
            _npu_allocator_mod = _npu_alloc
        alloc = _npu_allocator_mod.get_allocator(self.device.index or 0)
        self._finalizer = weakref.finalize(self, alloc.free, self._device_ptr, None)

    def nbytes(self):
        return self._nbytes

    def data_ptr(self):
        return self._device_ptr

    def is_pinned(self):
        return False

    def buffer(self):
        raise RuntimeError("Cannot get buffer of NPU storage on CPU")

    def resize_(self, new_nbytes):
        new_nbytes = int(new_nbytes)
        if new_nbytes == self._nbytes:
            return self
        from ._backends.npu import allocator as npu_allocator
        from ._backends.npu import runtime as npu_runtime
        from ._backends.npu import state as npu_state

        device_id = self.device.index or 0
        runtime = npu_runtime.get_runtime(device_id)
        stream = npu_state.current_stream(device_id).stream
        alloc = npu_allocator.get_allocator(device_id)
        runtime.activate()
        new_ptr = alloc.malloc(new_nbytes, stream=stream)
        if self._device_ptr:
            copy_bytes = min(self._nbytes, new_nbytes)
            if copy_bytes:
                npu_runtime.memcpy_d2d(
                    new_ptr,
                    copy_bytes,
                    self._device_ptr,
                    runtime=runtime,
                    stream=stream,
                )
        alloc.free(self._device_ptr, stream=stream)
        self._device_ptr = int(new_ptr)
        self._nbytes = new_nbytes
        self._finalizer.detach()
        self._finalizer = weakref.finalize(self, alloc.free, self._device_ptr, None)
        return self


class _MPSUntypedStorage(UntypedStorage):
    def __init__(self, metal_buffer, nbytes, device=None):
        super().__init__(device or Device("mps"))
        self._metal_buffer = metal_buffer
        self._nbytes = int(nbytes)
        from ._backends.mps.runtime import buffer_contents
        self._contents_ptr = buffer_contents(metal_buffer)

    def nbytes(self):
        return self._nbytes

    def data_ptr(self):
        return self._contents_ptr

    def is_pinned(self):
        return False

    def buffer(self):
        return np.ctypeslib.as_array(
            (ctypes.c_uint8 * self._nbytes).from_address(self._contents_ptr)
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
            ctypes.memmove(new_ptr, self._contents_ptr, copy_bytes)
        self._metal_buffer = new_buf
        self._contents_ptr = new_ptr
        self._nbytes = new_nbytes
        return self


class _MetaUntypedStorage(UntypedStorage):
    def __init__(self, nbytes, device=None):
        super().__init__(device or Device("meta"))
        self._nbytes = int(nbytes)

    def nbytes(self):
        return self._nbytes

    def data_ptr(self):
        raise RuntimeError("meta tensor has no data")

    def is_pinned(self):
        return False

    def resize_(self, new_nbytes):
        self._nbytes = int(new_nbytes)
        return self


UntypedStorage.register(CyCPUUntypedStorage)
UntypedStorage.register(CyPinnedCPUUntypedStorage)


class TypedStorage:
    """Public typed storage wrapper over an untyped runtime owner.

    Runtime pointer ownership lives in the referenced untyped storage object.
    TypedStorage adds dtype/size interpretation for the public Python API while
    methods like data_ptr(), device, is_shared(), and is_pinned() delegate to the
    runtime-owned untyped backing storage.
    """

    def __init__(self, untyped, dtype=None, size=0, data=None):
        self._untyped = untyped
        self.dtype = dtype or float32
        self._size = int(size)
        self._data = data

    @property
    def device(self):
        return self._untyped.device

    def __repr__(self):
        if self.device.type in ("cpu", "mps"):
            values = [f" {item}" for item in self._data.tolist()]
            body = "\n".join(values) if values else " "
            return (
                f"{body}\n"
                f"[torch.storage.TypedStorage(dtype=torch.{self.dtype.name}, device={self.device.type}) of size {self._size}]"
            )
        return object.__repr__(self)

    def __len__(self):
        return self._size

    def __iter__(self):
        if self.device.type == "cpu":
            return iter(self._data.tolist())
        if self.device.type == "mps":
            return iter(self._data.tolist())
        raise RuntimeError("storage has no CPU data")

    def __getitem__(self, index):
        from ._tensor import Tensor

        if isinstance(index, bool):
            raise TypeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'bool'>"
            )
        if isinstance(index, np.generic):
            raise RuntimeError(
                f"can't index a <class 'torch.storage.TypedStorage'> with <class 'numpy.{type(index).__name__}'>"
            )
        if isinstance(index, float):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'float'>"
            )
        if isinstance(index, complex):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'complex'>"
            )
        if isinstance(index, str):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'str'>"
            )
        if isinstance(index, bytes):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'bytes'>"
            )
        if isinstance(index, bytearray):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'bytearray'>"
            )
        if isinstance(index, memoryview):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'memoryview'>"
            )
        if isinstance(index, range):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'range'>"
            )
        index_type = type(index)
        if (
            index_type.__module__.startswith("torch")
            and index_type.__name__ == "Tensor"
        ):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'torch.Tensor'>"
            )
        if isinstance(index, Tensor):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'torch.Tensor'>"
            )
        if isinstance(index, list):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'list'>"
            )
        if isinstance(index, tuple):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'tuple'>"
            )
        if isinstance(index, np.ndarray):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'numpy.ndarray'>"
            )
        if index is None:
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'NoneType'>"
            )
        if index is Ellipsis:
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'ellipsis'>"
            )
        if isinstance(index, slice):
            raise RuntimeError("slices are only supported in UntypedStorage.__getitem__")
        if self.device.type == "cpu":
            try:
                return self._data[index].item()
            except IndexError as exc:
                raise IndexError(f"index {index} out of range for storage of size {self._size}") from exc
        if self.device.type == "mps":
            try:
                return self._data[index].item()
            except IndexError as exc:
                raise IndexError(f"index {index} out of range for storage of size {self._size}") from exc
        raise RuntimeError("storage has no CPU data")

    def __setitem__(self, index, value):
        from ._tensor import Tensor

        if isinstance(index, np.generic):
            raise RuntimeError(
                f"can't index a <class 'torch.storage.TypedStorage'> with <class 'numpy.{type(index).__name__}'>"
            )
        if isinstance(index, float):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'float'>"
            )
        if isinstance(index, complex):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'complex'>"
            )
        if isinstance(index, str):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'str'>"
            )
        if isinstance(index, bytes):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'bytes'>"
            )
        if isinstance(index, bytearray):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'bytearray'>"
            )
        if isinstance(index, memoryview):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'memoryview'>"
            )
        if isinstance(index, range):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'range'>"
            )
        index_type = type(index)
        if (
            index_type.__module__.startswith("torch")
            and index_type.__name__ == "Tensor"
        ):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'torch.Tensor'>"
            )
        if isinstance(index, Tensor):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'torch.Tensor'>"
            )
        if isinstance(index, list):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'list'>"
            )
        if isinstance(index, tuple):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'tuple'>"
            )
        if isinstance(index, np.ndarray):
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'numpy.ndarray'>"
            )
        if index is None:
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'NoneType'>"
            )
        if isinstance(index, slice):
            if self.device.type == "cpu":
                self._data[index] = value
                return
            if self.device.type == "mps":
                self._data[index] = value
                return
            raise RuntimeError("storage has no CPU data")
        if index is Ellipsis:
            raise RuntimeError(
                "can't index a <class 'torch.storage.TypedStorage'> with <class 'ellipsis'>"
            )
        if self.device.type == "cpu":
            try:
                self._data[index] = value
            except IndexError as exc:
                raise IndexError(
                    f"index {index} is out of bounds for dimension 0 with size {self._size}"
                ) from exc
            return
        if self.device.type == "mps":
            try:
                self._data[index] = value
            except IndexError as exc:
                raise IndexError(
                    f"index {index} is out of bounds for dimension 0 with size {self._size}"
                ) from exc
            return
        raise RuntimeError("storage has no CPU data")

    def size(self):
        return self._size

    def nbytes(self):
        itemsize = np.dtype(to_numpy_dtype(self.dtype)).itemsize
        return int(self._size * itemsize)

    def data_ptr(self):
        """Return the runtime-owned pointer exposed through the typed shell."""
        return self._untyped.data_ptr()

    def untyped_storage(self):
        """Return the runtime owner backing this typed public storage view."""
        return self._untyped

    def is_shared(self):
        return self._untyped.is_shared()

    def is_pinned(self):
        return self._untyped.is_pinned()

    @property
    def data(self):
        if self.device.type not in ("cpu", "mps"):
            raise RuntimeError("storage has no CPU data")
        return self._data

    def clone(self):
        if self.device.type == "cpu":
            return typed_storage_from_numpy(np.copy(self._data), self.dtype, device=self.device)
        if self.device.type == "npu":
            from ._backends.npu import runtime as npu_runtime

            size = self.nbytes()
            runtime = npu_runtime.get_runtime(self.device.index or 0)
            dst_ptr = npu_runtime._alloc_device(size, runtime=runtime)
            runtime.activate()
            npu_runtime.memcpy_d2d(
                dst_ptr,
                size,
                self.data_ptr(),
                runtime=runtime,
            )
            untyped = _NPUUntypedStorage(dst_ptr, size, device=self.device)
            return TypedStorage(untyped, self.dtype, self._size)
        if self.device.type == "mps":
            src_buf = self._untyped.buffer()
            arr = np.array(src_buf, copy=True)
            return mps_typed_storage_from_numpy(
                np.frombuffer(arr, dtype=to_numpy_dtype(self.dtype), count=self._size),
                self.dtype,
                device=self.device,
            )
        if self.device.type == "meta":
            return meta_typed_storage_from_size(self._size, self.dtype, device=self.device)
        if self.device.type == "cuda":
            arr = cuda_typed_storage_to_numpy(self, (self._size,), self.dtype)
            return cuda_typed_storage_from_numpy(arr, self.dtype, device=self.device)
        raise NotImplementedError(f"Unsupported device: {self.device}")

    def copy_(self, other):
        if self.device.type != other.device.type:
            raise NotImplementedError("cross-device copy_ not supported")
        if self.device.type == "cpu":
            np.copyto(self._data, other._data)
            return self
        if self.device.type == "mps":
            dst_buf = self._untyped.buffer()
            src_buf = other._untyped.buffer()
            copy_bytes = min(dst_buf.nbytes, src_buf.nbytes)
            dst_buf[:copy_bytes] = src_buf[:copy_bytes]
            return self
        if self.device.type == "npu":
            from ._backends.npu import runtime as npu_runtime

            size = min(self.nbytes(), other.nbytes())
            runtime = npu_runtime.get_runtime(self.device.index or 0)
            runtime.activate()
            npu_runtime.memcpy_d2d(
                self.data_ptr(),
                size,
                other.data_ptr(),
                runtime=runtime,
            )
            return self
        if self.device.type == "cuda":
            from ._backends.cuda import storage as cuda_storage

            cuda_storage.copy_untyped(self.untyped_storage(), other.untyped_storage())
            return self
        raise NotImplementedError(f"Unsupported device: {self.device}")

    def _reinterpret(self, dtype):
        if dtype == self.dtype:
            return self
        if self.device.type == "cpu":
            itemsize = np.dtype(to_numpy_dtype(dtype)).itemsize
            size = int(self.nbytes() // itemsize)
            data = self._data.view(to_numpy_dtype(dtype))
            data = data.reshape(size)
            return TypedStorage(self._untyped, dtype, size, data=data)
        if self.device.type == "mps":
            itemsize = np.dtype(to_numpy_dtype(dtype)).itemsize
            size = int(self.nbytes() // itemsize)
            data = self._data.view(to_numpy_dtype(dtype))
            data = data.reshape(size)
            return TypedStorage(self._untyped, dtype, size, data=data)
        if self.device.type in ("npu", "cuda"):
            itemsize = np.dtype(to_numpy_dtype(dtype)).itemsize
            size = int(self.nbytes() // itemsize)
            return TypedStorage(self._untyped, dtype, size)
        if self.device.type == "meta":
            itemsize = np.dtype(to_numpy_dtype(dtype)).itemsize
            size = int(self.nbytes() // itemsize)
            return TypedStorage(self._untyped, dtype, size)
        raise NotImplementedError(f"Unsupported device: {self.device}")

    def resize_(self, new_size):
        itemsize = np.dtype(to_numpy_dtype(self.dtype)).itemsize
        self._untyped.resize_(int(new_size) * itemsize)
        self._size = int(new_size)
        if self.device.type == "cpu":
            buf = self._untyped.buffer()
            self._data = np.frombuffer(buf, dtype=to_numpy_dtype(self.dtype), count=self._size)
        elif self.device.type == "mps":
            buf = self._untyped.buffer()
            self._data = np.frombuffer(buf, dtype=to_numpy_dtype(self.dtype), count=self._size)
        return self


class Storage(TypedStorage):
    pass


def typed_storage_from_numpy(arr, dtype, device=None):
    arr = np.ascontiguousarray(arr, dtype=to_numpy_dtype(dtype))
    untyped = CyCPUUntypedStorage(arr.view(np.uint8), device=device)
    return TypedStorage(untyped, dtype, arr.size, data=arr)


def typed_storage_from_numpy_view(arr, dtype, device=None):
    arr = np.asarray(arr, dtype=to_numpy_dtype(dtype))
    if not arr.flags.c_contiguous:
        raise ValueError("expected contiguous numpy view")
    untyped = _CPUUntypedStorage(arr.view(np.uint8), device=device)
    return TypedStorage(untyped, dtype, arr.size, data=arr)


def empty_cpu_typed_storage(shape, dtype, device=None):
    arr = np.empty(shape, dtype=to_numpy_dtype(dtype))
    return typed_storage_from_numpy(arr, dtype, device=device)


def meta_typed_storage_from_shape(shape, dtype, device=None):
    size = int(np.prod(shape))
    return meta_typed_storage_from_size(size, dtype, device=device)


def meta_typed_storage_from_size(size, dtype, device=None):
    itemsize = np.dtype(to_numpy_dtype(dtype)).itemsize
    untyped = _MetaUntypedStorage(int(size) * itemsize, device=device)
    return TypedStorage(untyped, dtype, int(size))


def npu_typed_storage_from_ptr(device_ptr, size, dtype, device=None):
    from ._cython._storage import cy_npu_storage_from_ptr  # pylint: disable=import-error,no-name-in-module
    return cy_npu_storage_from_ptr(device_ptr, size, dtype, device=device)


def mps_typed_storage_from_numpy(arr, dtype, device=None):
    """Create MPS TypedStorage from a numpy array (copies data into a Metal buffer)."""
    from ._backends.mps.runtime import get_runtime, buffer_contents

    arr = np.ascontiguousarray(arr, dtype=to_numpy_dtype(dtype))
    rt = get_runtime()
    nbytes = int(arr.nbytes)
    metal_buf = rt.create_buffer(max(nbytes, 1))
    ptr = buffer_contents(metal_buf)
    if nbytes > 0:
        ctypes.memmove(ptr, arr.ctypes.data, nbytes)
    untyped = _MPSUntypedStorage(metal_buf, nbytes, device=device)
    data = np.ctypeslib.as_array(
        (ctypes.c_uint8 * nbytes).from_address(ptr)
    )
    typed_data = np.frombuffer(data, dtype=to_numpy_dtype(dtype), count=arr.size)
    return TypedStorage(untyped, dtype, arr.size, data=typed_data)


def mps_typed_storage_from_ptr(metal_buffer, size, dtype, device=None):
    """Create MPS TypedStorage from an existing Metal buffer."""
    from ._backends.mps.runtime import buffer_contents

    itemsize = np.dtype(to_numpy_dtype(dtype)).itemsize
    nbytes = int(size) * itemsize
    untyped = _MPSUntypedStorage(metal_buffer, nbytes, device=device)
    ptr = buffer_contents(metal_buffer)
    data_buf = np.ctypeslib.as_array(
        (ctypes.c_uint8 * nbytes).from_address(ptr)
    )
    typed_data = np.frombuffer(data_buf, dtype=to_numpy_dtype(dtype), count=int(size))
    return TypedStorage(untyped, dtype, int(size), data=typed_data)


def cuda_typed_storage_from_numpy(arr, dtype, device=None, stream=None):
    arr = np.ascontiguousarray(arr, dtype=to_numpy_dtype(dtype))
    from ._backends.cuda import storage as cuda_storage

    untyped = cuda_storage.untyped_from_numpy(arr, device=device, stream=stream)
    return TypedStorage(untyped, dtype, arr.size)


def empty_cuda_typed_storage(shape, dtype, device=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    size = int(np.prod(shape))
    from ._backends.cuda import storage as cuda_storage

    untyped = cuda_storage.empty_untyped(shape, dtype, device=device)
    return TypedStorage(untyped, dtype, size)


def cuda_typed_storage_to_numpy(storage, shape, dtype, stream=None):
    from ._backends.cuda import storage as cuda_storage

    return cuda_storage.to_numpy(storage.untyped_storage(), dtype, shape=shape, stream=stream)


class PendingStorage:
    def __init__(self, shape, dtype, device):
        if isinstance(device, str):
            device = Device(device)
        self._shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        size = 1
        for d in self._shape:
            size *= d
        self._size = int(size)

    def size(self):
        return self._size

    def nbytes(self):
        itemsize = np.dtype(to_numpy_dtype(self.dtype)).itemsize
        return int(self._size * itemsize)

    def data_ptr(self):
        raise RuntimeError("pending tensor has no data")

    @property
    def data(self):
        raise RuntimeError(
            "PendingStorage has no data. Call flush() on the pipeline context "
            "to materialize the storage, or move the tensor to a device."
        )

    def untyped_storage(self):
        return self

    def is_shared(self):
        return False

    def is_pinned(self):
        return False



def pinned_cpu_typed_storage_from_numpy(arr, dtype, device=None):
    from ._backends.npu import runtime as npu_runtime

    arr = np.ascontiguousarray(arr, dtype=to_numpy_dtype(dtype))
    size = int(arr.nbytes)
    host_ptr = npu_runtime.alloc_host(size)
    buf = np.ctypeslib.as_array((ctypes.c_uint8 * size).from_address(int(host_ptr)))
    buf[:] = arr.view(np.uint8).reshape(-1)
    raw = np.frombuffer(buf, dtype=np.uint8)
    untyped = CyPinnedCPUUntypedStorage(raw, host_ptr, device=device)
    data = np.frombuffer(raw, dtype=to_numpy_dtype(dtype), count=arr.size)
    return TypedStorage(untyped, dtype, arr.size, data=data)
