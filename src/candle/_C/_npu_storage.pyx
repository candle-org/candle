# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython cdef classes for NPU device storage.

Replaces Python _NPUUntypedStorage + TypedStorage for the NPU path with
cdef classes that use __dealloc__ for deterministic device memory cleanup,
eliminating weakref.finalize overhead (~8-10 us per NPU op).
"""
from libc.stdint cimport int64_t

cdef object _allocator_mod = None
cdef object _runtime_mod = None


cdef inline object _get_alloc(int device_id):
    global _allocator_mod
    if _allocator_mod is None:
        from candle._backends.npu import allocator as _am
        _allocator_mod = _am
    return _allocator_mod.get_allocator(device_id)


cdef inline object _get_runtime(int device_id):
    global _runtime_mod
    if _runtime_mod is None:
        from candle._backends.npu import runtime as _rt
        _runtime_mod = _rt
    return _runtime_mod.get_runtime(device_id)


cdef class FastNPUStorage:
    """NPU device storage with deterministic cleanup via __dealloc__.

    Replaces _NPUUntypedStorage. __dealloc__ is called synchronously by
    CPython refcount when the object is deallocated — no GC delay, no
    weakref overhead.

    Exposes the same interface as _NPUUntypedStorage for drop-in compatibility.
    """
    cdef public int64_t _device_ptr
    cdef public int64_t _nbytes
    cdef public object device
    cdef object _alloc

    def __cinit__(self, int64_t ptr, int64_t nbytes, object device):
        self._device_ptr = ptr
        self._nbytes = nbytes
        self.device = device
        try:
            self._alloc = _get_alloc(device.index or 0)
        except Exception:  # pylint: disable=broad-except
            import sys
            if not sys.is_finalizing():
                import warnings
                warnings.warn(
                    f"FastNPUStorage: allocator unavailable for device {device}, "
                    "device memory will not be freed",
                    RuntimeWarning, stacklevel=2)
            self._alloc = None

    def __dealloc__(self):
        """Synchronous Python object teardown, but asynchronous device free.

        We cannot safely return the device pointer directly to the allocator's
        free-list here because the producing stream may still have in-flight
        kernels touching the buffer. Instead, mirror the runtime.py behavior:
        defer the free into runtime._deferred_frees and let synchronize() drain
        it after the stream sync point.
        """
        if self._device_ptr != 0 and self._alloc is not None:
            try:
                runtime = _get_runtime(self.device.index or 0)
                runtime.defer_free(self._device_ptr)
            except Exception:  # pylint: disable=broad-except
                # If runtime is unavailable at interpreter shutdown, leak rather than crash.
                pass
            self._device_ptr = 0

    def data_ptr(self):
        return self._device_ptr

    def nbytes(self):
        return self._nbytes

    def is_pinned(self):
        return False

    def is_shared(self):
        return False

    def buffer(self):
        raise RuntimeError("Cannot get buffer of NPU storage on CPU")

    def untyped_storage(self):
        return self

    def size(self):
        """Byte count (for untyped storage, size == nbytes)."""
        return self._nbytes

    @property
    def dtype(self):
        """Untyped storage has no dtype — returns None for compat."""
        return None

    @property
    def _finalizer(self):
        """No-op shim — FastNPUStorage has no weakref finalizer.
        Code that calls _finalizer.detach() (e.g. resize_) will work.
        """
        return _NoopFinalizer()

    def resize_(self, new_nbytes):
        """In-place resize: allocate new buffer, D2D copy, free old on stream."""
        from candle._backends.npu import runtime as npu_runtime
        from candle._backends.npu import state as npu_state
        new_nbytes = int(new_nbytes)
        if new_nbytes == self._nbytes:
            return self
        device_id = self.device.index or 0
        runtime = npu_runtime.get_runtime(device_id)
        stream = npu_state.current_stream(device_id).stream
        alloc = _get_alloc(device_id)
        runtime.activate()
        new_ptr = alloc.malloc(new_nbytes, stream=stream)
        copy_bytes = min(self._nbytes, new_nbytes)
        if copy_bytes:
            npu_runtime.memcpy_d2d(new_ptr, copy_bytes, self._device_ptr,
                                   runtime=runtime, stream=stream)
        alloc.free(self._device_ptr, stream=stream)
        self._device_ptr = new_ptr
        self._nbytes = new_nbytes
        return self


class _NoopFinalizer:
    """No-op shim for weakref finalizer compatibility."""
    def detach(self):
        pass
    def __bool__(self):
        return False


cdef class FastTypedStorage:
    """Typed NPU storage wrapping FastNPUStorage.

    Replaces Python TypedStorage for the NPU path. Exposes the same
    interface as far as NPU callers require: device, dtype, data_ptr(),
    untyped_storage(), size(), nbytes(), copy_(), _reinterpret(), resize_(), clone().
    """
    cdef public FastNPUStorage _untyped
    cdef public object _dtype
    cdef public int64_t _size

    def __cinit__(self, FastNPUStorage untyped, object dtype, int64_t size):
        self._untyped = untyped
        self._dtype = dtype
        self._size = size

    @property
    def device(self):
        return self._untyped.device

    @property
    def dtype(self):
        return self._dtype

    def data_ptr(self):
        return self._untyped._device_ptr

    def untyped_storage(self):
        return self._untyped

    def size(self):
        return self._size

    def nbytes(self):
        return int(self._size * getattr(self._dtype, 'itemsize', 4))

    def is_pinned(self):
        return False

    def is_shared(self):
        return False

    def copy_(self, other):
        """Copy from another storage of the same device type."""
        if self.device.type != other.device.type:
            raise NotImplementedError("cross-device copy_ not supported")
        if self.device.type == "npu":
            from candle._backends.npu import runtime as npu_runtime
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
        raise NotImplementedError(f"Unsupported device: {self.device}")

    def _reinterpret(self, dtype):
        """Return a view of the same bytes with a different dtype."""
        if dtype == self._dtype:
            return self
        itemsize = getattr(dtype, 'itemsize', 4)
        size = int(self.nbytes() // itemsize)
        return FastTypedStorage(self._untyped, dtype, size)

    def resize_(self, new_size):
        """Resize typed storage by delegating to untyped resize_."""
        cdef object itemsize = getattr(self._dtype, 'itemsize', 4)
        self._untyped.resize_(int(new_size) * itemsize)
        self._size = int(new_size)
        return self

    def clone(self):
        """Return a copy with independently-owned device memory."""
        from candle._backends.npu import runtime as npu_runtime
        from candle._backends.npu import state as npu_state
        device_id = self.device.index or 0
        runtime = npu_runtime.get_runtime(device_id)
        stream = npu_state.current_stream(device_id).stream
        nbytes = self.nbytes()
        alloc = _get_alloc(device_id)
        runtime.activate()
        new_ptr = alloc.malloc(nbytes, stream=stream)
        try:
            if nbytes:
                npu_runtime.memcpy_d2d(new_ptr, nbytes, self._untyped._device_ptr,
                                       runtime=runtime, stream=stream)
            new_untyped = FastNPUStorage(new_ptr, nbytes, self.device)
        except Exception:  # pylint: disable=broad-except
            alloc.free(new_ptr, stream)
            raise
        return FastTypedStorage(new_untyped, self._dtype, self._size)
