# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython TensorImpl — C-level storage for Tensor metadata.

Mirrors PyTorch's TensorImpl: shape/stride/device/dtype/autograd fields
are stored as C-typed attributes for zero-overhead access from Cython code.
VersionCounter is inlined as a C int64.
"""

from libc.stdint cimport int64_t

DEF MAX_NDIM = 64


cdef class TensorImpl:
    """C-level base for Tensor. All hot fields are cdef-typed."""

    # -- shape / stride (C arrays + cached Python tuples) --
    cdef int64_t _c_shape[MAX_NDIM]
    cdef int64_t _c_stride[MAX_NDIM]
    cdef public int _ndim
    cdef public int64_t _c_numel
    cdef public int64_t _c_offset

    # -- device (int enum + cached object) --
    cdef public int _device_type       # 0=cpu 1=npu 2=cuda 3=mps 4=meta
    cdef public int _device_index
    cdef public object _device_obj     # cached device instance

    # -- dtype (int code + cached object) --
    cdef public int _dtype_code
    cdef public int _itemsize
    cdef public object _dtype_obj      # cached DType instance

    # -- storage --
    cdef public object _storage

    # -- autograd --
    cdef public bint requires_grad
    cdef public object grad
    cdef public object grad_fn
    cdef public int64_t _version_value   # inlined VersionCounter
    cdef public object _base
    cdef public object _view_meta
    cdef public bint _pending
    cdef public bint _retain_grad
    cdef public object _backward_hooks

    # -- Python tuple caches (avoid re-creating every access) --
    cdef public tuple _shape_tuple
    cdef public object _stride_tuple   # _StrideTuple instance

    # -- allow dynamic attrs (__dict__) for _fw_tangents etc. --
    cdef dict __dict__

    # -- cached version counter proxy --
    cdef public object _vc_proxy

    # ---------------------------------------------------------------
    # Initialisation helpers
    # ---------------------------------------------------------------

    cdef inline void _set_shape(self, tuple shape):
        cdef int n = len(shape)
        cdef int i
        cdef int64_t numel = 1
        self._ndim = n
        for i in range(n):
            self._c_shape[i] = <int64_t>shape[i]
            numel *= self._c_shape[i]
        self._c_numel = numel
        self._shape_tuple = shape

    cdef inline void _set_stride(self, object stride):
        cdef int n = len(stride)
        cdef int i
        for i in range(n):
            self._c_stride[i] = <int64_t>stride[i]
        self._stride_tuple = stride

    cdef inline void _set_device_from_obj(self, object dev):
        """Cache device object and extract type_code/index."""
        self._device_obj = dev
        cdef str dt = getattr(dev, "type", None)
        if dt is None:
            dt = str(dev)
        if dt == "cpu":
            self._device_type = 0
        elif dt == "npu":
            self._device_type = 1
        elif dt == "cuda":
            self._device_type = 2
        elif dt == "mps":
            self._device_type = 3
        elif dt == "meta":
            self._device_type = 4
        else:
            self._device_type = -1
        cdef object idx = getattr(dev, "index", None)
        self._device_index = <int>(idx if idx is not None else -1)

    cdef inline void _set_dtype_from_obj(self, object dtype):
        """Cache dtype object and extract code/itemsize."""
        self._dtype_obj = dtype
        self._itemsize = <int>getattr(dtype, "itemsize", 4)
        # Assign a numeric code based on dtype name for fast comparison
        cdef str name = getattr(dtype, "name", "")
        if name == "float32":
            self._dtype_code = 0
        elif name == "float16":
            self._dtype_code = 1
        elif name == "float64":
            self._dtype_code = 2
        elif name == "bfloat16":
            self._dtype_code = 3
        elif name == "int32":
            self._dtype_code = 4
        elif name == "int64":
            self._dtype_code = 5
        elif name == "int16":
            self._dtype_code = 6
        elif name == "int8":
            self._dtype_code = 7
        elif name == "uint8":
            self._dtype_code = 8
        elif name == "bool":
            self._dtype_code = 9
        else:
            self._dtype_code = -1

    # ---------------------------------------------------------------
    # Properties — zero-overhead access to cached Python objects
    # ---------------------------------------------------------------

    @property
    def shape(self):
        return self._shape_tuple

    @shape.setter
    def shape(self, value):
        self._set_shape(tuple(value))

    @property
    def stride(self):
        return self._stride_tuple

    @stride.setter
    def stride(self, value):
        self._stride_tuple = value
        cdef int n = len(value)
        cdef int i
        for i in range(n):
            self._c_stride[i] = <int64_t>value[i]

    @property
    def offset(self):
        return self._c_offset

    @offset.setter
    def offset(self, value):
        self._c_offset = <int64_t>value

    @property
    def device(self):
        return self._storage.device

    @property
    def dtype(self):
        return self._storage.dtype

    # ---------------------------------------------------------------
    # VersionCounter — inlined as C int64
    # For views (_base is set), delegate to the base tensor so that
    # base._version_counter is view._version_counter (identity).
    # ---------------------------------------------------------------

    @property
    def _version_counter(self):
        if self._base is not None:
            return self._base._version_counter
        cdef object proxy = self._vc_proxy
        if proxy is not None:
            return proxy
        proxy = _VersionCounterProxy.__new__(_VersionCounterProxy, self)
        self._vc_proxy = proxy
        return proxy

    @_version_counter.setter
    def _version_counter(self, value):
        # When setting from a proxy, share the underlying impl's value
        if isinstance(value, _VersionCounterProxy):
            self._version_value = (<_VersionCounterProxy>value)._impl._version_value
        else:
            self._version_value = <int64_t>getattr(value, "value", 0)
        # Invalidate cached proxy
        self._vc_proxy = None

    def _bump_version(self):
        if self._base is not None:
            self._base._bump_version()
        else:
            self._version_value += 1

    # ---------------------------------------------------------------
    # Storage access
    # ---------------------------------------------------------------

    def storage(self):
        return self._storage

    # ---------------------------------------------------------------
    # Fast dim / numel
    # ---------------------------------------------------------------

    def dim(self):
        return self._ndim

    def numel(self):
        return self._c_numel

    def element_size(self):
        return self._itemsize

    # ---------------------------------------------------------------
    # Pickle support
    # ---------------------------------------------------------------

    def __reduce__(self):
        # Delegate to Tensor's own __reduce__ if available
        reduce_fn = getattr(type(self), "_tensor_reduce", None)
        if reduce_fn is not None:
            return reduce_fn(self)
        raise TypeError(f"cannot pickle {type(self).__name__}")


# -------------------------------------------------------------------
# VersionCounter proxy — lightweight wrapper around TensorImpl._version_value
# -------------------------------------------------------------------

cdef class _VersionCounterProxy:
    cdef TensorImpl _impl

    def __cinit__(self, TensorImpl impl):
        self._impl = impl

    @property
    def value(self):
        return self._impl._version_value

    @value.setter
    def value(self, int64_t v):
        self._impl._version_value = v

    cpdef void bump(self):
        self._impl._version_value += 1
