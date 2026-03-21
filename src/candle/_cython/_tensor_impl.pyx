# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython TensorImpl — C-level storage for Tensor metadata.

Mirrors PyTorch's TensorImpl: shape/stride/device/dtype/autograd fields
are stored as C-typed attributes for zero-overhead access from Cython code.
VersionCounter is inlined as a C int64.
"""

from libc.stdint cimport int64_t

import candle._dtype as _dtype_mod

DEF MAX_NDIM = 64

# Dispatch key bit values (must match keys.py DispatchKey enum)
DEF _DK_CPU = 1 << 15
DEF _DK_NPU = 1 << 13
DEF _DK_CUDA = 1 << 14
DEF _DK_MPS = 1 << 21       # PrivateUse2
DEF _DK_META = 1 << 12
DEF _DK_AUTOGRAD_CPU = 1 << 6
DEF _DK_AUTOGRAD_NPU = 1 << 7
DEF _DK_AUTOGRAD_CUDA = 1 << 8
DEF _DK_AUTOGRAD_MPS = 1 << 22  # PrivateUse3
DEF _DK_AUTOGRAD_META = 1 << 10
DEF _DK_ADINPLACEORVIEW = 1 << 4
DEF _DK_AUTOGRAD = 1 << 11


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

    # -- precomputed dispatch key bits (Step 4) --
    # Bitmask of DispatchKey values based on device_type and requires_grad.
    # Updated by _set_device_from_obj and requires_grad setter.
    cdef public unsigned int _dispatch_keys

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
        """Cache device object and extract type_code/index, compute dispatch keys."""
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
        self._recompute_dispatch_keys()

    cdef inline void _recompute_dispatch_keys(self):
        """Recompute _dispatch_keys from _device_type and requires_grad."""
        cdef unsigned int dk = 0
        cdef int devt = self._device_type
        if devt == 0:    # cpu
            dk = _DK_CPU
        elif devt == 1:  # npu
            dk = _DK_NPU
        elif devt == 2:  # cuda
            dk = _DK_CUDA
        elif devt == 3:  # mps
            dk = _DK_MPS
        elif devt == 4:  # meta
            dk = _DK_META
        else:
            dk = _DK_CPU
        if self.requires_grad:
            dk |= _DK_ADINPLACEORVIEW | _DK_AUTOGRAD
            if devt == 0:
                dk |= _DK_AUTOGRAD_CPU
            elif devt == 1:
                dk |= _DK_AUTOGRAD_NPU
            elif devt == 2:
                dk |= _DK_AUTOGRAD_CUDA
            elif devt == 3:
                dk |= _DK_AUTOGRAD_MPS
            elif devt == 4:
                dk |= _DK_AUTOGRAD_META
        self._dispatch_keys = dk

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
    # Safe read-only metadata / scalar helpers (migrated from Tensor)
    # ---------------------------------------------------------------

    @property
    def output_nr(self):
        return 0

    @property
    def is_cuda(self):
        return self._device_type == 2

    @property
    def is_cpu(self):
        return self._device_type == 0

    @property
    def is_npu(self):
        return self._device_type == 1

    @property
    def is_meta(self):
        return self._device_type == 4

    @property
    def is_leaf(self):
        return self.grad_fn is None

    @property
    def is_sparse(self):
        return bool(getattr(self, "_is_sparse", False))

    @property
    def layout(self):
        return getattr(self, "_layout", "strided")

    @layout.setter
    def layout(self, value):
        self._layout = value

    @property
    def is_quantized(self):
        return False

    def storage_offset(self):
        return self._c_offset

    def get_device(self):
        if self._device_type == 0:
            return -1
        return self._device_index if self._device_index >= 0 else 0

    def ndimension(self):
        return self._ndim

    def size(self, dim=None):
        if dim is None:
            return self._shape_tuple
        if dim < 0:
            dim += self._ndim
        if dim < 0 or dim >= self._ndim:
            raise IndexError("Dimension out of range")
        return self._c_shape[dim]

    def nelement(self):
        return self._c_numel

    def item(self):
        if self._c_numel != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        if self._device_type != 0:
            return self.to("cpu").item()
        return self._numpy_view().flat[0].item()

    def tolist(self):
        if self._device_type != 0:
            return self.to("cpu").tolist()
        return self._numpy_view().tolist()

    def __int__(self):
        return int(self.item())

    def __float__(self):
        return float(self.item())

    def __bool__(self):
        if self._c_numel == 0:
            raise RuntimeError("Boolean value of Tensor with no values is ambiguous")
        if self._c_numel > 1:
            raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
        return bool(self.item())

    def __repr__(self):
        from candle._printing import format_tensor
        return format_tensor(self)

    def __str__(self):
        from candle._printing import format_tensor
        return format_tensor(self)

    def __len__(self):
        if self._ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self._c_shape[0]

    def __iter__(self):
        if self._ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        cdef int64_t i
        cdef int64_t n = self._c_shape[0]
        for i in range(n):
            yield self[i]

    @staticmethod
    def _is_scalar_comparable(other):
        return isinstance(other, (int, float, bool))

    def __hash__(self):
        return id(self)

    # ---------------------------------------------------------------
    # Dtype shorthand wrappers (migrated from Tensor)
    # ---------------------------------------------------------------

    def float(self):
        return self._to_dtype(_dtype_mod.float32) if self.dtype != _dtype_mod.float32 else self

    def half(self):
        return self._to_dtype(_dtype_mod.float16) if self.dtype != _dtype_mod.float16 else self

    def double(self):
        return self._to_dtype(_dtype_mod.float64) if self.dtype != _dtype_mod.float64 else self

    def bfloat16(self):
        return self._to_dtype(_dtype_mod.bfloat16) if self.dtype != _dtype_mod.bfloat16 else self

    def long(self):
        return self._to_dtype(_dtype_mod.int64) if self.dtype != _dtype_mod.int64 else self

    def int(self):
        return self._to_dtype(_dtype_mod.int32) if self.dtype != _dtype_mod.int32 else self

    def short(self):
        return self._to_dtype(_dtype_mod.int16) if self.dtype != _dtype_mod.int16 else self

    def char(self):
        return self._to_dtype(_dtype_mod.int8) if self.dtype != _dtype_mod.int8 else self

    def byte(self):
        return self._to_dtype(_dtype_mod.uint8) if self.dtype != _dtype_mod.uint8 else self

    def bool(self):
        return self._to_dtype(_dtype_mod.bool) if self.dtype != _dtype_mod.bool else self

    # ---------------------------------------------------------------
    # Comparison dunders (migrated from Tensor)
    # ---------------------------------------------------------------
    # Cython cdef classes require __richcmp__ instead of individual
    # __eq__/__ne__/__lt__/__le__/__gt__/__ge__ methods.

    def __richcmp__(self, other, int op):
        from candle._tensor import Tensor
        from candle._functional import eq as eq_dispatch
        from candle._functional import ne as ne_dispatch
        from candle._functional import lt as lt_dispatch
        from candle._functional import le as le_dispatch
        from candle._functional import gt as gt_dispatch
        from candle._functional import ge as ge_dispatch

        if not (isinstance(other, Tensor) or self._is_scalar_comparable(other)):
            if op == 2:    # Py_EQ
                return False
            if op == 3:    # Py_NE
                return True
            return NotImplemented

        if op == 0:        # Py_LT
            return lt_dispatch(self, other)
        if op == 1:        # Py_LE
            return le_dispatch(self, other)
        if op == 2:        # Py_EQ
            return eq_dispatch(self, other)
        if op == 3:        # Py_NE
            return ne_dispatch(self, other)
        if op == 4:        # Py_GT
            return gt_dispatch(self, other)
        if op == 5:        # Py_GE
            return ge_dispatch(self, other)
        return NotImplemented

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
