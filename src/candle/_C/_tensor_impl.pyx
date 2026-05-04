# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython TensorImpl — runtime-owned tensor metadata and cached runtime state.

TensorImpl is the internal owner for tensor shape/stride/device/dtype/autograd
state and other cached runtime fields used by the dispatcher and lifecycle code.
Python Tensor remains the public shell that exposes the user-facing API.
VersionCounter is inlined as a C int64.
"""

from libc.stdint cimport int64_t

import candle._dtype as _dtype_mod


class _StrideTuple(tuple):
    """Tuple subclass that also supports torch-style stride() calls."""

    def __call__(self, dim=None):
        if dim is None:
            return tuple(self)
        if dim < 0:
            dim += len(self)
        return self[dim]


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

cdef object _StrideTuple_cls = None

cdef inline object _coerce_stride_tuple(object stride):
    global _StrideTuple_cls
    if _StrideTuple_cls is None:
        from candle._C import _StrideTuple
        _StrideTuple_cls = _StrideTuple
    if isinstance(stride, _StrideTuple_cls):
        return stride
    return _StrideTuple_cls(stride)


cdef class TensorImpl:
    """Internal runtime owner for tensor metadata and cached state.

    Python Tensor is the public shell; TensorImpl holds the runtime-owned fields
    that back that shell and keep hot-path metadata in Cython-managed storage.
    """
    # Field and method declarations in _tensor_impl.pxd

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
        stride = _coerce_stride_tuple(stride)
        cdef int n = len(stride)
        cdef int i
        for i in range(n):
            self._c_stride[i] = <int64_t>stride[i]
        self._stride_tuple = _StrideTuple(stride)

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
        self._stride_tuple = _StrideTuple(value)
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
        if isinstance(value, _VersionCounterProxy):
            self._vc_proxy = value
            self._version_value = (<_VersionCounterProxy>value).value
        else:
            self._version_value = <int64_t>getattr(value, "value", 0)
            self._vc_proxy = None

    def _bump_version(self):
        cdef object proxy
        if self._base is not None:
            self._base._bump_version()
            return
        proxy = self._vc_proxy
        if proxy is not None:
            (<_VersionCounterProxy>proxy).bump()
            return
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
        return self._output_nr

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

    # ---------------------------------------------------------------
    # View operations — share storage, only update layout
    # ---------------------------------------------------------------

    cpdef object cy_detach(self):
        cdef object tensor_type = type(self)
        cdef TensorImpl out = <TensorImpl>tensor_type.__new__(tensor_type)
        cy_init_tensor_fields(
            out,
            self._storage,
            self._shape_tuple,
            self._stride_tuple,
            self._c_offset,
            False,
            None,
            None,
            None,
            None,
            self._pending,
            False,
            None,
            self._version_value,
            self._version_counter,
        )
        return out

    cdef inline void _attach_view_runtime_truth(self, TensorImpl view):
        view._version_value = self._version_value
        view._base = self._base if self._base is not None else self
        view._vc_proxy = None
        view._view_meta = None
        view._view_func = None
        view._rev_view_func = None
        view._pending = False
        view._retain_grad = False
        view._backward_hooks = None
        view._output_nr = 0

    cpdef object cy_view(self, tuple new_shape):
        cdef int64_t new_numel = 1
        cdef int i
        cdef int new_ndim = len(new_shape)
        cdef object tensor_type = type(self)
        for i in range(new_ndim):
            new_numel *= <int64_t>new_shape[i]
        if new_numel != self._c_numel:
            raise RuntimeError(
                f"shape '{new_shape}' is invalid for input of size {self._c_numel}")
        cdef TensorImpl v = <TensorImpl>tensor_type.__new__(tensor_type)
        v._storage = self._storage
        v._set_shape(new_shape)
        cdef list strides = [0] * new_ndim
        cdef int64_t acc = 1
        for i in range(new_ndim - 1, -1, -1):
            strides[i] = acc
            acc *= <int64_t>new_shape[i]
        v._set_stride(tuple(strides))
        v._c_offset = self._c_offset
        v._device_type = self._device_type
        v._device_index = self._device_index
        v._device_obj = self._device_obj
        v._dtype_code = self._dtype_code
        v._itemsize = self._itemsize
        v._dtype_obj = self._dtype_obj
        v._dispatch_keys = self._dispatch_keys
        v.requires_grad = self.requires_grad
        v.grad = None
        v.grad_fn = self.grad_fn
        self._attach_view_runtime_truth(v)
        return v

    cpdef object cy_as_strided(self, tuple size, tuple stride, int64_t storage_offset):
        cdef object tensor_type = type(self)
        cdef TensorImpl v = <TensorImpl>tensor_type.__new__(tensor_type)
        v._storage = self._storage
        v._set_shape(size)
        v._set_stride(stride)
        v._c_offset = storage_offset
        v._device_type = self._device_type
        v._device_index = self._device_index
        v._device_obj = self._device_obj
        v._dtype_code = self._dtype_code
        v._itemsize = self._itemsize
        v._dtype_obj = self._dtype_obj
        v._dispatch_keys = self._dispatch_keys
        v.requires_grad = self.requires_grad
        v.grad = None
        v.grad_fn = self.grad_fn
        self._attach_view_runtime_truth(v)
        return v

    cpdef object cy_transpose(self, int dim0, int dim1):
        cdef int ndim = self._ndim
        if dim0 < 0:
            dim0 += ndim
        if dim1 < 0:
            dim1 += ndim
        if dim0 < 0 or dim0 >= ndim or dim1 < 0 or dim1 >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[-{ndim}, {ndim-1}], but got {dim0} and {dim1})")
        cdef list new_shape = list(self._shape_tuple)
        cdef list new_stride = list(self._stride_tuple)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_stride[dim0], new_stride[dim1] = new_stride[dim1], new_stride[dim0]
        return self.cy_as_strided(tuple(new_shape), tuple(new_stride), self._c_offset)

    cpdef object cy_set_runtime_truth(
        self,
        object typed_storage,
        tuple size,
        object stride,
        int64_t storage_offset,
    ):
        cdef object device = getattr(typed_storage, "device", None)
        cdef object dtype = getattr(typed_storage, "dtype", None)
        self._storage = typed_storage
        self._set_shape(size)
        self._set_stride(stride)
        self._c_offset = storage_offset
        if device is not None:
            self._set_device_from_obj(device)
        if dtype is not None:
            self._set_dtype_from_obj(dtype)
        self._bump_version()
        return self

    cpdef object cy_set_data_runtime_truth_from(self, object other):
        # Internal fast path: callers already validated other as Tensor/TensorImpl.
        # Keep this helper as a trusted copy boundary rather than re-validating here.
        cdef TensorImpl src = <TensorImpl>other
        self._storage = src._storage
        self._set_shape(src._shape_tuple)
        self._set_stride(src._stride_tuple)
        self._c_offset = src._c_offset
        # Copy cached runtime metadata directly so set_data_ preserves the source
        # tensor's already-resolved device/dtype truth without re-deriving it from
        # Python objects or re-running the broader cy_set_runtime_truth path.
        self._device_type = src._device_type
        self._device_index = src._device_index
        self._device_obj = src._device_obj
        self._dtype_code = src._dtype_code
        self._itemsize = src._itemsize
        self._dtype_obj = src._dtype_obj
        self._recompute_dispatch_keys()
        self._bump_version()
        return self


# -------------------------------------------------------------------
# Module-level tensor factory functions
# -------------------------------------------------------------------

cpdef void cy_init_tensor_fields(
    TensorImpl t,
    object storage,
    tuple shape,
    object stride,
    int64_t offset,
    bint requires_grad,
    object grad,
    object grad_fn,
    object base,
    object view_meta,
    bint pending,
    bint retain_grad,
    object backward_hooks,
    int64_t version_value,
    object vc_proxy,
):
    t._storage = storage
    t._set_shape(shape)
    t._set_stride(stride)
    t._c_offset = offset
    t.requires_grad = requires_grad
    t.grad = grad
    t.grad_fn = grad_fn
    t._base = base
    t._view_meta = view_meta
    t._view_func = None
    t._rev_view_func = None
    t._pending = pending
    t._retain_grad = retain_grad
    t._backward_hooks = backward_hooks
    t._version_value = version_value
    t._vc_proxy = vc_proxy
    t._output_nr = 0

    cdef object dev = getattr(storage, "device", None)
    if dev is not None:
        t._set_device_from_obj(dev)
    cdef object dtype = getattr(storage, "dtype", None)
    if dtype is not None:
        t._set_dtype_from_obj(dtype)
    t._recompute_dispatch_keys()


cpdef object cy_make_tensor_from_storage(
    object storage,
    tuple shape,
    object stride,
    int64_t offset=0,
    bint requires_grad=False,
):
    from candle._tensor import Tensor

    cdef TensorImpl t = Tensor.__new__(Tensor)
    cy_init_tensor_fields(
        t,
        storage,
        shape,
        stride,
        offset,
        requires_grad,
        None,
        None,
        None,
        None,
        False,
        False,
        None,
        0,
        None,
    )
    return t


cpdef object cy_make_view_tensor(
    object base,
    object storage,
    tuple shape,
    object stride,
    int64_t offset=0,
):
    from candle._tensor import Tensor

    cdef TensorImpl b = <TensorImpl>base
    cdef object root = b._base if b._base is not None else base
    cdef TensorImpl t = Tensor.__new__(Tensor)
    cy_init_tensor_fields(
        t,
        storage,
        shape,
        tuple(stride),
        offset,
        b.requires_grad,
        None,
        b.grad_fn,
        root,
        None,
        False,
        False,
        None,
        (<TensorImpl>root)._version_value,
        None,
    )
    return t


# -------------------------------------------------------------------
# VersionCounter proxy — lightweight wrapper around TensorImpl._version_value
# -------------------------------------------------------------------

cdef class _VersionCounterProxy:
    # Field declarations in _tensor_impl.pxd

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


# -------------------------------------------------------------------
# Standalone VersionCounter — used in contexts that do not have a
# TensorImpl as a base.  Mirrors torch's Variable._version field shape.
# -------------------------------------------------------------------

class VersionCounter:
    """Standalone version counter (used when TensorImpl is not the base)."""
    __slots__ = ("value",)

    def __init__(self, value=0):
        self.value = int(value)

    def bump(self):
        self.value += 1
