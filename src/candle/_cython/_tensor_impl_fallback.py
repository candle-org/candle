"""Pure-Python fallback for _tensor_impl.pyx.

Provides TensorImpl base class and _VersionCounterProxy when Cython is
not available.  The Tensor class inherits from this unconditionally —
when Cython IS available, the .pyx version replaces this module.
"""

import builtins as _builtins

from candle._dtype import float32, float16, float64, bfloat16, int8, int16, int32, int64, uint8
from candle._dtype import bool as dtype_bool
from candle._functional import eq as eq_dispatch, ne as ne_dispatch, lt as lt_dispatch, le as le_dispatch, gt as gt_dispatch, ge as ge_dispatch
from candle._printing import format_tensor


class _VersionCounterProxy:
    """Lightweight proxy wrapping TensorImpl._version_value."""
    __slots__ = ("_impl",)

    def __init__(self, impl):
        self._impl = impl

    @property
    def value(self):
        return self._impl._version_value

    @value.setter
    def value(self, v):
        self._impl._version_value = int(v)

    def bump(self):
        self._impl._version_value += 1


class TensorImpl:
    """Pure-Python mirror of the Cython TensorImpl cdef class."""
    __slots__ = (
        "_storage",
        "_shape_tuple", "_stride_tuple",
        "_c_offset", "_c_numel", "_ndim",
        "_device_type", "_device_index", "_device_obj",
        "_dtype_code", "_itemsize", "_dtype_obj",
        "requires_grad", "grad", "grad_fn",
        "_version_value", "_vc_proxy",
        "_base", "_view_meta",
        "_pending", "_retain_grad", "_backward_hooks",
        "__dict__",
    )

    # -- shape --
    @property
    def shape(self):
        return self._shape_tuple

    @shape.setter
    def shape(self, value):
        t = tuple(value)
        self._shape_tuple = t
        self._ndim = len(t)
        numel = 1
        for d in t:
            numel *= d
        self._c_numel = numel

    # -- stride --
    @property
    def stride(self):
        return self._stride_tuple

    @stride.setter
    def stride(self, value):
        self._stride_tuple = value

    # -- offset --
    @property
    def offset(self):
        return self._c_offset

    @offset.setter
    def offset(self, value):
        self._c_offset = int(value)

    # -- device (always from storage, matching PyTorch TensorImpl) --
    @property
    def device(self):
        return self._storage.device

    # -- dtype (always from storage, matching PyTorch TensorImpl) --
    @property
    def dtype(self):
        return self._storage.dtype

    # -- version counter (inlined, views delegate to _base) --
    @property
    def _version_counter(self):
        if self._base is not None:
            return self._base._version_counter
        proxy = self._vc_proxy
        if proxy is not None:
            return proxy
        proxy = _VersionCounterProxy(self)
        self._vc_proxy = proxy
        return proxy

    @_version_counter.setter
    def _version_counter(self, value):
        if isinstance(value, _VersionCounterProxy):
            self._version_value = value._impl._version_value
        else:
            self._version_value = int(getattr(value, "value", 0))
        self._vc_proxy = None

    def _bump_version(self):
        if self._base is not None:
            self._base._bump_version()
        else:
            self._version_value += 1

    # -- storage --
    def storage(self):
        return self._storage

    # -- fast dim/numel --
    def dim(self):
        return self._ndim

    def numel(self):
        return self._c_numel

    def element_size(self):
        return self._itemsize

    @property
    def output_nr(self):
        return 0

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    @property
    def is_cpu(self):
        return self.device.type == "cpu"

    @property
    def is_npu(self):
        return self.device.type == "npu"

    @property
    def is_meta(self):
        return self.device.type == "meta"

    @property
    def is_leaf(self):
        return self.grad_fn is None

    @property
    def is_sparse(self):
        return _builtins.bool(getattr(self, "_is_sparse", False))

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
        return self.offset

    def get_device(self):
        if self.device.type == "cpu":
            return -1
        return self.device.index if self.device.index is not None else 0

    def ndimension(self):
        return self._ndim

    def size(self, dim=None):
        if dim is None:
            return self.shape
        if dim < 0:
            dim += len(self.shape)
        if dim < 0 or dim >= len(self.shape):
            raise IndexError("Dimension out of range")
        return self.shape[dim]

    def nelement(self):
        return self.numel()

    def item(self):
        if self.numel() != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        if self.device.type != "cpu":
            return self.to("cpu").item()
        return self._numpy_view().flat[0].item()

    def tolist(self):
        if self.device.type != "cpu":
            return self.to("cpu").tolist()
        return self._numpy_view().tolist()

    def __int__(self):
        return _builtins.int(self.item())

    def __float__(self):
        return _builtins.float(self.item())

    def __bool__(self):
        if self.numel() == 0:
            raise RuntimeError("Boolean value of Tensor with no values is ambiguous")
        if self.numel() > 1:
            raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
        return _builtins.bool(self.item())

    def __repr__(self):
        return format_tensor(self)

    def __str__(self):
        return format_tensor(self)

    def __len__(self):
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __iter__(self):
        if self.dim() == 0:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def _is_scalar_comparable(other):
        return isinstance(other, (_builtins.int, _builtins.float, _builtins.bool))

    def float(self):
        return self._to_dtype(float32) if self.dtype != float32 else self

    def half(self):
        return self._to_dtype(float16) if self.dtype != float16 else self

    def double(self):
        return self._to_dtype(float64) if self.dtype != float64 else self

    def bfloat16(self):
        return self._to_dtype(bfloat16) if self.dtype != bfloat16 else self

    def long(self):
        return self._to_dtype(int64) if self.dtype != int64 else self

    def int(self):
        return self._to_dtype(int32) if self.dtype != int32 else self

    def short(self):
        return self._to_dtype(int16) if self.dtype != int16 else self

    def char(self):
        return self._to_dtype(int8) if self.dtype != int8 else self

    def byte(self):
        return self._to_dtype(uint8) if self.dtype != uint8 else self

    def bool(self):
        return self._to_dtype(dtype_bool) if self.dtype != dtype_bool else self

    def __gt__(self, other):
        from candle._tensor import Tensor
        if isinstance(other, Tensor) or self._is_scalar_comparable(other):
            return gt_dispatch(self, other)
        return NotImplemented

    def __lt__(self, other):
        from candle._tensor import Tensor
        if isinstance(other, Tensor) or self._is_scalar_comparable(other):
            return lt_dispatch(self, other)
        return NotImplemented

    def __ge__(self, other):
        from candle._tensor import Tensor
        if isinstance(other, Tensor) or self._is_scalar_comparable(other):
            return ge_dispatch(self, other)
        return NotImplemented

    def __le__(self, other):
        from candle._tensor import Tensor
        if isinstance(other, Tensor) or self._is_scalar_comparable(other):
            return le_dispatch(self, other)
        return NotImplemented

    def __eq__(self, other):
        from candle._tensor import Tensor
        if isinstance(other, Tensor) or self._is_scalar_comparable(other):
            return eq_dispatch(self, other)
        return False

    def __ne__(self, other):
        from candle._tensor import Tensor
        if isinstance(other, Tensor) or self._is_scalar_comparable(other):
            return ne_dispatch(self, other)
        return True

    def __hash__(self):
        return id(self)
