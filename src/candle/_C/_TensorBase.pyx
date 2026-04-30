# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython TensorBase class — torch._C.TensorBase equivalent.

This class provides the base for torch.Tensor. Cython hot-path methods from
_tensor_api.pyx are installed via _install_tensor_api() after import.
"""

from ._tensor_impl cimport TensorImpl
from ._tensor_impl import _StrideTuple, cy_init_tensor_fields

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


class TensorBase(TensorImpl):
    """torch._C.TensorBase equivalent.

    Inherits from TensorImpl (Cython cdef class).
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
        from candle.storage import _warn_typed_storage_removal
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
        from candle._functional import pow as _pow
        return _pow(self, exponent)

    def pow_(self, exponent):
        from candle._functional import pow as _pow
        result = _pow(self, exponent)
        self.copy_(result)
        return self

    def positive(self):
        return self

    def neg(self):
        from candle._functional import neg as _neg
        return _neg(self)

    def abs(self):
        from candle._functional import abs as _abs
        return _abs(self)

    def __idiv__(self, other):
        from candle._functional import div as _div
        result = _div(self, other)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def as_subclass(self, cls):
        return self

    def is_contiguous(self, memory_format=None):
        expected = _compute_strides(self.shape)
        return self.stride == expected

    def contiguous(self, memory_format=None):
        if self.is_contiguous():
            return self
        from candle._dispatch import dispatch
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
            from candle._functional import reshape as reshape_dispatch
            return reshape_dispatch(self, shape)
        from candle._dispatch import dispatch
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
            from candle._dispatch import dispatch
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
            from candle._dispatch import dispatch
            return dispatch("flatten", self.device.type, self, start_dim, end_dim)
        from candle._functional import flatten as flatten_dispatch
        return flatten_dispatch(self, start_dim, end_dim)

    def _transpose_view(self, dim0, dim1):
        return self.cy_transpose(dim0, dim1)

    def transpose(self, dim0, dim1):
        if self.requires_grad:
            from candle._dispatch import dispatch
            return dispatch("transpose", self.device.type, self, dim0, dim1)
        from candle._functional import transpose as transpose_dispatch
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
        from candle.storage import TypedStorage
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

        storage_offset = int(storage_offset)
        size = tuple(int(dim) for dim in size)
        stride = tuple(int(step) for step in stride)

        if len(size) != len(stride):
            raise RuntimeError(
                f"mismatch in length of strides and shape: {len(stride)} != {len(size)}"
            )

        required = 0
        if size and all(dim > 0 for dim in size):
            max_index = storage_offset
            for dim, step in zip(size, stride):
                if dim > 0:
                    max_index += (dim - 1) * step
            required = max_index + 1

        storage_size = typed_storage._size()
        if required > storage_size:
            itemsize = self.dtype.itemsize if getattr(self, 'dtype', None) is not None else typed_storage.dtype.itemsize
            raise RuntimeError(
                f"setStorage: sizes {list(size)}, strides {list(stride)}, storage offset {storage_offset}, "
                f"and itemsize {itemsize} requiring a storage size of {required * itemsize} are out of bounds "
                f"for storage of size {storage_size * itemsize}"
            )

        self.cy_set_runtime_truth(typed_storage, size, stride, storage_offset)
        return self

    def as_strided(self, size, stride, storage_offset=None):
        if storage_offset is None:
            storage_offset = self.offset
        return self.cy_as_strided(size, stride, storage_offset)

    def _ones_like(self):
        from candle._functional import ones_like
        return ones_like(self)

    def record_stream(self, stream):
        pass

    def numpy(self):
        from candle._dtype import bfloat16
        arr = self._numpy_view()
        if self.dtype == bfloat16:
            arr = _bf16_to_f32(arr)
        return arr

    def backward(self, gradient=None, retain_graph=False, create_graph=False, inputs=None):
        from candle.autograd.engine import backward as _backward
        _backward(self, gradient, retain_graph, create_graph, inputs=inputs)

    def pin_memory(self):
        from candle._C import pinned_cpu_typed_storage_from_numpy
        storage = pinned_cpu_typed_storage_from_numpy(self._numpy_view(), self.dtype, device=self.device)
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
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
            if self.grad_fn is not None and hasattr(self.grad_fn, '_register_hook_dict'):
                self.grad_fn._register_hook_dict(self)
        from candle.utils.hooks import RemovableHandle
        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def _is_view(self):
        return self._base is not None

    def _check_inplace(self, other):
        pass

    def add_(self, other, *, alpha=1):
        from candle._functional import add as add_dispatch
        result = add_dispatch(self, other, alpha=alpha)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def mul_(self, other):
        from candle._functional import mul as mul_dispatch
        result = mul_dispatch(self, other)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def relu_(self):
        from candle._functional import relu as relu_dispatch
        result = relu_dispatch(self)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def zero_(self):
        from candle._functional import zeros_like
        result = zeros_like(self)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def fill_(self, value):
        from candle._functional import full_like
        result = full_like(self, value)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def copy_(self, source, non_blocking=None):
        from candle._functional import copy as copy_dispatch
        result = copy_dispatch(self, source)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def new_empty(self, size, *, dtype=None, device=None, requires_grad=False):
        from candle._functional import empty
        return empty(size, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_tensor(self, data, *, dtype=None, device=None, requires_grad=False):
        from candle._functional import tensor
        return tensor(data, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_empty_strided(self, size, stride, *, dtype=None, device=None, requires_grad=False):
        from candle._functional import empty_strided
        return empty_strided(size, stride, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_ones(self, size, *, dtype=None, device=None, requires_grad=False):
        from candle._functional import ones
        return ones(size, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_zeros(self, size, *, dtype=None, device=None, requires_grad=False):
        from candle._functional import zeros
        return zeros(size, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_full(self, size, fill_value, *, dtype=None, device=None, requires_grad=False):
        from candle._functional import full
        return full(size, fill_value, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def var_mean(self, dim=None, keepdim=False, unbiased=True):
        from candle._functional import var_mean as var_mean_dispatch
        return var_mean_dispatch(self, dim, keepdim=keepdim, unbiased=unbiased)

    def __rsub__(self, other):
        from candle._functional import sub as sub_dispatch
        return sub_dispatch(other, self)

    def __getitem__(self, index):
        from candle._functional import getitem as getitem_dispatch
        return getitem_dispatch(self, index)

    def __setitem__(self, index, value):
        from candle._functional import setitem as setitem_dispatch
        return setitem_dispatch(self, index, value)

    def __iadd__(self, other):
        return self.add_(other)

    def __isub__(self, other):
        from candle._functional import sub as sub_dispatch
        result = sub_dispatch(self, other)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def __imul__(self, other):
        return self.mul_(other)

    def __itruediv__(self, other):
        from candle._functional import true_divide as true_divide_dispatch
        result = true_divide_dispatch(self, other)
        self.cy_set_data_runtime_truth_from(result)
        return self

    def __neg__(self):
        from candle._functional import neg as neg_dispatch
        return neg_dispatch(self)

    def clone(self, *, memory_format=None):
        from . import _tensor_api as _tensor_api_mod
        return _tensor_api_mod.tensor_clone(self, memory_format=memory_format)

    def to(self, *args, **kwargs):
        from candle._functional import to as to_dispatch
        return to_dispatch(self, *args, **kwargs)

    def _to_dtype(self, dtype):
        from candle._functional import to_dtype
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
        from candle._tensor_str import _str
        return _str(self)

    def __str__(self):
        from candle._tensor_str import _str
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


def _install_tensor_api(TensorBase):
    """Install Cython tensor API methods on TensorBase."""
    from . import _tensor_api as _tensor_api_mod
    TensorBase._set_device_from_storage = _tensor_api_mod.tensor_set_device_from_storage
    TensorBase._set_dtype_from_storage = _tensor_api_mod.tensor_set_dtype_from_storage
    TensorBase.data = property(TensorBase.data.fget, _tensor_api_mod.tensor_set_data)
    TensorBase.__delattr__ = _tensor_api_mod.tensor_delattr
    TensorBase._fw_get = _tensor_api_mod.tensor_fw_get
    TensorBase._fw_set = _tensor_api_mod.tensor_fw_set
    TensorBase._fw_clear = _tensor_api_mod.tensor_fw_clear
    TensorBase._fw_has = _tensor_api_mod.tensor_fw_has
    TensorBase.untyped_storage = _tensor_api_mod.tensor_untyped_storage
    TensorBase.record_stream = _tensor_api_mod.tensor_record_stream
    TensorBase.is_pinned = _tensor_api_mod.tensor_is_pinned

    TensorBase.__add__ = _tensor_api_mod.tensor_add
    TensorBase.__sub__ = _tensor_api_mod.tensor_sub
    TensorBase.__mul__ = _tensor_api_mod.tensor_mul
    TensorBase.__matmul__ = _tensor_api_mod.tensor_matmul
    TensorBase.__getitem__ = _tensor_api_mod.tensor_getitem
    TensorBase.__setitem__ = _tensor_api_mod.tensor_setitem
    TensorBase.__iadd__ = _tensor_api_mod.tensor_iadd
    TensorBase.__isub__ = _tensor_api_mod.tensor_isub
    TensorBase.__imul__ = _tensor_api_mod.tensor_imul
    TensorBase.__itruediv__ = _tensor_api_mod.tensor_itruediv
    TensorBase.__neg__ = _tensor_api_mod.tensor_neg
    TensorBase.neg = _tensor_api_mod.tensor_neg

    TensorBase.clone = _tensor_api_mod.tensor_clone
    TensorBase.detach = _tensor_api_mod.tensor_detach
    TensorBase.detach_ = _tensor_api_mod.tensor_detach_
    TensorBase.to = _tensor_api_mod.tensor_to
    TensorBase._to_dtype = _tensor_api_mod.tensor_to_dtype
    TensorBase.cpu = _tensor_api_mod.tensor_cpu
    TensorBase.npu = _tensor_api_mod.tensor_npu
    TensorBase.mps = _tensor_api_mod.tensor_mps
    TensorBase.cuda = _tensor_api_mod.tensor_cuda
    TensorBase.backward = _tensor_api_mod.tensor_backward
    TensorBase.relu = _tensor_api_mod.tensor_relu
    TensorBase.is_contiguous = _tensor_api_mod.tensor_is_contiguous
    TensorBase.contiguous = _tensor_api_mod.tensor_contiguous
    TensorBase.reshape = _tensor_api_mod.tensor_reshape
    TensorBase.transpose = _tensor_api_mod.tensor_transpose
    TensorBase.view = _tensor_api_mod.tensor_view
    TensorBase.flatten = _tensor_api_mod.tensor_flatten
    TensorBase.t = _tensor_api_mod.tensor_t
    TensorBase.as_strided = _tensor_api_mod.tensor_as_strided
    TensorBase.size = _tensor_api_mod.tensor_size
    TensorBase.dim = _tensor_api_mod.tensor_dim

    TensorBase.retain_grad = _tensor_api_mod.tensor_retain_grad
    TensorBase.requires_grad_ = _tensor_api_mod.tensor_requires_grad_
    TensorBase.register_hook = _tensor_api_mod.tensor_register_hook

    TensorBase._is_view = _tensor_api_mod.tensor_is_view
    TensorBase._check_inplace = _tensor_api_mod.tensor_check_inplace

    TensorBase.add_ = _tensor_api_mod.tensor_add_
    TensorBase.mul_ = _tensor_api_mod.tensor_mul_
    TensorBase.relu_ = _tensor_api_mod.tensor_relu_
    TensorBase.zero_ = _tensor_api_mod.tensor_zero_
    TensorBase.fill_ = _tensor_api_mod.tensor_fill_
    TensorBase.copy_ = _tensor_api_mod.tensor_copy_

    TensorBase.abs_ = _tensor_api_mod.tensor_abs_
    TensorBase.neg_ = _tensor_api_mod.tensor_neg_
    TensorBase.exp_ = _tensor_api_mod.tensor_exp_
    TensorBase.log_ = _tensor_api_mod.tensor_log_
    TensorBase.log2_ = _tensor_api_mod.tensor_log2_
    TensorBase.log10_ = _tensor_api_mod.tensor_log10_
    TensorBase.sqrt_ = _tensor_api_mod.tensor_sqrt_
    TensorBase.sin_ = _tensor_api_mod.tensor_sin_
    TensorBase.cos_ = _tensor_api_mod.tensor_cos_
    TensorBase.tan_ = _tensor_api_mod.tensor_tan_
    TensorBase.tanh_ = _tensor_api_mod.tensor_tanh_
    TensorBase.sigmoid_ = _tensor_api_mod.tensor_sigmoid_
    TensorBase.floor_ = _tensor_api_mod.tensor_floor_
    TensorBase.ceil_ = _tensor_api_mod.tensor_ceil_
    TensorBase.round_ = _tensor_api_mod.tensor_round_
    TensorBase.trunc_ = _tensor_api_mod.tensor_trunc_
    TensorBase.pow_ = _tensor_api_mod.tensor_pow_
    TensorBase.reciprocal_ = _tensor_api_mod.tensor_reciprocal_
    TensorBase.erfinv_ = _tensor_api_mod.tensor_erfinv_

    TensorBase.sub_ = _tensor_api_mod.tensor_sub_
    TensorBase.clamp_ = _tensor_api_mod.tensor_clamp_
    TensorBase.uniform_ = _tensor_api_mod.tensor_uniform_
    TensorBase.normal_ = _tensor_api_mod.tensor_normal_
    TensorBase.random_ = _tensor_api_mod.tensor_random_
    TensorBase.randint_ = _tensor_api_mod.tensor_randint_
    TensorBase.bernoulli_ = _tensor_api_mod.tensor_bernoulli_
    TensorBase.exponential_ = _tensor_api_mod.tensor_exponential_
    TensorBase.log_normal_ = _tensor_api_mod.tensor_log_normal_
    TensorBase.cauchy_ = _tensor_api_mod.tensor_cauchy_
    TensorBase.geometric_ = _tensor_api_mod.tensor_geometric_

    TensorBase.transpose_ = _tensor_api_mod.tensor_transpose_
    TensorBase.t_ = _tensor_api_mod.tensor_t_
    TensorBase.squeeze_ = _tensor_api_mod.tensor_squeeze_
    TensorBase.unsqueeze_ = _tensor_api_mod.tensor_unsqueeze_
    TensorBase.as_strided_ = _tensor_api_mod.tensor_as_strided_
    TensorBase.swapdims_ = _tensor_api_mod.tensor_swapdims_
    TensorBase.swapaxes_ = _tensor_api_mod.tensor_swapaxes_

    TensorBase.scatter_add = _tensor_api_mod.tensor_scatter_add
    TensorBase.index_fill = _tensor_api_mod.tensor_index_fill
    TensorBase.index_copy = _tensor_api_mod.tensor_index_copy
    TensorBase.index_add = _tensor_api_mod.tensor_index_add
    TensorBase.put_ = _tensor_api_mod.tensor_put_
    TensorBase.scatter_ = _tensor_api_mod.tensor_scatter_
    TensorBase.scatter_add_ = _tensor_api_mod.tensor_scatter_add_
    TensorBase.masked_fill_ = _tensor_api_mod.tensor_masked_fill_
    TensorBase.masked_scatter_ = _tensor_api_mod.tensor_masked_scatter_
    TensorBase.index_put_ = _tensor_api_mod.tensor_index_put_
    TensorBase.index_copy_ = _tensor_api_mod.tensor_index_copy_
    TensorBase.index_fill_ = _tensor_api_mod.tensor_index_fill_
    TensorBase.index_add_ = _tensor_api_mod.tensor_index_add_

    TensorBase.new_empty = _tensor_api_mod.tensor_new_empty
    TensorBase.new_tensor = _tensor_api_mod.tensor_new_tensor
    TensorBase.new_empty_strided = _tensor_api_mod.tensor_new_empty_strided
    TensorBase._ones_like = _tensor_api_mod.tensor_ones_like
    TensorBase.new_ones = _tensor_api_mod.tensor_new_ones
    TensorBase.new_zeros = _tensor_api_mod.tensor_new_zeros
    TensorBase.new_full = _tensor_api_mod.tensor_new_full
    TensorBase.div_ = _tensor_api_mod.tensor_div_
    TensorBase.unflatten = _tensor_api_mod.tensor_unflatten
    TensorBase.bitwise_and_ = _tensor_api_mod.tensor_bitwise_and_
    TensorBase.bitwise_or_ = _tensor_api_mod.tensor_bitwise_or_
    TensorBase.bitwise_xor_ = _tensor_api_mod.tensor_bitwise_xor_
    TensorBase.type = _tensor_api_mod.tensor_type
    TensorBase.type_as = _tensor_api_mod.tensor_type_as
    TensorBase.reshape_as = _tensor_api_mod.tensor_reshape_as
    TensorBase.permute = _tensor_api_mod.tensor_permute
    TensorBase.mean = _tensor_api_mod.tensor_mean
    TensorBase.std = _tensor_api_mod.tensor_std
    TensorBase.repeat = _tensor_api_mod.tensor_repeat
    TensorBase.tile = _tensor_api_mod.tensor_tile
    TensorBase.flip = _tensor_api_mod.tensor_flip
    TensorBase.logsumexp = _tensor_api_mod.tensor_logsumexp
    TensorBase.trace = _tensor_api_mod.tensor_trace
    TensorBase.det = _tensor_api_mod.tensor_det
    TensorBase.matrix_power = _tensor_api_mod.tensor_matrix_power
    TensorBase.dist = _tensor_api_mod.tensor_dist
    TensorBase.renorm = _tensor_api_mod.tensor_renorm
    TensorBase.nansum = _tensor_api_mod.tensor_nansum
    TensorBase.nanmean = _tensor_api_mod.tensor_nanmean
    TensorBase.argwhere = _tensor_api_mod.tensor_argwhere
    TensorBase.baddbmm = _tensor_api_mod.tensor_baddbmm
    TensorBase.vsplit = _tensor_api_mod.tensor_vsplit
    TensorBase.hsplit = _tensor_api_mod.tensor_hsplit
    TensorBase.dsplit = _tensor_api_mod.tensor_dsplit
    TensorBase.take_along_dim = _tensor_api_mod.tensor_take_along_dim
    TensorBase.cummin = _tensor_api_mod.tensor_cummin
    TensorBase.log1p = _tensor_api_mod.tensor_log1p
    TensorBase.expm1 = _tensor_api_mod.tensor_expm1
    TensorBase.lt = _tensor_api_mod.tensor_lt
    TensorBase.le = _tensor_api_mod.tensor_le
    TensorBase.gt = _tensor_api_mod.tensor_gt
    TensorBase.ge = _tensor_api_mod.tensor_ge
    TensorBase.abs = _tensor_api_mod.tensor_abs
    TensorBase.exp = _tensor_api_mod.tensor_exp
    TensorBase.log = _tensor_api_mod.tensor_log
    TensorBase.sqrt = _tensor_api_mod.tensor_sqrt
    TensorBase.sin = _tensor_api_mod.tensor_sin
    TensorBase.cos = _tensor_api_mod.tensor_cos
    TensorBase.tan = _tensor_api_mod.tensor_tan
    TensorBase.tanh = _tensor_api_mod.tensor_tanh
    TensorBase.sigmoid = _tensor_api_mod.tensor_sigmoid
    TensorBase.floor = _tensor_api_mod.tensor_floor
    TensorBase.ceil = _tensor_api_mod.tensor_ceil
    TensorBase.round = _tensor_api_mod.tensor_round
    TensorBase.trunc = _tensor_api_mod.tensor_trunc
    TensorBase.frac = _tensor_api_mod.tensor_frac
    TensorBase.log2 = _tensor_api_mod.tensor_log2
    TensorBase.log10 = _tensor_api_mod.tensor_log10
    TensorBase.exp2 = _tensor_api_mod.tensor_exp2
    TensorBase.rsqrt = _tensor_api_mod.tensor_rsqrt
    TensorBase.sign = _tensor_api_mod.tensor_sign
    TensorBase.signbit = _tensor_api_mod.tensor_signbit
    TensorBase.square = _tensor_api_mod.tensor_square
    TensorBase.isnan = _tensor_api_mod.tensor_isnan
    TensorBase.isinf = _tensor_api_mod.tensor_isinf
    TensorBase.isfinite = _tensor_api_mod.tensor_isfinite
    TensorBase.sinh = _tensor_api_mod.tensor_sinh
    TensorBase.cosh = _tensor_api_mod.tensor_cosh
    TensorBase.asinh = _tensor_api_mod.tensor_asinh
    TensorBase.acosh = _tensor_api_mod.tensor_acosh
    TensorBase.atanh = _tensor_api_mod.tensor_atanh
    TensorBase.erf = _tensor_api_mod.tensor_erf
    TensorBase.erfc = _tensor_api_mod.tensor_erfc
    TensorBase.reciprocal = _tensor_api_mod.tensor_reciprocal
    TensorBase.tril = _tensor_api_mod.tensor_tril
    TensorBase.triu = _tensor_api_mod.tensor_triu
    TensorBase.diag = _tensor_api_mod.tensor_diag
    TensorBase.add = _tensor_api_mod.tensor_add_method
    TensorBase.sub = _tensor_api_mod.tensor_sub_method
    TensorBase.mul = _tensor_api_mod.tensor_mul_method
    TensorBase.div = _tensor_api_mod.tensor_div_method
    TensorBase.pow = _tensor_api_mod.tensor_pow_method
    TensorBase.matmul = _tensor_api_mod.tensor_matmul_method
    TensorBase.__rsub__ = _tensor_api_mod.tensor_rsub
    TensorBase.__rmul__ = _tensor_api_mod.tensor_rmul
    TensorBase.__truediv__ = _tensor_api_mod.tensor_truediv
    TensorBase.__rtruediv__ = _tensor_api_mod.tensor_rtruediv
    TensorBase.__pow__ = _tensor_api_mod.tensor_pow_op
    TensorBase.__rpow__ = _tensor_api_mod.tensor_rpow
    TensorBase.__floordiv__ = _tensor_api_mod.tensor_floordiv
    TensorBase.__rfloordiv__ = _tensor_api_mod.tensor_rfloordiv
    TensorBase.__mod__ = _tensor_api_mod.tensor_mod
    TensorBase.__rmod__ = _tensor_api_mod.tensor_rmod
    TensorBase.__rmatmul__ = _tensor_api_mod.tensor_rmatmul
    TensorBase.__rlshift__ = _tensor_api_mod.tensor_rlshift
    TensorBase.__rrshift__ = _tensor_api_mod.tensor_rrshift
    TensorBase.__and__ = _tensor_api_mod.tensor_and
    TensorBase.__or__ = _tensor_api_mod.tensor_or
    TensorBase.__xor__ = _tensor_api_mod.tensor_xor
    TensorBase.all = _tensor_api_mod.tensor_all_method
    TensorBase.any = _tensor_api_mod.tensor_any_method
    TensorBase.sum = _tensor_api_mod.tensor_sum_method
    TensorBase.prod = _tensor_api_mod.tensor_prod_method
    TensorBase.var = _tensor_api_mod.tensor_var_method
    TensorBase.var_mean = _tensor_api_mod.tensor_var_mean_method
    TensorBase.norm = _tensor_api_mod.tensor_norm_method
    TensorBase.count_nonzero = _tensor_api_mod.tensor_count_nonzero_method
    TensorBase.cumsum = _tensor_api_mod.tensor_cumsum_method
    TensorBase.cumprod = _tensor_api_mod.tensor_cumprod_method
    TensorBase.cummax = _tensor_api_mod.tensor_cummax_method
    TensorBase.argsort = _tensor_api_mod.tensor_argsort_method
    TensorBase.sort = _tensor_api_mod.tensor_sort_method
    TensorBase.topk = _tensor_api_mod.tensor_topk_method
    TensorBase.eq = _tensor_api_mod.tensor_eq_method
    TensorBase.ne = _tensor_api_mod.tensor_ne_method
    TensorBase.allclose = _tensor_api_mod.tensor_allclose_method
    TensorBase.isclose = _tensor_api_mod.tensor_isclose_method
    TensorBase.equal = _tensor_api_mod.tensor_equal_method
    TensorBase.view_as = _tensor_api_mod.tensor_view_as
    TensorBase.expand = _tensor_api_mod.tensor_expand_method
    TensorBase.expand_as = _tensor_api_mod.tensor_expand_as_method
    TensorBase.expand_copy = _tensor_api_mod.tensor_expand_copy_method
    TensorBase.narrow = _tensor_api_mod.tensor_narrow_method
    TensorBase.select = _tensor_api_mod.tensor_select_method
    TensorBase.unfold = _tensor_api_mod.tensor_unfold_method
    TensorBase.moveaxis = _tensor_api_mod.tensor_moveaxis_method
    TensorBase.swapdims = _tensor_api_mod.tensor_swapdims_method
    TensorBase.swapaxes = _tensor_api_mod.tensor_swapaxes_method
    TensorBase.gather = _tensor_api_mod.tensor_gather_method
    TensorBase.scatter = _tensor_api_mod.tensor_scatter_method
    TensorBase.index_select = _tensor_api_mod.tensor_index_select_method
    TensorBase.take = _tensor_api_mod.tensor_take_method
    TensorBase.masked_fill = _tensor_api_mod.tensor_masked_fill_method
    TensorBase.masked_select = _tensor_api_mod.tensor_masked_select_method
    TensorBase.index_put = _tensor_api_mod.tensor_index_put_method
    TensorBase.slice = _tensor_api_mod.tensor_slice_method
    TensorBase.slice_copy = _tensor_api_mod.tensor_slice_copy_method
    TensorBase.slice_scatter = _tensor_api_mod.tensor_slice_scatter_method
    TensorBase.nonzero = _tensor_api_mod.tensor_nonzero_method
    TensorBase.sum_to_size = _tensor_api_mod.tensor_sum_to_size_method
    TensorBase.softplus = _tensor_api_mod.tensor_softplus_method
    TensorBase.clamp = _tensor_api_mod.tensor_clamp_method
    TensorBase.relu6 = _tensor_api_mod.tensor_relu6_method
    TensorBase.hardtanh = _tensor_api_mod.tensor_hardtanh_method
    TensorBase.min = _tensor_api_mod.tensor_min_method
    TensorBase.max = _tensor_api_mod.tensor_max_method
    TensorBase.amin = _tensor_api_mod.tensor_amin_method
    TensorBase.amax = _tensor_api_mod.tensor_amax_method
    TensorBase.addmm = _tensor_api_mod.tensor_addmm_method
    TensorBase.bmm = _tensor_api_mod.tensor_bmm_method
    TensorBase.mm = _tensor_api_mod.tensor_mm_method
    TensorBase.chunk = _tensor_api_mod.tensor_chunk_method
    TensorBase.split = _tensor_api_mod.tensor_split_method
    TensorBase.roll = _tensor_api_mod.tensor_roll_method
    TensorBase.rot90 = _tensor_api_mod.tensor_rot90_method
    TensorBase.addcdiv = _tensor_api_mod.tensor_addcdiv_method
    TensorBase.addcmul = _tensor_api_mod.tensor_addcmul_method
    TensorBase.hypot = _tensor_api_mod.tensor_hypot_method
    TensorBase.lerp = _tensor_api_mod.tensor_lerp_method
    TensorBase.atan2 = _tensor_api_mod.tensor_atan2_method
    TensorBase.asin = _tensor_api_mod.tensor_asin_method
    TensorBase.acos = _tensor_api_mod.tensor_acos_method
    TensorBase.atan = _tensor_api_mod.tensor_atan_method
    TensorBase.as_strided_copy = _tensor_api_mod.tensor_as_strided_copy_method
    TensorBase.as_strided_scatter = _tensor_api_mod.tensor_as_strided_scatter_method
    TensorBase.multinomial = _tensor_api_mod.tensor_multinomial_method
    TensorBase.ndim = property(_tensor_api_mod.tensor_ndim_fget)
    TensorBase.T = property(_tensor_api_mod.tensor_T_fget)
    TensorBase.is_floating_point = _tensor_api_mod.tensor_is_floating_point
    TensorBase.is_complex = _tensor_api_mod.tensor_is_complex
    TensorBase.clamp_min = _tensor_api_mod.tensor_clamp_min_method
    TensorBase.clamp_max = _tensor_api_mod.tensor_clamp_max_method
    TensorBase.fmin = _tensor_api_mod.tensor_fmin_method
    TensorBase.fmax = _tensor_api_mod.tensor_fmax_method
    TensorBase.where = _tensor_api_mod.tensor_where_method
    TensorBase.logaddexp = _tensor_api_mod.tensor_logaddexp_method
    TensorBase.logaddexp2 = _tensor_api_mod.tensor_logaddexp2_method
    TensorBase.remainder = _tensor_api_mod.tensor_remainder_method
    TensorBase.fmod = _tensor_api_mod.tensor_fmod_method
    TensorBase.squeeze = _tensor_api_mod.tensor_squeeze_method
    TensorBase.unsqueeze = _tensor_api_mod.tensor_unsqueeze_method
    TensorBase.argmax = _tensor_api_mod.tensor_argmax_method
    TensorBase.argmin = _tensor_api_mod.tensor_argmin_method
    TensorBase.logical_and = _tensor_api_mod.tensor_logical_and
    TensorBase.logical_or = _tensor_api_mod.tensor_logical_or
    TensorBase.logical_xor = _tensor_api_mod.tensor_logical_xor
    TensorBase.logical_not = _tensor_api_mod.tensor_logical_not
    TensorBase.bitwise_and = _tensor_api_mod.tensor_bitwise_and
    TensorBase.bitwise_or = _tensor_api_mod.tensor_bitwise_or
    TensorBase.bitwise_xor = _tensor_api_mod.tensor_bitwise_xor
    TensorBase.bitwise_not = _tensor_api_mod.tensor_bitwise_not
    TensorBase.movedim = _tensor_api_mod.tensor_movedim
    TensorBase.diagonal = _tensor_api_mod.tensor_diagonal
    TensorBase.unbind = _tensor_api_mod.tensor_unbind

    TensorBase.numpy = _tensor_api_mod.tensor_numpy
    TensorBase._numpy_view = _tensor_api_mod.tensor_numpy_view
    TensorBase.pin_memory = _tensor_api_mod.tensor_pin_memory
