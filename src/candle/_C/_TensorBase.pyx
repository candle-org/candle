# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython TensorBase class — torch._C.TensorBase equivalent.

This class provides the base for torch.Tensor. Cython hot-path methods from
_tensor_api.pyx are installed via _install_tensor_api() after import.
"""

from ._tensor_impl cimport TensorImpl
from ._tensor_impl import _StrideTuple, cy_init_tensor_fields


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
        from candle._C import _compute_strides
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
        from candle._C import _compute_strides
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
        from candle._functional import ones_like
        return ones_like(self)

    def record_stream(self, stream):
        pass

    def numpy(self):
        from candle._C import _bf16_to_f32
        from candle._dtype import bfloat16
        arr = self._numpy_view()
        if self.dtype == bfloat16:
            arr = _bf16_to_f32(arr)
        return arr

    def backward(self, gradient=None, retain_graph=False, create_graph=False, inputs=None):
        from candle.autograd.engine import backward as _backward
        _backward(self, gradient, retain_graph, create_graph, inputs=inputs)

    def pin_memory(self):
        from candle._C import pinned_cpu_typed_storage_from_numpy, _compute_strides
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

    def clone(self):
        from candle._functional import clone as clone_dispatch
        return clone_dispatch(self)

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
