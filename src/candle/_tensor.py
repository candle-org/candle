import numpy as np

from ._cython._tensor_impl import cy_make_tensor_from_storage, cy_make_view_tensor  # pylint: disable=import-error,no-name-in-module
from ._storage import (
    Storage,
    empty_cpu_typed_storage,
    meta_typed_storage_from_shape,
    npu_typed_storage_from_ptr,
    pinned_cpu_typed_storage_from_numpy,
    typed_storage_from_numpy,
)
from ._device import _default_device, device as Device
from ._dtype import float32, float16, float64, bfloat16, int8, int16, int32, int64, uint8
from ._dtype import bool as dtype_bool
from ._dtype import to_numpy_dtype
from ._functional import add, mul, matmul, relu, sum, mean as mean_dispatch, std as std_dispatch, true_divide as true_divide_dispatch, repeat as repeat_dispatch, chunk as chunk_dispatch, split as split_dispatch, vsplit as vsplit_dispatch, hsplit as hsplit_dispatch, dsplit as dsplit_dispatch, unbind as unbind_dispatch, abs as abs_dispatch, neg as neg_dispatch
from ._functional import sub as sub_dispatch, div as div_dispatch
from ._functional import exp as exp_dispatch, log as log_dispatch, sqrt as sqrt_dispatch
from ._functional import sin as sin_dispatch, cos as cos_dispatch, tan as tan_dispatch
from ._functional import tanh as tanh_dispatch, sigmoid as sigmoid_dispatch
from ._functional import floor as floor_dispatch, ceil as ceil_dispatch, round as round_dispatch
from ._functional import trunc as trunc_dispatch, frac as frac_dispatch
from ._functional import pow as pow_dispatch, log2 as log2_dispatch, log10 as log10_dispatch
from ._functional import exp2 as exp2_dispatch, rsqrt as rsqrt_dispatch
from ._functional import sign as sign_dispatch, signbit as signbit_dispatch, square as square_dispatch
from ._functional import isnan as isnan_dispatch, isinf as isinf_dispatch, isfinite as isfinite_dispatch
from ._functional import sinh as sinh_dispatch, cosh as cosh_dispatch
from ._functional import asinh as asinh_dispatch, acosh as acosh_dispatch, atanh as atanh_dispatch
from ._functional import erf as erf_dispatch, erfc as erfc_dispatch, softplus as softplus_dispatch
from ._functional import clamp as clamp_dispatch, clamp_min as clamp_min_dispatch, clamp_max as clamp_max_dispatch
from ._functional import relu6 as relu6_dispatch, hardtanh as hardtanh_dispatch
from ._functional import min as min_dispatch, max as max_dispatch
from ._functional import amin as amin_dispatch, amax as amax_dispatch
from ._functional import fmin as fmin_dispatch, fmax as fmax_dispatch
from ._functional import where as where_dispatch
from ._functional import atan as atan_dispatch, atan2 as atan2_dispatch
from ._functional import asin as asin_dispatch, acos as acos_dispatch
from ._functional import lerp as lerp_dispatch
from ._functional import addcmul as addcmul_dispatch, addcdiv as addcdiv_dispatch
from ._functional import logaddexp as logaddexp_dispatch, logaddexp2 as logaddexp2_dispatch
from ._functional import hypot as hypot_dispatch, remainder as remainder_dispatch, fmod as fmod_dispatch
from ._functional import all as all_dispatch, any as any_dispatch, argmax as argmax_dispatch
from ._functional import argmin as argmin_dispatch, count_nonzero as count_nonzero_dispatch
from ._functional import allclose as allclose_dispatch, isclose as isclose_dispatch, equal as equal_dispatch
from ._functional import eq as eq_dispatch, ne as ne_dispatch, lt as lt_dispatch, le as le_dispatch, gt as gt_dispatch, ge as ge_dispatch
from ._functional import cumsum as cumsum_dispatch, cumprod as cumprod_dispatch, cummax as cummax_dispatch
from ._functional import argsort as argsort_dispatch, sort as sort_dispatch, topk as topk_dispatch
from ._functional import tril as tril_dispatch, triu as triu_dispatch, diag as diag_dispatch
from ._functional import reshape as reshape_dispatch
from ._functional import transpose as transpose_dispatch, view as view_dispatch, to as to_dispatch
from ._functional import nonzero as nonzero_dispatch, masked_select as masked_select_dispatch
from ._functional import gather as gather_dispatch, scatter as scatter_dispatch
from ._functional import index_select as index_select_dispatch, take as take_dispatch
from ._functional import narrow as narrow_dispatch, select as select_dispatch
from ._functional import expand as expand_dispatch
from ._functional import masked_fill_ as masked_fill__dispatch, masked_fill as masked_fill_dispatch
from ._functional import index_put_ as index_put__dispatch, index_put as index_put_dispatch
from ._functional import index_copy_ as index_copy__dispatch
from ._functional import index_fill_ as index_fill__dispatch
from ._functional import index_add_ as index_add__dispatch
from ._functional import scatter_ as scatter__dispatch, scatter_add_ as scatter_add__dispatch
from ._functional import masked_scatter_ as masked_scatter__dispatch
from ._functional import unfold as unfold_dispatch
from ._functional import squeeze as squeeze_dispatch, unsqueeze as unsqueeze_dispatch, permute as permute_dispatch
from ._functional import var as var_dispatch, norm as norm_dispatch, prod as prod_dispatch
from ._functional import mm as mm_dispatch, bmm as bmm_dispatch
from ._functional import floor_divide as floor_divide_dispatch
from ._functional import slice as slice_dispatch, slice_copy as slice_copy_dispatch, slice_scatter as slice_scatter_dispatch
from ._functional import expand_copy as expand_copy_dispatch
from ._functional import sum_to_size as sum_to_size_dispatch
from ._functional import as_strided_ as as_strided__dispatch
from ._functional import as_strided_copy as as_strided_copy_dispatch
from ._functional import as_strided_scatter as as_strided_scatter_dispatch
from ._functional import tile as tile_dispatch, flip as flip_dispatch, roll as roll_dispatch, rot90 as rot90_dispatch
from ._functional import movedim as movedim_dispatch, moveaxis as moveaxis_dispatch, diagonal as diagonal_dispatch
from ._functional import reciprocal as reciprocal_dispatch, addmm as addmm_dispatch
from ._functional import log1p as log1p_dispatch, expm1 as expm1_dispatch
from .autograd.engine import backward as _backward

# TensorImpl base class: compiled Cython extension required.
from ._cython._tensor_impl import TensorImpl as _TensorBase  # pylint: disable=import-error,no-name-in-module
from ._cython._tensor_impl import _VersionCounterProxy  # noqa: F401  # pylint: disable=import-error,no-name-in-module
from ._cython._tensor_impl import cy_init_tensor_fields  # pylint: disable=import-error,no-name-in-module


class _StrideTuple(tuple):
    """A tuple subclass that is also callable, matching PyTorch's stride() API.

    Supports both attribute access (t.stride) and method call (t.stride()),
    as well as per-dimension access (t.stride(dim)).
    """
    def __call__(self, dim=None):
        if dim is None:
            return tuple(self)
        if dim < 0:
            dim += len(self)
        return self[dim]


def _compute_strides(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return _StrideTuple(reversed(stride))


def _bf16_to_f32(arr):
    """Convert bfloat16 (stored as uint16) to float32."""
    u32 = arr.astype(np.uint32) << 16
    return u32.view(np.float32)


def _f32_to_bf16(arr):
    """Convert float32 to bfloat16 (stored as uint16), round-to-nearest-even."""
    u32 = arr.view(np.uint32)
    # Round to nearest even: add bias + (lsb of result)
    rounding_bias = (u32 >> 16) & 1
    u32 = u32 + 0x7FFF + rounding_bias
    return (u32 >> 16).astype(np.uint16)


class _HookHandle:
    _next_id = 0

    def __init__(self, hooks):
        self._hooks = hooks
        self.id = _HookHandle._next_id
        _HookHandle._next_id += 1

    def remove(self):
        if self._hooks is None:
            return
        self._hooks.pop(self.id, None)
        self._hooks = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


class Tensor(_TensorBase):
    def __init__(self, storage, shape, stride, offset=0, requires_grad=False):
        cy_init_tensor_fields(
            self,
            storage,
            tuple(shape),
            _StrideTuple(stride),
            int(offset),
            bool(requires_grad),
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

    _DEVICE_MAP = {"cpu": 0, "npu": 1, "cuda": 2, "mps": 3, "meta": 4}
    # Dispatch key bit values — must stay in sync with _tensor_impl.pyx
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
        """Returns the underlying data tensor (detached from autograd graph)."""
        return self.detach()

    @data.setter
    def data(self, new_data):
        """Replace the tensor's data with new_data (in-place)."""
        if not isinstance(new_data, Tensor):
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
        """Return the underlying untyped storage.

        This is needed for compatibility with safetensors which calls
        tensor.untyped_storage().nbytes() to determine storage size.
        """
        return self._storage.untyped_storage()

    def data_ptr(self):
        """Return the address of the first element of this tensor."""
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

    def is_contiguous(self, memory_format=None):
        """Check if tensor is contiguous in row-major order."""
        expected = _compute_strides(self.shape)
        return self.stride == expected

    def contiguous(self, memory_format=None):
        """Return contiguous tensor (copy if not already contiguous)."""
        if self.is_contiguous():
            return self

        # Use dispatch to stay on device (avoid numpy round-trip)
        from ._dispatch import dispatch
        return dispatch("contiguous", self.device.type, self)

    def _numpy_view(self):
        if self.device.type == "meta":
            raise RuntimeError("meta tensor has no data")
        if self.device.type != "cpu":
            # Convert to CPU to get numpy view
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
            return reshape_dispatch(self, shape)
        from ._dispatch import dispatch
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
            from ._dispatch import dispatch
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
            from ._dispatch import dispatch
            return dispatch("flatten", self.device.type, self, start_dim, end_dim)
        ndim = len(self.shape)
        if ndim == 0:
            return self.reshape((1,))
        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim
        if start_dim < 0 or start_dim >= ndim:
            raise IndexError("Dimension out of range")
        if end_dim < 0 or end_dim >= ndim:
            raise IndexError("Dimension out of range")
        if start_dim > end_dim:
            raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")

        flattened = 1
        for d in self.shape[start_dim:end_dim + 1]:
            flattened *= d
        new_shape = self.shape[:start_dim] + (flattened,) + self.shape[end_dim + 1:]
        return self.reshape(new_shape)

    def _transpose_view(self, dim0, dim1):
        view = self.cy_transpose(dim0, dim1)
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
            "op": "transpose",
            "shape": tuple(view.shape),
            "stride": tuple(view.stride),
            "offset": int(view.offset),
            "creation_mode": creation_mode,
            "creation_kind": creation_kind,
        }
        return view

    def transpose(self, dim0, dim1):
        if self.requires_grad:
            from ._dispatch import dispatch
            return dispatch("transpose", self.device.type, self, dim0, dim1)
        return self._transpose_view(dim0, dim1)

    def transpose_(self, dim0, dim1):
        ndim = len(self.shape)
        d0 = dim0 if dim0 >= 0 else dim0 + ndim
        d1 = dim1 if dim1 >= 0 else dim1 + ndim
        if d0 < 0 or d0 >= ndim or d1 < 0 or d1 >= ndim:
            raise IndexError("Dimension out of range")
        shape = list(self.shape)
        stride = list(self.stride)
        shape[d0], shape[d1] = shape[d1], shape[d0]
        stride[d0], stride[d1] = stride[d1], stride[d0]
        return self.as_strided_(tuple(shape), tuple(stride))

    def t(self):
        """Transpose for 2D tensors. Expects input to be <= 2-D tensor and transposes dimensions 0 and 1."""
        if len(self.shape) > 2:
            raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {len(self.shape)}D")
        if len(self.shape) < 2:
            return self
        if self.requires_grad:
            from ._dispatch import dispatch
            return dispatch("transpose", self.device.type, self, 0, 1)
        return self._transpose_view(0, 1)

    def t_(self):
        """In-place transpose for 2D tensors."""
        if len(self.shape) > 2:
            raise RuntimeError(f"t_() expects a tensor with <= 2 dimensions, but self is {len(self.shape)}D")
        self._check_inplace()
        if len(self.shape) < 2:
            return self
        shape = list(self.shape)
        stride = list(self.stride)
        shape[0], shape[1] = shape[1], shape[0]
        stride[0], stride[1] = stride[1], stride[0]
        self.shape = tuple(shape)
        self.stride = _StrideTuple(tuple(stride))
        return self

    @property
    def T(self):
        if len(self.shape) > 2:
            raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {len(self.shape)}D")
        if len(self.shape) < 2:
            return self
        if self.requires_grad:
            from ._dispatch import dispatch
            return dispatch("transpose", self.device.type, self, 0, 1)
        return self._transpose_view(0, 1)

    def view_as(self, other):
        """Reshape this tensor to the same shape as other."""
        return self.view(other.shape)

    def var_mean(self, dim=None, keepdim=False, unbiased=True):
        from ._functional import var_mean
        return var_mean(self, dim=dim, keepdim=keepdim, unbiased=unbiased)

    def new_empty(self, size, *, dtype=None, device=None, requires_grad=False):
        """Create a new empty tensor with the same dtype and device as self."""
        from ._creation import empty
        dt = dtype if dtype is not None else self.dtype
        dev = device if device is not None else self.device
        return empty(size, dtype=dt, device=dev)

    def new_tensor(self, data, *, dtype=None, device=None, requires_grad=False):
        """Create a new tensor with the given data using this tensor's dtype and device by default."""
        from ._creation import tensor
        dt = dtype if dtype is not None else self.dtype
        dev = device if device is not None else self.device
        return tensor(data, dtype=dt, device=dev)

    def new_empty_strided(self, size, stride, *, dtype=None, device=None, requires_grad=False):
        """Create a new empty tensor with the given size and stride."""
        dt = dtype if dtype is not None else self.dtype
        dev = device if device is not None else self.device
        if dev.type == "cpu":
            numel = 1
            for s in size:
                numel *= s
            arr = np.empty(numel, dtype=to_numpy_dtype(dt))
            storage = typed_storage_from_numpy(arr, dt, device=dev)
            return cy_make_tensor_from_storage(storage, tuple(size), tuple(stride), 0, requires_grad)
        else:
            from ._creation import empty
            t = empty(size, dtype=dt, device=dev)
            return t

    def set_(self, typed_storage, storage_offset, size, stride):
        from ._storage import TypedStorage

        if not isinstance(typed_storage, TypedStorage):
            raise TypeError("set_() currently only supports TypedStorage input")
        if not isinstance(storage_offset, int) or storage_offset < 0:
            raise RuntimeError("storage_offset must be a non-negative integer")
        if not isinstance(size, (tuple, list)) or not isinstance(stride, (tuple, list)):
            raise RuntimeError("size and stride must be tuple-like")
        if len(size) != len(stride):
            raise RuntimeError("size and stride must have the same length")
        if any(not isinstance(dim, int) for dim in size):
            raise RuntimeError("size values must be integers")
        if any(not isinstance(step, int) for step in stride):
            raise RuntimeError("stride values must be integers")
        if any(dim < 0 for dim in size):
            raise RuntimeError("size values must be non-negative")
        if any(step < 0 for step in stride):
            raise RuntimeError("stride values must be non-negative")

        size = tuple(size)
        stride = tuple(stride)
        device = getattr(typed_storage, "device", None)
        dtype = getattr(typed_storage, "dtype", None)
        if device is None or dtype is None:
            raise RuntimeError("typed_storage must have device and dtype")

        if not any(dim == 0 for dim in size):
            max_index = storage_offset
            for dim, step in zip(size, stride):
                max_index += (dim - 1) * step
            if max_index >= typed_storage.size():
                raise RuntimeError("set_() view exceeds storage bounds")

        self._check_inplace()
        return self.cy_set_runtime_truth(
            typed_storage,
            size,
            _StrideTuple(stride),
            int(storage_offset),
        )

    def as_strided(self, size, stride, storage_offset=None):
        """Create a view of the tensor with given size, stride, and storage_offset."""
        offset = storage_offset if storage_offset is not None else self.offset
        return self.cy_as_strided(tuple(size), tuple(stride), offset)

    def _ones_like(self):
        if self.device.type == "meta":
            storage = meta_typed_storage_from_shape(self.shape, self.dtype, device=self.device)
            return cy_make_tensor_from_storage(storage, self.shape, self.stride, 0, False)
        arr = np.ones(self.shape, dtype=to_numpy_dtype(self.dtype))
        storage = typed_storage_from_numpy(arr, self.dtype, device=self.device if self.device.type == "cpu" else None)
        stride = tuple(np.array(arr.strides) // arr.itemsize)
        tensor = cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)
        if self.device.type != "cpu":
            return tensor.to(self.device)
        return tensor

    def record_stream(self, stream):
        if self.device.type != "npu":
            return
        from ._backends.npu import allocator as npu_allocator

        alloc = npu_allocator.get_allocator(self.device.index or 0)
        alloc.record_stream(self.storage().data_ptr(), stream.stream)

    def numpy(self):
        if self._pending:
            from ._dispatch.pipeline import current_pipeline

            pipe = current_pipeline()
            if pipe is not None:
                pipe.flush()
        if self.device.type == "meta":
            raise RuntimeError("meta tensor has no data")
        if self.device.type != "cpu":
            raise RuntimeError("numpy() is only available for CPU tensors")
        return self._numpy_view()

    def backward(self, gradient=None, retain_graph=False, create_graph=False, inputs=None):
        if self._pending:
            from ._dispatch.pipeline import current_pipeline

            pipe = current_pipeline()
            if pipe is not None:
                pipe.flush()
        _backward(
            self,
            gradient,
            retain_graph=retain_graph,
            create_graph=create_graph,
            inputs=inputs,
        )

    def pin_memory(self):
        if self.device.type != "cpu":
            raise RuntimeError("pin_memory only supports CPU tensors")
        from . import npu as npu_api

        if not npu_api.is_available():
            raise RuntimeError("Cannot access accelerator device when none is available.")
        if self.is_pinned():
            return self
        storage = pinned_cpu_typed_storage_from_numpy(self._numpy_view(), self.dtype, device=self.device)
        return cy_make_tensor_from_storage(storage, self.shape, self.stride, self.offset, self.requires_grad)

    def is_pinned(self):
        return self._storage.is_pinned()

    def retain_grad(self):
        self._retain_grad = True

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = bool(requires_grad)
        if not self.requires_grad:
            self.grad_fn = None
        return self

    def detach(self):
        return self.cy_detach()

    def detach_(self):
        self.requires_grad = False
        self.grad_fn = None
        self._retain_grad = False
        return self

    def register_hook(self, hook):
        if not callable(hook):
            raise TypeError("hook must be callable")
        hooks = getattr(self, "_backward_hooks", None)
        if hooks is None:
            hooks = {}
            self._backward_hooks = hooks
        handle = _HookHandle(hooks)
        hooks[handle.id] = hook
        return handle

    def _is_view(self):
        return self._base is not None

    def _check_inplace(self):
        from .autograd.grad_mode import is_grad_enabled

        if not is_grad_enabled():
            return
        if not self.requires_grad:
            return
        if self._is_view() and self._base is not None:
            view_meta = getattr(self, "_view_meta", None) or {}
            creation_mode = view_meta.get("creation_mode")
            creation_kind = view_meta.get("creation_kind")
            grad_fn_name = self.grad_fn.name() if self.grad_fn is not None and hasattr(self.grad_fn, "name") else "<unknown>"
            if creation_kind == "multi_view":
                raise RuntimeError(
                    f"Output 0 of {grad_fn_name} is a view and is being modified inplace. This view is the output of a function that returns multiple views. Such functions do not allow the output views to be modified inplace. You should replace the inplace operation by an out-of-place one."
                )
            if creation_kind == "custom_function":
                display_name = grad_fn_name.removesuffix("Backward0") if grad_fn_name.endswith("Backward0") else grad_fn_name
                raise RuntimeError(
                    f"Output 0 of {display_name} is a view and is being modified inplace. This view was created inside a custom Function (or because an input was returned as-is) and the autograd logic to handle view+inplace would override the custom backward associated with the custom Function, leading to incorrect gradients. This behavior is forbidden. You can fix this by cloning the output of the custom Function."
                )
            if creation_mode == "no_grad":
                if creation_kind == "view_of_view":
                    raise RuntimeError(
                        "a view of a view which is being modified inside the no_grad block."
                    )
                if creation_kind == "view":
                    raise RuntimeError(
                        "A view was created in no_grad mode and is being modified inplace with grad mode enabled."
                    )
            if creation_mode == "inference_mode":
                if creation_kind == "view_of_view":
                    raise RuntimeError(
                        "a view of a view which is being modified inside the inference_mode."
                    )
                if creation_kind == "view":
                    raise RuntimeError(
                        "A view was created in inference_mode and is being modified inplace in normal mode."
                    )
            if self._base.grad_fn is None and self._base.requires_grad:
                raise RuntimeError("a view of a leaf Variable that requires grad is being used in an in-place operation.")
        if self.grad_fn is None and not self._is_view():
            raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")

    def add_(self, other, *, alpha=1):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        if alpha != 1:
            other = mul(other, alpha)
        out = dispatch("add_", self.device.type, self, other)
        return out

    def add(self, other, *, alpha=1):
        return add(self, other, alpha=alpha)

    def sub(self, other, *, alpha=1):
        return sub_dispatch(self, other, alpha=alpha)

    def mul(self, other):
        return mul(self, other)

    def div(self, other, *, rounding_mode=None):
        return div_dispatch(self, other)

    def mul_(self, other):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("mul_", self.device.type, self, other)
        return out

    def relu_(self):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("relu_", self.device.type, self)
        return out

    def zero_(self):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("zero_", self.device.type, self)
        return out

    def uniform_(self, low=0.0, high=1.0, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("uniform_", self.device.type, self, low, high, generator=generator)
        return out

    def normal_(self, mean=0.0, std=1.0, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("normal_", self.device.type, self, mean, std, generator=generator)
        return out

    def random_(self, from_=0, to=None, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("random_", self.device.type, self, from_, to, generator=generator)
        return out

    def randint_(self, low, high=None, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("randint_", self.device.type, self, low, high, generator=generator)
        return out

    def bernoulli_(self, p=0.5, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("bernoulli_", self.device.type, self, p, generator=generator)
        return out

    def multinomial(self, num_samples, replacement=False, *, generator=None):
        from ._random import multinomial
        return multinomial(self, num_samples, replacement=replacement, generator=generator)

    def exponential_(self, lambd=1.0, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("exponential_", self.device.type, self, lambd, generator=generator)
        return out

    def log_normal_(self, mean=1.0, std=2.0, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("log_normal_", self.device.type, self, mean, std, generator=generator)
        return out

    def cauchy_(self, median=0.0, sigma=1.0, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("cauchy_", self.device.type, self, median, sigma, generator=generator)
        return out

    def geometric_(self, p, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("geometric_", self.device.type, self, p, generator=generator)
        return out

    def fill_(self, value):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("fill_", self.device.type, self, value)
        return out

    def clamp_(self, min=None, max=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("clamp_", self.device.type, self, min, max)
        return out

    def copy_(self, src):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("copy_", self.device.type, self, src)
        return out

    def erfinv_(self):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("erfinv_", self.device.type, self)
        return out

    def sub_(self, other, *, alpha=1):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        if alpha != 1:
            other = mul(other, alpha)
        out = dispatch("sub_", self.device.type, self, other)
        return out

    def abs_(self):
        """In-place absolute value."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("abs_", self.device.type, self)

    def neg_(self):
        """In-place negation."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("neg_", self.device.type, self)

    def exp_(self):
        """In-place exponential."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("exp_", self.device.type, self)

    def log_(self):
        """In-place natural logarithm."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("log_", self.device.type, self)

    def log2_(self):
        """In-place base-2 logarithm."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("log2_", self.device.type, self)

    def log10_(self):
        """In-place base-10 logarithm."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("log10_", self.device.type, self)

    def sqrt_(self):
        """In-place square root."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("sqrt_", self.device.type, self)

    def sin_(self):
        """In-place sine."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("sin_", self.device.type, self)

    def cos_(self):
        """In-place cosine."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("cos_", self.device.type, self)

    def tan_(self):
        """In-place tangent."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("tan_", self.device.type, self)

    def tanh_(self):
        """In-place hyperbolic tangent."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("tanh_", self.device.type, self)

    def sigmoid_(self):
        """In-place sigmoid."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("sigmoid_", self.device.type, self)

    def floor_(self):
        """In-place floor."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("floor_", self.device.type, self)

    def ceil_(self):
        """In-place ceiling."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("ceil_", self.device.type, self)

    def round_(self):
        """In-place rounding."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("round_", self.device.type, self)

    def trunc_(self):
        """In-place truncation."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("trunc_", self.device.type, self)

    def pow_(self, exponent):
        """In-place power."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("pow_", self.device.type, self, exponent)

    def reciprocal_(self):
        """In-place reciprocal."""
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        return dispatch("reciprocal_", self.device.type, self)

    def to(self, *args, **kwargs):
        if self._pending:
            from ._dispatch.pipeline import current_pipeline

            pipe = current_pipeline()
            if pipe is not None:
                pipe.flush()
        # Parse arguments: to(device), to(dtype), to(device, dtype), to(dtype=, device=)
        device = None
        dtype = None
        non_blocking = kwargs.get("non_blocking", False)
        copy = kwargs.get("copy", False)
        memory_format = kwargs.get("memory_format", None)
        for arg in args:
            if isinstance(arg, Device):
                device = arg
            elif isinstance(arg, str):
                from ._dtype import from_name
                dt = from_name(arg)
                if dt is not None:
                    dtype = dt
                else:
                    device = Device(arg)
            elif hasattr(arg, 'name') and hasattr(arg, 'itemsize'):
                dtype = arg
            else:
                device = Device(str(arg))
        if "device" in kwargs:
            device = kwargs["device"]
            if isinstance(device, str):
                device = Device(device)
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        result = self
        if dtype is not None and dtype != self.dtype:
            result = result._to_dtype(dtype)
        if device is not None:
            result = to_dispatch(
                result,
                device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
                memory_format=memory_format,
            )
        if result is self and dtype is None and device is None:
            return self
        return result

    def cpu(self, memory_format=None):
        if memory_format is None:
            return self.to("cpu")
        return self.to("cpu", memory_format=memory_format)

    def npu(self, device=None, non_blocking=False, memory_format=None):
        if device is None:
            device = "npu"
        return self.to(device, non_blocking=non_blocking, memory_format=memory_format)

    def mps(self, memory_format=None):
        if memory_format is None:
            return self.to("mps")
        return self.to("mps", memory_format=memory_format)

    def cuda(self, device=None, non_blocking=False, memory_format=None):
        if device is None:
            target = "cuda"
        elif isinstance(device, str):
            target = device
        else:
            target = f"cuda:{int(device)}"
        return self.to(target, non_blocking=non_blocking, memory_format=memory_format)

    def _to_dtype(self, dtype):
        if self.device.type == "cpu":
            arr = self._numpy_view()
            src_dtype = self.dtype
            target_np = to_numpy_dtype(dtype)
            if src_dtype == bfloat16:
                # bfloat16 -> target: first convert uint16 bits to float32
                arr = _bf16_to_f32(arr)
            if dtype == bfloat16:
                # source -> bfloat16: convert to float32 then to uint16 bits
                arr = arr.astype(np.float32)
                arr = _f32_to_bf16(arr)
            else:
                arr = arr.astype(target_np)
            storage = typed_storage_from_numpy(arr, dtype, device=self.device)
            stride = tuple(np.array(arr.strides) // arr.itemsize)
            return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)
        elif self.device.type == "npu":
            from ._backends.npu.ops._helpers import _cast_tensor_dtype
            return _cast_tensor_dtype(self, dtype)
        elif self.device.type == "mps":
            from ._storage import mps_typed_storage_from_numpy
            arr = self._numpy_view()
            src_dtype = self.dtype
            target_np = to_numpy_dtype(dtype)
            if src_dtype == bfloat16:
                arr = _bf16_to_f32(arr)
            if dtype == bfloat16:
                arr = arr.astype(np.float32)
                arr = _f32_to_bf16(arr)
            else:
                arr = arr.astype(target_np)
            storage = mps_typed_storage_from_numpy(
                np.ascontiguousarray(arr), dtype, device=self.device
            )
            stride = tuple(np.array(arr.strides) // arr.itemsize) if arr.ndim > 0 else ()
            return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)
        elif self.device.type == "meta":
            storage = meta_typed_storage_from_shape(self.shape, dtype, device=self.device)
            return cy_make_tensor_from_storage(storage, self.shape, _compute_strides(self.shape), 0, False)
        else:
            raise RuntimeError(
                f"dtype conversion not yet supported on device {self.device.type}"
            )

    def new_ones(self, size, dtype=None, device=None):
        from ._creation import ones

        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return ones(size, dtype=dtype, device=device)

    def new_zeros(self, size, dtype=None, device=None):
        from ._creation import zeros

        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return zeros(size, dtype=dtype, device=device)

    def type(self, dtype=None):
        if dtype is None:
            return f"torch.{self.dtype.name.capitalize()}Tensor"
        if isinstance(dtype, str):
            from ._dtype import from_name
            _type_map = {
                "torch.FloatTensor": float32,
                "torch.DoubleTensor": float64,
                "torch.HalfTensor": float16,
                "torch.BFloat16Tensor": bfloat16,
                "torch.LongTensor": int64,
                "torch.IntTensor": int32,
                "torch.ShortTensor": int16,
                "torch.CharTensor": int8,
                "torch.ByteTensor": uint8,
                "torch.BoolTensor": dtype_bool,
            }
            dt = _type_map.get(dtype) or from_name(dtype)
            if dt is None:
                raise RuntimeError(f"Unknown type: {dtype}")
            return self._to_dtype(dt)
        return self._to_dtype(dtype)

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return add(self, neg_dispatch(other))
        return add(self, -other)

    def __rsub__(self, other):
        return add(neg_dispatch(self), other)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return true_divide_dispatch(self, other)

    def __rtruediv__(self, other):
        return true_divide_dispatch(other, self)

    def __pow__(self, exponent):
        return pow_dispatch(self, exponent)

    def __rpow__(self, base):
        from ._dispatch.dispatcher import dispatch
        return dispatch("pow", self.device.type, base, self)

    def __floordiv__(self, other):
        return floor_divide_dispatch(self, other)

    def __rfloordiv__(self, other):
        from ._dispatch.dispatcher import dispatch
        return dispatch("floor_divide", self.device.type, other, self)

    def __mod__(self, other):
        return remainder_dispatch(self, other)

    def __rmod__(self, other):
        from ._dispatch.dispatcher import dispatch
        return dispatch("remainder", self.device.type, other, self)

    def __iadd__(self, other):
        self._check_inplace()
        self.add_(other)
        return self

    def __isub__(self, other):
        self._check_inplace()
        self.sub_(other)
        return self

    def __imul__(self, other):
        self._check_inplace()
        self.mul_(other)
        return self

    def __itruediv__(self, other):
        self._check_inplace()
        self.div_(other)
        return self

    def __neg__(self):
        return neg_dispatch(self)

    def clone(self):
        from ._functional import to as to_dispatch

        return to_dispatch(self, self.device, copy=True)

    def matmul(self, other):
        return matmul(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def relu(self):
        return relu(self)

    def abs(self):
        return abs_dispatch(self)

    def neg(self):
        return neg_dispatch(self)

    def exp(self):
        return exp_dispatch(self)

    def log(self):
        return log_dispatch(self)

    def sqrt(self):
        return sqrt_dispatch(self)

    def tril(self, diagonal=0):
        return tril_dispatch(self, diagonal)

    def triu(self, diagonal=0):
        return triu_dispatch(self, diagonal)

    def diag(self, diagonal=0):
        return diag_dispatch(self, diagonal)

    def sin(self):
        return sin_dispatch(self)

    def cos(self):
        return cos_dispatch(self)

    def tan(self):
        return tan_dispatch(self)

    def tanh(self):
        return tanh_dispatch(self)

    def sigmoid(self):
        return sigmoid_dispatch(self)

    def floor(self):
        return floor_dispatch(self)

    def ceil(self):
        return ceil_dispatch(self)

    def round(self):
        return round_dispatch(self)

    def trunc(self):
        return trunc_dispatch(self)

    def frac(self):
        return frac_dispatch(self)

    def pow(self, exponent):
        return pow_dispatch(self, exponent)

    def log2(self):
        return log2_dispatch(self)

    def log10(self):
        return log10_dispatch(self)

    def exp2(self):
        return exp2_dispatch(self)

    def rsqrt(self):
        return rsqrt_dispatch(self)

    def sign(self):
        return sign_dispatch(self)

    def signbit(self):
        return signbit_dispatch(self)

    def square(self):
        return square_dispatch(self)

    def isnan(self):
        return isnan_dispatch(self)

    def isinf(self):
        return isinf_dispatch(self)

    def isfinite(self):
        return isfinite_dispatch(self)

    def sinh(self):
        return sinh_dispatch(self)

    def cosh(self):
        return cosh_dispatch(self)

    def asinh(self):
        return asinh_dispatch(self)

    def acosh(self):
        return acosh_dispatch(self)

    def atanh(self):
        return atanh_dispatch(self)

    def erf(self):
        return erf_dispatch(self)

    def erfc(self):
        return erfc_dispatch(self)

    def softplus(self):
        return softplus_dispatch(self)

    def clamp(self, min_val=None, max_val=None):
        return clamp_dispatch(self, min_val, max_val)

    def clamp_min(self, min_val):
        return clamp_min_dispatch(self, min_val)

    def clamp_max(self, max_val):
        return clamp_max_dispatch(self, max_val)

    def clamp_(self, min=None, max=None):
        self._check_inplace()
        out = clamp_dispatch(self, min, max)
        return self.copy_(out)

    def relu6(self):
        return relu6_dispatch(self)

    def hardtanh(self, min_val=-1.0, max_val=1.0):
        return hardtanh_dispatch(self, min_val, max_val)

    def min(self, dim=None, keepdim=False):
        from ._dispatch.dispatcher import dispatch
        if dim is None:
            return amin_dispatch(self)
        if isinstance(dim, Tensor):
            return min_dispatch(self, dim)
        return dispatch("min", self.device.type, self, dim, keepdim)

    def max(self, dim=None, keepdim=False):
        from ._dispatch.dispatcher import dispatch
        if dim is None:
            return amax_dispatch(self)
        if isinstance(dim, Tensor):
            return max_dispatch(self, dim)
        return dispatch("max", self.device.type, self, dim, keepdim)

    def amin(self, dim=None, keepdim=False):
        return amin_dispatch(self, dim=dim, keepdim=keepdim)

    def amax(self, dim=None, keepdim=False):
        return amax_dispatch(self, dim=dim, keepdim=keepdim)

    def fmin(self, other):
        return fmin_dispatch(self, other)

    def fmax(self, other):
        return fmax_dispatch(self, other)

    def where(self, condition, other):
        return where_dispatch(condition, self, other)

    def atan(self):
        return atan_dispatch(self)

    def atan2(self, other):
        return atan2_dispatch(self, other)

    def asin(self):
        return asin_dispatch(self)

    def acos(self):
        return acos_dispatch(self)

    def lerp(self, other, weight):
        return lerp_dispatch(self, other, weight)

    def addcmul(self, tensor1, tensor2, value=1.0):
        return addcmul_dispatch(self, tensor1, tensor2, value=value)

    def addcdiv(self, tensor1, tensor2, value=1.0):
        return addcdiv_dispatch(self, tensor1, tensor2, value=value)

    def logaddexp(self, other):
        return logaddexp_dispatch(self, other)

    def logaddexp2(self, other):
        return logaddexp2_dispatch(self, other)

    def hypot(self, other):
        return hypot_dispatch(self, other)

    def remainder(self, other):
        return remainder_dispatch(self, other)

    def fmod(self, other):
        return fmod_dispatch(self, other)

    def squeeze(self, dim=None):
        return squeeze_dispatch(self, dim)

    def squeeze_(self, dim=None):
        if dim is not None:
            if isinstance(dim, (list, tuple)):
                if dim:
                    ndim = len(self.shape)
                    targets = set()
                    for item in dim:
                        d = item if item >= 0 else item + ndim
                        targets.add(d)
                    pairs = [
                        (s, st)
                        for idx, (s, st) in enumerate(zip(self.shape, self.stride))
                        if idx not in targets or s != 1
                    ]
                    shape = [p[0] for p in pairs]
                    stride = [p[1] for p in pairs]
                else:
                    shape = list(self.shape)
                    stride = list(self.stride)
            else:
                d = dim if dim >= 0 else dim + len(self.shape)
                shape = list(self.shape)
                stride = list(self.stride)
                if 0 <= d < len(shape) and shape[d] == 1:
                    del shape[d]
                    del stride[d]
        else:
            pairs = [(s, st) for s, st in zip(self.shape, self.stride) if s != 1]
            shape = [p[0] for p in pairs]
            stride = [p[1] for p in pairs]
        return self.as_strided_(tuple(shape), tuple(stride))

    def unsqueeze(self, dim):
        return unsqueeze_dispatch(self, dim)

    def unsqueeze_(self, dim):
        ndim = len(self.shape)
        d = dim if dim >= 0 else dim + ndim + 1
        if d < 0 or d > ndim:
            raise IndexError("Dimension out of range")
        shape = list(self.shape)
        stride = list(self.stride)
        new_stride = stride[d] * shape[d] if d < ndim else 1
        shape.insert(d, 1)
        stride.insert(d, new_stride)
        return self.as_strided_(tuple(shape), tuple(stride))

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return permute_dispatch(self, dims)

    def var(self, dim=None, keepdim=False, unbiased=True):
        return var_dispatch(self, dim=dim, keepdim=keepdim, unbiased=unbiased)

    def norm(self, p=2, dim=None, keepdim=False):
        return norm_dispatch(self, p=p, dim=dim, keepdim=keepdim)

    def prod(self, dim=None, keepdim=False):
        return prod_dispatch(self, dim=dim, keepdim=keepdim)

    def mm(self, mat2):
        return mm_dispatch(self, mat2)

    def bmm(self, batch2):
        return bmm_dispatch(self, batch2)
    def sum(self, dim=None, keepdim=False, *, dtype=None):
        return sum(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def mean(self, dim=None, keepdim=False, *, dtype=None, axis=None):
        if axis is not None:
            dim = axis
        return mean_dispatch(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def std(self, dim=None, keepdim=False, unbiased=True, axis=None):
        if axis is not None:
            dim = axis
        return std_dispatch(self, dim=dim, keepdim=keepdim, unbiased=unbiased)

    def all(self, dim=None, keepdim=False):
        return all_dispatch(self, dim=dim, keepdim=keepdim)

    def any(self, dim=None, keepdim=False):
        return any_dispatch(self, dim=dim, keepdim=keepdim)

    def argmax(self, dim=None, keepdim=False):
        return argmax_dispatch(self, dim=dim, keepdim=keepdim)

    def argmin(self, dim=None, keepdim=False):
        return argmin_dispatch(self, dim=dim, keepdim=keepdim)

    def count_nonzero(self, dim=None, keepdim=False):
        return count_nonzero_dispatch(self, dim=dim, keepdim=keepdim)

    def cumsum(self, dim=0):
        return cumsum_dispatch(self, dim)

    def cumprod(self, dim=0):
        return cumprod_dispatch(self, dim)

    def cummax(self, dim=0):
        return cummax_dispatch(self, dim)

    def argsort(self, dim=-1, descending=False, stable=False):
        return argsort_dispatch(self, dim=dim, descending=descending, stable=stable)

    def sort(self, dim=-1, descending=False, stable=False):
        return sort_dispatch(self, dim=dim, descending=descending, stable=stable)

    def topk(self, k, dim=-1, largest=True, sorted=True):
        return topk_dispatch(self, k, dim=dim, largest=largest, sorted=sorted)

    def split(self, split_size_or_sections, dim=0):
        return split_dispatch(self, split_size_or_sections, dim=dim)

    def chunk(self, chunks, dim=0):
        return chunk_dispatch(self, chunks, dim=dim)

    def repeat(self, *repeats):
        if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)):
            repeats = tuple(repeats[0])
        return repeat_dispatch(self, repeats)

    def tile(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return tile_dispatch(self, dims)

    def flip(self, dims):
        if isinstance(dims, int):
            dims = [dims]
        return flip_dispatch(self, dims)

    def roll(self, shifts, dims=None):
        return roll_dispatch(self, shifts, dims)

    def rot90(self, k=1, dims=(0, 1)):
        return rot90_dispatch(self, k, dims)

    def reciprocal(self):
        return reciprocal_dispatch(self)

    def log1p(self):
        """Returns a new tensor with the natural logarithm of (1 + input)."""
        return log1p_dispatch(self)

    def expm1(self):
        """Returns a new tensor with the exponential of the elements minus 1."""
        return expm1_dispatch(self)

    def logsumexp(self, dim, keepdim=False):
        """Returns the log of summed exponentials of each row of the input tensor in the given dimension dim."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("logsumexp", self.device.type, self, dim, keepdim)

    def trace(self):
        """Returns the sum of the elements of the diagonal of the input 2-D matrix."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("trace", self.device.type, self)

    def det(self):
        """Returns the determinant of a square matrix."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("det", self.device.type, self)

    def matrix_power(self, n):
        """Returns the matrix raised to the power n for square matrices."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("matrix_power", self.device.type, self, n)

    def dist(self, other, p=2):
        """Returns the p-norm of (self - other)."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("dist", self.device.type, self, other, p)

    def renorm(self, p, dim, maxnorm):
        """Returns a tensor where each sub-tensor along dimension dim is normalized."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("renorm", self.device.type, self, p, dim, maxnorm)

    def nansum(self, dim=None, keepdim=False):
        """Returns the sum of all elements, treating NaNs as zero."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("nansum", self.device.type, self, dim, keepdim)

    def nanmean(self, dim=None, keepdim=False):
        """Returns the mean of all elements, treating NaNs as zero."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("nanmean", self.device.type, self, dim, keepdim)

    def argwhere(self):
        """Returns a tensor containing the indices of all non-zero elements."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("argwhere", self.device.type, self)

    def addmm(self, mat1, mat2, *, beta=1, alpha=1):
        return addmm_dispatch(self, mat1, mat2, beta=beta, alpha=alpha)

    def baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
        """Performs a batch matrix-matrix product with added input.

        out = beta * self + alpha * (batch1 @ batch2)

        Args:
            batch1: First batch of matrices (B x N x M)
            batch2: Second batch of matrices (B x M x P)
            beta: Multiplier for self (default: 1)
            alpha: Multiplier for batch1 @ batch2 (default: 1)

        Returns:
            Tensor of shape (B x N x P)
        """
        from ._dispatch.dispatcher import dispatch
        return dispatch("baddbmm", self.device.type, self, batch1, batch2, beta=beta, alpha=alpha)

    def type_as(self, other):
        return self.to(other.dtype)

    def reshape_as(self, other):
        return self.cy_view(other.shape)

    def new_full(self, size, fill_value, *, dtype=None, device=None, requires_grad=False):
        from ._creation import full
        dt = dtype if dtype is not None else self.dtype
        dev = device if device is not None else self.device
        return full(size, fill_value, dtype=dt, device=dev)

    def div_(self, other):
        from ._dispatch.dispatcher import dispatch
        self._check_inplace()
        out = dispatch("div_", self.device.type, self, other)
        return out

    def unflatten(self, dim, sizes):
        ndim = len(self.shape)
        if dim < 0:
            dim += ndim
        new_shape = self.shape[:dim] + tuple(sizes) + self.shape[dim + 1:]
        return self.view(new_shape)

    # -----------------------------------------------------------------------
    # Indexing / selection methods
    # -----------------------------------------------------------------------

    def narrow(self, dim, start, length):
        return narrow_dispatch(self, dim, start, length)

    def select(self, dim, index):
        return select_dispatch(self, dim, index)

    def expand(self, *sizes):
        return expand_dispatch(self, *sizes)

    def expand_copy(self, *sizes):
        return expand_copy_dispatch(self, sizes)

    def sum_to_size(self, *size):
        return sum_to_size_dispatch(self, *size)

    def expand_as(self, other):
        return expand_dispatch(self, *other.shape)

    def nonzero(self, as_tuple=False):
        return nonzero_dispatch(self, as_tuple=as_tuple)

    def masked_select(self, mask):
        return masked_select_dispatch(self, mask)

    def slice(self, dim, start=0, end=9223372036854775807, step=1):
        return slice_dispatch(self, dim, start, end, step)

    def slice_copy(self, dim, start=0, end=9223372036854775807, step=1):
        return slice_copy_dispatch(self, dim, start, end, step)

    def slice_scatter(self, src, dim, start=0, end=9223372036854775807, step=1):
        return slice_scatter_dispatch(self, src, dim, start, end, step)

    def gather(self, dim, index):
        return gather_dispatch(self, dim, index)

    def scatter(self, dim, index, src):
        return scatter_dispatch(self, dim, index, src)

    def scatter_(self, dim, index, src):
        self._check_inplace()
        return scatter__dispatch(self, dim, index, src)

    def scatter_add_(self, dim, index, src):
        self._check_inplace()
        return scatter_add__dispatch(self, dim, index, src)

    def index_select(self, dim, index):
        return index_select_dispatch(self, dim, index)

    def take(self, index):
        return take_dispatch(self, index)

    def as_strided_(self, size, stride, storage_offset=None):
        self._check_inplace()
        return as_strided__dispatch(self, size, stride, storage_offset)

    def as_strided_copy(self, size, stride, storage_offset=None):
        return as_strided_copy_dispatch(self, size, stride, storage_offset)

    def as_strided_scatter(self, src, size, stride, storage_offset=None):
        return as_strided_scatter_dispatch(self, src, size, stride, storage_offset)

    def masked_fill(self, mask, value):
        return masked_fill_dispatch(self, mask, value)

    def masked_fill_(self, mask, value):
        self._check_inplace()
        return masked_fill__dispatch(self, mask, value)

    def masked_scatter_(self, mask, source):
        self._check_inplace()
        return masked_scatter__dispatch(self, mask, source)

    def index_put_(self, indices, values, accumulate=False):
        self._check_inplace()
        return index_put__dispatch(self, indices, values, accumulate)

    def index_put(self, indices, values, accumulate=False):
        return index_put_dispatch(self, indices, values, accumulate)

    def index_copy_(self, dim, index, source):
        self._check_inplace()
        return index_copy__dispatch(self, dim, index, source)

    def index_fill_(self, dim, index, value):
        self._check_inplace()
        return index_fill__dispatch(self, dim, index, value)

    def index_add_(self, dim, index, source, alpha=1.0):
        self._check_inplace()
        return index_add__dispatch(self, dim, index, source, alpha)

    def unfold(self, dimension, size, step):
        return unfold_dispatch(self, dimension, size, step)

    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return allclose_dispatch(self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return isclose_dispatch(self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def equal(self, other):
        return equal_dispatch(self, other)

    def eq(self, other):
        return self.__eq__(other)

    def ne(self, other):
        return self.__ne__(other)

    def lt(self, other):
        """Element-wise less-than comparison."""
        return lt_dispatch(self, other)

    def le(self, other):
        """Element-wise less-than-or-equal comparison."""
        return le_dispatch(self, other)

    def gt(self, other):
        """Element-wise greater-than comparison."""
        return gt_dispatch(self, other)

    def ge(self, other):
        """Element-wise greater-than-or-equal comparison."""
        return ge_dispatch(self, other)

    def logical_and(self, other):
        """Element-wise logical AND."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("logical_and", self.device.type, self, other)

    def logical_or(self, other):
        """Element-wise logical OR."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("logical_or", self.device.type, self, other)

    def logical_xor(self, other):
        """Element-wise logical XOR."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("logical_xor", self.device.type, self, other)

    def logical_not(self):
        """Element-wise logical NOT."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("logical_not", self.device.type, self)

    def bitwise_and(self, other):
        """Element-wise bitwise AND."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("bitwise_and", self.device.type, self, other)

    def bitwise_or(self, other):
        """Element-wise bitwise OR."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("bitwise_or", self.device.type, self, other)

    def bitwise_xor(self, other):
        """Element-wise bitwise XOR."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("bitwise_xor", self.device.type, self, other)

    def bitwise_not(self):
        """Element-wise bitwise NOT."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("bitwise_not", self.device.type, self)

    def bitwise_and_(self, other):
        """In-place element-wise bitwise AND."""
        self._check_inplace()
        from ._dispatch.dispatcher import dispatch
        return dispatch("bitwise_and_", self.device.type, self, other)

    def bitwise_or_(self, other):
        """In-place element-wise bitwise OR."""
        self._check_inplace()
        from ._dispatch.dispatcher import dispatch
        return dispatch("bitwise_or_", self.device.type, self, other)

    def bitwise_xor_(self, other):
        """In-place element-wise bitwise XOR."""
        self._check_inplace()
        from ._dispatch.dispatcher import dispatch
        return dispatch("bitwise_xor_", self.device.type, self, other)

    def movedim(self, source, destination):
        """Move dimensions to new positions."""
        return movedim_dispatch(self, source, destination)

    def moveaxis(self, source, destination):
        """Alias for movedim."""
        return moveaxis_dispatch(self, source, destination)

    def swapdims(self, dim0, dim1):
        """Swap two dimensions (alias for transpose with positional args)."""
        return self.cy_transpose(dim0, dim1)

    def swapdims_(self, dim0, dim1):
        ndim = len(self.shape)
        d0 = dim0 if dim0 >= 0 else dim0 + ndim
        d1 = dim1 if dim1 >= 0 else dim1 + ndim
        if d0 < 0 or d0 >= ndim or d1 < 0 or d1 >= ndim:
            raise IndexError("Dimension out of range")
        shape = list(self.shape)
        stride = list(self.stride)
        shape[d0], shape[d1] = shape[d1], shape[d0]
        stride[d0], stride[d1] = stride[d1], stride[d0]
        self.shape = tuple(shape)
        self.stride = _StrideTuple(tuple(stride))
        return self

    def swapaxes(self, axis0, axis1):
        """Alias for swapdims."""
        return self.cy_transpose(axis0, axis1)

    def swapaxes_(self, axis0, axis1):
        ndim = len(self.shape)
        d0 = axis0 if axis0 >= 0 else axis0 + ndim
        d1 = axis1 if axis1 >= 0 else axis1 + ndim
        if d0 < 0 or d0 >= ndim or d1 < 0 or d1 >= ndim:
            raise IndexError("Dimension out of range")
        shape = list(self.shape)
        stride = list(self.stride)
        shape[d0], shape[d1] = shape[d1], shape[d0]
        stride[d0], stride[d1] = stride[d1], stride[d0]
        self.shape = tuple(shape)
        self.stride = _StrideTuple(tuple(stride))
        return self

    def diagonal(self, offset=0, dim1=0, dim2=1):
        """Returns partial view of input with the diagonal elements of input."""
        return diagonal_dispatch(self, offset, dim1, dim2)

    def unbind(self, dim=0):
        """Remove a tensor dimension, returning a tuple of all slices along dim."""
        return unbind_dispatch(self, dim=dim)

    def vsplit(self, split_size_or_sections):
        """Split a tensor into multiple sub-tensors vertically (row-wise)."""
        return vsplit_dispatch(self, split_size_or_sections)

    def hsplit(self, split_size_or_sections):
        """Split a tensor into multiple sub-tensors horizontally (column-wise)."""
        return hsplit_dispatch(self, split_size_or_sections)

    def dsplit(self, split_size_or_sections):
        """Split a tensor into multiple sub-tensors along the third axis."""
        return dsplit_dispatch(self, split_size_or_sections)

    def take_along_dim(self, indices, dim):
        """Take values along an axis at the given indices."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("take_along_dim", self.device.type, self, indices, dim)

    def scatter_add(self, dim, index, src):
        """Non-inplace scatter_add: adds all values from src into self at index positions."""
        out = self.clone()
        out.scatter_add_(dim, index, src)
        return out

    def index_fill(self, dim, index, value):
        """Non-inplace version of index_fill_: fills self tensor with value along dim at index."""
        out = self.clone()
        out.index_fill_(dim, index, value)
        return out

    def index_copy(self, dim, index, source):
        """Non-inplace version of index_copy_: copies values from source into self along dim."""
        out = self.clone()
        out.index_copy_(dim, index, source)
        return out

    def index_add(self, dim, index, source, alpha=1):
        """Non-inplace version of index_add_: adds values from source (scaled by alpha) into self along dim."""
        out = self.clone()
        out.index_add_(dim, index, source, alpha)
        return out

    def put_(self, indices, values, accumulate=False):
        """Copies elements from values into self tensor at positions specified by indices.

        Treats self as a flat (1-D) tensor and uses flat indices.
        """
        self._check_inplace()
        # Work on a contiguous version for flat indexing
        if not self.is_contiguous():
            cont = self.contiguous()
            self._storage = cont._storage
            self.stride = cont.stride
        numel_idx = indices.numel()
        shape = self.shape
        for i in range(numel_idx):
            idx = int(indices.reshape((numel_idx,))[i].item())
            val = values.reshape((numel_idx,))[i]
            # Calculate multi-dim index from flat index
            multi_idx = []
            tmp = idx
            for d in reversed(shape):
                multi_idx.append(tmp % d)
                tmp //= d
            multi_idx = tuple(reversed(multi_idx))
            if accumulate:
                self[multi_idx] = self[multi_idx] + val
            else:
                self[multi_idx] = val
        self._bump_version()
        return self

    def cummin(self, dim):
        """Returns a namedtuple (values, indices) of cumulative minimum of elements along dim."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("cummin", self.device.type, self, dim)

    def __getitem__(self, key):
        from ._dispatch.dispatcher import dispatch

        return dispatch("getitem", self.device.type, self, key)

    def __setitem__(self, key, value):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        dispatch("setitem", self.device.type, self, key, value)

    def __and__(self, other):
        return mul(self.bool(), other.bool() if isinstance(other, Tensor) else bool(other))

    def __or__(self, other):
        return add(self.bool(), other.bool() if isinstance(other, Tensor) else bool(other))

    def __xor__(self, other):
        return ne_dispatch(self.bool(), other.bool() if isinstance(other, Tensor) else bool(other))

    def __hash__(self):
        return id(self)


from . import _cython as _cython_mod

if getattr(_cython_mod, "_HAS_CYTHON_TENSOR_API", False):
    Tensor._set_device_from_storage = _cython_mod.tensor_set_device_from_storage
    Tensor._set_dtype_from_storage = _cython_mod.tensor_set_dtype_from_storage
    Tensor.data = property(Tensor.data.fget, _cython_mod.tensor_set_data)
    Tensor.__delattr__ = _cython_mod.tensor_delattr
    Tensor._fw_get = _cython_mod.tensor_fw_get
    Tensor._fw_set = _cython_mod.tensor_fw_set
    Tensor._fw_clear = _cython_mod.tensor_fw_clear
    Tensor._fw_has = _cython_mod.tensor_fw_has
    Tensor.untyped_storage = _cython_mod.tensor_untyped_storage
    Tensor.record_stream = _cython_mod.tensor_record_stream
    Tensor.is_pinned = _cython_mod.tensor_is_pinned

    _python_tensor_min = Tensor.min
    _python_tensor_max = Tensor.max

    Tensor.__add__ = _cython_mod.tensor_add
    Tensor.__sub__ = _cython_mod.tensor_sub
    Tensor.__mul__ = _cython_mod.tensor_mul
    Tensor.__matmul__ = _cython_mod.tensor_matmul
    Tensor.__getitem__ = _cython_mod.tensor_getitem
    Tensor.__setitem__ = _cython_mod.tensor_setitem
    Tensor.__iadd__ = _cython_mod.tensor_iadd
    Tensor.__isub__ = _cython_mod.tensor_isub
    Tensor.__imul__ = _cython_mod.tensor_imul
    Tensor.__itruediv__ = _cython_mod.tensor_itruediv
    Tensor.__neg__ = _cython_mod.tensor_neg
    Tensor.neg = _cython_mod.tensor_neg

    Tensor.clone = _cython_mod.tensor_clone
    Tensor.detach = _cython_mod.tensor_detach
    Tensor.detach_ = _cython_mod.tensor_detach_
    Tensor.to = _cython_mod.tensor_to
    Tensor._to_dtype = _cython_mod.tensor_to_dtype
    Tensor.cpu = _cython_mod.tensor_cpu
    Tensor.npu = _cython_mod.tensor_npu
    Tensor.mps = _cython_mod.tensor_mps
    Tensor.cuda = _cython_mod.tensor_cuda
    Tensor.backward = _cython_mod.tensor_backward
    Tensor.relu = _cython_mod.tensor_relu
    Tensor.is_contiguous = _cython_mod.tensor_is_contiguous
    Tensor.contiguous = _cython_mod.tensor_contiguous
    Tensor.reshape = _cython_mod.tensor_reshape
    Tensor.transpose = _cython_mod.tensor_transpose
    Tensor.view = _cython_mod.tensor_view
    Tensor.flatten = _cython_mod.tensor_flatten
    Tensor.t = _cython_mod.tensor_t
    Tensor.as_strided = _cython_mod.tensor_as_strided
    Tensor.size = _cython_mod.tensor_size
    Tensor.dim = _cython_mod.tensor_dim

    Tensor.retain_grad = _cython_mod.tensor_retain_grad
    Tensor.requires_grad_ = _cython_mod.tensor_requires_grad_
    Tensor.register_hook = _cython_mod.tensor_register_hook
    Tensor._is_view = _cython_mod.tensor_is_view
    Tensor._check_inplace = _cython_mod.tensor_check_inplace

    Tensor.add_ = _cython_mod.tensor_add_
    Tensor.mul_ = _cython_mod.tensor_mul_
    Tensor.relu_ = _cython_mod.tensor_relu_
    Tensor.zero_ = _cython_mod.tensor_zero_
    Tensor.fill_ = _cython_mod.tensor_fill_
    Tensor.copy_ = _cython_mod.tensor_copy_

    Tensor.abs_ = _cython_mod.tensor_abs_
    Tensor.neg_ = _cython_mod.tensor_neg_
    Tensor.exp_ = _cython_mod.tensor_exp_
    Tensor.log_ = _cython_mod.tensor_log_
    Tensor.log2_ = _cython_mod.tensor_log2_
    Tensor.log10_ = _cython_mod.tensor_log10_
    Tensor.sqrt_ = _cython_mod.tensor_sqrt_
    Tensor.sin_ = _cython_mod.tensor_sin_
    Tensor.cos_ = _cython_mod.tensor_cos_
    Tensor.tan_ = _cython_mod.tensor_tan_
    Tensor.tanh_ = _cython_mod.tensor_tanh_
    Tensor.sigmoid_ = _cython_mod.tensor_sigmoid_
    Tensor.floor_ = _cython_mod.tensor_floor_
    Tensor.ceil_ = _cython_mod.tensor_ceil_
    Tensor.round_ = _cython_mod.tensor_round_
    Tensor.trunc_ = _cython_mod.tensor_trunc_
    Tensor.pow_ = _cython_mod.tensor_pow_
    Tensor.reciprocal_ = _cython_mod.tensor_reciprocal_
    Tensor.erfinv_ = _cython_mod.tensor_erfinv_

    Tensor.sub_ = _cython_mod.tensor_sub_
    Tensor.clamp_ = _cython_mod.tensor_clamp_
    Tensor.uniform_ = _cython_mod.tensor_uniform_
    Tensor.normal_ = _cython_mod.tensor_normal_
    Tensor.random_ = _cython_mod.tensor_random_
    Tensor.randint_ = _cython_mod.tensor_randint_
    Tensor.bernoulli_ = _cython_mod.tensor_bernoulli_
    Tensor.exponential_ = _cython_mod.tensor_exponential_
    Tensor.log_normal_ = _cython_mod.tensor_log_normal_
    Tensor.cauchy_ = _cython_mod.tensor_cauchy_
    Tensor.geometric_ = _cython_mod.tensor_geometric_

    Tensor.transpose_ = _cython_mod.tensor_transpose_
    Tensor.t_ = _cython_mod.tensor_t_
    Tensor.squeeze_ = _cython_mod.tensor_squeeze_
    Tensor.unsqueeze_ = _cython_mod.tensor_unsqueeze_
    Tensor.as_strided_ = _cython_mod.tensor_as_strided_
    Tensor.swapdims_ = _cython_mod.tensor_swapdims_
    Tensor.swapaxes_ = _cython_mod.tensor_swapaxes_

    Tensor.scatter_add = _cython_mod.tensor_scatter_add
    Tensor.index_fill = _cython_mod.tensor_index_fill
    Tensor.index_copy = _cython_mod.tensor_index_copy
    Tensor.index_add = _cython_mod.tensor_index_add
    Tensor.put_ = _cython_mod.tensor_put_
    Tensor.scatter_ = _cython_mod.tensor_scatter_
    Tensor.scatter_add_ = _cython_mod.tensor_scatter_add_
    Tensor.masked_fill_ = _cython_mod.tensor_masked_fill_
    Tensor.masked_scatter_ = _cython_mod.tensor_masked_scatter_
    Tensor.index_put_ = _cython_mod.tensor_index_put_
    Tensor.index_copy_ = _cython_mod.tensor_index_copy_
    Tensor.index_fill_ = _cython_mod.tensor_index_fill_
    Tensor.index_add_ = _cython_mod.tensor_index_add_

    Tensor.new_empty = _cython_mod.tensor_new_empty
    Tensor.new_tensor = _cython_mod.tensor_new_tensor
    Tensor.new_empty_strided = _cython_mod.tensor_new_empty_strided
    Tensor._ones_like = _cython_mod.tensor_ones_like
    Tensor.new_ones = _cython_mod.tensor_new_ones
    Tensor.new_zeros = _cython_mod.tensor_new_zeros
    Tensor.new_full = _cython_mod.tensor_new_full
    Tensor.div_ = _cython_mod.tensor_div_
    Tensor.unflatten = _cython_mod.tensor_unflatten
    Tensor.bitwise_and_ = _cython_mod.tensor_bitwise_and_
    Tensor.bitwise_or_ = _cython_mod.tensor_bitwise_or_
    Tensor.bitwise_xor_ = _cython_mod.tensor_bitwise_xor_
    Tensor.type = _cython_mod.tensor_type
    Tensor.type_as = _cython_mod.tensor_type_as
    Tensor.reshape_as = _cython_mod.tensor_reshape_as
    Tensor.permute = _cython_mod.tensor_permute
    Tensor.mean = _cython_mod.tensor_mean
    Tensor.std = _cython_mod.tensor_std
    Tensor.repeat = _cython_mod.tensor_repeat
    Tensor.tile = _cython_mod.tensor_tile
    Tensor.flip = _cython_mod.tensor_flip
    Tensor.logsumexp = _cython_mod.tensor_logsumexp
    Tensor.trace = _cython_mod.tensor_trace
    Tensor.det = _cython_mod.tensor_det
    Tensor.matrix_power = _cython_mod.tensor_matrix_power
    Tensor.dist = _cython_mod.tensor_dist
    Tensor.renorm = _cython_mod.tensor_renorm
    Tensor.nansum = _cython_mod.tensor_nansum
    Tensor.nanmean = _cython_mod.tensor_nanmean
    Tensor.argwhere = _cython_mod.tensor_argwhere
    Tensor.baddbmm = _cython_mod.tensor_baddbmm
    Tensor.vsplit = _cython_mod.tensor_vsplit
    Tensor.hsplit = _cython_mod.tensor_hsplit
    Tensor.dsplit = _cython_mod.tensor_dsplit
    Tensor.take_along_dim = _cython_mod.tensor_take_along_dim
    Tensor.cummin = _cython_mod.tensor_cummin
    Tensor.log1p = _cython_mod.tensor_log1p
    Tensor.expm1 = _cython_mod.tensor_expm1
    Tensor.lt = _cython_mod.tensor_lt
    Tensor.le = _cython_mod.tensor_le
    Tensor.gt = _cython_mod.tensor_gt
    Tensor.ge = _cython_mod.tensor_ge
    Tensor.abs = _cython_mod.tensor_abs
    Tensor.exp = _cython_mod.tensor_exp
    Tensor.log = _cython_mod.tensor_log
    Tensor.sqrt = _cython_mod.tensor_sqrt
    Tensor.sin = _cython_mod.tensor_sin
    Tensor.cos = _cython_mod.tensor_cos
    Tensor.tan = _cython_mod.tensor_tan
    Tensor.tanh = _cython_mod.tensor_tanh
    Tensor.sigmoid = _cython_mod.tensor_sigmoid
    Tensor.floor = _cython_mod.tensor_floor
    Tensor.ceil = _cython_mod.tensor_ceil
    Tensor.round = _cython_mod.tensor_round
    Tensor.trunc = _cython_mod.tensor_trunc
    Tensor.frac = _cython_mod.tensor_frac
    Tensor.log2 = _cython_mod.tensor_log2
    Tensor.log10 = _cython_mod.tensor_log10
    Tensor.exp2 = _cython_mod.tensor_exp2
    Tensor.rsqrt = _cython_mod.tensor_rsqrt
    Tensor.sign = _cython_mod.tensor_sign
    Tensor.signbit = _cython_mod.tensor_signbit
    Tensor.square = _cython_mod.tensor_square
    Tensor.isnan = _cython_mod.tensor_isnan
    Tensor.isinf = _cython_mod.tensor_isinf
    Tensor.isfinite = _cython_mod.tensor_isfinite
    Tensor.sinh = _cython_mod.tensor_sinh
    Tensor.cosh = _cython_mod.tensor_cosh
    Tensor.asinh = _cython_mod.tensor_asinh
    Tensor.acosh = _cython_mod.tensor_acosh
    Tensor.atanh = _cython_mod.tensor_atanh
    Tensor.erf = _cython_mod.tensor_erf
    Tensor.erfc = _cython_mod.tensor_erfc
    Tensor.reciprocal = _cython_mod.tensor_reciprocal
    Tensor.tril = _cython_mod.tensor_tril
    Tensor.triu = _cython_mod.tensor_triu
    Tensor.diag = _cython_mod.tensor_diag
    Tensor.add = _cython_mod.tensor_add_method
    Tensor.sub = _cython_mod.tensor_sub_method
    Tensor.mul = _cython_mod.tensor_mul_method
    Tensor.div = _cython_mod.tensor_div_method
    Tensor.pow = _cython_mod.tensor_pow_method
    Tensor.matmul = _cython_mod.tensor_matmul_method
    Tensor.__rsub__ = _cython_mod.tensor_rsub
    Tensor.__rmul__ = _cython_mod.tensor_rmul
    Tensor.__truediv__ = _cython_mod.tensor_truediv
    Tensor.__rtruediv__ = _cython_mod.tensor_rtruediv
    Tensor.__pow__ = _cython_mod.tensor_pow_op
    Tensor.__rpow__ = _cython_mod.tensor_rpow
    Tensor.__floordiv__ = _cython_mod.tensor_floordiv
    Tensor.__rfloordiv__ = _cython_mod.tensor_rfloordiv
    Tensor.__mod__ = _cython_mod.tensor_mod
    Tensor.__rmod__ = _cython_mod.tensor_rmod
    Tensor.__rmatmul__ = _cython_mod.tensor_rmatmul
    Tensor.__and__ = _cython_mod.tensor_and
    Tensor.__or__ = _cython_mod.tensor_or
    Tensor.__xor__ = _cython_mod.tensor_xor
    Tensor.all = _cython_mod.tensor_all_method
    Tensor.any = _cython_mod.tensor_any_method
    Tensor.sum = _cython_mod.tensor_sum_method
    Tensor.prod = _cython_mod.tensor_prod_method
    Tensor.var = _cython_mod.tensor_var_method
    Tensor.var_mean = _cython_mod.tensor_var_mean_method
    Tensor.norm = _cython_mod.tensor_norm_method
    Tensor.count_nonzero = _cython_mod.tensor_count_nonzero_method
    Tensor.cumsum = _cython_mod.tensor_cumsum_method
    Tensor.cumprod = _cython_mod.tensor_cumprod_method
    Tensor.cummax = _cython_mod.tensor_cummax_method
    Tensor.argsort = _cython_mod.tensor_argsort_method
    Tensor.sort = _cython_mod.tensor_sort_method
    Tensor.topk = _cython_mod.tensor_topk_method
    Tensor.eq = _cython_mod.tensor_eq_method
    Tensor.ne = _cython_mod.tensor_ne_method
    Tensor.allclose = _cython_mod.tensor_allclose_method
    Tensor.isclose = _cython_mod.tensor_isclose_method
    Tensor.equal = _cython_mod.tensor_equal_method
    Tensor.view_as = _cython_mod.tensor_view_as
    Tensor.expand = _cython_mod.tensor_expand_method
    Tensor.expand_as = _cython_mod.tensor_expand_as_method
    Tensor.expand_copy = _cython_mod.tensor_expand_copy_method
    Tensor.narrow = _cython_mod.tensor_narrow_method
    Tensor.select = _cython_mod.tensor_select_method
    Tensor.unfold = _cython_mod.tensor_unfold_method
    Tensor.moveaxis = _cython_mod.tensor_moveaxis_method
    Tensor.swapdims = _cython_mod.tensor_swapdims_method
    Tensor.swapaxes = _cython_mod.tensor_swapaxes_method
    Tensor.gather = _cython_mod.tensor_gather_method
    Tensor.scatter = _cython_mod.tensor_scatter_method
    Tensor.index_select = _cython_mod.tensor_index_select_method
    Tensor.take = _cython_mod.tensor_take_method
    Tensor.masked_fill = _cython_mod.tensor_masked_fill_method
    Tensor.masked_select = _cython_mod.tensor_masked_select_method
    Tensor.index_put = _cython_mod.tensor_index_put_method
    Tensor.slice = _cython_mod.tensor_slice_method
    Tensor.slice_copy = _cython_mod.tensor_slice_copy_method
    Tensor.slice_scatter = _cython_mod.tensor_slice_scatter_method
    Tensor.nonzero = _cython_mod.tensor_nonzero_method
    Tensor.sum_to_size = _cython_mod.tensor_sum_to_size_method
    Tensor.softplus = _cython_mod.tensor_softplus_method
    Tensor.clamp = _cython_mod.tensor_clamp_method
    Tensor.relu6 = _cython_mod.tensor_relu6_method
    Tensor.hardtanh = _cython_mod.tensor_hardtanh_method
    Tensor.min = _cython_mod.tensor_min_method
    Tensor.max = _cython_mod.tensor_max_method
    Tensor.amin = _cython_mod.tensor_amin_method
    Tensor.amax = _cython_mod.tensor_amax_method
    Tensor.addmm = _cython_mod.tensor_addmm_method
    Tensor.bmm = _cython_mod.tensor_bmm_method
    Tensor.mm = _cython_mod.tensor_mm_method
    Tensor.chunk = _cython_mod.tensor_chunk_method
    Tensor.split = _cython_mod.tensor_split_method
    Tensor.roll = _cython_mod.tensor_roll_method
    Tensor.rot90 = _cython_mod.tensor_rot90_method
    Tensor.addcdiv = _cython_mod.tensor_addcdiv_method
    Tensor.addcmul = _cython_mod.tensor_addcmul_method
    Tensor.hypot = _cython_mod.tensor_hypot_method
    Tensor.lerp = _cython_mod.tensor_lerp_method
    Tensor.atan2 = _cython_mod.tensor_atan2_method
    Tensor.asin = _cython_mod.tensor_asin_method
    Tensor.acos = _cython_mod.tensor_acos_method
    Tensor.atan = _cython_mod.tensor_atan_method
    Tensor.as_strided_copy = _cython_mod.tensor_as_strided_copy_method
    Tensor.as_strided_scatter = _cython_mod.tensor_as_strided_scatter_method
    Tensor.multinomial = _cython_mod.tensor_multinomial_method
    Tensor.ndim = property(_cython_mod.tensor_ndim_fget)
    Tensor.T = property(_cython_mod.tensor_T_fget)
    Tensor.is_floating_point = _cython_mod.tensor_is_floating_point
    Tensor.is_complex = _cython_mod.tensor_is_complex
    Tensor.clamp_min = _cython_mod.tensor_clamp_min_method
    Tensor.clamp_max = _cython_mod.tensor_clamp_max_method
    Tensor.fmin = _cython_mod.tensor_fmin_method
    Tensor.fmax = _cython_mod.tensor_fmax_method
    Tensor.where = _cython_mod.tensor_where_method
    Tensor.logaddexp = _cython_mod.tensor_logaddexp_method
    Tensor.logaddexp2 = _cython_mod.tensor_logaddexp2_method
    Tensor.remainder = _cython_mod.tensor_remainder_method
    Tensor.fmod = _cython_mod.tensor_fmod_method
    Tensor.squeeze = _cython_mod.tensor_squeeze_method
    Tensor.unsqueeze = _cython_mod.tensor_unsqueeze_method
    Tensor.argmax = _cython_mod.tensor_argmax_method
    Tensor.argmin = _cython_mod.tensor_argmin_method
    Tensor.logical_and = _cython_mod.tensor_logical_and
    Tensor.logical_or = _cython_mod.tensor_logical_or
    Tensor.logical_xor = _cython_mod.tensor_logical_xor
    Tensor.logical_not = _cython_mod.tensor_logical_not
    Tensor.bitwise_and = _cython_mod.tensor_bitwise_and
    Tensor.bitwise_or = _cython_mod.tensor_bitwise_or
    Tensor.bitwise_xor = _cython_mod.tensor_bitwise_xor
    Tensor.bitwise_not = _cython_mod.tensor_bitwise_not
    Tensor.movedim = _cython_mod.tensor_movedim
    Tensor.diagonal = _cython_mod.tensor_diagonal
    Tensor.unbind = _cython_mod.tensor_unbind

    Tensor.numpy = _cython_mod.tensor_numpy
    Tensor._numpy_view = _cython_mod.tensor_numpy_view
    Tensor.pin_memory = _cython_mod.tensor_pin_memory
