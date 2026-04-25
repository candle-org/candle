"""Candle's C extension layer — Cython accelerators for hot paths.

This package provides Cython implementations of performance-critical code
paths (dispatcher, allocator, storage creation, NPU ops, ACLNN FFI,
TensorImpl, dispatcher core, device, dtype, autograd node, autograd graph,
autograd function, autograd ops, functional wrappers, fast ops, tensor API).

Feature flags (set after import):
    _HAS_CYTHON_DISPATCH        — True if _dispatch.pyx compiled successfully
    _HAS_CYTHON_ALLOCATOR       — True if _allocator.pyx compiled successfully
    _HAS_CYTHON_STORAGE         — True if _storage.pyx compiled successfully
    _HAS_CYTHON_NPU_OPS         — True if _npu_ops.pyx compiled successfully
    _HAS_CYTHON_ACLNN_FFI       — True if _aclnn_ffi.pyx compiled successfully
    _HAS_CYTHON_DISPATCHER_CORE — True if _dispatcher_core.pyx compiled
    _HAS_CYTHON_DEVICE          — True if _device.pyx compiled successfully
    _HAS_CYTHON_DTYPE           — True if _dtype.pyx compiled successfully
    _HAS_CYTHON_AUTOGRAD_NODE   — always True (hard import, no fallback)
    _HAS_CYTHON_AUTOGRAD_GRAPH  — always True (hard import, no fallback)
    _HAS_CYTHON_AUTOGRAD_ENGINE — always True (hard import, no fallback)
    _HAS_CYTHON_AUTOGRAD_FUNCTION — always True (hard import, no fallback)
    _HAS_CYTHON_AUTOGRAD_OPS    — always True (hard import, no fallback)
    _HAS_CYTHON_FUNCTIONAL_OPS  — True if _functional_ops.pyx compiled successfully
    _HAS_CYTHON_FAST_OPS        — True if _fast_ops.pyx compiled successfully
    _HAS_CYTHON_TENSOR_API      — True if _tensor_api.pyx compiled successfully
    _HAS_CYTHON_STORAGE_IMPL    — True if _storage_impl.pyx compiled successfully
"""

_HAS_CYTHON_DISPATCH = False
_HAS_CYTHON_ALLOCATOR = False
_HAS_CYTHON_STORAGE = False
_HAS_CYTHON_NPU_OPS = False
_HAS_CYTHON_ACLNN_FFI = False
_HAS_CYTHON_DISPATCHER_CORE = False
_HAS_CYTHON_DEVICE = False
_HAS_CYTHON_DTYPE = False
# Autograd core modules are required (hard imports below -- no fallback).
_HAS_CYTHON_AUTOGRAD_NODE = False
_HAS_CYTHON_AUTOGRAD_GRAPH = False
_HAS_CYTHON_AUTOGRAD_ENGINE = False
_HAS_CYTHON_AUTOGRAD_FUNCTION = False
_HAS_CYTHON_AUTOGRAD_OPS = False
_HAS_CYTHON_FUNCTIONAL_OPS = False
_HAS_CYTHON_FAST_OPS = False
_HAS_CYTHON_TENSOR_API = False

try:
    from candle._dispatch import cy_dispatch, cy_dispatch_with_keyset  # noqa: F401
    _HAS_CYTHON_DISPATCH = True
except ImportError:
    pass

try:
    from ._allocator import FastNpuAllocator  # noqa: F401
    _HAS_CYTHON_ALLOCATOR = True
except ImportError:
    pass

try:
    from ._storage import cy_npu_storage_from_ptr  # noqa: F401
    _HAS_CYTHON_STORAGE = True
except ImportError:
    pass

try:
    from ._npu_ops import fast_binary_op  # noqa: F401
    _HAS_CYTHON_NPU_OPS = True
except ImportError:
    pass

try:
    from ._aclnn_ffi import (  # noqa: F401
        init as aclnn_ffi_init,
        create_tensor, destroy_tensor,
        create_scalar, destroy_scalar,
        create_int_array, destroy_int_array,
        destroy_executor, resolve_op, execute,
        binary_op_with_alpha, binary_op_no_alpha,
    )
    _HAS_CYTHON_ACLNN_FFI = True
except ImportError:
    pass

from ._tensor_impl import TensorImpl, _VersionCounterProxy  # noqa: F401  # pylint: disable=import-error,no-name-in-module

try:
    from ._dispatcher_core import cy_dispatch_with_keyset_fast  # noqa: F401
    _HAS_CYTHON_DISPATCHER_CORE = True
except ImportError:
    pass

try:
    from candle._device import FastDevice  # noqa: F401
    _HAS_CYTHON_DEVICE = True
except ImportError:
    pass

try:
    from candle._dtype import FastDType  # noqa: F401
    _HAS_CYTHON_DTYPE = True
except ImportError:
    pass

try:
    from ._autograd_node import (  # pylint: disable=import-error,no-name-in-module
        AccumulateGrad,  # noqa: F401
        FastNode,  # noqa: F401
        InputMetadata,  # noqa: F401
        Node,  # noqa: F401
        SavedTensor,  # noqa: F401
        _NodeHookHandle,  # noqa: F401
        _SavedValue,  # noqa: F401
    )
    _HAS_CYTHON_AUTOGRAD_NODE = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_NODE = False

try:
    from ._autograd_graph import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
        GradientEdge,
        current_saved_tensors_hooks,
        get_gradient_edge,
        saved_tensors_hooks,
    )
    _HAS_CYTHON_AUTOGRAD_GRAPH = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_GRAPH = False

try:
    from ._autograd_engine import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
        _GraphTask,
        _build_dependencies,
        _run_backward,
        backward,
        current_anomaly_parent,
        grad,
        is_anomaly_check_nan_enabled,
        is_anomaly_enabled,
        is_create_graph_enabled,
        pop_anomaly_config,
        pop_evaluating_node,
        push_anomaly_config,
        push_evaluating_node,
    )
    _HAS_CYTHON_AUTOGRAD_ENGINE = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_ENGINE = False

try:
    from ._autograd_function import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
        FunctionCtx,
        _function_apply,
    )
    _HAS_CYTHON_AUTOGRAD_FUNCTION = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_FUNCTION = False

try:
    from ._autograd_ops import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
        _strip_autograd_keys,
        _grad_context,
        _backward_dispatch_keyset,
        _autograd_unary_passthrough,
        _autograd_binary,
        _autograd_binary_args,
        _autograd_unary_args,
        _norm_extract_weight_bias,
        _autograd_norm,
    )
    _HAS_CYTHON_AUTOGRAD_OPS = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_OPS = False

try:
    from ._functional_ops import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
        _has_torch_function as cy_has_torch_function,
        _handle_torch_function as cy_handle_torch_function,
        add as functional_add,
        mul as functional_mul,
        matmul as functional_matmul,
        relu as functional_relu,
        transpose as functional_transpose,
        reshape as functional_reshape,
        neg as functional_neg,
    )
    _HAS_CYTHON_FUNCTIONAL_OPS = True
except ImportError:
    pass

try:
    import importlib
    _legacy_fast_ops = importlib.import_module(f"{__name__}._fast_ops")  # noqa: F401
    _HAS_CYTHON_FAST_OPS = True
except ImportError:
    pass

try:
    from ._TensorBase import TensorBase, _TensorBase  # noqa: F401  # pylint: disable=import-error,no-name-in-module
    _HAS_CYTHON_TENSOR_API = True
except ImportError:
    _HAS_CYTHON_TENSOR_API = False

_HAS_CYTHON_STORAGE_IMPL = False

try:
    from ._storage_impl import StorageImpl  # noqa: F401
    _HAS_CYTHON_STORAGE_IMPL = True
except ImportError:
    pass

# =============================================================================
# torch._C stubs and TensorBase (from _C_stubs.py)
# =============================================================================

# pylint: disable=import-error,no-name-in-module,possibly-unused-variable
import abc


def _add_docstr(obj, docstr):
    """Minimal torch._C._add_docstr stub."""
    obj.__doc__ = docstr
    return obj


class _disabled_torch_dispatch_impl:
    """Minimal torch._C._disabled_torch_dispatch_impl context manager."""
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


_torch_function_enabled = True


class DisableTorchFunctionSubclass:
    """Minimal torch._C.DisableTorchFunctionSubclass context manager."""
    def __init__(self): pass
    def __enter__(self):
        global _torch_function_enabled
        self._prev = _torch_function_enabled
        _torch_function_enabled = False
        return self
    def __exit__(self, *args):
        global _torch_function_enabled
        _torch_function_enabled = self._prev


def _has_storage(tensor):
    """Minimal torch._C._has_storage stub."""
    return hasattr(tensor, '_storage') and tensor._storage is not None


def _get_tracing_state():
    """Minimal torch._C._get_tracing_state stub."""
    return None


def _get_privateuse():
    """Minimal torch._C._get_privateuse stub."""
    return "npu"


class _dlpack_exchange_api:
    """Minimal torch._C._dlpack_exchange_api stub."""
    @staticmethod
    def to_dlpack(tensor): raise NotImplementedError("DLPack not supported")
    @staticmethod
    def from_dlpack(dlpack): raise NotImplementedError("DLPack not supported")


def _to_dlpack(tensor):
    raise NotImplementedError("DLPack not supported")


def _to_dlpack_versioned(tensor, version):
    raise NotImplementedError("DLPack not supported")


class _VariableFunctions:
    """Minimal torch._C._VariableFunctions stub."""
    @staticmethod
    def rsub(tensor, other):
        import numpy as np
        if isinstance(other, (int, float, bool, complex, np.integer, np.floating)):
            from candle._functional import mul as _mul, sub as _sub
            result = _mul(tensor, -1)
            return result + other
        from candle._functional import sub as _sub
        return _sub(other, tensor)


def _get_PyTorchFileReader():
    from ._stream import PyTorchFileReader
    return PyTorchFileReader


def _get_PyTorchFileWriter():
    from ._stream import PyTorchFileWriter
    return PyTorchFileWriter


def _get_privateuse1_backend_name():
    return "npu"


from ._Storage import (
    StorageBase,
    _NPUUntypedStorage,
    _MPSUntypedStorage,
    _MetaUntypedStorage,
    PendingStorage,
    typed_storage_from_numpy,
    empty_cpu_typed_storage,
    meta_typed_storage_from_shape,
    meta_typed_storage_from_size,
    npu_typed_storage_from_ptr,
    mps_typed_storage_from_numpy,
    mps_typed_storage_from_ptr,
    cuda_typed_storage_from_numpy,
    empty_cuda_typed_storage,
    cuda_typed_storage_to_numpy,
    pinned_cpu_typed_storage_from_numpy,
    _get_storage_classes,
    _install_typed_storage_compat,
    _make_legacy_classes,
    __getattr__ as _storage_getattr,
    _FloatStorage,
    _DoubleStorage,
    _HalfStorage,
    _LongStorage,
    _IntStorage,
    _ShortStorage,
    _ByteStorage,
    _BoolStorage,
    _BFloat16Storage,
    _ComplexFloatStorage,
    _ComplexDoubleStorage,
    _Storage,
)


def __getattr__(name):
    return _storage_getattr(name)


# =============================================================================
# TensorBase — torch._C.TensorBase equivalent for candle/tensor.py
# Must be defined AFTER all storage factories so _tensor_impl can import from _C.
# =============================================================================

from ._tensor_impl import TensorImpl, _StrideTuple

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




def _install_tensor_api():
    """Install Cython tensor API methods on TensorBase (called after all modules loaded)."""
    if not _HAS_CYTHON_TENSOR_API:
        return
    from . import _tensor_api as _tensor_api_mod  # pylint: disable=import-self
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


_install_tensor_api()

