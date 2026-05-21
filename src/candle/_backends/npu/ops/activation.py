"""Activation functions and embedding for NPU."""

try:
    from candle._C._npu_ops import fast_relu as _fast_relu_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_RELU = True
except ImportError:
    _fast_relu_impl = None  # type: ignore[assignment]
    _HAS_FAST_RELU = False

try:
    from candle._C._npu_ops import fast_hardtanh as _fast_hardtanh_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_HARDTANH = True
except ImportError:
    _fast_hardtanh_impl = None  # type: ignore[assignment]
    _HAS_FAST_HARDTANH = False

try:
    from candle._C._npu_ops import fast_dropout as _fast_dropout_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_DROPOUT = True
except ImportError:
    _fast_dropout_impl = None  # type: ignore[assignment]
    _HAS_FAST_DROPOUT = False

try:
    from candle._C._npu_ops import fast_embedding as _fast_embedding_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_EMBEDDING = True
except ImportError:
    _fast_embedding_impl = None  # type: ignore[assignment]
    _HAS_FAST_EMBEDDING = False

try:
    from candle._C._npu_ops import fast_prelu as _fast_prelu_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_PRELU = True
except ImportError:
    _fast_prelu_impl = None  # type: ignore[assignment]
    _HAS_FAST_PRELU = False

try:
    from candle._C._npu_ops import fast_softplus as _fast_softplus_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_SOFTPLUS = True
except ImportError:
    _fast_softplus_impl = None  # type: ignore[assignment]
    _HAS_FAST_SOFTPLUS = False

try:
    from candle._C._npu_ops import fast_softmax as _fast_softmax_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_SOFTMAX = True
except ImportError:
    _fast_softmax_impl = None  # type: ignore[assignment]
    _HAS_FAST_SOFTMAX = False

try:
    from candle._C._npu_ops import fast_log_softmax as _fast_log_softmax_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_LOG_SOFTMAX = True
except ImportError:
    _fast_log_softmax_impl = None  # type: ignore[assignment]
    _HAS_FAST_LOG_SOFTMAX = False

try:
    from candle._C._npu_ops import fast_leaky_relu as _fast_leaky_relu_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_LEAKY_RELU = True
except ImportError:
    _fast_leaky_relu_impl = None  # type: ignore[assignment]
    _HAS_FAST_LEAKY_RELU = False

try:
    from candle._C._npu_ops import fast_elu as _fast_elu_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_ELU = True
except ImportError:
    _fast_elu_impl = None  # type: ignore[assignment]
    _HAS_FAST_ELU = False

try:
    from candle._C._npu_ops import fast_silu as _fast_silu_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_SILU = True
except ImportError:
    _fast_silu_impl = None  # type: ignore[assignment]
    _HAS_FAST_SILU = False

try:
    from candle._C._npu_ops import fast_gelu as _fast_gelu_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_GELU = True
except ImportError:
    _fast_gelu_impl = None  # type: ignore[assignment]
    _HAS_FAST_GELU = False

try:
    from candle._C._npu_ops import fast_mish as _fast_mish_impl  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_MISH = True
except ImportError:
    _fast_mish_impl = None  # type: ignore[assignment]
    _HAS_FAST_MISH = False

try:
    from candle._C._npu_ops import (  # pylint: disable=import-error,no-name-in-module
        fast_relu6 as _fast_relu6_impl,
        fast_selu as _fast_selu_impl,
        fast_celu as _fast_celu_impl,
        fast_threshold as _fast_threshold_impl,
        fast_hardshrink as _fast_hardshrink_impl,
        fast_softshrink as _fast_softshrink_impl,
        fast_hardswish as _fast_hardswish_impl,
        fast_hardsigmoid as _fast_hardsigmoid_impl,
        fast_softsign as _fast_softsign_impl,
        fast_rrelu as _fast_rrelu_impl,
    )
    _HAS_FAST_ACTIVATION_COMPOSITES = True
except ImportError:
    _fast_relu6_impl = None  # type: ignore[assignment]
    _fast_selu_impl = None  # type: ignore[assignment]
    _fast_celu_impl = None  # type: ignore[assignment]
    _fast_threshold_impl = None  # type: ignore[assignment]
    _fast_hardshrink_impl = None  # type: ignore[assignment]
    _fast_softshrink_impl = None  # type: ignore[assignment]
    _fast_hardswish_impl = None  # type: ignore[assignment]
    _fast_hardsigmoid_impl = None  # type: ignore[assignment]
    _fast_softsign_impl = None  # type: ignore[assignment]
    _fast_rrelu_impl = None  # type: ignore[assignment]
    _HAS_FAST_ACTIVATION_COMPOSITES = False

from ._helpers import (
    _unwrap_storage, _wrap_tensor, _unary_op, _binary_op,
    _broadcast_shape, _broadcast_shape_checked,
    _numel, _dtype_itemsize, _use_soc_fallback,
    _scalar_to_npu_tensor, _scalar_to_npu_tensor_no_add,
    _npu_broadcast_to, _npu_arange_1d, _npu_linear_index,
    _npu_add_scalar_, npu_index_put_impl,
    _normalize_reduction_dims, _reduce_out_shape,
    _cast_tensor_dtype, _normalize_tensor_sequence_args,
    _matmul_out_shape,
    _iter_indices, _broadcast_index, _batch_offset,
    bool_dtype, int32_dtype, int64_dtype, float_dtype,
    npu_typed_storage_from_ptr, reshape,
    aclnn, npu_runtime, npu_state, ops_soc,
)
from .comparison import gt, lt
from .elementwise import clamp, where
from .math import abs, add, div, exp, frac, log, mul, neg, sin, sub, tanh
from .reduce import maximum, minimum


def relu(a):
    if _HAS_FAST_RELU:
        return _fast_relu_impl(a)
    raise RuntimeError("Cython NPU relu implementation is unavailable")


def relu_(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU relu_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)

    if _HAS_FAST_RELU:
        out = _fast_relu_impl(a)
        npu_runtime.memcpy_d2d(
            a_storage.data_ptr(),
            out_size,
            _unwrap_storage(out).data_ptr(),
            runtime=runtime,
        )
        return a

    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.relu(
        a_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    npu_runtime.memcpy_d2d(
        a_storage.data_ptr(),
        out_size,
        out_ptr,
        runtime=runtime,
    )
    npu_runtime.get_runtime((a.device.index or 0)).defer_free(out_ptr)
    return a


def relu6(a):
    if _HAS_FAST_ACTIVATION_COMPOSITES:
        return _fast_relu6_impl(a)
    raise RuntimeError("Cython NPU relu6 implementation is unavailable")


def softplus(a, beta=1.0, threshold=20.0):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU softplus expects NPU tensors")

    if _use_soc_fallback("softplus"):
        beta = float(beta)
        threshold = float(threshold)
        bx = mul(a, beta)
        base = add(relu(bx), log(add(exp(neg(abs(bx))), 1)))
        out = div(base, beta)
        if threshold > 0:
            thr = _scalar_to_npu_tensor(threshold, bx)
            mask = gt(bx, thr)
            out = where(mask, a, out)
        return out

    if _HAS_FAST_SOFTPLUS:
        return _fast_softplus_impl(a, beta, threshold)

    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    aclnn.softplus(
        storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        beta,
        threshold,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, a.shape, a.stride)


def hardtanh(a, min_val=-1.0, max_val=1.0):
    if _HAS_FAST_HARDTANH:
        try:
            return _fast_hardtanh_impl(a, min_val, max_val)
        except RuntimeError as exc:
            if "561103" not in str(exc):
                raise
            # Fallback to clamp when hardtanh kernel is unsupported.
            return clamp(a, min_val, max_val)

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU hardtanh expects NPU tensors")
    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    try:
        aclnn.hardtanh(
            storage.data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            a.dtype,
            min_val,
            max_val,
            runtime,
            stream=stream.stream,
        )
    except RuntimeError as exc:
        if "561103" not in str(exc):
            raise
        # Fallback to clamp when hardtanh kernel is unsupported.
        return clamp(a, min_val, max_val)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def silu(a):
    if _HAS_FAST_SILU:
        return _fast_silu_impl(a)
    raise RuntimeError("Cython NPU silu implementation is unavailable")


def gelu(a):
    if _HAS_FAST_GELU:
        return _fast_gelu_impl(a)
    raise RuntimeError("Cython NPU gelu implementation is unavailable")


def leaky_relu(a, negative_slope=0.01):
    if _HAS_FAST_LEAKY_RELU:
        return _fast_leaky_relu_impl(a, negative_slope)
    raise RuntimeError("Cython NPU leaky_relu implementation is unavailable")


def elu(a, alpha=1.0):
    if _HAS_FAST_ELU:
        return _fast_elu_impl(a, alpha)
    raise RuntimeError("Cython NPU elu implementation is unavailable")


def mish(a):
    """Compute Mish activation using aclnnMish."""
    if _use_soc_fallback("mish"):
        return mul(a, tanh(softplus(a)))
    if not aclnn.mish_symbols_ok():
        raise RuntimeError("aclnnMish not available")
    if _HAS_FAST_MISH:
        return _fast_mish_impl(a)
    return _unary_op(a, aclnn.mish, "mish")


def prelu(a, weight):
    if _HAS_FAST_PRELU:
        return _fast_prelu_impl(a, weight)
    raise RuntimeError("Cython NPU prelu implementation is unavailable")


def selu_op(a):
    """SELU activation: scale * (max(0,x) + min(0, alpha*(exp(x)-1)))."""
    if _HAS_FAST_ACTIVATION_COMPOSITES:
        return _fast_selu_impl(a)
    raise RuntimeError("Cython NPU selu implementation is unavailable")


def celu_op(a, alpha=1.0):
    """CELU activation: max(0,x) + min(0, alpha*(exp(x/alpha)-1))."""
    if _HAS_FAST_ACTIVATION_COMPOSITES:
        return _fast_celu_impl(a, float(alpha))
    raise RuntimeError("Cython NPU celu implementation is unavailable")


def threshold_op(a, threshold_val, value):
    """Threshold: x if x > threshold else value."""
    if _HAS_FAST_ACTIVATION_COMPOSITES:
        return _fast_threshold_impl(a, float(threshold_val), float(value))
    raise RuntimeError("Cython NPU threshold implementation is unavailable")


def hardshrink_op(a, lambd=0.5):
    """Hard shrink: x if |x| > lambd else 0."""
    if _HAS_FAST_ACTIVATION_COMPOSITES:
        return _fast_hardshrink_impl(a, float(lambd))
    raise RuntimeError("Cython NPU hardshrink implementation is unavailable")


def softshrink_op(a, lambd=0.5):
    """Soft shrink: x-lambd if x>lambd, x+lambd if x<-lambd, else 0."""
    if _HAS_FAST_ACTIVATION_COMPOSITES:
        return _fast_softshrink_impl(a, float(lambd))
    raise RuntimeError("Cython NPU softshrink implementation is unavailable")


def hardswish_op(a):
    """HardSwish: x * clamp(x + 3, 0, 6) / 6."""
    if _HAS_FAST_ACTIVATION_COMPOSITES:
        return _fast_hardswish_impl(a)
    raise RuntimeError("Cython NPU hardswish implementation is unavailable")


def hardsigmoid_op(a):
    """HardSigmoid: clamp(x + 3, 0, 6) / 6."""
    if _HAS_FAST_ACTIVATION_COMPOSITES:
        return _fast_hardsigmoid_impl(a)
    raise RuntimeError("Cython NPU hardsigmoid implementation is unavailable")


def softsign_op(a):
    """Softsign: x / (1 + |x|)."""
    if _HAS_FAST_ACTIVATION_COMPOSITES:
        return _fast_softsign_impl(a)
    raise RuntimeError("Cython NPU softsign implementation is unavailable")


def rrelu_op(a, lower=0.125, upper=0.3333333333333333, training=False):
    """RReLU: if training, random slope from [lower, upper]; else fixed (lower+upper)/2."""
    if _HAS_FAST_ACTIVATION_COMPOSITES:
        return _fast_rrelu_impl(a, float(lower), float(upper), bool(training))
    raise RuntimeError("Cython NPU rrelu implementation is unavailable")


def softmax(a, dim=-1):
    """Compute softmax along a dimension using aclnnSoftmax."""
    if not aclnn.softmax_symbols_ok():
        raise RuntimeError("aclnnSoftmax not available")
    if _HAS_FAST_SOFTMAX:
        return _fast_softmax_impl(a, dim)

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    # Normalize dim
    if dim < 0:
        dim += len(a.shape)

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    aclnn.softmax(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape, a.stride, a.dtype,
        dim,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def log_softmax(a, dim=-1):
    """Compute log_softmax along a dimension using aclnnLogSoftmax."""
    if not aclnn.log_softmax_symbols_ok():
        raise RuntimeError("aclnnLogSoftmax not available")
    if _HAS_FAST_LOG_SOFTMAX:
        return _fast_log_softmax_impl(a, dim)

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    # Normalize dim
    if dim < 0:
        dim += len(a.shape)

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    aclnn.log_softmax(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape, a.stride, a.dtype,
        dim,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def embedding(weight, indices, padding_idx=None, scale_grad_by_freq=False, sparse=False):
    """Compute embedding lookup using aclnnEmbedding."""
    runtime = npu_runtime.get_runtime((weight.device.index or 0))
    stream = npu_state.current_stream((weight.device.index or 0))

    if not aclnn.embedding_symbols_ok():
        raise RuntimeError("aclnnEmbedding not available")
    if _HAS_FAST_EMBEDDING:
        return _fast_embedding_impl(weight, indices)

    # Output shape: indices.shape + (embedding_dim,)
    embedding_dim = weight.shape[1] if len(weight.shape) > 1 else weight.shape[0]
    out_shape = indices.shape + (embedding_dim,)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(weight.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    # Note: aclnnEmbedding doesn't support padding_idx, scale_grad_by_freq, sparse parameters
    # These are ignored for now
    aclnn.embedding(
        _unwrap_storage(weight).data_ptr(),
        _unwrap_storage(indices).data_ptr(),
        out_ptr,
        weight.shape, weight.stride,
        indices.shape, indices.stride,
        out_shape, out_stride,
        weight.dtype,
        indices.dtype,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, weight.dtype, device=weight.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def _dropout_310b_mask(a, keep_prob):
    from ..creation import empty_create
    from .... import npu as npu_mod

    numel = _numel(a.shape)
    if numel == 0:
        return empty_create(a.shape, dtype=bool_dtype, device=a.device)

    idx = _npu_arange_1d(numel, a.device)
    idx_f = _cast_tensor_dtype(idx, float_dtype)

    seed, offset = npu_mod._get_and_advance_offset(device_index=(a.device.index or 0), increment=10)
    seed_t = _scalar_to_npu_tensor(float(seed + offset), idx_f)

    val = sin(add(mul(idx_f, 12.9898), mul(seed_t, 78.233)))
    val = abs(mul(val, 43758.5453))
    val = frac(val)
    val = reshape(val, a.shape)

    keep_t = _scalar_to_npu_tensor(float(keep_prob), val)
    return lt(val, keep_t)


def dropout(a, p=0.5, training=True):
    """Compute dropout using aclnnDropoutGenMask + aclnnDropoutDoMask."""
    if not training or p == 0:
        return a

    if _use_soc_fallback("dropout"):
        if p >= 1:
            from ..creation import zeros_create
            return zeros_create(a.shape, dtype=a.dtype, device=a.device)
        if not getattr(a.dtype, "is_floating_point", True):
            raise ValueError("NPU dropout expects floating-point tensors")
        keep_prob = 1.0 - float(p)
        keep = _dropout_310b_mask(a, keep_prob)
        out = where(keep, a, 0)
        return mul(out, 1.0 / keep_prob)

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.dropout_symbols_ok():
        raise RuntimeError("aclnnDropout symbols not available")

    if _HAS_FAST_DROPOUT:
        out, mask_ptr, mask_numel = _fast_dropout_impl(a, p)
        out._backward_data = {"mask_ptr": mask_ptr, "mask_numel": mask_numel, "p": p}
        return out

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    # Allocate mask (bit-packed: align(numel, 128) / 8 bytes)
    mask_numel = (out_numel + 127) // 128 * 128 // 8
    mask_ptr = npu_runtime._alloc_device(mask_numel, runtime=runtime)

    # Get seed and offset from npu module
    from .... import npu as npu_mod
    seed, offset = npu_mod._get_and_advance_offset(device_index=(a.device.index or 0), increment=10)

    # Step 1: Generate mask
    aclnn.dropout_gen_mask(
        a.shape, p, seed, offset,
        mask_ptr, mask_numel,
        runtime, stream=stream.stream
    )

    # Step 2: Apply mask
    aclnn.dropout_do_mask(
        _unwrap_storage(a).data_ptr(),
        mask_ptr,
        out_ptr,
        a.shape, a.stride, a.dtype,
        mask_numel, p,
        runtime, stream=stream.stream
    )

    # Save mask for backward (dropout backward reuses the same mask)
    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    out = _wrap_tensor(out_storage, out_shape, out_stride)
    out._backward_data = {"mask_ptr": mask_ptr, "mask_numel": mask_numel, "p": p}
    return out
