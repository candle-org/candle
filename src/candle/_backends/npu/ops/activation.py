"""Activation functions and embedding for NPU."""
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
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU relu expects NPU tensors")

    a_storage = _unwrap_storage(a)
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
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

    storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(storage, a.shape, a.stride)


def relu_(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU relu_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
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
    return clamp(a, 0.0, 6.0)


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
    """Compute SiLU (Swish) activation using aclnnSilu."""
    if not aclnn.silu_symbols_ok():
        raise RuntimeError("aclnnSilu not available")
    return _unary_op(a, aclnn.silu, "silu")


def gelu(a):
    """Compute GELU activation using aclnnGelu."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.gelu_symbols_ok():
        raise RuntimeError("aclnnGelu not available")

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    aclnn.gelu(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape, a.stride, a.dtype,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def leaky_relu(a, negative_slope=0.01):
    """Compute Leaky ReLU activation using aclnnLeakyRelu."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.leaky_relu_symbols_ok():
        raise RuntimeError("aclnnLeakyRelu not available")

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    aclnn.leaky_relu(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape, a.stride, a.dtype,
        negative_slope,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def elu(a, alpha=1.0):
    """Compute ELU activation using aclnnElu."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.elu_symbols_ok():
        raise RuntimeError("aclnnElu not available")

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    aclnn.elu(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape, a.stride, a.dtype,
        alpha,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def mish(a):
    """Compute Mish activation using aclnnMish."""
    if _use_soc_fallback("mish"):
        return mul(a, tanh(softplus(a)))
    if not aclnn.mish_symbols_ok():
        raise RuntimeError("aclnnMish not available")
    return _unary_op(a, aclnn.mish, "mish")


def prelu(a, weight):
    """Compute PReLU activation using aclnnPrelu."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.prelu_symbols_ok():
        raise RuntimeError("aclnnPrelu not available")

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    aclnn.prelu(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(weight).data_ptr(),
        out_ptr,
        a.shape, a.stride,
        weight.shape, weight.stride,
        a.dtype,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def selu_op(a):
    """SELU activation: scale * (max(0,x) + min(0, alpha*(exp(x)-1)))."""
    _alpha = 1.6732632423543772848170429916717
    _scale = 1.0507009873554804934193349852946
    return mul(elu(a, alpha=_alpha), _scalar_to_npu_tensor(_scale, a))


def celu_op(a, alpha=1.0):
    """CELU activation: max(0,x) + min(0, alpha*(exp(x/alpha)-1))."""
    inv_alpha = _scalar_to_npu_tensor(1.0 / alpha, a)
    alpha_t = _scalar_to_npu_tensor(alpha, a)
    one = _scalar_to_npu_tensor(1.0, a)
    zero = _scalar_to_npu_tensor(0.0, a)
    # exp(x / alpha) - 1
    exp_part = sub(exp(mul(a, inv_alpha)), one)
    neg_part = mul(alpha_t, minimum(exp_part, zero))
    pos_part = maximum(a, zero)
    return add(pos_part, neg_part)


def threshold_op(a, threshold_val, value):
    """Threshold: x if x > threshold else value."""
    thresh_t = _scalar_to_npu_tensor(threshold_val, a)
    value_t = _scalar_to_npu_tensor(value, a)
    mask = gt(a, thresh_t)
    return where(mask, a, value_t)


def hardshrink_op(a, lambd=0.5):
    """Hard shrink: x if |x| > lambd else 0."""
    zero = _scalar_to_npu_tensor(0.0, a)
    lambd_t = _scalar_to_npu_tensor(lambd, a)
    mask = gt(abs(a), lambd_t)
    return where(mask, a, zero)


def softshrink_op(a, lambd=0.5):
    """Soft shrink: x-lambd if x>lambd, x+lambd if x<-lambd, else 0."""
    zero = _scalar_to_npu_tensor(0.0, a)
    lambd_t = _scalar_to_npu_tensor(lambd, a)
    neg_lambd_t = _scalar_to_npu_tensor(-lambd, a)
    pos_mask = gt(a, lambd_t)
    neg_mask = lt(a, neg_lambd_t)
    result = where(pos_mask, sub(a, lambd_t), zero)
    return where(neg_mask, add(a, lambd_t), result)


def hardswish_op(a):
    """HardSwish: x * clamp(x + 3, 0, 6) / 6."""
    three = _scalar_to_npu_tensor(3.0, a)
    six = _scalar_to_npu_tensor(6.0, a)
    return div(mul(a, clamp(add(a, three), min_val=0.0, max_val=6.0)), six)


def hardsigmoid_op(a):
    """HardSigmoid: clamp(x + 3, 0, 6) / 6."""
    six = _scalar_to_npu_tensor(6.0, a)
    three = _scalar_to_npu_tensor(3.0, a)
    return div(clamp(add(a, three), min_val=0.0, max_val=6.0), six)


def softsign_op(a):
    """Softsign: x / (1 + |x|)."""
    one = _scalar_to_npu_tensor(1.0, a)
    return div(a, add(one, abs(a)))


def rrelu_op(a, lower=0.125, upper=0.3333333333333333, training=False):
    """RReLU: if training, random slope from [lower, upper]; else fixed (lower+upper)/2."""
    zero = _scalar_to_npu_tensor(0.0, a)
    slope = (lower + upper) / 2.0
    slope_t = _scalar_to_npu_tensor(slope, a)
    mask = gt(a, zero)
    return where(mask, a, mul(a, slope_t))


def softmax(a, dim=-1):
    """Compute softmax along a dimension using aclnnSoftmax."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.softmax_symbols_ok():
        raise RuntimeError("aclnnSoftmax not available")

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
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.log_softmax_symbols_ok():
        raise RuntimeError("aclnnLogSoftmax not available")

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
