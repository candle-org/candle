"""Normalization operations for NPU."""
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
from .math import add, div, mul, rsqrt, sqrt, sub
from .random import copy_
from .reduce import maximum, mean, norm_


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Compute layer normalization using aclnnLayerNorm."""
    if _use_soc_fallback("layer_norm"):
        return _layer_norm_310b_fallback(input, normalized_shape, weight=weight, bias=bias, eps=eps)

    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    if not aclnn.layer_norm_symbols_ok():
        raise RuntimeError("aclnnLayerNorm not available")

    # Compute stats shape (all dims except normalized dims)
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    num_normalized_dims = len(normalized_shape)
    # Stats (mean/rstd) must have same rank as input, with normalized dims replaced by 1
    if num_normalized_dims > 0:
        stats_shape = tuple(
            s if i < len(input.shape) - num_normalized_dims else 1
            for i, s in enumerate(input.shape)
        )
    else:
        stats_shape = input.shape
    stats_stride = npu_runtime._contiguous_stride(stats_shape)
    stats_numel = _numel(stats_shape)

    out_shape = input.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)

    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)
    # Allocate mean/rstd for backward pass (layer_norm backward needs them)
    stats_numel_val = max(stats_numel, 1)
    float_dtype = input.dtype  # same dtype for stats
    mean_ptr = npu_runtime._alloc_device(stats_numel_val * 4, runtime=runtime)  # float32
    rstd_ptr = npu_runtime._alloc_device(stats_numel_val * 4, runtime=runtime)  # float32
    # Wrap in Storage to prevent early deallocation
    mean_storage = npu_typed_storage_from_ptr(mean_ptr, stats_numel_val, float_dtype, device=input.device)
    rstd_storage = npu_typed_storage_from_ptr(rstd_ptr, stats_numel_val, float_dtype, device=input.device)

    weight_ptr = _unwrap_storage(weight).data_ptr() if weight is not None else None
    bias_ptr = _unwrap_storage(bias).data_ptr() if bias is not None else None

    aclnn.layer_norm(
        _unwrap_storage(input).data_ptr(),
        weight_ptr,
        bias_ptr,
        out_ptr,
        mean_ptr,
        rstd_ptr,
        input.shape, input.stride,
        weight.shape if weight is not None else (),
        weight.stride if weight is not None else (),
        bias.shape if bias is not None else (),
        bias.stride if bias is not None else (),
        out_shape, out_stride,
        stats_shape, stats_stride,
        normalized_shape,
        eps,
        input.dtype,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, input.dtype, device=input.device)
    out = _wrap_tensor(out_storage, out_shape, out_stride)
    # Attach mean/rstd for backward pass
    out._backward_data = {
        "mean_ptr": mean_ptr, "rstd_ptr": rstd_ptr,
        "mean_storage": mean_storage, "rstd_storage": rstd_storage,
        "stats_shape": stats_shape, "stats_stride": stats_stride,
        "normalized_shape": tuple(normalized_shape),
    }
    return out


def _layer_norm_310b_fallback(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    n_norm = len(normalized_shape)
    if n_norm == 0:
        return input

    axis_dims = tuple(range(input.dim() - n_norm, input.dim()))
    lead = input.dim() - n_norm
    stats_shape = (1,) * lead + tuple(normalized_shape)

    x = input if input.dtype == float_dtype else _cast_tensor_dtype(input, float_dtype)
    mean_t = mean(x, dim=axis_dims, keepdim=True)
    diff = sub(x, mean_t)
    var = mean(mul(diff, diff), dim=axis_dims, keepdim=True)
    eps_t = _scalar_to_npu_tensor(float(eps), var)
    inv_std = rsqrt(add(var, eps_t))
    out = mul(diff, inv_std)

    if weight is not None:
        w = weight if weight.dtype == float_dtype else _cast_tensor_dtype(weight, float_dtype)
        w = reshape(w, stats_shape)
        out = mul(out, w)
    if bias is not None:
        b = bias if bias.dtype == float_dtype else _cast_tensor_dtype(bias, float_dtype)
        b = reshape(b, stats_shape)
        out = add(out, b)

    if input.dtype != float_dtype:
        out = _cast_tensor_dtype(out, input.dtype)
    return out


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    """Compute batch normalization using aclnnBatchNorm."""
    if _use_soc_fallback("batch_norm"):
        return _batch_norm_310b_fallback(input, running_mean, running_var, weight=weight, bias=bias,
                                         training=training, momentum=momentum, eps=eps)

    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    if not aclnn.batch_norm_symbols_ok():
        raise RuntimeError("aclnnBatchNorm not available")

    out_shape = input.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    weight_ptr = _unwrap_storage(weight).data_ptr() if weight is not None else None
    bias_ptr = _unwrap_storage(bias).data_ptr() if bias is not None else None
    running_mean_ptr = _unwrap_storage(running_mean).data_ptr() if running_mean is not None else None
    running_var_ptr = _unwrap_storage(running_var).data_ptr() if running_var is not None else None

    # Allocate save_mean/save_invstd externally for backward pass
    C = input.shape[1] if len(input.shape) >= 2 else 1
    save_mean_ptr = npu_runtime._alloc_device(C * 4, runtime=runtime)
    save_invstd_ptr = npu_runtime._alloc_device(C * 4, runtime=runtime)
    # Wrap in Storage to prevent GC
    save_mean_storage = npu_typed_storage_from_ptr(save_mean_ptr, C, input.dtype, device=input.device)
    save_invstd_storage = npu_typed_storage_from_ptr(save_invstd_ptr, C, input.dtype, device=input.device)

    aclnn.batch_norm(
        _unwrap_storage(input).data_ptr(),
        weight_ptr,
        bias_ptr,
        running_mean_ptr,
        running_var_ptr,
        out_ptr,
        input.shape, input.stride,
        weight.shape if weight is not None else (),
        weight.stride if weight is not None else (),
        bias.shape if bias is not None else (),
        bias.stride if bias is not None else (),
        running_mean.shape if running_mean is not None else (),
        running_mean.stride if running_mean is not None else (),
        running_var.shape if running_var is not None else (),
        running_var.stride if running_var is not None else (),
        out_shape, out_stride,
        training, momentum, eps,
        input.dtype,
        runtime, stream=stream.stream,
        ext_save_mean_ptr=save_mean_ptr,
        ext_save_invstd_ptr=save_invstd_ptr,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, input.dtype, device=input.device)
    out = _wrap_tensor(out_storage, out_shape, out_stride)
    out._backward_data = {
        "save_mean_ptr": save_mean_ptr, "save_invstd_ptr": save_invstd_ptr,
        "save_mean_storage": save_mean_storage, "save_invstd_storage": save_invstd_storage,
        "C": C, "training": training, "eps": eps,
    }
    return out


def _batch_norm_310b_fallback(input, running_mean, running_var, weight=None, bias=None,
                               training=False, momentum=0.1, eps=1e-5):
    if input.dim() < 2:
        raise ValueError("batch_norm expects input with at least 2 dims")

    C = int(input.shape[1])
    stats_shape = (1, C) + (1,) * (input.dim() - 2)

    if training or running_mean is None or running_var is None:
        dims = [0] + list(range(2, input.dim()))
        mean_t = mean(input, dim=dims, keepdim=True)
        diff = sub(input, mean_t)
        var_t = mean(mul(diff, diff), dim=dims, keepdim=True)

        if running_mean is not None:
            mean_reshaped = reshape(mean_t, (C,))
            new_rm = add(mul(running_mean, (1.0 - float(momentum))), mul(mean_reshaped, float(momentum)))
            copy_(running_mean, new_rm)
        if running_var is not None:
            var_reshaped = reshape(var_t, (C,))
            new_rv = add(mul(running_var, (1.0 - float(momentum))), mul(var_reshaped, float(momentum)))
            copy_(running_var, new_rv)
    else:
        mean_t = reshape(running_mean, stats_shape)
        var_t = reshape(running_var, stats_shape)

    eps_t = _scalar_to_npu_tensor(float(eps), mean_t)
    denom = sqrt(add(var_t, eps_t))
    out = div(sub(input, mean_t), denom)

    if weight is not None:
        w = reshape(weight, stats_shape)
        out = mul(out, w)
    if bias is not None:
        b = reshape(bias, stats_shape)
        out = add(out, b)
    return out


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """Compute group normalization using aclnnLayerNorm (composite implementation).

    This avoids the aclnnGroupNorm state contamination bug in CANN 8.3.RC2.
    Algorithm:
    1. Reshape input from (N, C, H, W) to (N*num_groups, C//num_groups * H * W)
    2. Apply layer_norm over the last dimension (normalizes each group independently)
    3. Reshape back to (N, C, H, W)
    4. Apply affine transform: result * weight + bias
    """
    if not aclnn.layer_norm_symbols_ok():
        raise RuntimeError("aclnnLayerNorm not available (required for group_norm)")

    # Extract dimensions
    N = input.shape[0]
    C = input.shape[1]
    spatial_dims = input.shape[2:]
    spatial_size = 1
    for dim in spatial_dims:
        spatial_size *= dim

    if C % num_groups != 0:
        raise ValueError(f"num_channels ({C}) must be divisible by num_groups ({num_groups})")

    channels_per_group = C // num_groups

    # Step 1: Reshape to (N*num_groups, channels_per_group * spatial_size)
    reshaped_shape = (N * num_groups, channels_per_group * spatial_size)
    reshaped = reshape(input, reshaped_shape)

    # Step 2: Apply layer_norm over the last dimension (no weight/bias yet)
    normalized_shape = (channels_per_group * spatial_size,)
    normalized = layer_norm(reshaped, normalized_shape, weight=None, bias=None, eps=eps)

    # Step 3: Reshape back to original shape
    result = reshape(normalized, input.shape)

    # Step 4: Apply affine transform if weight/bias provided
    if weight is not None:
        # Reshape weight from (C,) to (1, C, 1, 1, ...) for broadcasting
        weight_shape = (1, C) + (1,) * len(spatial_dims)
        weight_reshaped = reshape(weight, weight_shape)
        result = mul(result, weight_reshaped)

    if bias is not None:
        # Reshape bias from (C,) to (1, C, 1, 1, ...) for broadcasting
        bias_shape = (1, C) + (1,) * len(spatial_dims)
        bias_reshaped = reshape(bias, bias_shape)
        result = add(result, bias_reshaped)

    return result


def instance_norm(input, weight=None, bias=None, running_mean=None, running_var=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5, cudnn_enabled=False):
    """Instance normalization as composite of existing dispatched ops.

    Note: aclnnInstanceNorm returns 161002 on CANN 8.3.RC2 (Ascend910B),
    so we use composite implementation.
    """
    if input.dim() < 2:
        raise ValueError("instance_norm expects input with at least 2 dims")

    C = int(input.shape[1])
    ndim = input.dim()
    spatial_axes = list(range(2, ndim))

    if use_input_stats:
        mean_t = mean(input, dim=spatial_axes, keepdim=True)
        diff = sub(input, mean_t)
        var_t = mean(mul(diff, diff), dim=spatial_axes, keepdim=True)

        if running_mean is not None:
            batch_dims = [0] + spatial_axes
            global_mean = mean(input, dim=batch_dims, keepdim=False)
            new_rm = add(mul(running_mean, (1.0 - float(momentum))), mul(global_mean, float(momentum)))
            copy_(running_mean, new_rm)
        if running_var is not None:
            batch_dims = [0] + spatial_axes
            global_diff = sub(input, mean(input, dim=batch_dims, keepdim=True))
            global_var = mean(mul(global_diff, global_diff), dim=batch_dims, keepdim=False)
            new_rv = add(mul(running_var, (1.0 - float(momentum))), mul(global_var, float(momentum)))
            copy_(running_var, new_rv)
    else:
        stats_shape = (1, C) + (1,) * (ndim - 2)
        mean_t = reshape(running_mean, stats_shape)
        var_t = reshape(running_var, stats_shape)
        diff = sub(input, mean_t)

    eps_t = _scalar_to_npu_tensor(float(eps), mean_t)
    denom = sqrt(add(var_t, eps_t))
    out = div(diff, denom)

    if weight is not None:
        w_shape = (1, C) + (1,) * (ndim - 2)
        w = reshape(weight, w_shape)
        out = mul(out, w)
    if bias is not None:
        b_shape = (1, C) + (1,) * (ndim - 2)
        b = reshape(bias, b_shape)
        out = add(out, b)
    return out


# --- P1 ops ---


def rms_norm(input, normalized_shape, weight=None, eps=1e-6):
    """Compute RMS normalization using aclnnRmsNorm."""
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    norm_shape = tuple(normalized_shape)
    y_shape = input.shape
    y_stride = npu_runtime._contiguous_stride(y_shape)
    y_numel = _numel(y_shape)

    # rstd shape: input shape with normalized dims reduced to 1
    rstd_shape = list(input.shape)
    for i in range(len(norm_shape)):
        rstd_shape[-(i + 1)] = 1
    rstd_shape = tuple(rstd_shape)
    rstd_stride = npu_runtime._contiguous_stride(rstd_shape)
    rstd_numel = _numel(rstd_shape)

    itemsize = _dtype_itemsize(input.dtype)
    y_ptr = npu_runtime._alloc_device(max(y_numel, 1) * itemsize, runtime=runtime)
    rstd_ptr = npu_runtime._alloc_device(max(rstd_numel, 1) * itemsize, runtime=runtime)

    gamma_ptr = _unwrap_storage(weight).data_ptr() if weight is not None else None
    gamma_shape = weight.shape if weight is not None else ()
    gamma_stride = weight.stride if weight is not None else ()

    if gamma_ptr is None:
        # aclnnRmsNorm requires gamma; create ones tensor
        from ...._creation import ones as _ones
        w = _ones(norm_shape, dtype=input.dtype, device=input.device)
        gamma_ptr = _unwrap_storage(w).data_ptr()
        gamma_shape = w.shape
        gamma_stride = w.stride

    aclnn.rms_norm(
        _unwrap_storage(input).data_ptr(), gamma_ptr, eps, y_ptr, rstd_ptr,
        input.shape, input.stride, gamma_shape, gamma_stride,
        y_shape, y_stride, rstd_shape, rstd_stride,
        input.dtype,
        runtime, stream=stream.stream,
    )

    y_storage = npu_typed_storage_from_ptr(y_ptr, max(y_numel, 1), input.dtype, device=input.device)
    rstd_storage = npu_typed_storage_from_ptr(rstd_ptr, max(rstd_numel, 1), input.dtype, device=input.device)
    out = _wrap_tensor(y_storage, y_shape, y_stride)
    out._backward_data = {
        "rstd_ptr": rstd_ptr, "rstd_storage": rstd_storage,
        "rstd_shape": rstd_shape, "rstd_stride": rstd_stride,
        "normalized_shape": tuple(normalized_shape),
    }
    return out


def normalize_op(a, p=2.0, dim=1, eps=1e-12):
    """Normalize along dim: x / max(norm(x, p, dim, keepdim=True), eps)."""
    norm_val = norm_(a, p=p, dim=dim, keepdim=True)
    eps_t = _scalar_to_npu_tensor(eps, norm_val)
    denom = maximum(norm_val, eps_t)
    return div(a, denom)
