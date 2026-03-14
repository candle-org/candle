import math
import ctypes
import struct
import numpy as np

from ._helpers import (
    _can_use_gpu, _metal_buf, _kernel_suffix, _scalar_fmt, _itemsize,
    _alloc_output_buf, _metal_buf_to_bytes, _from_metal_buffer,
    _get_dispatcher, _dispatch_unary_gpu, _dispatch_unary_predicate_gpu,
    _scalar_value, _dispatch_binary_gpu,
    _to_numpy, _from_numpy,
    _compute_reduce_dims, _reduce_shape, _gpu_reduce_single_dim,
    _normalize_tensor_sequence_args,
    _can_use_blas, _blas_gemm,
    float32_dtype, float16_dtype, float64_dtype,
    int32_dtype, int64_dtype, bool_dtype,
    to_numpy_dtype, Tensor,
    mps_typed_storage_from_numpy, _MPSUntypedStorage, TypedStorage,
    _accel,
)

def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    ndim = len(input.shape)

    # GPU path: f32/f16, contiguous, ndim >= 2
    if (_can_use_gpu(input) and input.is_contiguous()
            and input.dtype in (float32_dtype, float16_dtype) and ndim >= 2):
        N = input.shape[0]
        C = input.shape[1]
        spatial_size = 1
        for i in range(2, ndim):
            spatial_size *= input.shape[i]
        total = input.numel()

        d = _get_dispatcher()
        sfx = _kernel_suffix(input.dtype)
        from ..runtime import get_runtime, buffer_contents
        rt = get_runtime()

        if training:
            # Compute stats on GPU
            mean_buf = rt.create_buffer(C * 4)  # float32 stats
            var_buf = rt.create_buffer(C * 4)
            d.dispatch_batch_norm_stats(
                f"batch_norm_stats_{sfx}", _metal_buf(input),
                mean_buf, var_buf, N, C, spatial_size)

            # Read stats back to update running_mean/running_var on CPU
            mean_ptr = buffer_contents(mean_buf)
            var_ptr = buffer_contents(var_buf)
            mean_np = np.frombuffer(
                (ctypes.c_uint8 * (C * 4)).from_address(mean_ptr),
                dtype=np.float32, count=C).copy()
            var_np = np.frombuffer(
                (ctypes.c_uint8 * (C * 4)).from_address(var_ptr),
                dtype=np.float32, count=C).copy()

            if running_mean is not None:
                rm_np = running_mean._storage.data.ravel().astype(np.float32)
                new_rm = ((1 - momentum) * rm_np + momentum * mean_np).astype(
                    running_mean._storage.data.dtype)
                running_mean._storage.data[:] = new_rm
            if running_var is not None:
                rv_np = running_var._storage.data.ravel().astype(np.float32)
                new_rv = ((1 - momentum) * rv_np + momentum * var_np).astype(
                    running_var._storage.data.dtype)
                running_var._storage.data[:] = new_rv
        else:
            # Eval mode: use running stats
            rm_np = _to_numpy(running_mean).astype(np.float32)
            rv_np = _to_numpy(running_var).astype(np.float32)
            mean_buf = rt.create_buffer(C * 4)
            var_buf = rt.create_buffer(C * 4)
            mean_ptr = buffer_contents(mean_buf)
            var_ptr = buffer_contents(var_buf)
            ctypes.memmove(mean_ptr, rm_np.ctypes.data, C * 4)
            ctypes.memmove(var_ptr, rv_np.ctypes.data, C * 4)

        # Apply normalization
        out_buf = _alloc_output_buf(total, input.dtype)

        has_weight = 0
        if weight is not None and _can_use_gpu(weight):
            weight_buf = _metal_buf(weight)
            has_weight = 1
        else:
            weight_buf = rt.create_buffer(4)

        has_bias = 0
        if bias is not None and _can_use_gpu(bias):
            bias_buf = _metal_buf(bias)
            has_bias = 1
        else:
            bias_buf = rt.create_buffer(4)

        d.dispatch_batch_norm_apply(
            f"batch_norm_apply_{sfx}", _metal_buf(input), mean_buf,
            var_buf, weight_buf, bias_buf, out_buf,
            C, spatial_size, float(eps), has_weight, has_bias, total)

        from ...._tensor import _compute_strides
        out_shape = tuple(input.shape)
        out_stride = _compute_strides(out_shape)
        return _from_metal_buffer(out_buf, out_shape, out_stride, input.dtype, input.device)

    # Numpy fallback
    arr = _to_numpy(input)

    if training:
        axes = (0,) + tuple(range(2, ndim))
        mean = arr.mean(axis=axes)
        var = arr.var(axis=axes)
        if running_mean is not None:
            rm = running_mean._storage.data.ravel()
            new_rm = (1 - momentum) * rm + momentum * mean
            running_mean._storage.data[:] = new_rm.astype(rm.dtype)
        if running_var is not None:
            rv = running_var._storage.data.ravel()
            new_rv = (1 - momentum) * rv + momentum * var
            running_var._storage.data[:] = new_rv.astype(rv.dtype)
    else:
        mean = _to_numpy(running_mean)
        var = _to_numpy(running_var)

    shape = [1, -1] + [1] * (ndim - 2)
    normalized = (arr - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps)

    if weight is not None:
        normalized = normalized * _to_numpy(weight).reshape(shape)
    if bias is not None:
        normalized = normalized + _to_numpy(bias).reshape(shape)

    return _from_numpy(normalized, input.dtype, input.device)

def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    # GPU path: float32/float16, contiguous, ndim >= 2
    if (_can_use_gpu(input) and input.is_contiguous()
            and input.dtype in (float32_dtype, float16_dtype)
            and len(input.shape) >= 2
            and input.shape[1] % num_groups == 0):
        N, C = input.shape[0], input.shape[1]
        spatial_size = 1
        for d in input.shape[2:]:
            spatial_size *= d
        total = N * C * spatial_size
        sfx = _kernel_suffix(input.dtype)
        d = _get_dispatcher()
        from ..runtime import get_runtime
        rt = get_runtime()
        out_buf = _alloc_output_buf(total, input.dtype)
        has_w = 1 if weight is not None else 0
        has_b = 1 if bias is not None else 0
        w_buf = _metal_buf(weight) if weight is not None else rt.create_buffer(4)
        b_buf = _metal_buf(bias) if bias is not None else rt.create_buffer(4)
        d.dispatch_group_norm(
            f"group_norm_{sfx}", _metal_buf(input), w_buf, b_buf, out_buf,
            N, C, spatial_size, num_groups, float(eps), has_w, has_b, total)
        return _from_metal_buffer(out_buf, tuple(input.shape),
                                  tuple(input.stride()), input.dtype, input.device)

    arr = _to_numpy(input)
    N, C = arr.shape[:2]
    spatial = arr.shape[2:]

    grouped = arr.reshape(N, num_groups, C // num_groups, *spatial)
    axes = tuple(range(2, len(grouped.shape)))
    mean = grouped.mean(axis=axes, keepdims=True)
    var = grouped.var(axis=axes, keepdims=True)
    normalized = (grouped - mean) / np.sqrt(var + eps)
    normalized = normalized.reshape(arr.shape)

    if weight is not None:
        shape = [1, C] + [1] * len(spatial)
        normalized = normalized * _to_numpy(weight).reshape(shape)
    if bias is not None:
        shape = [1, C] + [1] * len(spatial)
        normalized = normalized + _to_numpy(bias).reshape(shape)

    return _from_numpy(normalized, input.dtype, input.device)

def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    norm_shape = tuple(normalized_shape)
    if len(norm_shape) == 0:
        raise ValueError("normalized_shape must be non-empty")
    if tuple(input.shape[-len(norm_shape):]) != norm_shape:
        raise ValueError("normalized_shape mismatch")

    # GPU path: f32/f16, contiguous
    if (_can_use_gpu(input) and input.is_contiguous()
            and input.dtype in (float32_dtype, float16_dtype)):
        inner_size = 1
        for s in norm_shape:
            inner_size *= s
        outer_size = input.numel() // inner_size

        d = _get_dispatcher()
        sfx = _kernel_suffix(input.dtype)
        out_buf = _alloc_output_buf(input.numel(), input.dtype)

        from ..runtime import get_runtime
        has_weight = 0
        if weight is not None and _can_use_gpu(weight):
            weight_buf = _metal_buf(weight)
            has_weight = 1
        else:
            weight_buf = get_runtime().create_buffer(4)

        has_bias = 0
        if bias is not None and _can_use_gpu(bias):
            bias_buf = _metal_buf(bias)
            has_bias = 1
        else:
            bias_buf = get_runtime().create_buffer(4)

        d.dispatch_layer_norm(
            f"layer_norm_{sfx}", _metal_buf(input), weight_buf, bias_buf,
            out_buf, outer_size, inner_size, float(eps), has_weight, has_bias)

        from ...._tensor import _compute_strides
        out_shape = tuple(input.shape)
        out_stride = _compute_strides(out_shape)
        return _from_metal_buffer(out_buf, out_shape, out_stride, input.dtype, input.device)

    # Numpy fallback
    arr = _to_numpy(input)
    axis = tuple(range(arr.ndim - len(norm_shape), arr.ndim))
    mean = arr.mean(axis=axis, keepdims=True)
    var = arr.var(axis=axis, keepdims=True)
    out = (arr - mean) / np.sqrt(var + eps)

    if weight is not None:
        out = out * _to_numpy(weight).reshape((1,) * (arr.ndim - len(norm_shape)) + norm_shape)
    if bias is not None:
        out = out + _to_numpy(bias).reshape((1,) * (arr.ndim - len(norm_shape)) + norm_shape)

    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)

def instance_norm(input, weight=None, bias=None, running_mean=None, running_var=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5, cudnn_enabled=False):
    arr = _to_numpy(input)
    ndim = len(arr.shape)
    N = arr.shape[0]
    C = arr.shape[1] if ndim >= 2 else 1
    spatial_axes = tuple(range(2, ndim))

    if use_input_stats:
        mean = arr.mean(axis=spatial_axes, keepdims=True)
        var = arr.var(axis=spatial_axes, keepdims=True)
        if running_mean is not None:
            rm = _to_numpy(running_mean)
            batch_mean = mean.reshape(N, C).mean(axis=0)
            rm[:] = (1 - momentum) * rm + momentum * batch_mean
        if running_var is not None:
            rv = _to_numpy(running_var)
            batch_var = var.reshape(N, C).mean(axis=0)
            rv[:] = (1 - momentum) * rv + momentum * batch_var
    else:
        rm = _to_numpy(running_mean)
        rv = _to_numpy(running_var)
        shape = [1, C] + [1] * (ndim - 2)
        mean = rm.reshape(shape)
        var = rv.reshape(shape)

    normalized = (arr - mean) / np.sqrt(var + eps)

    if weight is not None:
        shape = [1, C] + [1] * (ndim - 2)
        normalized = normalized * _to_numpy(weight).reshape(shape)
    if bias is not None:
        shape = [1, C] + [1] * (ndim - 2)
        normalized = normalized + _to_numpy(bias).reshape(shape)

    return _from_numpy(normalized, input.dtype, input.device)

def rms_norm(input, normalized_shape, weight=None, eps=1e-6):
    norm_shape = tuple(normalized_shape)
    inner_size = 1
    for s in norm_shape:
        inner_size *= s

    # GPU path: f32/f16, contiguous
    if (_can_use_gpu(input) and input.is_contiguous()
            and input.dtype in (float32_dtype, float16_dtype)):
        outer_size = input.numel() // inner_size

        d = _get_dispatcher()
        sfx = _kernel_suffix(input.dtype)
        out_buf = _alloc_output_buf(input.numel(), input.dtype)

        from ..runtime import get_runtime
        has_weight = 0
        if weight is not None and _can_use_gpu(weight):
            weight_buf = _metal_buf(weight)
            has_weight = 1
        else:
            weight_buf = get_runtime().create_buffer(4)

        d.dispatch_rms_norm(
            f"rms_norm_{sfx}", _metal_buf(input), weight_buf,
            out_buf, outer_size, inner_size, float(eps), has_weight)

        from ...._tensor import _compute_strides
        out_shape = tuple(input.shape)
        out_stride = _compute_strides(out_shape)
        return _from_metal_buffer(out_buf, out_shape, out_stride, input.dtype, input.device)

    # Numpy fallback
    arr = _to_numpy(input)
    axis = tuple(range(arr.ndim - len(norm_shape), arr.ndim))
    variance = np.mean(arr ** 2, axis=axis, keepdims=True)
    out = arr / np.sqrt(variance + eps)
    if weight is not None:
        out = out * _to_numpy(weight).reshape((1,) * (arr.ndim - len(norm_shape)) + norm_shape)
    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)

def normalize(a, p=2.0, dim=1, eps=1e-12):
    arr = _to_numpy(a).astype(np.float64)
    norm = np.linalg.norm(arr, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)
    out = arr / norm
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype)), a.dtype, a.device)

