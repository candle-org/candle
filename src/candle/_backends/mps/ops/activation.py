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
from .math import add, sub, mul, div, log
from .shape import _ensure_integer_indices


def relu(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "relu")
    return _from_numpy(np.maximum(_to_numpy(a), 0), a.dtype, a.device)

def gelu(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "gelu")
    arr = _to_numpy(a)
    out = 0.5 * arr * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def softplus(a):
    # GPU composite: log(1 + exp(x))
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        exp_a = _dispatch_unary_gpu(a, "exp")
        sum_val = add(exp_a, 1.0)
        return _dispatch_unary_gpu(sum_val, "log")
    arr = _to_numpy(a)
    out = np.log1p(np.exp(arr))
    return _from_numpy(out, a.dtype, a.device)

def silu(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "silu")
    arr = _to_numpy(a)
    out = arr / (1.0 + np.exp(-arr))
    return _from_numpy(out, a.dtype, a.device)

def leaky_relu(a, negative_slope=0.01):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        scalar = float(negative_slope)
        if a.is_contiguous():
            d.dispatch_binary_scalar(f"leaky_relu_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        else:
            d.dispatch_binary_scalar_strided(
                f"leaky_relu_scalar_strided_{sfx}", _metal_buf(a), scalar,
                out_buf, numel, list(a.shape), list(a.stride),
                len(a.shape), scalar_fmt=_scalar_fmt(a.dtype))
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape),
                                  a.dtype, a.device)
    arr = _to_numpy(a)
    out = np.where(arr > 0, arr, negative_slope * arr)
    return _from_numpy(out, a.dtype, a.device)

def elu(a, alpha=1.0):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # elu(x) = x if x > 0, alpha*(exp(x)-1) otherwise
        relu_a = _dispatch_unary_gpu(a, "relu")
        exp_a = _dispatch_unary_gpu(a, "exp")
        elu_part = mul(sub(exp_a, 1.0), alpha)
        neg_part = clamp(elu_part, None, 0.0)
        return add(relu_a, neg_part)
    arr = _to_numpy(a)
    out = np.where(arr > 0, arr, alpha * (np.exp(arr) - 1))
    return _from_numpy(out, a.dtype, a.device)

def mish(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
        sp = softplus(a)
        return mul(a, _dispatch_unary_gpu(sp, "tanh"))
    arr = _to_numpy(a)
    out = arr * np.tanh(np.log1p(np.exp(arr)))
    return _from_numpy(out, a.dtype, a.device)

def prelu(a, weight):
    arr = _to_numpy(a)
    weight_arr = _to_numpy(weight)
    out = np.where(arr > 0, arr, arr * weight_arr)
    return _from_numpy(out, a.dtype, a.device)

def clamp(a, min_val=None, max_val=None):
    if min_val is not None and max_val is not None:
        if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
            d = _get_dispatcher()
            sfx = _kernel_suffix(a.dtype)
            numel = a.numel()
            out_buf = _alloc_output_buf(numel, a.dtype)
            s_min = _scalar_value(min_val, a.dtype)
            s_max = _scalar_value(max_val, a.dtype)
            fmt = _scalar_fmt(a.dtype)
            if a.is_contiguous():
                d.dispatch_clamp(f"clamp_{sfx}", _metal_buf(a),
                                 s_min, s_max, out_buf, numel,
                                 scalar_fmt=fmt)
            else:
                d.dispatch_clamp_strided(
                    f"clamp_strided_{sfx}", _metal_buf(a),
                    s_min, s_max, out_buf, numel,
                    list(a.shape), list(a.stride), len(a.shape),
                    scalar_fmt=fmt)
            from ...._tensor import _compute_strides
            return _from_metal_buffer(out_buf, a.shape,
                                      _compute_strides(a.shape),
                                      a.dtype, a.device)
        arr = _to_numpy(a)
        out = np.clip(arr, min_val, max_val)
        return _from_numpy(out, a.dtype, a.device)
    if min_val is not None:
        return clamp_min(a, min_val)
    if max_val is not None:
        return clamp_max(a, max_val)
    return a

def clamp_min(a, min_val):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        scalar = _scalar_value(min_val, a.dtype)
        if a.is_contiguous():
            d.dispatch_binary_scalar(f"clamp_min_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        else:
            d.dispatch_binary_scalar_strided(
                f"clamp_min_scalar_strided_{sfx}", _metal_buf(a), scalar,
                out_buf, numel, list(a.shape), list(a.stride),
                len(a.shape), scalar_fmt=_scalar_fmt(a.dtype))
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape),
                                  a.dtype, a.device)
    arr = _to_numpy(a)
    out = np.maximum(arr, min_val)
    return _from_numpy(out, a.dtype, a.device)

def clamp_max(a, max_val):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        scalar = _scalar_value(max_val, a.dtype)
        if a.is_contiguous():
            d.dispatch_binary_scalar(f"clamp_max_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        else:
            d.dispatch_binary_scalar_strided(
                f"clamp_max_scalar_strided_{sfx}", _metal_buf(a), scalar,
                out_buf, numel, list(a.shape), list(a.stride),
                len(a.shape), scalar_fmt=_scalar_fmt(a.dtype))
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape),
                                  a.dtype, a.device)
    arr = _to_numpy(a)
    out = np.minimum(arr, max_val)
    return _from_numpy(out, a.dtype, a.device)

def relu6(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
        return clamp(a, 0.0, 6.0)
    arr = _to_numpy(a)
    out = np.minimum(np.maximum(arr, 0.0), 6.0)
    return _from_numpy(out, a.dtype, a.device)

def hardtanh(a, min_val=-1.0, max_val=1.0):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
        return clamp(a, min_val, max_val)
    arr = _to_numpy(a)
    out = np.clip(arr, min_val, max_val)
    return _from_numpy(out, a.dtype, a.device)

def selu(a):
    ALPHA = 1.6732632423543772
    SCALE = 1.0507009873554805
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # scale * where(x > 0, x, alpha * (exp(x) - 1))
        exp_a = _dispatch_unary_gpu(a, "exp")
        elu_part = mul(sub(exp_a, 1.0), ALPHA)
        relu_a = _dispatch_unary_gpu(a, "relu")
        # where(x > 0, x, elu_part) = relu(x) + min(0, elu_part)
        # Simpler: use the where op if available, or compute via masks
        # relu(x) - relu(-elu_part) + elu_part = ... too complex
        # Just use: selu = scale * (relu(x) + min(0, alpha*(exp(x)-1)))
        neg_part = clamp(elu_part, None, 0.0)
        result = add(relu_a, neg_part)
        return mul(result, SCALE)
    arr = _to_numpy(a)
    out = SCALE * np.where(arr > 0, arr, ALPHA * (np.exp(arr) - 1))
    return _from_numpy(out, a.dtype, a.device)

def celu(a, alpha=1.0):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
        relu_a = _dispatch_unary_gpu(a, "relu")
        scaled = div(a, alpha)
        exp_scaled = _dispatch_unary_gpu(scaled, "exp")
        elu_part = mul(sub(exp_scaled, 1.0), alpha)
        neg_part = clamp(elu_part, None, 0.0)
        return add(relu_a, neg_part)
    arr = _to_numpy(a)
    out = np.maximum(arr, 0.0) + np.minimum(0.0, alpha * (np.exp(arr / alpha) - 1))
    return _from_numpy(out, a.dtype, a.device)

def threshold(a, threshold_val, value):
    arr = _to_numpy(a)
    out = np.where(arr > threshold_val, arr, value)
    return _from_numpy(out, a.dtype, a.device)

def hardshrink(a, lambd=0.5):
    arr = _to_numpy(a)
    out = np.where(np.abs(arr) > lambd, arr, 0.0)
    return _from_numpy(out, a.dtype, a.device)

def softshrink(a, lambd=0.5):
    arr = _to_numpy(a)
    out = np.sign(arr) * np.maximum(np.abs(arr) - lambd, 0.0)
    return _from_numpy(out, a.dtype, a.device)

def rrelu(a, lower=1.0 / 8, upper=1.0 / 3, training=False):
    arr = _to_numpy(a)
    if training:
        slope = np.random.uniform(lower, upper, size=arr.shape).astype(arr.dtype)
    else:
        slope = np.full_like(arr, (lower + upper) / 2.0)
    out = np.where(arr >= 0, arr, arr * slope)
    result = _from_numpy(out, a.dtype, a.device)
    result._rrelu_slope = _from_numpy(slope, a.dtype, a.device)
    return result

def hardswish(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # hardswish(x) = x * clamp(x + 3, 0, 6) / 6
        shifted = add(a, 3.0)
        clamped = clamp(shifted, 0.0, 6.0)
        return div(mul(a, clamped), 6.0)
    arr = _to_numpy(a).astype(np.float64)
    out = arr * np.clip(arr + 3.0, 0.0, 6.0) / 6.0
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype)), a.dtype, a.device)

def hardsigmoid(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # hardsigmoid(x) = clamp(x + 3, 0, 6) / 6
        shifted = add(a, 3.0)
        clamped = clamp(shifted, 0.0, 6.0)
        return div(clamped, 6.0)
    arr = _to_numpy(a).astype(np.float64)
    out = np.clip(arr + 3.0, 0.0, 6.0) / 6.0
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype)), a.dtype, a.device)

def softsign(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # softsign(x) = x / (1 + |x|)
        abs_a = _dispatch_unary_gpu(a, "abs")
        denom = add(abs_a, 1.0)
        return div(a, denom)
    arr = _to_numpy(a).astype(np.float64)
    out = arr / (1.0 + np.abs(arr))
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype)), a.dtype, a.device)

def softmax(a, dim):
    # GPU path: float32/float16
    ndim = len(a.shape)
    actual_dim = dim if dim >= 0 else dim + ndim
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype) and ndim >= 1:
        if actual_dim == ndim - 1 and a.is_contiguous():
            # Fast path: softmax over last dim
            d = _get_dispatcher()
            sfx = _kernel_suffix(a.dtype)
            numel = a.numel()
            cols = a.shape[-1]
            rows = numel // cols
            out_buf = _alloc_output_buf(numel, a.dtype)
            d.dispatch_softmax_2d(f"softmax_{sfx}", _metal_buf(a), out_buf,
                                  rows, cols)
            return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
        # Non-last dim: permute target to last → softmax → permute back
        perm = list(range(ndim))
        perm[actual_dim], perm[-1] = perm[-1], perm[actual_dim]
        a_t = a.permute(*perm).contiguous()
        out_t = softmax(a_t, -1)
        return out_t.permute(*perm).contiguous()
    arr = _to_numpy(a)
    exp_arr = np.exp(arr - np.max(arr, axis=dim, keepdims=True))
    result = exp_arr / np.sum(exp_arr, axis=dim, keepdims=True)
    return _from_numpy(result, a.dtype, a.device)

def log_softmax(a, dim):
    # GPU composite: log(softmax(x))
    ndim = len(a.shape)
    actual_dim = dim if dim >= 0 else dim + ndim
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype) and ndim >= 1:
        s = softmax(a, dim)
        return log(s)
    arr = _to_numpy(a)
    max_arr = np.max(arr, axis=dim, keepdims=True)
    exp_arr = np.exp(arr - max_arr)
    log_sum_exp = np.log(np.sum(exp_arr, axis=dim, keepdims=True))
    result = arr - max_arr - log_sum_exp
    return _from_numpy(result, a.dtype, a.device)

def embedding(weight, indices, padding_idx=None, scale_grad_by_freq=False, sparse=False):
    # GPU path: reuse index_select on dim=0
    if (_can_use_gpu(weight) and weight.is_contiguous()
            and weight.dtype in (float32_dtype, float16_dtype)):
        idx_np = _ensure_integer_indices(_to_numpy(indices), "indices").astype(np.int64, copy=False)
        if idx_np.size and (idx_np.min() < 0 or idx_np.max() >= weight.shape[0]):
            raise IndexError("index out of range in self")
        flat_idx = idx_np.reshape(-1)
        flat_idx_i32 = flat_idx.astype(np.int32)
        idx_tensor = _from_numpy(flat_idx_i32, int32_dtype, weight.device)
        d = _get_dispatcher()
        sfx = _kernel_suffix(weight.dtype)
        vocab, dim = weight.shape[0], weight.shape[1]
        out_numel = len(flat_idx) * dim
        out_buf = _alloc_output_buf(out_numel, weight.dtype)
        d.dispatch_index_gather(f"index_select_{sfx}", _metal_buf(weight),
                                _metal_buf(idx_tensor), out_buf,
                                1, len(flat_idx), dim,
                                vocab, out_numel)
        out_shape = tuple(indices.shape) + (dim,) if hasattr(indices, 'shape') else (len(flat_idx), dim)
        out_shape = tuple(idx_np.shape) + (weight.shape[1],)
        s = 1
        out_stride = ()
        for d_ in reversed(out_shape):
            out_stride = (s,) + out_stride
            s *= d_
        return _from_metal_buffer(out_buf, out_shape, out_stride, weight.dtype, weight.device)
    weight_arr = _to_numpy(weight)
    idx = _ensure_integer_indices(_to_numpy(indices), "indices").astype(np.int64, copy=False)
    if idx.size and (idx.min() < 0 or idx.max() >= weight_arr.shape[0]):
        raise IndexError("index out of range in self")
    out = weight_arr[idx]
    return _from_numpy(np.ascontiguousarray(out), weight.dtype, weight.device)

def dropout(a, p=0.5, training=True):
    if not training or p == 0:
        return a
    if p == 1.0:
        return _from_numpy(np.zeros(_to_numpy(a).shape, dtype=_to_numpy(a).dtype), a.dtype, a.device)
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from ....mps import _get_default_generator
        gen = _get_default_generator()
        numel = a.numel()
        increment = (numel + 3) // 4
        seed, offset = gen.philox_engine_inputs(increment)
        seed_lo, seed_hi = seed & 0xffffffff, (seed >> 32) & 0xffffffff
        sfx = _kernel_suffix(a.dtype)
        scale = 1.0 / (1.0 - p)
        out_buf = _alloc_output_buf(numel, a.dtype)
        _get_dispatcher().dispatch_philox_dropout(
            f"philox_dropout_{sfx}", _metal_buf(a), out_buf,
            float(p), float(scale), seed_lo, seed_hi, offset, numel)
        stride = tuple(a.stride())
        return _from_metal_buffer(out_buf, tuple(a.shape), stride, a.dtype, a.device)
    from ...._random import _get_cpu_rng
    rng = _get_cpu_rng()
    arr = _to_numpy(a)
    mask = (rng.random(arr.shape) >= p).astype(arr.dtype)
    return _from_numpy(arr * mask / (1.0 - p), a.dtype, a.device)

