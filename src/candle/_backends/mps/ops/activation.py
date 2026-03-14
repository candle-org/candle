import numpy as np

from ._helpers import (
    _can_use_gpu, _empty_like, _unsupported_dtype,
    _metal_buf, _kernel_suffix, _scalar_fmt, _itemsize,
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
from .comparison import gt
from .elementwise import where
from .shape import _ensure_integer_indices


def relu(a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "relu")
    _unsupported_dtype("relu", a)

def gelu(a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "gelu")
    _unsupported_dtype("gelu", a)

def softplus(a):
    # GPU composite: log(1 + exp(x))
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        exp_a = _dispatch_unary_gpu(a, "exp")
        sum_val = add(exp_a, 1.0)
        return _dispatch_unary_gpu(sum_val, "log")
    _unsupported_dtype("softplus", a)

def silu(a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "silu")
    _unsupported_dtype("silu", a)

def leaky_relu(a, negative_slope=0.01):
    if a.numel() == 0:
        return _empty_like(a)
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
    _unsupported_dtype("leaky_relu", a)

def elu(a, alpha=1.0):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # elu(x) = x if x > 0, alpha*(exp(x)-1) otherwise
        relu_a = _dispatch_unary_gpu(a, "relu")
        exp_a = _dispatch_unary_gpu(a, "exp")
        elu_part = mul(sub(exp_a, 1.0), alpha)
        neg_part = clamp(elu_part, None, 0.0)
        return add(relu_a, neg_part)
    _unsupported_dtype("elu", a)

def mish(a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
        sp = softplus(a)
        return mul(a, _dispatch_unary_gpu(sp, "tanh"))
    _unsupported_dtype("mish", a)

def prelu(a, weight):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        a_c = a.contiguous() if not a.is_contiguous() else a
        mask = gt(a_c, 0)
        w_np = _to_numpy(weight)
        # Broadcast weight to match a's shape
        w_expanded = _from_numpy(
            np.broadcast_to(w_np, a_c.shape).copy(), a.dtype, a.device)
        neg = mul(a_c, w_expanded)
        return where(mask, a_c, neg)
    _unsupported_dtype("prelu", a)

def clamp(a, min_val=None, max_val=None):
    if a.numel() == 0:
        return _empty_like(a)
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
        _unsupported_dtype("clamp", a)
    if min_val is not None:
        return clamp_min(a, min_val)
    if max_val is not None:
        return clamp_max(a, max_val)
    return a

def clamp_min(a, min_val):
    if a.numel() == 0:
        return _empty_like(a)
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
    _unsupported_dtype("clamp_min", a)

def clamp_max(a, max_val):
    if a.numel() == 0:
        return _empty_like(a)
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
    _unsupported_dtype("clamp_max", a)

def relu6(a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
        return clamp(a, 0.0, 6.0)
    _unsupported_dtype("relu6", a)

def hardtanh(a, min_val=-1.0, max_val=1.0):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
        return clamp(a, min_val, max_val)
    _unsupported_dtype("hardtanh", a)

def selu(a):
    ALPHA = 1.6732632423543772
    SCALE = 1.0507009873554805
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # scale * where(x > 0, x, alpha * (exp(x) - 1))
        exp_a = _dispatch_unary_gpu(a, "exp")
        elu_part = mul(sub(exp_a, 1.0), ALPHA)
        relu_a = _dispatch_unary_gpu(a, "relu")
        neg_part = clamp(elu_part, None, 0.0)
        result = add(relu_a, neg_part)
        return mul(result, SCALE)
    _unsupported_dtype("selu", a)

def celu(a, alpha=1.0):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
        relu_a = _dispatch_unary_gpu(a, "relu")
        scaled = div(a, alpha)
        exp_scaled = _dispatch_unary_gpu(scaled, "exp")
        elu_part = mul(sub(exp_scaled, 1.0), alpha)
        neg_part = clamp(elu_part, None, 0.0)
        return add(relu_a, neg_part)
    _unsupported_dtype("celu", a)

def threshold(a, threshold_val, value):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        a_c = a.contiguous() if not a.is_contiguous() else a
        mask = gt(a_c, threshold_val)
        fill = mul(add(mul(a_c, 0.0), 1.0), value)
        return where(mask, a_c, fill)
    _unsupported_dtype("threshold", a)

def hardshrink(a, lambd=0.5):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        a_c = a.contiguous() if not a.is_contiguous() else a
        abs_a = _dispatch_unary_gpu(a_c, "abs")
        mask = gt(abs_a, lambd)
        return where(mask, a_c, mul(a_c, 0.0))
    _unsupported_dtype("hardshrink", a)

def softshrink(a, lambd=0.5):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        a_c = a.contiguous() if not a.is_contiguous() else a
        s = _dispatch_unary_gpu(a_c, "sign")
        abs_a = _dispatch_unary_gpu(a_c, "abs")
        shifted = sub(abs_a, lambd)
        clamped = clamp(shifted, 0.0, None)
        return mul(s, clamped)
    _unsupported_dtype("softshrink", a)

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
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # hardswish(x) = x * clamp(x + 3, 0, 6) / 6
        shifted = add(a, 3.0)
        clamped = clamp(shifted, 0.0, 6.0)
        return div(mul(a, clamped), 6.0)
    _unsupported_dtype("hardswish", a)

def hardsigmoid(a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # hardsigmoid(x) = clamp(x + 3, 0, 6) / 6
        shifted = add(a, 3.0)
        clamped = clamp(shifted, 0.0, 6.0)
        return div(clamped, 6.0)
    _unsupported_dtype("hardsigmoid", a)

def softsign(a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        # softsign(x) = x / (1 + |x|)
        abs_a = _dispatch_unary_gpu(a, "abs")
        denom = add(abs_a, 1.0)
        return div(a, denom)
    _unsupported_dtype("softsign", a)

def softmax(a, dim):
    # GPU path: float32/float16
    ndim = len(a.shape)
    if a.numel() == 0:
        return _empty_like(a)
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
    _unsupported_dtype("softmax", a)

def log_softmax(a, dim):
    # GPU composite: log(softmax(x))
    if a.numel() == 0:
        return _empty_like(a)
    ndim = len(a.shape)
    actual_dim = dim if dim >= 0 else dim + ndim
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype) and ndim >= 1:
        s = softmax(a, dim)
        return log(s)
    _unsupported_dtype("log_softmax", a)

def embedding(weight, indices, padding_idx=None, scale_grad_by_freq=False, sparse=False):
    # GPU path: reuse index_select on dim=0
    if weight.numel() == 0:
        return _empty_like(weight)
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
        out_shape = tuple(idx_np.shape) + (weight.shape[1],)
        s = 1
        out_stride = ()
        for d_ in reversed(out_shape):
            out_stride = (s,) + out_stride
            s *= d_
        return _from_metal_buffer(out_buf, out_shape, out_stride, weight.dtype, weight.device)
    _unsupported_dtype("embedding", weight)

def dropout(a, p=0.5, training=True):
    if not training or p == 0:
        return a
    if p == 1.0:
        return mul(a, 0.0)
    if a.numel() == 0:
        return _empty_like(a)
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
        result = _from_metal_buffer(out_buf, tuple(a.shape), stride, a.dtype, a.device)
        result._backward_data = {
            'seed_lo': seed_lo,
            'seed_hi': seed_hi,
            'offset': offset,
            'p': float(p),
            'scale': float(scale),
        }
        return result
    _unsupported_dtype("dropout", a)

