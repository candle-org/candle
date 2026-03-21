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

def add_(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_inplace_binary(f"add_inplace_{sfx}", _metal_buf(a),
                                      _metal_buf(b), numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_inplace_binary_scalar(f"add_inplace_scalar_{sfx}",
                                              _metal_buf(a), scalar, numel,
                                              scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr += _to_numpy(b) if isinstance(b, Tensor) else b
    return a

def mul_(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_inplace_binary(f"mul_inplace_{sfx}", _metal_buf(a),
                                      _metal_buf(b), numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_inplace_binary_scalar(f"mul_inplace_scalar_{sfx}",
                                              _metal_buf(a), scalar, numel,
                                              scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr *= _to_numpy(b) if isinstance(b, Tensor) else b
    return a

def sub_(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_inplace_binary(f"sub_inplace_{sfx}", _metal_buf(a),
                                      _metal_buf(b), numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_inplace_binary_scalar(f"sub_inplace_scalar_{sfx}",
                                              _metal_buf(a), scalar, numel,
                                              scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr -= _to_numpy(b) if isinstance(b, Tensor) else b
    return a

def div_(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_inplace_binary(f"div_inplace_{sfx}", _metal_buf(a),
                                      _metal_buf(b), numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_inplace_binary_scalar(f"div_inplace_scalar_{sfx}",
                                              _metal_buf(a), scalar, numel,
                                              scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    arr /= b_np
    return a

def relu_(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        d.dispatch_inplace_unary(f"relu_inplace_{sfx}", _metal_buf(a), a.numel())
        return a
    arr = _to_numpy(a)
    np.maximum(arr, 0, out=arr)
    return a

def zero_(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        d.dispatch_fill(f"fill_{sfx}", _metal_buf(a), 0.0, a.numel(),
                        scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr.fill(0)
    return a

def uniform_(a, low=0.0, high=1.0, generator=None):
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from ....mps import _get_default_generator
        gen = generator if (generator is not None and hasattr(generator, 'device') and generator.device.type == 'mps') else _get_default_generator()
        numel = a.numel()
        increment = (numel + 3) // 4
        seed, offset = gen.philox_engine_inputs(increment)
        seed_lo, seed_hi = seed & 0xffffffff, (seed >> 32) & 0xffffffff
        sfx = _kernel_suffix(a.dtype)
        fmt = _scalar_fmt(a.dtype)
        _get_dispatcher().dispatch_philox_fill(
            f"philox_uniform_{sfx}", _metal_buf(a),
            seed_lo, seed_hi, offset, low, high, numel, param_fmt=fmt)
        return a
    from ...._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[:] = rng.uniform(low, high, arr.shape).astype(arr.dtype)
    return a

def normal_(a, mean=0.0, std=1.0, generator=None):
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from ....mps import _get_default_generator
        gen = generator if (generator is not None and hasattr(generator, 'device') and generator.device.type == 'mps') else _get_default_generator()
        numel = a.numel()
        increment = (numel + 3) // 4
        seed, offset = gen.philox_engine_inputs(increment)
        seed_lo, seed_hi = seed & 0xffffffff, (seed >> 32) & 0xffffffff
        sfx = _kernel_suffix(a.dtype)
        fmt = _scalar_fmt(a.dtype)
        _get_dispatcher().dispatch_philox_fill(
            f"philox_normal_{sfx}", _metal_buf(a),
            seed_lo, seed_hi, offset, mean, std, numel, param_fmt=fmt)
        return a
    from ...._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[:] = rng.normal(mean, std, arr.shape).astype(arr.dtype)
    return a

def bernoulli_(a, p=0.5, generator=None):
    is_scalar_p = not hasattr(p, '_numpy_view') and not hasattr(p, 'numpy')
    if is_scalar_p and _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from ....mps import _get_default_generator
        gen = generator if (generator is not None and hasattr(generator, 'device') and generator.device.type == 'mps') else _get_default_generator()
        numel = a.numel()
        increment = (numel + 3) // 4
        seed, offset = gen.philox_engine_inputs(increment)
        seed_lo, seed_hi = seed & 0xffffffff, (seed >> 32) & 0xffffffff
        sfx = _kernel_suffix(a.dtype)
        _get_dispatcher().dispatch_philox_bernoulli(
            f"philox_bernoulli_{sfx}", _metal_buf(a), float(p),
            seed_lo, seed_hi, offset, numel)
        return a
    # GPU composite for tensor p: uniform_(a) → lt(a, p) → cast to float → copy_
    if (not is_scalar_p and _can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)
            and isinstance(p, Tensor) and _can_use_gpu(p)):
        from .comparison import lt
        from .elementwise import where
        uniform_(a, 0.0, 1.0, generator=generator)
        mask = lt(a, p)
        # where needs x as Tensor; fill a with 1.0 to use as the "true" branch
        fill_(a, 1.0)
        result = where(mask, a, 0.0)
        copy_(a, result)
        return a
    from ...._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    if hasattr(p, '_numpy_view'):
        probs = p._numpy_view()
    elif hasattr(p, 'numpy'):
        probs = p.numpy()
    else:
        probs = float(p)
    arr[...] = (rng.uniform(0, 1, arr.shape) < probs).astype(arr.dtype)
    return a

def exponential_(a, lambd=1.0, generator=None):
    # GPU composite: uniform_(a) → -log(a)/λ → copy back
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from .math import neg, log, div
        uniform_(a, 0.0, 1.0, generator=generator)
        result = div(neg(log(a)), lambd)
        copy_(a, result)
        return a
    from ...._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = rng.exponential(1.0 / lambd, arr.shape).astype(arr.dtype)
    return a

def log_normal_(a, mean=1.0, std=2.0, generator=None):
    # GPU composite: normal_(a,0,1) → exp(a*σ + μ) → copy back
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from .math import mul, add, exp
        normal_(a, 0.0, 1.0, generator=generator)
        result = exp(add(mul(a, std), mean))
        copy_(a, result)
        return a
    from ...._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = rng.lognormal(mean, std, arr.shape).astype(arr.dtype)
    return a

def cauchy_(a, median=0.0, sigma=1.0, generator=None):
    # GPU composite: uniform_(a) → median + σ*tan(π*(a-0.5)) → copy back
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        import math as _math
        from .math import mul, add, sub, tan
        uniform_(a, 0.0, 1.0, generator=generator)
        result = add(median, mul(sigma, tan(mul(_math.pi, sub(a, 0.5)))))
        copy_(a, result)
        return a
    from ...._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = (median + sigma * np.tan(np.pi * (rng.uniform(0, 1, arr.shape) - 0.5))).astype(arr.dtype)
    return a

def geometric_(a, p, generator=None):
    # GPU composite: uniform_(a) → ceil(log(1-a)/log(1-p)) → copy back
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        import math as _math
        from .math import sub, log, div, ceil
        uniform_(a, 0.0, 1.0, generator=generator)
        result = ceil(div(log(sub(1.0, a)), _math.log(1.0 - p)))
        copy_(a, result)
        return a
    from ...._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = rng.geometric(p, arr.shape).astype(arr.dtype)
    return a

def fill_(a, value):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        d.dispatch_fill(f"fill_{sfx}", _metal_buf(a), float(value), a.numel(),
                        scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr.fill(value)
    return a

def clamp_(a, min_val=None, max_val=None):
    if _can_use_gpu(a):
        from .activation import clamp
        clamped = clamp(a, min_val, max_val)
        # Copy result back into a's buffer using identity kernel
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        d.dispatch_unary(f"identity_{sfx}", _metal_buf(clamped),
                         _metal_buf(a), a.numel())
        return a
    arr = _to_numpy(a)
    np.clip(arr, min_val, max_val, out=arr)
    return a

def copy_(a, src):
    if _can_use_gpu(a) and _can_use_gpu(src):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = min(a.numel(), src.numel())
        # Use identity kernel (unary copy) — dispatch_copy has a bug
        d.dispatch_unary(f"identity_{sfx}", _metal_buf(src), _metal_buf(a), numel)
        return a
    arr = _to_numpy(a)
    src_arr = _to_numpy(src)
    np.copyto(arr, src_arr)
    return a

def erfinv_(a):
    # GPU composite: Beasley-Springer-Moro rational approximation on-device
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from .math import mul, add, sub, div, sqrt, log, neg, abs as abs_
        from .comparison import le, ge, gt, lt
        from .elementwise import where
        _sqrt2 = 1.4142135623730951
        # Map erfinv domain [-1,1] to probit domain [0,1]: p = (a+1)/2
        p = div(add(a, 1.0), 2.0)
        q = sub(p, 0.5)
        abs_q = abs_(q)
        central_mask = le(abs_q, 0.425)
        # --- Central region: rational approx in q ---
        r2 = mul(q, q)
        # num = ((a3*r2 + a2)*r2 + a1)*r2 + a0
        t = add(mul(r2, -25.44106049637), 41.39119773534)
        t = add(mul(t, r2), -18.61500062529)
        num = add(mul(t, r2), 2.50662823884)
        # den = (((b3*r2 + b2)*r2 + b1)*r2 + b0)*r2 + 1
        t = add(mul(r2, 3.13082909833), -21.06224101826)
        t = add(mul(t, r2), 23.08336743743)
        t = add(mul(t, r2), -8.47351093090)
        den = add(mul(t, r2), 1.0)
        central_val = div(mul(q, num), den)
        # --- Tail region ---
        one_minus_p = add(neg(p), 1.0)  # 1 - p
        p_lt_half = le(p, 0.5)
        pp = where(p_lt_half, p, one_minus_p)
        r = sqrt(mul(log(pp), -2.0))
        # t_num = (c2*r + c1)*r + c0
        t = add(mul(r, 0.010328), 0.802853)
        t_num = add(mul(t, r), 2.515517)
        # t_den = ((d2*r + d1)*r + d0)*r + 1
        t = add(mul(r, 0.001308), 0.189269)
        t = add(mul(t, r), 1.432788)
        t_den = add(mul(t, r), 1.0)
        tail_abs = sub(r, div(t_num, t_den))
        tail_val = where(p_lt_half, neg(tail_abs), tail_abs)
        # Combine central + tail, scale by 1/sqrt(2)
        result = div(where(central_mask, central_val, tail_val), _sqrt2)
        # Boundary: erfinv(-1)=-inf, erfinv(1)=inf (scalar must be y in where)
        result = where(gt(a, -1.0), result, float('-inf'))
        result = where(lt(a, 1.0), result, float('inf'))
        copy_(a, result)
        return a
    arr = _to_numpy(a)
    arr[:] = _ndtr_inv((arr + 1.0) / 2.0) / np.sqrt(2.0)
    return a

def _ndtr_inv(p):
    """Inverse normal CDF (probit function) using rational approximation.
    Used to compute erfinv: erfinv(x) = ndtr_inv((x+1)/2) / sqrt(2)."""
    p = np.asarray(p, dtype=np.float64)
    result = np.zeros_like(p)

    # Central region: |p - 0.5| <= 0.425
    q = p - 0.5
    mask_central = np.abs(q) <= 0.425
    if np.any(mask_central):
        r = q[mask_central]
        r2 = r * r
        # Rational approximation coefficients (Beasley-Springer-Moro)
        a = np.array([
            2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637
        ])
        b = np.array([
            -8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833
        ])
        num = ((a[3] * r2 + a[2]) * r2 + a[1]) * r2 + a[0]
        den = (((b[3] * r2 + b[2]) * r2 + b[1]) * r2 + b[0]) * r2 + 1.0
        result[mask_central] = r * num / den

    # Tail regions
    mask_tail = ~mask_central & (p > 0) & (p < 1)
    if np.any(mask_tail):
        pp = np.where(p[mask_tail] < 0.5, p[mask_tail], 1.0 - p[mask_tail])
        r = np.sqrt(-2.0 * np.log(pp))
        c = np.array([
            2.515517, 0.802853, 0.010328
        ])
        d = np.array([
            1.432788, 0.189269, 0.001308
        ])
        num = (c[2] * r + c[1]) * r + c[0]
        den = ((d[2] * r + d[1]) * r + d[0]) * r + 1.0
        val = r - num / den
        val = np.where(p[mask_tail] < 0.5, -val, val)
        result[mask_tail] = val

    # Boundary cases
    result[p <= 0] = -np.inf
    result[p >= 1] = np.inf
    return result

def randint_(a, low, high=None, generator=None):
    """In-place randint — fills tensor a with random integers from [low, high)."""
    if high is None:
        low, high = 0, low
    from ...._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = rng.randint(int(low), int(high), size=arr.shape)
    return a

def random_(a, from_=0, to=None, generator=None):
    """In-place random — fills tensor with random values from [from_, to)."""
    from ...._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    if to is None:
        if np.issubdtype(arr.dtype, np.floating):
            to = 2**24 if arr.dtype == np.float32 else 2**53
        else:
            to = int(np.iinfo(arr.dtype).max) + 1
    arr[...] = rng.randint(int(from_), int(to), size=arr.shape).astype(arr.dtype)
    return a


# ---------------------------------------------------------------------------
# Group 5: Shape ops
# ---------------------------------------------------------------------------

try:
    from candle._cython._mps_ops import (  # pylint: disable=import-error,no-name-in-module
        add_, mul_, sub_, div_, relu_, zero_, fill_, clamp_, copy_,
        uniform_, normal_, bernoulli_, exponential_, log_normal_,
        cauchy_, geometric_, erfinv_, randint_, random_,
    )
except ImportError:
    pass