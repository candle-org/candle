"""Special mathematical functions and FFT for NPU."""
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
from .comparison import eq, gt
from .elementwise import clamp, where
from .linalg import matmul
from .math import add, cos, div, erfc, exp, log, log1p, mul, neg, sin, sqrt, sub
from .reduce import maximum
from .shape import contiguous, index_select


def special_digamma(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    s = _unwrap_storage(a)
    out_ptr = npu_runtime._alloc_device(s.nbytes, runtime=runtime)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    aclnn.digamma(s.data_ptr(), out_ptr, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return _wrap_tensor(out_storage, a.shape, a.stride)


def special_erfinv(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    s = _unwrap_storage(a)
    out_ptr = npu_runtime._alloc_device(s.nbytes, runtime=runtime)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    aclnn.erfinv(s.data_ptr(), out_ptr, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return _wrap_tensor(out_storage, a.shape, a.stride)


def special_gammaln(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    s = _unwrap_storage(a)
    out_ptr = npu_runtime._alloc_device(s.nbytes, runtime=runtime)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    aclnn.lgamma(s.data_ptr(), out_ptr, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return _wrap_tensor(out_storage, a.shape, a.stride)


def special_sinc(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    s = _unwrap_storage(a)
    out_ptr = npu_runtime._alloc_device(s.nbytes, runtime=runtime)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    aclnn.sinc(s.data_ptr(), out_ptr, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return _wrap_tensor(out_storage, a.shape, a.stride)


def special_entr_op(a):
    """Entropy: -x * log(x) for x > 0, 0 for x == 0, -inf for x < 0."""
    zero = _scalar_to_npu_tensor(0.0, a)
    neg_inf = _scalar_to_npu_tensor(float('-inf'), a)
    pos_mask = gt(a, zero)
    eq_mask = eq(a, zero)
    # -x * log(x) where x > 0
    entr_val = neg(mul(a, log(maximum(a, _scalar_to_npu_tensor(1e-38, a)))))
    result = where(pos_mask, entr_val, neg_inf)
    return where(eq_mask, zero, result)


def special_erfcx_op(a):
    """Scaled complementary error function: exp(x^2) * erfc(x)."""
    return mul(exp(mul(a, a)), erfc(a))


def special_logit_op(a, eps=None):
    """Logit function: log(x / (1 - x))."""
    one = _scalar_to_npu_tensor(1.0, a)
    if eps is not None:
        a = clamp(a, min_val=eps, max_val=1.0 - eps)
    return log(div(a, sub(one, a)))


def special_ndtr_op(a):
    """Normal CDF: 0.5 * erfc(-x / sqrt(2))."""
    import math
    half = _scalar_to_npu_tensor(0.5, a)
    inv_sqrt2 = _scalar_to_npu_tensor(-1.0 / math.sqrt(2.0), a)
    return mul(half, erfc(mul(a, inv_sqrt2)))


def special_log_ndtr_op(a):
    """Log of normal CDF: log(0.5 * erfc(-x / sqrt(2)))."""
    return log(special_ndtr_op(a))


def special_xlogy_op(a, b):
    """x * log(y), with 0 where x == 0."""
    zero = _scalar_to_npu_tensor(0.0, a)
    eq_mask = eq(a, zero)
    result = mul(a, log(maximum(b, _scalar_to_npu_tensor(1e-38, b))))
    return where(eq_mask, zero, result)


def special_xlog1py_op(a, b):
    """x * log1p(y), with 0 where x == 0."""
    zero = _scalar_to_npu_tensor(0.0, a)
    eq_mask = eq(a, zero)
    result = mul(a, log1p(b))
    return where(eq_mask, zero, result)


def special_multigammaln_op(a, p):
    """Multivariate log-gamma: sum_{i=0}^{p-1} lgamma(a - i/2) + p*(p-1)/4*log(pi)."""
    import math
    result = _scalar_to_npu_tensor(p * (p - 1) / 4.0 * math.log(math.pi), a)
    for i in range(p):
        offset = _scalar_to_npu_tensor(i / 2.0, a)
        result = add(result, special_gammaln(sub(a, offset)))
    return result


# ===========================================================================
# Phase 6: Linalg composites
# ===========================================================================


def _chebyshev_eval(x, coeffs, ref):
    """Evaluate Chebyshev polynomial: sum(c_i * x^i) using Horner's method."""
    result = _scalar_to_npu_tensor(coeffs[-1], ref)
    for c in reversed(coeffs[:-1]):
        result = add(mul(result, x), _scalar_to_npu_tensor(c, ref))
    return result


def special_i0_op(a):
    """Modified Bessel function I0 via CEPHES Chebyshev polynomial approximation."""
    from ...._dispatch.dispatcher import dispatch
    abs_x = dispatch("abs", "npu", a)
    # Coefficients from CEPHES for |x| <= 8
    A = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813]
    # Coefficients for |x| > 8
    B = [0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281,
         -0.02057706, 0.02635537, -0.01647633, 0.00392377]

    # For |x| <= 8: I0(x) = sum(A[i] * (x/3.75)^(2i))
    t_small = div(abs_x, _scalar_to_npu_tensor(3.75, abs_x))
    t2_small = mul(t_small, t_small)
    result_small = _chebyshev_eval(t2_small, A, a)

    # For |x| > 8: I0(x) = exp(x)/sqrt(x) * sum(B[i] * (3.75/x)^i)
    t_large = div(_scalar_to_npu_tensor(3.75, abs_x), abs_x)
    poly_large = _chebyshev_eval(t_large, B, a)
    exp_x = dispatch("exp", "npu", abs_x)
    sqrt_x = dispatch("sqrt", "npu", abs_x)
    result_large = mul(div(exp_x, sqrt_x), poly_large)

    # Select based on |x| <= 8
    threshold = _scalar_to_npu_tensor(8.0, abs_x)
    mask = dispatch("le", "npu", abs_x, threshold)
    return where(mask, result_small, result_large)


def special_i0e_op(a):
    """Exponentially scaled I0: i0(x) * exp(-|x|)."""
    from ...._dispatch.dispatcher import dispatch
    i0_val = special_i0_op(a)
    abs_x = dispatch("abs", "npu", a)
    neg_abs = dispatch("neg", "npu", abs_x)
    return mul(i0_val, dispatch("exp", "npu", neg_abs))


def special_i1_op(a):
    """Modified Bessel function I1 via CEPHES Chebyshev polynomial approximation."""
    from ...._dispatch.dispatcher import dispatch
    abs_x = dispatch("abs", "npu", a)
    # Coefficients for |x| <= 8
    A = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.02658733, 0.00301532, 0.00032411]
    # Coefficients for |x| > 8
    B = [0.39894228, -0.03988024, -0.00362018, 0.00163801, -0.01031555,
         0.02282967, -0.02895312, 0.01787654, -0.00420059]

    t_small = div(abs_x, _scalar_to_npu_tensor(3.75, abs_x))
    t2_small = mul(t_small, t_small)
    result_small = mul(abs_x, _chebyshev_eval(t2_small, A, a))

    t_large = div(_scalar_to_npu_tensor(3.75, abs_x), abs_x)
    poly_large = _chebyshev_eval(t_large, B, a)
    exp_x = dispatch("exp", "npu", abs_x)
    sqrt_x = dispatch("sqrt", "npu", abs_x)
    result_large = mul(div(exp_x, sqrt_x), poly_large)

    threshold = _scalar_to_npu_tensor(8.0, abs_x)
    mask = dispatch("le", "npu", abs_x, threshold)
    result = where(mask, result_small, result_large)
    # I1 is odd: I1(-x) = -I1(x)
    sign = dispatch("sign", "npu", a)
    return mul(sign, result)


def special_i1e_op(a):
    """Exponentially scaled I1: i1(x) * exp(-|x|)."""
    from ...._dispatch.dispatcher import dispatch
    i1_val = special_i1_op(a)
    abs_x = dispatch("abs", "npu", a)
    neg_abs = dispatch("neg", "npu", abs_x)
    return mul(i1_val, dispatch("exp", "npu", neg_abs))


def special_ndtri_op(a):
    """Inverse normal CDF via Beasley-Springer-Moro algorithm."""
    from ...._dispatch.dispatcher import dispatch
    import math
    # Rational approximation for the central region
    # Split into 3 regions based on p
    p = a
    half = _scalar_to_npu_tensor(0.5, p)
    t = sub(p, half)
    # Central region coefficients (|t| <= 0.42)
    a0, a1, a2, a3 = 2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637
    b1, b2, b3, b4 = -8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833
    # Compute r = t^2
    r = mul(t, t)
    # Numerator: t * (a0 + r*(a1 + r*(a2 + r*a3)))
    num = mul(t, add(_scalar_to_npu_tensor(a0, t),
        mul(r, add(_scalar_to_npu_tensor(a1, t),
        mul(r, add(_scalar_to_npu_tensor(a2, t),
        mul(r, _scalar_to_npu_tensor(a3, t))))))))
    # Denominator: 1 + r*(b1 + r*(b2 + r*(b3 + r*b4)))
    den = add(_scalar_to_npu_tensor(1.0, t),
        mul(r, add(_scalar_to_npu_tensor(b1, t),
        mul(r, add(_scalar_to_npu_tensor(b2, t),
        mul(r, add(_scalar_to_npu_tensor(b3, t),
        mul(r, _scalar_to_npu_tensor(b4, t)))))))))
    result_central = div(num, den)

    # Tail approximation for |t| > 0.42
    # r = sqrt(-2 * log(min(p, 1-p)))
    one = _scalar_to_npu_tensor(1.0, p)
    one_minus_p = sub(one, p)
    min_p = dispatch("minimum", "npu", p, one_minus_p)
    eps = _scalar_to_npu_tensor(1e-30, p)
    min_p_safe = dispatch("maximum", "npu", min_p, eps)
    log_p = dispatch("log", "npu", min_p_safe)
    neg2log = mul(_scalar_to_npu_tensor(-2.0, log_p), log_p)
    r_tail = dispatch("sqrt", "npu", neg2log)
    # Tail coefficients
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    t_num = add(_scalar_to_npu_tensor(c0, r_tail),
        mul(r_tail, add(_scalar_to_npu_tensor(c1, r_tail),
        mul(r_tail, _scalar_to_npu_tensor(c2, r_tail)))))
    t_den = add(_scalar_to_npu_tensor(1.0, r_tail),
        mul(r_tail, add(_scalar_to_npu_tensor(d1, r_tail),
        mul(r_tail, add(_scalar_to_npu_tensor(d2, r_tail),
        mul(r_tail, _scalar_to_npu_tensor(d3, r_tail)))))))
    result_tail = sub(r_tail, div(t_num, t_den))
    # Negate for p < 0.5
    lt_half = dispatch("lt", "npu", p, half)
    neg_result = dispatch("neg", "npu", result_tail)
    result_tail = where(lt_half, neg_result, result_tail)

    # Select central vs tail based on |t| <= 0.42
    abs_t = dispatch("abs", "npu", t)
    central_mask = dispatch("le", "npu", abs_t, _scalar_to_npu_tensor(0.42, abs_t))
    return where(central_mask, result_central, result_tail)


def special_polygamma_op(n, a):
    """Polygamma function. n=0: digamma. n>=1: series approximation."""
    from ...._dispatch.dispatcher import dispatch
    if isinstance(n, int) and n == 0:
        return dispatch("digamma", "npu", a)
    # For n >= 1: psi^(n)(x) = (-1)^(n+1) * n! * sum_{k=0}^{N} 1/(x+k)^(n+1)
    n_val = int(n) if not hasattr(n, 'data_ptr') else n
    import math
    sign = (-1) ** (n_val + 1)
    factorial_n = math.factorial(n_val)
    N_terms = 30  # number of series terms
    result = _scalar_to_npu_tensor(0.0, a)
    for k in range(N_terms):
        x_plus_k = add(a, _scalar_to_npu_tensor(float(k), a))
        term = dispatch("pow", "npu", x_plus_k, -(n_val + 1))
        result = add(result, term)
    return mul(result, _scalar_to_npu_tensor(float(sign * factorial_n), result))


def special_zeta_op(a, q):
    """Hurwitz zeta function via Euler-Maclaurin summation."""
    from ...._dispatch.dispatcher import dispatch
    # zeta(s, q) = sum_{k=0}^{N} 1/(q+k)^s + correction
    N_terms = 30
    result = _scalar_to_npu_tensor(0.0, q)
    for k in range(N_terms):
        q_plus_k = add(q, _scalar_to_npu_tensor(float(k), q))
        term = dispatch("pow", "npu", q_plus_k, dispatch("neg", "npu", a))
        result = add(result, term)
    # Euler-Maclaurin correction: 1/((s-1)*(q+N)^(s-1)) + 1/(2*(q+N)^s)
    q_N = add(q, _scalar_to_npu_tensor(float(N_terms), q))
    s_minus_1 = sub(a, _scalar_to_npu_tensor(1.0, a))
    correction1 = div(
        _scalar_to_npu_tensor(1.0, q_N),
        mul(s_minus_1, dispatch("pow", "npu", q_N, s_minus_1))
    )
    correction2 = div(
        _scalar_to_npu_tensor(0.5, q_N),
        dispatch("pow", "npu", q_N, a)
    )
    return add(result, add(correction1, correction2))


def special_gammainc_op(a, x):
    """Regularized lower incomplete gamma: P(a,x) via series expansion."""
    from ...._dispatch.dispatcher import dispatch
    # P(a,x) = e^{-x} * x^a * sum_{k=0}^{N} x^k / Gamma(a+k+1)
    # Use: sum_{k=0}^{N} x^k / prod_{j=1}^{k}(a+j) / Gamma(a+1)
    N_terms = 50
    term = div(_scalar_to_npu_tensor(1.0, x), a)  # 1/a
    s = contiguous(add(term, _scalar_to_npu_tensor(0.0, term)))  # clone
    for k in range(1, N_terms):
        a_plus_k = add(a, _scalar_to_npu_tensor(float(k), a))
        term = mul(term, div(x, a_plus_k))
        s = add(s, term)
    # P(a,x) = s * x^a * exp(-x)
    log_x = dispatch("log", "npu", dispatch("maximum", "npu", x, _scalar_to_npu_tensor(1e-30, x)))
    log_term = sub(mul(a, log_x), x)
    exp_term = dispatch("exp", "npu", log_term)
    return mul(s, exp_term)


def special_gammaincc_op(a, x):
    """Regularized upper incomplete gamma: Q(a,x) = 1 - P(a,x)."""
    return sub(_scalar_to_npu_tensor(1.0, a), special_gammainc_op(a, x))

# ---------- 3D conv/pool NPU composites ----------


def _build_dft_matrices(N, device, dtype, inverse=False):
    """Build real and imaginary parts of DFT matrix on NPU."""
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    import numpy as _np
    import math
    sign = 1.0 if inverse else -1.0
    # Build twiddle factors on CPU then copy to NPU
    angles = _np.zeros((N, N), dtype=_np.float32)
    for k in range(N):
        for n in range(N):
            angles[k, n] = sign * 2.0 * math.pi * k * n / N
    cos_vals = _np.cos(angles).astype(_np.float32)
    sin_vals = _np.sin(angles).astype(_np.float32)
    runtime = npu_runtime.get_runtime((device.index or 0))
    cos_ptr, _ = npu_runtime._copy_cpu_to_npu(cos_vals, runtime=runtime)
    sin_ptr, _ = npu_runtime._copy_cpu_to_npu(sin_vals, runtime=runtime)
    shape = (N, N)
    stride = npu_runtime._contiguous_stride(shape)
    cos_storage = npu_typed_storage_from_ptr(cos_ptr, N * N, float_dtype, device=device)
    sin_storage = npu_typed_storage_from_ptr(sin_ptr, N * N, float_dtype, device=device)
    Wr = _wrap_tensor(cos_storage, shape, stride)
    Wi = _wrap_tensor(sin_storage, shape, stride)
    if dtype != float_dtype:
        Wr = _cast_tensor_dtype(Wr, dtype)
        Wi = _cast_tensor_dtype(Wi, dtype)
    return Wr, Wi


def _apply_dft_1d(x_real, x_imag, dim, n, inverse, norm_mode):
    """Apply 1D DFT along a given dimension using matrix multiply."""
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    ndim = len(x_real.shape)
    N_in = x_real.shape[dim]
    N_out = n if n is not None else N_in
    device = x_real.device

    # Pad or truncate input to N_out along dim
    if N_in != N_out:
        if N_in < N_out:
            # Zero-pad
            pad_size = N_out - N_in
            pad_shape = list(x_real.shape)
            pad_shape[dim] = pad_size
            pad_real = dispatch("zeros", "npu", tuple(pad_shape), dtype=x_real.dtype, device=device)
            pad_imag = dispatch("zeros", "npu", tuple(pad_shape), dtype=x_real.dtype, device=device)
            x_real = dispatch("cat", "npu", [contiguous(x_real), pad_real], dim=dim)
            x_imag = dispatch("cat", "npu", [contiguous(x_imag), pad_imag], dim=dim)
        else:
            # Truncate
            from ...._creation import arange as _arange
            idx = _arange(0, N_out, dtype=int64_dtype, device=device)
            x_real = index_select(contiguous(x_real), dim, idx)
            x_imag = index_select(contiguous(x_imag), dim, idx)

    N = N_out
    Wr, Wi = _build_dft_matrices(N, device, x_real.dtype, inverse=inverse)

    # Move target dim to last, apply matmul, move back
    if dim < 0:
        dim = dim + ndim
    perm = list(range(ndim))
    if dim != ndim - 1:
        perm[dim], perm[ndim - 1] = perm[ndim - 1], perm[dim]
        x_real = view_backend.permute(contiguous(x_real), perm)
        x_imag = view_backend.permute(contiguous(x_imag), perm)

    # x is now (..., N) — apply W @ x via matmul
    # Need x as (..., N, 1) for matmul with (N, N)
    # Actually: result = x @ W^T (so each row of x gets multiplied)
    Wr_t = view_backend.permute(Wr, [1, 0])
    Wi_t = view_backend.permute(Wi, [1, 0])
    Wr_t = contiguous(Wr_t)
    Wi_t = contiguous(Wi_t)

    out_real = sub(matmul(contiguous(x_real), Wr_t), matmul(contiguous(x_imag), Wi_t))
    out_imag = add(matmul(contiguous(x_real), Wi_t), matmul(contiguous(x_imag), Wr_t))

    # Normalization
    if norm_mode == "ortho":
        scale = _scalar_to_npu_tensor(1.0 / (N ** 0.5), out_real)
        out_real = mul(out_real, scale)
        out_imag = mul(out_imag, scale)
    elif inverse and (norm_mode is None or norm_mode == "backward"):
        scale = _scalar_to_npu_tensor(1.0 / N, out_real)
        out_real = mul(out_real, scale)
        out_imag = mul(out_imag, scale)
    elif not inverse and norm_mode == "forward":
        scale = _scalar_to_npu_tensor(1.0 / N, out_real)
        out_real = mul(out_real, scale)
        out_imag = mul(out_imag, scale)

    # Permute back
    if dim != ndim - 1:
        out_real = view_backend.permute(contiguous(out_real), perm)
        out_imag = view_backend.permute(contiguous(out_imag), perm)

    return out_real, out_imag


def _pack_complex_as_last_dim(real, imag):
    """Pack real/imag into (..., 2) tensor for complex output."""
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    r = view_backend.reshape(contiguous(real), real.shape + (1,))
    i = view_backend.reshape(contiguous(imag), imag.shape + (1,))
    return dispatch("cat", "npu", [r, i], dim=-1)


def _unpack_complex(a):
    """Unpack (..., 2) complex tensor into (real, imag) pair."""
    from ...._creation import arange as _arange
    idx_r = _arange(0, 1, dtype=int64_dtype, device=a.device)
    idx_i = _arange(1, 2, dtype=int64_dtype, device=a.device)
    from ...common import view as view_backend
    real = view_backend.reshape(index_select(contiguous(a), -1, idx_r), a.shape[:-1])
    imag = view_backend.reshape(index_select(contiguous(a), -1, idx_i), a.shape[:-1])
    return real, imag


def _input_to_real_imag(a):
    """Convert input tensor to (real, imag) pair. Real input has imag=0."""
    from ...._dispatch.dispatcher import dispatch
    if len(a.shape) > 0 and a.shape[-1] == 2:
        # Could be complex stored as (..., 2)
        return _unpack_complex(a)
    # Real input
    imag = dispatch("zeros", "npu", a.shape, dtype=a.dtype, device=a.device)
    return a, imag


def fft_fft_op(a, n=None, dim=-1, norm=None):
    """1D FFT via DFT matrix multiply."""
    x_real, x_imag = _input_to_real_imag(a)
    out_r, out_i = _apply_dft_1d(x_real, x_imag, dim, n, inverse=False, norm_mode=norm)
    return _pack_complex_as_last_dim(out_r, out_i)


def fft_ifft_op(a, n=None, dim=-1, norm=None):
    """1D inverse FFT via DFT matrix multiply."""
    x_real, x_imag = _input_to_real_imag(a)
    out_r, out_i = _apply_dft_1d(x_real, x_imag, dim, n, inverse=True, norm_mode=norm)
    return _pack_complex_as_last_dim(out_r, out_i)


def fft_rfft_op(a, n=None, dim=-1, norm=None):
    """1D FFT of real input, returning only positive frequencies."""
    from ...._dispatch.dispatcher import dispatch
    from ...._creation import arange as _arange
    x_real = a
    x_imag = dispatch("zeros", "npu", a.shape, dtype=a.dtype, device=a.device)
    N = n if n is not None else a.shape[dim if dim >= 0 else dim + len(a.shape)]
    out_r, out_i = _apply_dft_1d(x_real, x_imag, dim, n, inverse=False, norm_mode=norm)
    # Keep only first N//2+1 frequencies
    half_n = N // 2 + 1
    d = dim if dim >= 0 else dim + len(out_r.shape)
    idx = _arange(0, half_n, dtype=int64_dtype, device=a.device)
    out_r = index_select(contiguous(out_r), d, idx)
    out_i = index_select(contiguous(out_i), d, idx)
    return _pack_complex_as_last_dim(out_r, out_i)


def fft_irfft_op(a, n=None, dim=-1, norm=None):
    """Inverse of rfft: reconstruct full spectrum, then ifft, return real."""
    from ...._dispatch.dispatcher import dispatch
    from ...._creation import arange as _arange
    from ...common import view as view_backend
    x_real, x_imag = _unpack_complex(a)
    d = dim if dim >= 0 else dim + len(x_real.shape)
    freq_len = x_real.shape[d]
    N = n if n is not None else 2 * (freq_len - 1)
    # Reconstruct full spectrum via conjugate symmetry
    if freq_len < N:
        # Conjugate mirror: X[N-k] = conj(X[k])
        idx_mirror = _arange(freq_len - 2, 0, step=-1, dtype=int64_dtype, device=a.device)
        mirror_real = index_select(contiguous(x_real), d, idx_mirror)
        mirror_imag = dispatch("neg", "npu", index_select(contiguous(x_imag), d, idx_mirror))
        x_real = dispatch("cat", "npu", [contiguous(x_real), mirror_real], dim=d)
        x_imag = dispatch("cat", "npu", [contiguous(x_imag), mirror_imag], dim=d)
    out_r, out_i = _apply_dft_1d(x_real, x_imag, d, N, inverse=True, norm_mode=norm)
    return out_r


def fft_fft2_op(a, s=None, dim=(-2, -1), norm=None):
    """2D FFT: sequential 1D FFT along each dim."""
    d0, d1 = dim
    s0 = s[0] if s is not None else None
    s1 = s[1] if s is not None else None
    x_real, x_imag = _input_to_real_imag(a)
    x_real, x_imag = _apply_dft_1d(x_real, x_imag, d1, s1, inverse=False, norm_mode=norm)
    x_real, x_imag = _apply_dft_1d(x_real, x_imag, d0, s0, inverse=False, norm_mode=norm)
    return _pack_complex_as_last_dim(x_real, x_imag)


def fft_ifft2_op(a, s=None, dim=(-2, -1), norm=None):
    """2D inverse FFT."""
    d0, d1 = dim
    s0 = s[0] if s is not None else None
    s1 = s[1] if s is not None else None
    x_real, x_imag = _input_to_real_imag(a)
    x_real, x_imag = _apply_dft_1d(x_real, x_imag, d0, s0, inverse=True, norm_mode=norm)
    x_real, x_imag = _apply_dft_1d(x_real, x_imag, d1, s1, inverse=True, norm_mode=norm)
    return _pack_complex_as_last_dim(x_real, x_imag)


def fft_rfft2_op(a, s=None, dim=(-2, -1), norm=None):
    """2D FFT of real input."""
    from ...._dispatch.dispatcher import dispatch
    from ...._creation import arange as _arange
    d0, d1 = dim
    s0 = s[0] if s is not None else None
    s1 = s[1] if s is not None else None
    x_real = a
    x_imag = dispatch("zeros", "npu", a.shape, dtype=a.dtype, device=a.device)
    # FFT along last dim first
    x_real, x_imag = _apply_dft_1d(x_real, x_imag, d1, s1, inverse=False, norm_mode=norm)
    # Keep only first N//2+1 along last dim
    d1_idx = d1 if d1 >= 0 else d1 + len(x_real.shape)
    N1 = s1 if s1 is not None else a.shape[d1_idx]
    half_n = N1 // 2 + 1
    idx = _arange(0, half_n, dtype=int64_dtype, device=a.device)
    x_real = index_select(contiguous(x_real), d1_idx, idx)
    x_imag = index_select(contiguous(x_imag), d1_idx, idx)
    # FFT along second-to-last dim
    x_real, x_imag = _apply_dft_1d(x_real, x_imag, d0, s0, inverse=False, norm_mode=norm)
    return _pack_complex_as_last_dim(x_real, x_imag)


def fft_irfft2_op(a, s=None, dim=(-2, -1), norm=None):
    """Inverse of rfft2."""
    d0, d1 = dim
    s0 = s[0] if s is not None else None
    s1 = s[1] if s is not None else None
    x_real, x_imag = _unpack_complex(a)
    # IFFT along second-to-last dim
    x_real, x_imag = _apply_dft_1d(x_real, x_imag, d0, s0, inverse=True, norm_mode=norm)
    # Reconstruct full spectrum along last dim and IFFT
    from ...._dispatch.dispatcher import dispatch
    from ...._creation import arange as _arange
    d1_idx = d1 if d1 >= 0 else d1 + len(x_real.shape)
    freq_len = x_real.shape[d1_idx]
    N1 = s1 if s1 is not None else 2 * (freq_len - 1)
    if freq_len < N1:
        idx_mirror = _arange(freq_len - 2, 0, step=-1, dtype=int64_dtype, device=a.device)
        mirror_real = index_select(contiguous(x_real), d1_idx, idx_mirror)
        mirror_imag = dispatch("neg", "npu", index_select(contiguous(x_imag), d1_idx, idx_mirror))
        x_real = dispatch("cat", "npu", [contiguous(x_real), mirror_real], dim=d1_idx)
        x_imag = dispatch("cat", "npu", [contiguous(x_imag), mirror_imag], dim=d1_idx)
    out_r, _ = _apply_dft_1d(x_real, x_imag, d1_idx, N1, inverse=True, norm_mode=norm)
    return out_r


def fft_fftn_op(a, s=None, dim=None, norm=None):
    """N-D FFT: sequential 1D FFT along each dim."""
    ndim = len(a.shape)
    if dim is None:
        dim = list(range(ndim))
    elif isinstance(dim, int):
        dim = [dim]
    else:
        dim = list(dim)
    x_real, x_imag = _input_to_real_imag(a)
    for i, d in enumerate(dim):
        n_d = s[i] if s is not None and i < len(s) else None
        x_real, x_imag = _apply_dft_1d(x_real, x_imag, d, n_d, inverse=False, norm_mode=norm)
    return _pack_complex_as_last_dim(x_real, x_imag)


def fft_ifftn_op(a, s=None, dim=None, norm=None):
    """N-D inverse FFT."""
    ndim = len(a.shape)
    if dim is None:
        dim = list(range(ndim))
    elif isinstance(dim, int):
        dim = [dim]
    else:
        dim = list(dim)
    x_real, x_imag = _input_to_real_imag(a)
    for i, d in enumerate(dim):
        n_d = s[i] if s is not None and i < len(s) else None
        x_real, x_imag = _apply_dft_1d(x_real, x_imag, d, n_d, inverse=True, norm_mode=norm)
    return _pack_complex_as_last_dim(x_real, x_imag)


def fft_rfftn_op(a, s=None, dim=None, norm=None):
    """N-D FFT of real input."""
    from ...._dispatch.dispatcher import dispatch
    from ...._creation import arange as _arange
    ndim = len(a.shape)
    if dim is None:
        dim = list(range(ndim))
    elif isinstance(dim, int):
        dim = [dim]
    else:
        dim = list(dim)
    x_real = a
    x_imag = dispatch("zeros", "npu", a.shape, dtype=a.dtype, device=a.device)
    for i, d in enumerate(dim):
        n_d = s[i] if s is not None and i < len(s) else None
        is_last = (i == len(dim) - 1)
        x_real, x_imag = _apply_dft_1d(x_real, x_imag, d, n_d, inverse=False, norm_mode=norm)
        if is_last:
            # Keep only first N//2+1 along last transformed dim
            d_idx = d if d >= 0 else d + len(x_real.shape)
            N_last = n_d if n_d is not None else a.shape[d_idx]
            half_n = N_last // 2 + 1
            idx = _arange(0, half_n, dtype=int64_dtype, device=a.device)
            x_real = index_select(contiguous(x_real), d_idx, idx)
            x_imag = index_select(contiguous(x_imag), d_idx, idx)
    return _pack_complex_as_last_dim(x_real, x_imag)


def fft_irfftn_op(a, s=None, dim=None, norm=None):
    """Inverse of rfftn."""
    from ...._dispatch.dispatcher import dispatch
    from ...._creation import arange as _arange
    x_real, x_imag = _unpack_complex(a)
    if dim is None:
        dim = list(range(len(x_real.shape)))
    elif isinstance(dim, int):
        dim = [dim]
    else:
        dim = list(dim)
    for i, d in enumerate(dim):
        n_d = s[i] if s is not None and i < len(s) else None
        is_last = (i == len(dim) - 1)
        if is_last:
            # Reconstruct full spectrum along last dim
            d_idx = d if d >= 0 else d + len(x_real.shape)
            freq_len = x_real.shape[d_idx]
            N = n_d if n_d is not None else 2 * (freq_len - 1)
            if freq_len < N:
                idx_mirror = _arange(freq_len - 2, 0, step=-1, dtype=int64_dtype, device=a.device)
                mirror_real = index_select(contiguous(x_real), d_idx, idx_mirror)
                mirror_imag = dispatch("neg", "npu", index_select(contiguous(x_imag), d_idx, idx_mirror))
                x_real = dispatch("cat", "npu", [contiguous(x_real), mirror_real], dim=d_idx)
                x_imag = dispatch("cat", "npu", [contiguous(x_imag), mirror_imag], dim=d_idx)
            n_d = N
        x_real, x_imag = _apply_dft_1d(x_real, x_imag, d, n_d, inverse=True, norm_mode=norm)
    return x_real


def fft_hfft_op(a, n=None, dim=-1, norm=None):
    """Hermitian FFT: irfft(conj(x)). Output is real."""
    x_real, x_imag = _unpack_complex(a)
    # conj: negate imag
    from ...._dispatch.dispatcher import dispatch
    x_imag_neg = dispatch("neg", "npu", x_imag)
    # irfft
    d = dim if dim >= 0 else dim + len(x_real.shape)
    from ...._creation import arange as _arange
    freq_len = x_real.shape[d]
    N = n if n is not None else 2 * (freq_len - 1)
    if freq_len < N:
        idx_mirror = _arange(freq_len - 2, 0, step=-1, dtype=int64_dtype, device=a.device)
        mirror_real = index_select(contiguous(x_real), d, idx_mirror)
        mirror_imag = dispatch("neg", "npu", index_select(contiguous(x_imag_neg), d, idx_mirror))
        x_real = dispatch("cat", "npu", [contiguous(x_real), mirror_real], dim=d)
        x_imag_neg = dispatch("cat", "npu", [contiguous(x_imag_neg), mirror_imag], dim=d)
    out_r, _ = _apply_dft_1d(x_real, x_imag_neg, d, N, inverse=True, norm_mode=norm)
    return out_r


def fft_ihfft_op(a, n=None, dim=-1, norm=None):
    """Inverse Hermitian FFT: conj(rfft(x))."""
    from ...._dispatch.dispatcher import dispatch
    from ...._creation import arange as _arange
    x_real = a
    x_imag = dispatch("zeros", "npu", a.shape, dtype=a.dtype, device=a.device)
    N = n if n is not None else a.shape[dim if dim >= 0 else dim + len(a.shape)]
    out_r, out_i = _apply_dft_1d(x_real, x_imag, dim, n, inverse=False, norm_mode=norm)
    # Keep only first N//2+1
    half_n = N // 2 + 1
    d = dim if dim >= 0 else dim + len(out_r.shape)
    idx = _arange(0, half_n, dtype=int64_dtype, device=a.device)
    out_r = index_select(contiguous(out_r), d, idx)
    out_i = index_select(contiguous(out_i), d, idx)
    # Conjugate
    out_i = dispatch("neg", "npu", out_i)
    return _pack_complex_as_last_dim(out_r, out_i)


def fft_fftshift_op(a, dim=None):
    """fftshift via roll — pure tensor op, no ACLNN needed."""
    from ...._dispatch.dispatcher import dispatch
    if dim is None:
        dim = list(range(len(a.shape)))
    elif isinstance(dim, int):
        dim = [dim]
    result = a
    for d in dim:
        n = a.shape[d]
        shift = n // 2
        result = dispatch("roll", "npu", result, shift, d)
    return result


def fft_ifftshift_op(a, dim=None):
    """ifftshift via roll."""
    from ...._dispatch.dispatcher import dispatch
    if dim is None:
        dim = list(range(len(a.shape)))
    elif isinstance(dim, int):
        dim = [dim]
    result = a
    for d in dim:
        n = a.shape[d]
        shift = -(n // 2)
        result = dispatch("roll", "npu", result, shift, d)
    return result


# ---------- Linalg NPU composites ----------
