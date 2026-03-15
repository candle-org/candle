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

def special_digamma(a):
    """Logarithmic derivative of the gamma function (psi function)."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.digamma(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_entr(a):
    """Entropy: -x * ln(x), 0 for x=0, -inf for x<0."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.where(arr > 0, -arr * np.log(arr), np.where(arr == 0, 0.0, -np.inf))
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_erfcx(a):
    """Scaled complementary error function: exp(x^2) * erfc(x)."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.erfcx(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_erfinv(a):
    """Inverse error function."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.erfinv(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_gammainc(a, b):
    """Regularized lower incomplete gamma function."""
    from scipy import special as sp
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    out = sp.gammainc(a_np, b_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_gammaincc(a, b):
    """Regularized upper incomplete gamma function."""
    from scipy import special as sp
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    out = sp.gammaincc(a_np, b_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_gammaln(a):
    """Log of the absolute value of the gamma function."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.gammaln(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_i0(a):
    """Zeroth order modified Bessel function of the first kind."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.i0(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_i0e(a):
    """Exponentially scaled zeroth order modified Bessel function."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.i0e(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_i1(a):
    """First order modified Bessel function of the first kind."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.i1(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_i1e(a):
    """Exponentially scaled first order modified Bessel function."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.i1e(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_log_ndtr(a):
    """Log of the area under the standard Gaussian PDF from -inf to x."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.log_ndtr(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_logit(a, eps=None):
    """Logit function: log(x / (1 - x))."""
    arr = _to_numpy(a).astype(np.float64)
    if eps is not None:
        arr = np.clip(arr, eps, 1.0 - eps)
    from scipy import special as sp
    out = sp.logit(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_multigammaln(a, p):
    """Multivariate log-gamma function with dimension p."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.multigammaln(arr, p)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_ndtr(a):
    """Area under the standard Gaussian PDF from -inf to x (normal CDF)."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.ndtr(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_ndtri(a):
    """Inverse of ndtr (quantile function of standard normal)."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.ndtri(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_polygamma(n, a):
    """N-th derivative of the digamma function."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.polygamma(n, arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_sinc(a):
    """Normalized sinc function: sin(pi*x) / (pi*x)."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.sinc(arr)  # np.sinc already computes sin(pi*x)/(pi*x)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_xlog1py(a, b):
    """x * log1p(y), with 0 when x=0."""
    from scipy import special as sp
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    out = sp.xlog1py(a_np, b_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_xlogy(a, b):
    """x * log(y), with 0 when x=0."""
    from scipy import special as sp
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    out = sp.xlogy(a_np, b_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def special_zeta(a, b):
    """Hurwitz zeta function."""
    from scipy import special as sp
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    out = sp.zeta(a_np, b_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


# ---------------------------------------------------------------------------
# F.affine_grid / F.grid_sample
# ---------------------------------------------------------------------------

def fft_fft(a, n=None, dim=-1, norm=None):
    """1D discrete Fourier Transform."""
    from ...._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    out = np.fft.fft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)

def fft_ifft(a, n=None, dim=-1, norm=None):
    """1D inverse discrete Fourier Transform."""
    from ...._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    out = np.fft.ifft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)

def fft_fft2(a, s=None, dim=(-2, -1), norm=None):
    """2D discrete Fourier Transform."""
    from ...._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.fft2(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)

def fft_ifft2(a, s=None, dim=(-2, -1), norm=None):
    """2D inverse discrete Fourier Transform."""
    from ...._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.ifft2(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)

def fft_fftn(a, s=None, dim=None, norm=None):
    """N-D discrete Fourier Transform."""
    from ...._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.fftn(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)

def fft_ifftn(a, s=None, dim=None, norm=None):
    """N-D inverse discrete Fourier Transform."""
    from ...._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.ifftn(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)

def fft_rfft(a, n=None, dim=-1, norm=None):
    """1D FFT of real-valued input."""
    from ...._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    out = np.fft.rfft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)

def fft_irfft(a, n=None, dim=-1, norm=None):
    """Inverse of rfft."""
    arr = _to_numpy(a)
    out = np.fft.irfft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def fft_rfft2(a, s=None, dim=(-2, -1), norm=None):
    """2D FFT of real-valued input."""
    from ...._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.rfft2(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)

def fft_irfft2(a, s=None, dim=(-2, -1), norm=None):
    """Inverse of rfft2."""
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.irfft2(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def fft_rfftn(a, s=None, dim=None, norm=None):
    """N-D FFT of real-valued input."""
    from ...._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.rfftn(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)

def fft_irfftn(a, s=None, dim=None, norm=None):
    """Inverse of rfftn."""
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.irfftn(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def fft_hfft(a, n=None, dim=-1, norm=None):
    """1D FFT of Hermitian symmetric signal (output is real)."""
    arr = _to_numpy(a)
    out = np.fft.hfft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def fft_ihfft(a, n=None, dim=-1, norm=None):
    """Inverse of hfft."""
    from ...._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    out = np.fft.ihfft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)

def fft_fftshift(a, dim=None):
    """Shift zero-frequency component to center."""
    # GPU composite: roll by n//2 per dim
    if _can_use_gpu(a):
        from .shape import roll
        ndim = len(a.shape)
        if dim is None:
            dims = list(range(ndim))
        elif isinstance(dim, int):
            dims = [dim]
        else:
            dims = list(dim)
        shifts = [a.shape[d if d >= 0 else d + ndim] // 2 for d in dims]
        return roll(a, shifts, dims)
    arr = _to_numpy(a)
    axes = None if dim is None else (tuple(dim) if isinstance(dim, (list, tuple)) else (dim,))
    out = np.fft.fftshift(arr, axes=axes)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def fft_ifftshift(a, dim=None):
    """Inverse of fftshift."""
    # GPU composite: roll by -(n//2) per dim
    if _can_use_gpu(a):
        from .shape import roll
        ndim = len(a.shape)
        if dim is None:
            dims = list(range(ndim))
        elif isinstance(dim, int):
            dims = [dim]
        else:
            dims = list(dim)
        shifts = [-(a.shape[d if d >= 0 else d + ndim] // 2) for d in dims]
        return roll(a, shifts, dims)
    arr = _to_numpy(a)
    axes = None if dim is None else (tuple(dim) if isinstance(dim, (list, tuple)) else (dim,))
    out = np.fft.ifftshift(arr, axes=axes)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


# ---------------------------------------------------------------------------
# torch.special ops
# ---------------------------------------------------------------------------

