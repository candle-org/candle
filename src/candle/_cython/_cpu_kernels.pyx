# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython CPU kernels — pure C loops, no NumPy arithmetic.

All kernels operate on flat C-contiguous 1-D typed memoryviews.
Callers are responsible for ensuring contiguity (np.ascontiguousarray +
np.ravel) before passing buffers in.
"""

cimport cython
from libc.math cimport (
    expf, logf, sqrtf, sinf, cosf, tanf, tanhf,
    sinhf, coshf, asinhf, acoshf, atanhf, erff, erfcf,
    exp2f, log2f, log10f, powf, fabsf,
    exp, log, sqrt, sin, cos, tan, tanh,
    sinh, cosh, asinh, acosh, atanh, erf, erfc,
    exp2, log2, log10, pow, fabs,
)


# ---------------------------------------------------------------------------
# Element-wise binary: float32
# ---------------------------------------------------------------------------

def add_f32(float[::1] a, float[::1] b, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] + b[i]


def sub_f32(float[::1] a, float[::1] b, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] - b[i]


def mul_f32(float[::1] a, float[::1] b, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] * b[i]


def div_f32(float[::1] a, float[::1] b, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] / b[i]


# ---------------------------------------------------------------------------
# Element-wise binary: float64
# ---------------------------------------------------------------------------

def add_f64(double[::1] a, double[::1] b, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] + b[i]


def sub_f64(double[::1] a, double[::1] b, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] - b[i]


def mul_f64(double[::1] a, double[::1] b, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] * b[i]


def div_f64(double[::1] a, double[::1] b, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] / b[i]


# ---------------------------------------------------------------------------
# Scalar binary: float32  (tensor op scalar)
# ---------------------------------------------------------------------------

def add_scalar_f32(float[::1] a, float b, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] + b


def sub_scalar_f32(float[::1] a, float b, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] - b


def mul_scalar_f32(float[::1] a, float b, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] * b


def div_scalar_f32(float[::1] a, float b, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] / b


# ---------------------------------------------------------------------------
# Scalar binary: float64
# ---------------------------------------------------------------------------

def add_scalar_f64(double[::1] a, double b, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] + b


def sub_scalar_f64(double[::1] a, double b, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] - b


def mul_scalar_f64(double[::1] a, double b, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] * b


def div_scalar_f64(double[::1] a, double b, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] / b


# ---------------------------------------------------------------------------
# Unary: float32
# ---------------------------------------------------------------------------

def neg_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = -a[i]


def abs_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = fabsf(a[i])


def relu_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] if a[i] > 0.0 else 0.0


def gelu_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    cdef float x
    cdef float inv_sqrt2 = 0.7071067811865476
    for i in range(n):
        x = a[i]
        out[i] = 0.5 * x * (1.0 + erff(x * inv_sqrt2))


def sigmoid_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = 1.0 / (1.0 + expf(-a[i]))


def exp_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = expf(a[i])


def log_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = logf(a[i])


def sqrt_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = sqrtf(a[i])


def rsqrt_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = 1.0 / sqrtf(a[i])


def sin_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = sinf(a[i])


def cos_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = cosf(a[i])


def tan_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = tanf(a[i])


def tanh_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = tanhf(a[i])


def sinh_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = sinhf(a[i])


def cosh_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = coshf(a[i])


def asinh_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = asinhf(a[i])


def acosh_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = acoshf(a[i])


def atanh_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = atanhf(a[i])


def erf_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = erff(a[i])


def erfc_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = erfcf(a[i])


def exp2_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = exp2f(a[i])


def log2_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = log2f(a[i])


def log10_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = log10f(a[i])


def pow_f32(float[::1] a, float b, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = powf(a[i], b)


def square_f32(float[::1] a, float[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] * a[i]


# ---------------------------------------------------------------------------
# Unary: float64
# ---------------------------------------------------------------------------

def neg_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = -a[i]


def abs_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = fabs(a[i])


def relu_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] if a[i] > 0.0 else 0.0


def gelu_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double x
    cdef double inv_sqrt2 = 0.7071067811865476
    for i in range(n):
        x = a[i]
        out[i] = 0.5 * x * (1.0 + erf(x * inv_sqrt2))


def sigmoid_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = 1.0 / (1.0 + exp(-a[i]))


def exp_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = exp(a[i])


def log_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = log(a[i])


def sqrt_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = sqrt(a[i])


def rsqrt_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = 1.0 / sqrt(a[i])


def sin_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = sin(a[i])


def cos_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = cos(a[i])


def tan_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = tan(a[i])


def tanh_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = tanh(a[i])


def sinh_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = sinh(a[i])


def cosh_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = cosh(a[i])


def asinh_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = asinh(a[i])


def acosh_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = acosh(a[i])


def atanh_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = atanh(a[i])


def erf_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = erf(a[i])


def erfc_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = erfc(a[i])


def exp2_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = exp2(a[i])


def log2_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = log2(a[i])


def log10_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = log10(a[i])


def pow_f64(double[::1] a, double b, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = pow(a[i], b)


def square_f64(double[::1] a, double[::1] out):
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] * a[i]


# matmul_f32 / matmul_f64 intentionally omitted:
# numpy matmul already dispatches to OpenBLAS sgemm/dgemm internally.
# cblas.h is not installed in this environment.
# ops.py keeps matmul on the numpy @ path.
