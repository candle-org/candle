# Cython CPU Ops Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace NumPy ufunc calls in `_backends/cpu/ops.py` with Cython kernels that operate directly on raw C memory, eliminating NumPy from the arithmetic hot path.

**Architecture:** Storage is numpy-backed (`_CPUUntypedStorage._array`). `t._numpy_view()` returns a zero-copy strided numpy array. Cython kernels receive C-contiguous typed `float[::1]` / `double[::1]` memoryviews (caller flattens+contiguous), compute in C loops with `libc.math` intrinsics, and write into a pre-allocated numpy output buffer. NumPy is retained only for memory allocation (`np.empty`) and buffer wrapping — never for arithmetic. A `_cpu_kernels_fallback.py` with `None` stubs keeps everything importable when the `.so` is absent.

**Tech Stack:** Cython 3.x, libc.math (expf/logf/erff/…), cblas (sgemm/dgemm), Python 3.11, Linux aarch64 + macOS arm64

---

## Chunk 1: Skeleton — new pyx file, fallback, build wiring

### Task 1: Create `_cpu_kernels_fallback.py`

**Files:**
- Create: `src/candle/_cython/_cpu_kernels_fallback.py`

- [ ] **Step 1: Write the file**

```python
"""Pure-Python fallback for _cpu_kernels.pyx.

All symbols are None; ops.py falls back to numpy when Cython .so is absent.
"""
# element-wise binary (float32)
add_f32 = None
sub_f32 = None
mul_f32 = None
div_f32 = None
# element-wise binary (float64)
add_f64 = None
sub_f64 = None
mul_f64 = None
div_f64 = None
# scalar binary (float32)
add_scalar_f32 = None
sub_scalar_f32 = None
mul_scalar_f32 = None
div_scalar_f32 = None
# scalar binary (float64)
add_scalar_f64 = None
sub_scalar_f64 = None
mul_scalar_f64 = None
div_scalar_f64 = None
# unary float32
neg_f32 = None
abs_f32 = None
relu_f32 = None
gelu_f32 = None
sigmoid_f32 = None
exp_f32 = None
log_f32 = None
sqrt_f32 = None
rsqrt_f32 = None
sin_f32 = None
cos_f32 = None
tan_f32 = None
tanh_f32 = None
sinh_f32 = None
cosh_f32 = None
asinh_f32 = None
acosh_f32 = None
atanh_f32 = None
erf_f32 = None
erfc_f32 = None
exp2_f32 = None
log2_f32 = None
log10_f32 = None
pow_f32 = None
square_f32 = None
# unary float64
neg_f64 = None
abs_f64 = None
relu_f64 = None
gelu_f64 = None
sigmoid_f64 = None
exp_f64 = None
log_f64 = None
sqrt_f64 = None
rsqrt_f64 = None
sin_f64 = None
cos_f64 = None
tan_f64 = None
tanh_f64 = None
sinh_f64 = None
cosh_f64 = None
asinh_f64 = None
acosh_f64 = None
atanh_f64 = None
erf_f64 = None
erfc_f64 = None
exp2_f64 = None
log2_f64 = None
log10_f64 = None
pow_f64 = None
square_f64 = None
# matmul
matmul_f32 = None
matmul_f64 = None
```

- [ ] **Step 2: Verify importable**

```bash
cd .worktrees/cython-cpu-ops
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -c "from candle._cython._cpu_kernels_fallback import add_f32; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/candle/_cython/_cpu_kernels_fallback.py
git commit -m "feat(cython-cpu): add _cpu_kernels_fallback skeleton"
```

---

### Task 2: Create `_cpu_kernels.pyx` with element-wise binary + unary kernels

**Files:**
- Create: `src/candle/_cython/_cpu_kernels.pyx`

- [ ] **Step 1: Write `_cpu_kernels.pyx`**

The file must:
- Use `# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True`
- Use `cimport cython` and `from libc.math cimport ...`
- Fused types `floating` = `float | double` (Cython fused type)
- All kernels take flat C-contiguous 1-D typed memoryviews `floating[::1]`
- All kernels return `void` (write into pre-allocated `out`)

```cython
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython CPU kernels — pure C loops, no NumPy arithmetic."""

cimport cython
from libc.math cimport (
    expf, logf, sqrtf, sinf, cosf, tanf, tanhf,
    sinhf, coshf, asinhf, acoshf, atanhf, erff, erfcf,
    exp2f, log2f, log10f, powf, fabsf,
    exp, log, sqrt, sin, cos, tan, tanh,
    sinh, cosh, asinh, acosh, atanh, erf, erfc,
    exp2, log2, log10, pow, fabs,
)

# Fused type covering float32 and float64
ctypedef fused floating:
    float
    double


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
# Scalar binary: float32  (a is tensor, b is Python scalar)
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
    cdef float M_SQRT1_2 = 0.7071067811865476
    for i in range(n):
        x = a[i]
        out[i] = 0.5 * x * (1.0 + erff(x * M_SQRT1_2))

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
    cdef double M_SQRT1_2 = 0.7071067811865476
    for i in range(n):
        x = a[i]
        out[i] = 0.5 * x * (1.0 + erf(x * M_SQRT1_2))

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
    cdef Py_ssize_t