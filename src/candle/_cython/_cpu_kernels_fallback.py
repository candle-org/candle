"""Pure-Python fallback for _cpu_kernels.pyx.

All symbols are None; ops.py falls back to numpy when the Cython .so is absent.
"""
# element-wise binary float32
add_f32 = None
sub_f32 = None
mul_f32 = None
div_f32 = None
# element-wise binary float64
add_f64 = None
sub_f64 = None
mul_f64 = None
div_f64 = None
# scalar binary float32
add_scalar_f32 = None
sub_scalar_f32 = None
mul_scalar_f32 = None
div_scalar_f32 = None
# scalar binary float64
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
