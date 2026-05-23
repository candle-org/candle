"""Random number generation and in-place initialization for NPU."""
from ._helpers import (
    _unwrap_storage, _wrap_tensor,
    _numel, _dtype_itemsize, _use_soc_fallback,
    _scalar_to_npu_tensor,
    _npu_arange_1d,
    _npu_add_scalar_,
    _cast_tensor_dtype,
    float_dtype,
    npu_typed_storage_from_ptr, reshape,
    aclnn, npu_runtime, npu_state,
)
from .comparison import lt
from .elementwise import clamp
from .math import abs, add, ceil, cos, exp, floor, frac, log, mul, neg, sin, sqrt, tan

try:
    from candle._C._npu_ops import (
        fast_add_inplace as _fast_add_inplace_impl,
        fast_ceil_inplace as _fast_ceil_inplace_impl,
        fast_clamp_inplace as _fast_clamp_inplace_impl,
        fast_copy_inplace as _fast_copy_inplace_impl,
        fast_div_inplace as _fast_div_inplace_impl,
        fast_erfinv_ as _fast_erfinv_inplace_impl,
        fast_exp_inplace as _fast_exp_inplace_impl,
        fast_fill_inplace as _fast_fill_inplace_impl,
        fast_floor_inplace as _fast_floor_inplace_impl,
        fast_log_inplace as _fast_log_inplace_impl,
        fast_mul_inplace as _fast_mul_inplace_impl,
        fast_neg_inplace as _fast_neg_inplace_impl,
        fast_sub_inplace as _fast_sub_inplace_impl,
        fast_tan_inplace as _fast_tan_inplace_impl,
    )  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_ADD_INPLACE = True
    _HAS_FAST_CEIL_INPLACE = True
    _HAS_FAST_CLAMP_INPLACE = True
    _HAS_FAST_COPY_INPLACE = True
    _HAS_FAST_DIV_INPLACE = True
    _HAS_FAST_ERFINV_INPLACE = True
    _HAS_FAST_EXP_INPLACE = True
    _HAS_FAST_FILL_INPLACE = True
    _HAS_FAST_FLOOR_INPLACE = True
    _HAS_FAST_LOG_INPLACE = True
    _HAS_FAST_MUL_INPLACE = True
    _HAS_FAST_NEG_INPLACE = True
    _HAS_FAST_SUB_INPLACE = True
    _HAS_FAST_TAN_INPLACE = True
except ImportError:
    _fast_add_inplace_impl = None  # type: ignore[assignment]
    _fast_ceil_inplace_impl = None  # type: ignore[assignment]
    _fast_clamp_inplace_impl = None  # type: ignore[assignment]
    _fast_copy_inplace_impl = None  # type: ignore[assignment]
    _fast_div_inplace_impl = None  # type: ignore[assignment]
    _fast_erfinv_inplace_impl = None  # type: ignore[assignment]
    _fast_exp_inplace_impl = None  # type: ignore[assignment]
    _fast_fill_inplace_impl = None  # type: ignore[assignment]
    _fast_floor_inplace_impl = None  # type: ignore[assignment]
    _fast_log_inplace_impl = None  # type: ignore[assignment]
    _fast_mul_inplace_impl = None  # type: ignore[assignment]
    _fast_neg_inplace_impl = None  # type: ignore[assignment]
    _fast_sub_inplace_impl = None  # type: ignore[assignment]
    _fast_tan_inplace_impl = None  # type: ignore[assignment]
    _HAS_FAST_ADD_INPLACE = False
    _HAS_FAST_CEIL_INPLACE = False
    _HAS_FAST_CLAMP_INPLACE = False
    _HAS_FAST_COPY_INPLACE = False
    _HAS_FAST_DIV_INPLACE = False
    _HAS_FAST_ERFINV_INPLACE = False
    _HAS_FAST_EXP_INPLACE = False
    _HAS_FAST_FILL_INPLACE = False
    _HAS_FAST_FLOOR_INPLACE = False
    _HAS_FAST_LOG_INPLACE = False
    _HAS_FAST_MUL_INPLACE = False
    _HAS_FAST_NEG_INPLACE = False
    _HAS_FAST_SUB_INPLACE = False
    _HAS_FAST_TAN_INPLACE = False


def randperm(n, dtype=None, device=None, generator=None):
    """Random permutation of integers from 0 to n-1."""
    if not aclnn.randperm_symbols_ok():
        raise RuntimeError("aclnnRandperm symbols not available")
    # Import device handling
    from ...._device import device as Device
    if device is None:
        device = Device("npu:0")
    elif isinstance(device, str):
        device = Device(device)
    if device.type != "npu":
        raise ValueError("NPU randperm only supports NPU device")

    if dtype is None:
        dtype = "int64"
    runtime = npu_runtime.get_runtime((device.index or 0))
    stream = npu_state.current_stream((device.index or 0))

    # Get deterministic seed
    if generator is not None and hasattr(generator, 'philox_engine_inputs'):
        seed, offset = generator.philox_engine_inputs(10)
    else:
        from .... import npu as npu_mod
        seed, offset = npu_mod._get_and_advance_offset(device_index=(device.index or 0), increment=10)

    itemsize = _dtype_itemsize(dtype)
    out_ptr = npu_runtime._alloc_device(n * itemsize, runtime=runtime)

    aclnn.randperm(n, out_ptr, dtype, runtime, stream=stream.stream, seed=seed, offset=offset)

    out_storage = npu_typed_storage_from_ptr(out_ptr, n, dtype, device=device)
    return _wrap_tensor(out_storage, (n,), (1,))


def zero_(a):
    fill_(a, 0.0)
    return a


def uniform_(a, low=0.0, high=1.0, generator=None):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if _use_soc_fallback("uniform_"):
        from .... import npu as npu_mod

        if generator is not None and hasattr(generator, 'philox_engine_inputs'):
            seed, offset = generator.philox_engine_inputs(10)
        else:
            seed, offset = npu_mod._get_and_advance_offset(device_index=(a.device.index or 0), increment=10)

        # Keep seed term in a compact range to avoid float32 precision collapse on 310B.
        seed_mod = float((int(seed) + int(offset)) % 1000003)
        idx = _cast_tensor_dtype(_npu_arange_1d(_numel(a.shape), a.device), float_dtype)
        u = sin(add(mul(idx, 12.9898), seed_mod * 78.233))
        u = frac(abs(mul(u, 43758.5453)))
        u = reshape(u, a.shape)

        scale = float(high) - float(low)
        if scale != 1.0:
            u = mul(u, scale)
        if float(low) != 0.0:
            u = add(u, float(low))

        if a.dtype != float_dtype:
            u = _cast_tensor_dtype(u, a.dtype)
        return copy_(a, u)

    if generator is not None and hasattr(generator, 'philox_engine_inputs'):
        seed, offset = generator.philox_engine_inputs(10)
    else:
        from .... import npu as npu_mod
        seed, offset = npu_mod._get_and_advance_offset(device_index=(a.device.index or 0), increment=10)

    a_storage = _unwrap_storage(a)
    aclnn.inplace_uniform(
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        float(low),
        float(high),
        seed,
        offset,
        runtime,
        stream=stream.stream,
    )
    return a


def normal_(a, mean=0.0, std=1.0, generator=None):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if _use_soc_fallback("normal_"):
        # Deterministic NPU-only fallback built from small ops.
        from .... import npu as npu_mod

        if generator is not None and hasattr(generator, 'philox_engine_inputs'):
            seed, offset = generator.philox_engine_inputs(10)
        else:
            seed, offset = npu_mod._get_and_advance_offset(device_index=(a.device.index or 0), increment=10)

        seed_mod = float((int(seed) + int(offset)) % 1000003)
        idx = _cast_tensor_dtype(_npu_arange_1d(_numel(a.shape), a.device), float_dtype)

        # Two decorrelated pseudo-uniform streams in (0, 1) for Box-Muller.
        u1 = sin(_npu_add_scalar_(mul(idx, 12.9898), seed_mod * 78.233))
        u1 = frac(abs(mul(u1, 43758.5453)))
        u2 = sin(_npu_add_scalar_(mul(_npu_add_scalar_(idx, 0.5), 93.9898), seed_mod * 67.345))
        u2 = frac(abs(mul(u2, 24634.6345)))

        eps = 1e-6
        u1 = clamp(u1, eps, 1.0 - eps)
        u2 = clamp(u2, eps, 1.0 - eps)

        # Box-Muller transform: z ~ N(0, 1).
        r = sqrt(mul(neg(log(u1)), 2.0))
        phi = mul(u2, 6.283185307179586)
        z = mul(r, cos(phi))
        z = reshape(z, a.shape)

        if float(std) != 1.0:
            z = mul(z, float(std))
        if float(mean) != 0.0:
            z = _npu_add_scalar_(z, float(mean))
        if a.dtype != float_dtype:
            z = _cast_tensor_dtype(z, a.dtype)
        return copy_(a, z)

    if generator is not None and hasattr(generator, 'philox_engine_inputs'):
        seed, offset = generator.philox_engine_inputs(10)
    else:
        from .... import npu as npu_mod
        seed, offset = npu_mod._get_and_advance_offset(device_index=(a.device.index or 0), increment=10)

    a_storage = _unwrap_storage(a)
    aclnn.inplace_normal(
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        float(mean),
        float(std),
        seed,
        offset,
        runtime,
        stream=stream.stream,
    )
    return a


def randint_(a, low, high=None, generator=None):
    """In-place randint — fills tensor with random integers from [low, high)."""
    if high is None:
        low, high = 0, low
    uniform_(a, float(low), float(high), generator=generator)
    if _HAS_FAST_FLOOR_INPLACE:
        return _fast_floor_inplace_impl(a)
    raise RuntimeError("Cython NPU randint_ floor implementation is unavailable")


def random_(a, from_=0, to=None, generator=None):
    """In-place random — fills tensor with random values from [from_, to)."""
    import numpy as np
    from ...._dtype import to_numpy_dtype
    np_dtype = to_numpy_dtype(a.dtype)
    if to is None:
        if np.issubdtype(np_dtype, np.floating):
            to = 2**24 if np_dtype == np.float32 else 2**53
        else:
            to = int(np.iinfo(np_dtype).max) + 1
    uniform_(a, float(from_), float(to), generator=generator)
    if _HAS_FAST_FLOOR_INPLACE:
        return _fast_floor_inplace_impl(a)
    raise RuntimeError("Cython NPU random_ floor implementation is unavailable")


def bernoulli_(a, p=0.5, generator=None):
    """In-place Bernoulli — fills tensor with 0/1 from Bernoulli(p)."""
    uniform_(a, 0.0, 1.0, generator=generator)
    if not _HAS_FAST_COPY_INPLACE:
        raise RuntimeError("Cython NPU bernoulli_ implementation is unavailable")
    if not hasattr(p, 'storage'):
        p = _scalar_to_npu_tensor(float(p), a)
    out = _cast_tensor_dtype(lt(a, p), a.dtype)
    return _fast_copy_inplace_impl(a, out)


def exponential_(a, lambd=1.0, generator=None):
    """In-place exponential — fills with samples from Exp(lambd)."""
    uniform_(a, 0.0, 1.0, generator=generator)
    if not (_HAS_FAST_LOG_INPLACE and _HAS_FAST_NEG_INPLACE):
        raise RuntimeError("Cython NPU exponential_ implementation is unavailable")
    _fast_log_inplace_impl(a)
    _fast_neg_inplace_impl(a)
    if lambd != 1.0:
        if not _HAS_FAST_MUL_INPLACE:
            raise RuntimeError("Cython NPU exponential_ scale implementation is unavailable")
        scale = _scalar_to_npu_tensor(1.0 / lambd, a)
        return _fast_mul_inplace_impl(a, scale)
    return a


def log_normal_(a, mean=1.0, std=2.0, generator=None):
    """In-place log-normal — fills with exp(N(mean, std))."""
    normal_(a, mean, std, generator=generator)
    if _HAS_FAST_EXP_INPLACE:
        return _fast_exp_inplace_impl(a)
    raise RuntimeError("Cython NPU log_normal_ implementation is unavailable")


def cauchy_(a, median=0.0, sigma=1.0, generator=None):
    """In-place Cauchy — fills with median + sigma * tan(pi * (U - 0.5))."""
    import math
    uniform_(a, 0.0, 1.0, generator=generator)
    if not (_HAS_FAST_SUB_INPLACE and _HAS_FAST_MUL_INPLACE and _HAS_FAST_TAN_INPLACE):
        raise RuntimeError("Cython NPU cauchy_ implementation is unavailable")
    _fast_sub_inplace_impl(a, _scalar_to_npu_tensor(0.5, a))
    _fast_mul_inplace_impl(a, _scalar_to_npu_tensor(math.pi, a))
    _fast_tan_inplace_impl(a)
    if sigma != 1.0:
        _fast_mul_inplace_impl(a, _scalar_to_npu_tensor(sigma, a))
    if median != 0.0:
        if not _HAS_FAST_ADD_INPLACE:
            raise RuntimeError("Cython NPU cauchy_ median implementation is unavailable")
        _fast_add_inplace_impl(a, _scalar_to_npu_tensor(median, a))
    return a
def geometric_(a, p, generator=None):
    """In-place geometric — fills with ceil(ln(U) / ln(1-p))."""
    import math
    uniform_(a, 0.0, 1.0, generator=generator)
    if not (_HAS_FAST_LOG_INPLACE and _HAS_FAST_DIV_INPLACE and _HAS_FAST_CEIL_INPLACE):
        raise RuntimeError("Cython NPU geometric_ implementation is unavailable")
    _fast_log_inplace_impl(a)
    divisor = _scalar_to_npu_tensor(math.log(1.0 - float(p)), a)
    _fast_div_inplace_impl(a, divisor)
    return _fast_ceil_inplace_impl(a)


def fill_(a, value):
    """In-place fill using scalar tensor copy-back."""
    if _HAS_FAST_FILL_INPLACE:
        return _fast_fill_inplace_impl(a, value)
    raise RuntimeError("Cython NPU fill_ implementation is unavailable")


def clamp_(a, min_val=None, max_val=None):
    """In-place clamp: output written back to a's storage."""
    if _HAS_FAST_CLAMP_INPLACE:
        return _fast_clamp_inplace_impl(a, min_val, max_val)
    raise RuntimeError("Cython NPU clamp_ implementation is unavailable")


def copy_(a, src):
    """In-place copy from src into a."""
    if _HAS_FAST_COPY_INPLACE:
        return _fast_copy_inplace_impl(a, src)
    raise RuntimeError("Cython NPU copy_ implementation is unavailable")


def erfinv_(a):
    """In-place erfinv using aclnnErfinv."""
    return _fast_erfinv_inplace_impl(a)


def reciprocal_(a):
    """In-place reciprocal: output written back to a's storage."""
    if not _HAS_FAST_COPY_INPLACE:
        raise RuntimeError("Cython NPU reciprocal_ implementation is unavailable")
    from .math import pow as pow_op
    out = pow_op(a, -1.0)
    return _fast_copy_inplace_impl(a, out)
