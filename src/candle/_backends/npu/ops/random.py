"""Random number generation and in-place initialization for NPU."""
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
from .comparison import lt
from .elementwise import clamp
from .math import abs, add, ceil, cos, div, exp, floor, frac, log, mul, neg, sin, sqrt, tan


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
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU zero_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    aclnn.inplace_zero(
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    return a


def uniform_(a, low=0.0, high=1.0, generator=None):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU uniform_ expects NPU tensors")

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
    if a.device.type != "npu":
        raise ValueError("NPU normal_ expects NPU tensors")

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
        u1 = sin(add(mul(idx, 12.9898), seed_mod * 78.233))
        u1 = frac(abs(mul(u1, 43758.5453)))
        u2 = sin(add(mul(add(idx, 0.5), 93.9898), seed_mod * 67.345))
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
            z = add(z, float(mean))
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
    # Fill with uniform [low, high), then floor to get integers
    uniform_(a, float(low), float(high), generator=generator)
    # In-place floor
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    aclnn.floor(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return a


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
    # Fill with uniform [from_, to), then floor
    uniform_(a, float(from_), float(to), generator=generator)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    aclnn.floor(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return a


def bernoulli_(a, p=0.5, generator=None):
    """In-place Bernoulli — fills tensor with 0/1 from Bernoulli(p)."""
    uniform_(a, 0.0, 1.0, generator=generator)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    numel = _numel(a.shape)
    if hasattr(p, 'storage'):
        p_storage = _unwrap_storage(p)
        p_shape, p_stride = p.shape, p.stride
    else:
        p_tensor = _scalar_to_npu_tensor(float(p), a)
        p_storage = _unwrap_storage(p_tensor)
        p_shape, p_stride = p_tensor.shape, p_tensor.stride
    bool_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize("bool"), runtime=runtime)
    aclnn.lt(a_storage.data_ptr(), p_storage.data_ptr(), bool_ptr,
             a.shape, a.stride, p_shape, p_stride, a.shape, a.stride,
             a.dtype, runtime, stream=stream.stream)
    aclnn.cast(bool_ptr, a_storage.data_ptr(), a.shape, a.stride, "bool", a.dtype, runtime, stream=stream.stream)
    runtime.defer_free(bool_ptr)
    return a


def exponential_(a, lambd=1.0, generator=None):
    """In-place exponential — fills with samples from Exp(lambd)."""
    uniform_(a, 0.0, 1.0, generator=generator)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    aclnn.log(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    aclnn.neg(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    if lambd != 1.0:
        scale = _scalar_to_npu_tensor(1.0 / lambd, a)
        scale_storage = _unwrap_storage(scale)
        numel = _numel(a.shape)
        tmp_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(a.dtype), runtime=runtime)
        aclnn.mul(a_storage.data_ptr(), scale_storage.data_ptr(), tmp_ptr,
                  a.shape, a.stride, scale.shape, scale.stride, a.shape, a.stride,
                  a.dtype, runtime, stream=stream.stream)
        aclnn.inplace_copy(a_storage.data_ptr(), tmp_ptr, a.shape, a.stride, a.dtype, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
        runtime.defer_free(tmp_ptr)
    return a


def log_normal_(a, mean=1.0, std=2.0, generator=None):
    """In-place log-normal — fills with exp(N(mean, std))."""
    normal_(a, mean, std, generator=generator)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    aclnn.exp(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return a


def cauchy_(a, median=0.0, sigma=1.0, generator=None):
    """In-place Cauchy — fills with median + sigma * tan(pi * (U - 0.5))."""
    import math
    uniform_(a, 0.0, 1.0, generator=generator)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    numel = _numel(a.shape)
    # sub 0.5
    aclnn.sub_scalar(a_storage.data_ptr(), 0.5, a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    # mul pi
    pi_tensor = _scalar_to_npu_tensor(math.pi, a)
    pi_storage = _unwrap_storage(pi_tensor)
    tmp_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(a.dtype), runtime=runtime)
    aclnn.mul(a_storage.data_ptr(), pi_storage.data_ptr(), tmp_ptr,
              a.shape, a.stride, pi_tensor.shape, pi_tensor.stride, a.shape, a.stride,
              a.dtype, runtime, stream=stream.stream)
    aclnn.inplace_copy(a_storage.data_ptr(), tmp_ptr, a.shape, a.stride, a.dtype, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    runtime.defer_free(tmp_ptr)
    # tan in-place
    aclnn.tan(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    # mul sigma
    if sigma != 1.0:
        sigma_tensor = _scalar_to_npu_tensor(sigma, a)
        sigma_storage = _unwrap_storage(sigma_tensor)
        tmp_ptr2 = npu_runtime._alloc_device(numel * _dtype_itemsize(a.dtype), runtime=runtime)
        aclnn.mul(a_storage.data_ptr(), sigma_storage.data_ptr(), tmp_ptr2,
                  a.shape, a.stride, sigma_tensor.shape, sigma_tensor.stride, a.shape, a.stride,
                  a.dtype, runtime, stream=stream.stream)
        aclnn.inplace_copy(a_storage.data_ptr(), tmp_ptr2, a.shape, a.stride, a.dtype, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
        runtime.defer_free(tmp_ptr2)
    # add median
    if median != 0.0:
        aclnn.add_scalar(a_storage.data_ptr(), median, a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return a


def geometric_(a, p, generator=None):
    """In-place geometric — fills with ceil(ln(U) / ln(1-p))."""
    import math
    uniform_(a, 0.0, 1.0, generator=generator)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    aclnn.log(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    # divide by log(1-p)
    log_1_minus_p = math.log(1.0 - float(p))
    divisor = _scalar_to_npu_tensor(log_1_minus_p, a)
    divisor_storage = _unwrap_storage(divisor)
    numel = _numel(a.shape)
    tmp_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(a.dtype), runtime=runtime)
    aclnn.div(a_storage.data_ptr(), divisor_storage.data_ptr(), tmp_ptr,
              a.shape, a.stride, divisor.shape, divisor.stride, a.shape, a.stride,
              a.dtype, runtime, stream=stream.stream)
    aclnn.inplace_copy(a_storage.data_ptr(), tmp_ptr, a.shape, a.stride, a.dtype, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    runtime.defer_free(tmp_ptr)
    # ceil in-place
    aclnn.ceil(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return a


def fill_(a, value):
    """In-place fill using aclnnInplaceFillScalar."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU fill_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    aclnn.inplace_fill_scalar(
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        float(value),
        runtime,
        stream=stream.stream,
    )
    return a


def clamp_(a, min_val=None, max_val=None):
    """In-place clamp: output written back to a's storage."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU clamp_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    # Use clamp_scalar with output == input for in-place
    aclnn.clamp_scalar(
        a_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        min_val,
        max_val,
        runtime,
        stream=stream.stream,
    )
    return a


def copy_(a, src):
    """In-place copy from src into a."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU copy_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    src_storage = _unwrap_storage(src)
    aclnn.inplace_copy(
        a_storage.data_ptr(),
        src_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        src.shape,
        src.stride,
        src.dtype,
        runtime,
        stream=stream.stream,
    )
    return a


def erfinv_(a):
    """In-place erfinv using aclnnErfinv."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU erfinv_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    # erfinv: output to same storage for in-place
    aclnn.erfinv(
        a_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    return a


def reciprocal_(a):
    return _unary_op(a, aclnn.reciprocal, "reciprocal")
