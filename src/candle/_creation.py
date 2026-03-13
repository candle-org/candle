import numpy as np

from ._dtype import float32
from ._dtype import bool as bool_dtype
from ._functional import tensor as tensor_dispatch
from ._functional import zeros as zeros_dispatch
from ._functional import ones as ones_dispatch
from ._functional import empty as empty_dispatch
from ._functional import arange as arange_dispatch
from ._functional import linspace as linspace_dispatch
from ._functional import full as full_dispatch
from ._functional import logspace as logspace_dispatch
from ._functional import eye as eye_dispatch
from ._functional import range as range_dispatch
from ._functional import randn as randn_dispatch
from ._functional import rand as rand_dispatch
from ._functional import randint as randint_dispatch
from ._functional import randperm as randperm_dispatch
from ._functional import normal as normal_dispatch


def _infer_creation_dtype(data):
    if isinstance(data, (np.ndarray, np.generic)):
        return bool_dtype if data.dtype == np.bool_ else None
    if hasattr(data, "dtype"):
        return None
    try:
        arr = np.asarray(data)
    except Exception:
        return None
    return bool_dtype if arr.dtype == np.bool_ else None


def tensor(data, *, dtype=None, device=None, requires_grad=False):
    if dtype is None:
        dtype = _infer_creation_dtype(data)
    if dtype is None:
        from . import get_default_dtype
        dtype = get_default_dtype()
    return tensor_dispatch(data, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(*shape, dtype=float32, device=None, memory_format=None):
    return zeros_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format)


def ones(*shape, dtype=float32, device=None, memory_format=None):
    return ones_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format)


def empty(*shape, dtype=float32, device=None, memory_format=None):
    return empty_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format)


def arange(start, end=None, step=1, dtype=float32, device=None):
    return arange_dispatch(start, end=end, step=step, dtype=dtype, device=device)


def linspace(start, end, steps, dtype=float32, device=None):
    return linspace_dispatch(start, end, steps, dtype=dtype, device=device)


def full(*args, dtype=float32, device=None):
    return full_dispatch(*args, dtype=dtype, device=device)


def logspace(start, end, steps, dtype=float32, device=None):
    return logspace_dispatch(start, end, steps, dtype=dtype, device=device)


def eye(n, m=None, dtype=float32, device=None, out=None, requires_grad=False):
    return eye_dispatch(n, m, dtype=dtype, device=device, out=out)


def range(start, end, step=1, dtype=float32, device=None):
    return range_dispatch(start, end, step=step, dtype=dtype, device=device)


def randn(*shape, dtype=float32, device=None, memory_format=None, generator=None, requires_grad=False):
    out = randn_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format, generator=generator)
    if requires_grad:
        out.requires_grad_(True)
    return out


def rand(*shape, dtype=float32, device=None, memory_format=None, generator=None, requires_grad=False):
    out = rand_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format, generator=generator)
    if requires_grad:
        out.requires_grad_(True)
    return out


def randint(low, high=None, size=None, *, dtype=None, device=None, generator=None):
    return randint_dispatch(low, high=high, size=size, dtype=dtype, device=device, generator=generator)


def randperm(n, *, dtype=None, device=None, generator=None):
    return randperm_dispatch(n, dtype=dtype, device=device, generator=generator)


def from_numpy(ndarray):
    from ._dtype import (
        float16, float32, float64, int8, int16, int32, int64,
        uint8, bool as bool_dtype, bfloat16,
    )
    _numpy_to_dtype = {
        np.float16: float16, np.float32: float32, np.float64: float64,
        np.int8: int8, np.int16: int16, np.int32: int32, np.int64: int64,
        np.uint8: uint8, np.bool_: bool_dtype,
    }
    dt = _numpy_to_dtype.get(ndarray.dtype.type, float32)
    return tensor_dispatch(ndarray, dtype=dt)


def as_tensor(data, dtype=None, device=None):
    from ._tensor import Tensor

    if isinstance(data, Tensor):
        if dtype is None and device is None:
            return data
        return data.to(device=device, dtype=dtype)

    if dtype is None:
        dtype = _infer_creation_dtype(data)
    if dtype is None:
        from . import get_default_dtype
        dtype = get_default_dtype()
    return tensor_dispatch(data, dtype=dtype, device=device)


def normal(mean, std, size=None, *, generator=None, out=None):
    return normal_dispatch(mean, std, size=size, generator=generator, out=out)
