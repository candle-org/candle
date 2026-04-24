import numpy as np

from ..._C import cuda_typed_storage_from_numpy, cuda_typed_storage_to_numpy
from ..._tensor import Tensor
from ..._cython._tensor_impl import cy_make_tensor_from_storage  # pylint: disable=import-error,no-name-in-module
from . import state as cuda_state


def _from_numpy(arr, dtype, device):
    storage = cuda_typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def _from_numpy_on_current_stream(arr, dtype, device, stream=None):
    storage = cuda_typed_storage_from_numpy(arr, dtype, device=device, stream=stream)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def add(a, b):
    stream = cuda_state.current_stream(a.device.index or 0).stream
    a_np = cuda_typed_storage_to_numpy(a.storage(), a.shape, a.dtype, stream=stream)
    b_np = cuda_typed_storage_to_numpy(b.storage(), b.shape, b.dtype, stream=stream) if isinstance(b, Tensor) else b
    out = np.ascontiguousarray(a_np + b_np)
    return _from_numpy_on_current_stream(out, a.dtype, a.device, stream=stream)
