"""Backward-compatibility shim.  All storage symbols now live in candle.storage
and candle._C.  This module exists only so that pre-compiled Cython extensions
that import from candle._storage continue to work without recompilation."""
from .storage import *
from ._C import (
    typed_storage_from_numpy, empty_cpu_typed_storage,
    meta_typed_storage_from_shape, meta_typed_storage_from_size,
    npu_typed_storage_from_ptr, mps_typed_storage_from_numpy,
    mps_typed_storage_from_ptr, cuda_typed_storage_from_numpy,
    empty_cuda_typed_storage, cuda_typed_storage_to_numpy,
    pinned_cpu_typed_storage_from_numpy, PendingStorage,
    FloatStorage, DoubleStorage, HalfStorage, LongStorage,
    IntStorage, ShortStorage, ByteStorage, BoolStorage,
    BFloat16Storage, ComplexFloatStorage, ComplexDoubleStorage,
    Storage,
)
