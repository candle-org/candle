"""Candle's C extension layer — Cython accelerators for hot paths.

This package provides Cython implementations of performance-critical code
paths (dispatcher, allocator, storage creation, NPU ops, ACLNN FFI,
TensorImpl, dispatcher core, device, dtype, autograd node, fast ops).
Each module has a pure-Python fallback so the framework works without
a C compiler.

Feature flags (set after import):
    _HAS_CYTHON_DISPATCH   — True if _dispatch.pyx compiled successfully
    _HAS_CYTHON_ALLOCATOR  — True if _allocator.pyx compiled successfully
    _HAS_CYTHON_STORAGE    — True if _storage.pyx compiled successfully
    _HAS_CYTHON_NPU_OPS    — True if _npu_ops.pyx compiled successfully
    _HAS_CYTHON_ACLNN_FFI  — True if _aclnn_ffi.pyx compiled successfully
    _HAS_CYTHON_TENSOR_IMPL — True if _tensor_impl.pyx compiled successfully
    _HAS_CYTHON_DISPATCHER_CORE — True if _dispatcher_core.pyx compiled
    _HAS_CYTHON_DEVICE     — True if _device.pyx compiled successfully
    _HAS_CYTHON_DTYPE      — True if _dtype.pyx compiled successfully
    _HAS_CYTHON_AUTOGRAD_NODE — True if _autograd_node.pyx compiled
    _HAS_CYTHON_FAST_OPS   — True if _fast_ops.pyx compiled successfully
"""

_HAS_CYTHON_DISPATCH = False
_HAS_CYTHON_ALLOCATOR = False
_HAS_CYTHON_STORAGE = False
_HAS_CYTHON_NPU_OPS = False
_HAS_CYTHON_ACLNN_FFI = False
_HAS_CYTHON_TENSOR_IMPL = False
_HAS_CYTHON_DISPATCHER_CORE = False
_HAS_CYTHON_DEVICE = False
_HAS_CYTHON_DTYPE = False
_HAS_CYTHON_AUTOGRAD_NODE = False
_HAS_CYTHON_FAST_OPS = False

try:
    from ._dispatch import cy_dispatch, cy_dispatch_with_keyset  # noqa: F401
    _HAS_CYTHON_DISPATCH = True
except ImportError:
    pass

try:
    from ._allocator import FastNpuAllocator  # noqa: F401
    _HAS_CYTHON_ALLOCATOR = True
except ImportError:
    pass

try:
    from ._storage import cy_npu_storage_from_ptr  # noqa: F401
    _HAS_CYTHON_STORAGE = True
except ImportError:
    pass

try:
    from ._npu_ops import fast_binary_op  # noqa: F401
    _HAS_CYTHON_NPU_OPS = True
except ImportError:
    pass

try:
    from ._aclnn_ffi import (  # noqa: F401
        init as aclnn_ffi_init,
        create_tensor, destroy_tensor,
        create_scalar, destroy_scalar,
        create_int_array, destroy_int_array,
        destroy_executor, resolve_op, execute,
        binary_op_with_alpha, binary_op_no_alpha,
    )
    _HAS_CYTHON_ACLNN_FFI = True
except ImportError:
    pass

try:
    from ._tensor_impl import TensorImpl, _VersionCounterProxy  # noqa: F401
    _HAS_CYTHON_TENSOR_IMPL = True
except ImportError:
    pass

try:
    from ._dispatcher_core import cy_dispatch_with_keyset_fast  # noqa: F401
    _HAS_CYTHON_DISPATCHER_CORE = True
except ImportError:
    pass

try:
    from ._device import FastDevice  # noqa: F401
    _HAS_CYTHON_DEVICE = True
except ImportError:
    pass

try:
    from ._dtype import FastDType  # noqa: F401
    _HAS_CYTHON_DTYPE = True
except ImportError:
    pass

try:
    from ._autograd_node import FastNode  # noqa: F401
    _HAS_CYTHON_AUTOGRAD_NODE = True
except ImportError:
    pass

try:
    from ._fast_ops import add, mul, matmul  # noqa: F401
    _HAS_CYTHON_FAST_OPS = True
except ImportError:
    pass
