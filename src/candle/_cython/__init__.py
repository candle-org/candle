"""Candle's C extension layer — Cython accelerators for hot paths.

This package provides Cython implementations of performance-critical code
paths (dispatcher, allocator, storage creation, NPU ops, ACLNN FFI,
TensorImpl, dispatcher core, device, dtype, autograd node, autograd graph,
autograd function, autograd ops, functional wrappers, fast ops, tensor API).

Feature flags (set after import):
    _HAS_CYTHON_DISPATCH        — True if _dispatch.pyx compiled successfully
    _HAS_CYTHON_ALLOCATOR       — True if _allocator.pyx compiled successfully
    _HAS_CYTHON_STORAGE         — True if _storage.pyx compiled successfully
    _HAS_CYTHON_NPU_OPS         — True if _npu_ops.pyx compiled successfully
    _HAS_CYTHON_ACLNN_FFI       — True if _aclnn_ffi.pyx compiled successfully
    _HAS_CYTHON_DISPATCHER_CORE — True if _dispatcher_core.pyx compiled
    _HAS_CYTHON_DEVICE          — True if _device.pyx compiled successfully
    _HAS_CYTHON_DTYPE           — True if _dtype.pyx compiled successfully
    _HAS_CYTHON_AUTOGRAD_NODE   — always True (hard import, no fallback)
    _HAS_CYTHON_AUTOGRAD_GRAPH  — always True (hard import, no fallback)
    _HAS_CYTHON_AUTOGRAD_ENGINE — always True (hard import, no fallback)
    _HAS_CYTHON_AUTOGRAD_FUNCTION — always True (hard import, no fallback)
    _HAS_CYTHON_AUTOGRAD_OPS    — always True (hard import, no fallback)
    _HAS_CYTHON_FUNCTIONAL_OPS  — True if _functional_ops.pyx compiled successfully
    _HAS_CYTHON_FAST_OPS        — True if _fast_ops.pyx compiled successfully
    _HAS_CYTHON_TENSOR_API      — True if _tensor_api.pyx compiled successfully
"""

_HAS_CYTHON_DISPATCH = False
_HAS_CYTHON_ALLOCATOR = False
_HAS_CYTHON_STORAGE = False
_HAS_CYTHON_NPU_OPS = False
_HAS_CYTHON_ACLNN_FFI = False
_HAS_CYTHON_DISPATCHER_CORE = False
_HAS_CYTHON_DEVICE = False
_HAS_CYTHON_DTYPE = False
# Autograd core modules are required (hard imports below -- no fallback).
_HAS_CYTHON_AUTOGRAD_NODE = False
_HAS_CYTHON_AUTOGRAD_GRAPH = False
_HAS_CYTHON_AUTOGRAD_ENGINE = False
_HAS_CYTHON_AUTOGRAD_FUNCTION = False
_HAS_CYTHON_AUTOGRAD_OPS = False
_HAS_CYTHON_FUNCTIONAL_OPS = False
_HAS_CYTHON_FAST_OPS = False
_HAS_CYTHON_TENSOR_API = False

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

from ._tensor_impl import TensorImpl, _VersionCounterProxy  # noqa: F401  # pylint: disable=import-error,no-name-in-module

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
    from ._autograd_node import (  # pylint: disable=import-error,no-name-in-module
        AccumulateGrad,  # noqa: F401
        FastNode,  # noqa: F401
        InputMetadata,  # noqa: F401
        Node,  # noqa: F401
        SavedTensor,  # noqa: F401
        _NodeHookHandle,  # noqa: F401
        _SavedValue,  # noqa: F401
    )
    _HAS_CYTHON_AUTOGRAD_NODE = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_NODE = False

from ._autograd_graph import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
    GradientEdge,
    current_saved_tensors_hooks,
    get_gradient_edge,
    saved_tensors_hooks,
)
_HAS_CYTHON_AUTOGRAD_GRAPH = True

from ._autograd_engine import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
    _GraphTask,
    _build_dependencies,
    _run_backward,
    backward,
    current_anomaly_parent,
    grad,
    is_anomaly_check_nan_enabled,
    is_anomaly_enabled,
    is_create_graph_enabled,
    pop_anomaly_config,
    pop_evaluating_node,
    push_anomaly_config,
    push_evaluating_node,
)
_HAS_CYTHON_AUTOGRAD_ENGINE = True

from ._autograd_function import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
    FunctionCtx,
    _function_apply,
)
_HAS_CYTHON_AUTOGRAD_FUNCTION = True

from ._autograd_ops import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
    _strip_autograd_keys,
    _grad_context,
    _backward_dispatch_keyset,
    _autograd_unary_passthrough,
    _autograd_binary,
    _autograd_binary_args,
    _autograd_unary_args,
    _norm_extract_weight_bias,
    _autograd_norm,
)
_HAS_CYTHON_AUTOGRAD_OPS = True

try:
    from ._functional_ops import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
        _has_torch_function as cy_has_torch_function,
        _handle_torch_function as cy_handle_torch_function,
        add as functional_add,
        mul as functional_mul,
        matmul as functional_matmul,
        relu as functional_relu,
        transpose as functional_transpose,
        reshape as functional_reshape,
        neg as functional_neg,
    )
    _HAS_CYTHON_FUNCTIONAL_OPS = True
except ImportError:
    pass

try:
    import importlib
    _legacy_fast_ops = importlib.import_module(f"{__name__}._fast_ops")  # noqa: F401
    _HAS_CYTHON_FAST_OPS = True
except ImportError:
    pass

try:
    from ._tensor_api import (  # noqa: F401  # pylint: disable=import-error,no-name-in-module
        tensor_add,
        tensor_backward,
        tensor_clone,
        tensor_detach,
        tensor_dim,
        tensor_iadd,
        tensor_imul,
        tensor_matmul,
        tensor_mul,
        tensor_neg,
        tensor_relu,
        tensor_reshape,
        tensor_size,
        tensor_to,
        tensor_transpose,
        tensor_view,
        tensor_sub,
    )
    _HAS_CYTHON_TENSOR_API = True
except ImportError:
    pass
