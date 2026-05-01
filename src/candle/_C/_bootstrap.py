"""Cython feature-flag bootstrap for candle._C package exports."""
# pylint: disable=import-error,no-name-in-module,possibly-unused-variable

# Default placeholders so package-level re-exports stay importable even when optional modules fail.
cy_dispatch = None
cy_dispatch_with_keyset = None
FastNpuAllocator = None
cy_npu_storage_from_ptr = None
fast_binary_op = None
aclnn_ffi_init = None
create_tensor = None
destroy_tensor = None
create_scalar = None
destroy_scalar = None
create_int_array = None
destroy_int_array = None
destroy_executor = None
resolve_op = None
execute = None
binary_op_with_alpha = None
binary_op_no_alpha = None
cy_dispatch_with_keyset_fast = None
FastDevice = None
FastDType = None
AccumulateGrad = None
FastNode = None
InputMetadata = None
Node = None
SavedTensor = None
_NodeHookHandle = None
_SavedValue = None
GradientEdge = None
current_saved_tensors_hooks = None
get_gradient_edge = None
saved_tensors_hooks = None
_GraphTask = None
_build_dependencies = None
_run_backward = None
backward = None
current_anomaly_parent = None
grad = None
is_anomaly_check_nan_enabled = None
is_anomaly_enabled = None
is_create_graph_enabled = None
pop_anomaly_config = None
pop_evaluating_node = None
push_anomaly_config = None
push_evaluating_node = None
FunctionCtx = None
_function_apply = None
_strip_autograd_keys = None
_grad_context = None
_backward_dispatch_keyset = None
_autograd_unary_passthrough = None
_autograd_binary = None
_autograd_binary_args = None
_autograd_unary_args = None
_norm_extract_weight_bias = None
_autograd_norm = None
cy_has_torch_function = None
cy_handle_torch_function = None
functional_add = None
functional_mul = None
functional_matmul = None
functional_relu = None
functional_transpose = None
functional_reshape = None
functional_neg = None
_legacy_fast_ops = None
TensorBase = None
_TensorBase = None
_StrideTuple = None
_bf16_to_f32 = None
_compute_strides = None
_f32_to_bf16 = None
_install_tensor_api = None
StorageImpl = None

_HAS_CYTHON_DISPATCH = False
_HAS_CYTHON_ALLOCATOR = False
_HAS_CYTHON_STORAGE = False
_HAS_CYTHON_NPU_OPS = False
_HAS_CYTHON_ACLNN_FFI = False
_HAS_CYTHON_DISPATCHER_CORE = False
_HAS_CYTHON_DEVICE = False
_HAS_CYTHON_DTYPE = False
_HAS_CYTHON_AUTOGRAD_NODE = False
_HAS_CYTHON_AUTOGRAD_GRAPH = False
_HAS_CYTHON_AUTOGRAD_ENGINE = False
_HAS_CYTHON_AUTOGRAD_FUNCTION = False
_HAS_CYTHON_AUTOGRAD_OPS = False
_HAS_CYTHON_GRAD_MODE_STATE = False
_HAS_CYTHON_FORWARD_AD = False
_HAS_CYTHON_FUNCTIONAL_OPS = False
_HAS_CYTHON_FAST_OPS = False
_HAS_CYTHON_TENSOR_API = False
_HAS_CYTHON_STORAGE_IMPL = False

try:
    from candle._dispatch import cy_dispatch, cy_dispatch_with_keyset  # noqa: F401
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
        create_tensor,
        destroy_tensor,
        create_scalar,
        destroy_scalar,
        create_int_array,
        destroy_int_array,
        destroy_executor,
        resolve_op,
        execute,
        binary_op_with_alpha,
        binary_op_no_alpha,
    )
    _HAS_CYTHON_ACLNN_FFI = True
except ImportError:
    pass

from ._tensor_impl import TensorImpl, _VersionCounterProxy  # noqa: F401

try:
    from ._dispatcher_core import cy_dispatch_with_keyset_fast  # noqa: F401
    _HAS_CYTHON_DISPATCHER_CORE = True
except ImportError:
    pass

try:
    from candle._device import FastDevice  # noqa: F401
    _HAS_CYTHON_DEVICE = True
except ImportError:
    pass

try:
    from candle._dtype import FastDType  # noqa: F401
    _HAS_CYTHON_DTYPE = True
except ImportError:
    pass

try:
    from ._autograd_node import (  # noqa: F401
        AccumulateGrad,
        FastNode,
        InputMetadata,
        Node,
        SavedTensor,
        _NodeHookHandle,
        _SavedValue,
    )
    _HAS_CYTHON_AUTOGRAD_NODE = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_NODE = False

try:
    from ._autograd_graph import (  # noqa: F401
        GradientEdge,
        current_saved_tensors_hooks,
        get_gradient_edge,
        saved_tensors_hooks,
    )
    _HAS_CYTHON_AUTOGRAD_GRAPH = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_GRAPH = False

try:
    from ._autograd_engine import (  # noqa: F401
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
except ImportError:
    _HAS_CYTHON_AUTOGRAD_ENGINE = False

try:
    from ._autograd_function import (  # noqa: F401
        FunctionCtx,
        _function_apply,
    )
    _HAS_CYTHON_AUTOGRAD_FUNCTION = True
except ImportError:
    _HAS_CYTHON_AUTOGRAD_FUNCTION = False

try:
    from ._autograd_ops import (  # noqa: F401
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
except ImportError:
    _HAS_CYTHON_AUTOGRAD_OPS = False

try:
    from ._grad_mode_state import (  # noqa: F401
        GradMode,
        current_creation_mode,
        enable_grad,
        get_creation_mode,
        get_enabled,
        inference_mode,
        is_grad_enabled,
        no_grad,
        set_creation_mode,
        set_enabled,
        set_grad_enabled,
    )
    _HAS_CYTHON_GRAD_MODE_STATE = True
except ImportError:
    _HAS_CYTHON_GRAD_MODE_STATE = False

try:
    from ._forward_ad import (  # noqa: F401
        _JVP_RULES,
        _STATE as _FORWARD_AD_STATE,
        _current_level,
        _disabled_levels,
        _level_stack,
        dual_level,
        enter_dual_level,
        exit_dual_level,
        get_jvp,
        get_tangent,
        is_level_disabled,
        register_jvp,
        temporarily_disable,
    )
    _HAS_CYTHON_FORWARD_AD = True
except ImportError:
    _HAS_CYTHON_FORWARD_AD = False

try:
    from ._functional_ops import (  # noqa: F401
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

    _legacy_fast_ops = importlib.import_module(f"{__package__}._fast_ops")  # noqa: F401
    _HAS_CYTHON_FAST_OPS = True
except ImportError:
    pass

try:
    from ._TensorBase import (  # noqa: F401
        TensorBase,
        _TensorBase,
        _StrideTuple,
        _bf16_to_f32,
        _compute_strides,
        _f32_to_bf16,
        _install_tensor_api,
    )
    _HAS_CYTHON_TENSOR_API = True
except ImportError:
    _HAS_CYTHON_TENSOR_API = False

try:
    from ._storage_impl import StorageImpl  # noqa: F401
    _HAS_CYTHON_STORAGE_IMPL = True
except ImportError:
    pass
