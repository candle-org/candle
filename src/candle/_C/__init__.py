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
    _HAS_CYTHON_STORAGE_IMPL    — True if _storage_impl.pyx compiled successfully
"""

from ._bootstrap import (
    _HAS_CYTHON_ACLNN_FFI,
    _HAS_CYTHON_ALLOCATOR,
    _HAS_CYTHON_AUTOGRAD_ENGINE,
    _HAS_CYTHON_AUTOGRAD_FUNCTION,
    _HAS_CYTHON_AUTOGRAD_GRAPH,
    _HAS_CYTHON_AUTOGRAD_NODE,
    _HAS_CYTHON_AUTOGRAD_OPS,
    _HAS_CYTHON_DEVICE,
    _HAS_CYTHON_DISPATCH,
    _HAS_CYTHON_DISPATCHER_CORE,
    _HAS_CYTHON_DTYPE,
    _HAS_CYTHON_FAST_OPS,
    _HAS_CYTHON_FUNCTIONAL_OPS,
    _HAS_CYTHON_NPU_OPS,
    _HAS_CYTHON_STORAGE,
    _HAS_CYTHON_STORAGE_IMPL,
    _HAS_CYTHON_TENSOR_API,
    AccumulateGrad,
    FastDevice,
    FastDType,
    FastNpuAllocator,
    FastNode,
    FunctionCtx,
    GradientEdge,
    InputMetadata,
    Node,
    SavedTensor,
    StorageImpl,
    TensorBase,
    TensorImpl,
    _StrideTuple,
    _TensorBase,
    _bf16_to_f32,
    _compute_strides,
    _f32_to_bf16,
    _install_tensor_api,
    aclnn_ffi_init,
    backward,
    binary_op_no_alpha,
    binary_op_with_alpha,
    create_int_array,
    create_scalar,
    create_tensor,
    cy_dispatch,
    cy_dispatch_with_keyset,
    cy_dispatch_with_keyset_fast,
    destroy_executor,
    destroy_int_array,
    destroy_scalar,
    destroy_tensor,
    execute,
    fast_binary_op,
    get_gradient_edge,
    grad,
    is_anomaly_check_nan_enabled,
    is_anomaly_enabled,
    is_create_graph_enabled,
    pop_anomaly_config,
    pop_evaluating_node,
    push_anomaly_config,
    push_evaluating_node,
    resolve_op,
    saved_tensors_hooks,
)


# =============================================================================
# torch._C stubs and TensorBase (from _C_stubs.py)
# =============================================================================

from ._stubs import (
    _add_docstr,
    _disabled_torch_dispatch_impl,
    DisableTorchFunctionSubclass,
    _has_storage,
    _get_tracing_state,
    _get_privateuse,
    _dlpack_exchange_api,
    _to_dlpack,
    _to_dlpack_versioned,
    _torch_function_enabled,
    _VariableFunctions,
    _get_PyTorchFileReader,
    _get_PyTorchFileWriter,
    _get_privateuse1_backend_name,
)

from ._Storage import (
    StorageBase,
    _MPSUntypedStorage,
    PendingStorage,
    typed_storage_from_numpy,
    empty_cpu_typed_storage,
    meta_typed_storage_from_shape,
    meta_typed_storage_from_size,
    npu_typed_storage_from_ptr,
    mps_typed_storage_from_numpy,
    mps_typed_storage_from_ptr,
    cuda_typed_storage_from_numpy,
    empty_cuda_typed_storage,
    cuda_typed_storage_to_numpy,
    pinned_cpu_typed_storage_from_numpy,
    _get_storage_classes,
    _install_typed_storage_compat,
    _make_legacy_classes,
    __getattr__ as _storage_getattr,
)


def __getattr__(name):
    return _storage_getattr(name)


if _HAS_CYTHON_TENSOR_API:
    _install_tensor_api(TensorBase)

    # Re-export the 11 tensor bootstrap helpers that contract tests expect
    # directly on candle._C (analogous to torch._C)._tensor_api helpers
    import candle._C._tensor_api as _tapi  # pylint: disable=import-error  # pylint: disable=import-error

    tensor_set_device_from_storage = _tapi.tensor_set_device_from_storage
    tensor_set_dtype_from_storage = _tapi.tensor_set_dtype_from_storage
    tensor_set_data = _tapi.tensor_set_data
    tensor_delattr = _tapi.tensor_delattr
    tensor_fw_get = _tapi.tensor_fw_get
    tensor_fw_set = _tapi.tensor_fw_set
    tensor_fw_clear = _tapi.tensor_fw_clear
    tensor_fw_has = _tapi.tensor_fw_has
    tensor_untyped_storage = _tapi.tensor_untyped_storage
    tensor_record_stream = _tapi.tensor_record_stream
    tensor_is_pinned = _tapi.tensor_is_pinned
