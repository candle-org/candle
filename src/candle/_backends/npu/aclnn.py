import ctypes
import inspect
import atexit
import os
import struct

from .acl_loader import ensure_acl
from . import cann_discovery

# Try to import the Cython FFI hot-path. Legacy Python/ctypes fallback is disabled.
try:
    from ..._cython import _aclnn_ffi as _ffi
except ImportError:
    _ffi = None  # pylint: disable=invalid-name

acl = None


def _launch_blocking_enabled():
    value = os.getenv("ACL_LAUNCH_BLOCKING", "").strip().lower()
    return value not in ("", "0", "false", "no")


def _maybe_sync(runtime):
    if runtime is None:
        return
    if _launch_blocking_enabled():
        runtime.synchronize()


def _get_lib_dirs():
    """Get ACLNN library directories from auto-discovery."""
    return cann_discovery.get_lib_dirs()


def _get_lib_names():
    """Get library names appropriate for the detected CANN version."""
    return cann_discovery.get_aclnn_lib_names()

_LIB_HANDLES = None
_BINDINGS = None
_ACLNN_INITIALIZED = False
_ACLNN_FINALIZED = False
_DEFERRED_EXECUTORS = []
_DEFERRED_EXECUTOR_CLEANUP = {}
_CLEANUP_REGISTERED = False


def _require_native_npu_ffi(op_name):
    if _ffi is None or not _ffi.is_initialized():
        raise RuntimeError(f"native NPU hot path unavailable for op {op_name}")


def _infer_ctypes_disabled_op_name(default_name="legacy_acl_op"):
    helper_names = {
        "_require_ctypes_npu_path_disabled",
        "_infer_ctypes_disabled_op_name",
        "_create_tensor",
        "_create_scalar",
        "_create_tensor_list",
        "_create_tensor_list_with_nones",
        "_unary_call",
    }
    frame = inspect.currentframe()
    try:
        caller = frame.f_back if frame is not None else None
        while caller is not None:
            if caller.f_globals.get("__name__") == __name__:
                caller_name = caller.f_code.co_name
                if caller_name not in helper_names:
                    return caller_name
            caller = caller.f_back
    finally:
        del frame
    return default_name


def _require_ctypes_npu_path_disabled(op_name=None):
    op_name = _infer_ctypes_disabled_op_name() if op_name is None else op_name
    raise RuntimeError(
        f"native NPU hot path unavailable for op {op_name}; python/ctypes fallback is disabled"
    )


def _npu_runtime_alloc_device(size, runtime=None):
    from . import runtime as npu_runtime_module
    return npu_runtime_module._alloc_device(size, runtime=runtime)


class AclnnBindings:
    def __init__(self, libs):  # pylint: disable=too-many-statements
        self.libs = libs
        self.aclnn_init = _bind_symbol(
            libs,
            "aclnnInit",
            ctypes.c_int32,
            [ctypes.c_char_p],
        )
        self.aclnn_finalize = _optional_symbol(
            libs,
            "aclnnFinalize",
            ctypes.c_int32,
            [],
        )
        self.acl_create_tensor = _bind_symbol(
            libs,
            "aclCreateTensor",
            ctypes.c_void_p,
            [
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_uint64,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int64,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_uint64,
                ctypes.c_void_p,
            ],
        )
        self.acl_create_scalar = _bind_symbol(
            libs,
            "aclCreateScalar",
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_int32],
        )
        self.acl_create_int_array = _bind_symbol(
            libs,
            "aclCreateIntArray",
            ctypes.c_void_p,
            [ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64],
        )
        self.acl_destroy_tensor = _bind_symbol(
            libs,
            "aclDestroyTensor",
            ctypes.c_int32,
            [ctypes.c_void_p],
        )
        self.acl_destroy_scalar = _bind_symbol(
            libs,
            "aclDestroyScalar",
            ctypes.c_int32,
            [ctypes.c_void_p],
        )
        self.acl_destroy_int_array = _bind_symbol(
            libs,
            "aclDestroyIntArray",
            ctypes.c_int32,
            [ctypes.c_void_p],
        )
        self.acl_create_bool_array = _bind_symbol(
            libs,
            "aclCreateBoolArray",
            ctypes.c_void_p,
            [ctypes.POINTER(ctypes.c_bool), ctypes.c_uint64],
        )
        self.acl_destroy_bool_array = _bind_symbol(
            libs,
            "aclDestroyBoolArray",
            ctypes.c_int32,
            [ctypes.c_void_p],
        )
        self.acl_destroy_executor = _bind_symbol(
            libs,
            "aclDestroyAclOpExecutor",
            ctypes.c_int32,
            [ctypes.c_void_p],
        )
        self.aclnn_add_get_workspace = _bind_symbol(
            libs,
            "aclnnAddGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_add = _bind_symbol(
            libs,
            "aclnnAdd",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_arange_get_workspace = _optional_symbol(
            libs,
            "aclnnArangeGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_arange = _optional_symbol(
            libs,
            "aclnnArange",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_linspace_get_workspace = _optional_symbol(
            libs,
            "aclnnLinspaceGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_linspace = _optional_symbol(
            libs,
            "aclnnLinspace",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_eye_get_workspace = _optional_symbol(
            libs,
            "aclnnEyeGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_eye = _optional_symbol(
            libs,
            "aclnnEye",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_range_get_workspace = _optional_symbol(
            libs,
            "aclnnRangeGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_range = _optional_symbol(
            libs,
            "aclnnRange",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_flip_get_workspace = _optional_symbol(
            libs,
            "aclnnFlipGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_flip = _optional_symbol(
            libs,
            "aclnnFlip",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_roll_get_workspace = _optional_symbol(
            libs,
            "aclnnRollGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_roll = _optional_symbol(
            libs,
            "aclnnRoll",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_cumsum_get_workspace = _optional_symbol(
            libs,
            "aclnnCumsumGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_cumsum = _optional_symbol(
            libs,
            "aclnnCumsum",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_cumprod_get_workspace = _optional_symbol(
            libs,
            "aclnnCumprodGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_cumprod = _optional_symbol(
            libs,
            "aclnnCumprod",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_cummax_get_workspace = _optional_symbol(
            libs,
            "aclnnCummaxGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_cummax = _optional_symbol(
            libs,
            "aclnnCummax",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_argsort_get_workspace = _optional_symbol(
            libs,
            "aclnnArgsortGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_argsort = _optional_symbol(
            libs,
            "aclnnArgsort",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sort_get_workspace = _optional_symbol(
            libs,
            "aclnnSortGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_sort = _optional_symbol(
            libs,
            "aclnnSort",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_topk_get_workspace = _optional_symbol(
            libs,
            "aclnnTopkGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_topk = _optional_symbol(
            libs,
            "aclnnTopk",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_tril_get_workspace = _optional_symbol(
            libs,
            "aclnnTrilGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_tril = _optional_symbol(
            libs,
            "aclnnTril",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_triu_get_workspace = _optional_symbol(
            libs,
            "aclnnTriuGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_triu = _optional_symbol(
            libs,
            "aclnnTriu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_nonzero_get_workspace = _optional_symbol(
            libs,
            "aclnnNonzeroGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_nonzero = _optional_symbol(
            libs,
            "aclnnNonzero",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_repeat_get_workspace = _optional_symbol(
            libs,
            "aclnnRepeatGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_repeat = _optional_symbol(
            libs,
            "aclnnRepeat",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_repeat_interleave_int_get_workspace = _optional_symbol(
            libs,
            "aclnnRepeatInterleaveIntGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_repeat_interleave_int = _optional_symbol(
            libs,
            "aclnnRepeatInterleaveInt",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_repeat_interleave_int_with_dim_get_workspace = _optional_symbol(
            libs,
            "aclnnRepeatInterleaveIntWithDimGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_repeat_interleave_int_with_dim = _optional_symbol(
            libs,
            "aclnnRepeatInterleaveIntWithDim",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_repeat_interleave_get_workspace = _optional_symbol(
            libs,
            "aclnnRepeatInterleaveGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_repeat_interleave = _optional_symbol(
            libs,
            "aclnnRepeatInterleave",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_repeat_interleave_with_dim_get_workspace = _optional_symbol(
            libs,
            "aclnnRepeatInterleaveWithDimGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_repeat_interleave_with_dim = _optional_symbol(
            libs,
            "aclnnRepeatInterleaveWithDim",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_scatter_get_workspace = _optional_symbol(
            libs,
            "aclnnScatterGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_scatter = _optional_symbol(
            libs,
            "aclnnScatter",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_diag_get_workspace = _optional_symbol(
            libs,
            "aclnnDiagGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_diag = _optional_symbol(
            libs,
            "aclnnDiag",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_mul_get_workspace = _bind_symbol(
            libs,
            "aclnnMulGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_mul = _bind_symbol(
            libs,
            "aclnnMul",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_index_put_impl_get_workspace = _optional_symbol(
            libs,
            "aclnnIndexPutImplGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_index_put_impl = _optional_symbol(
            libs,
            "aclnnIndexPutImpl",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnIndex (advanced indexing getitem)
        self.aclnn_index_get_workspace = _optional_symbol(
            libs,
            "aclnnIndexGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_index = _optional_symbol(
            libs,
            "aclnnIndex",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnSlice (strided slicing on any dim)
        self.aclnn_slice_get_workspace = _optional_symbol(
            libs,
            "aclnnSliceGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_slice = _optional_symbol(
            libs,
            "aclnnSlice",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnInplaceMaskedFillScalar
        self.aclnn_inplace_masked_fill_scalar_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceMaskedFillScalarGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,      # aclTensor* selfRef
                ctypes.c_void_p,      # const aclTensor* mask
                ctypes.c_void_p,      # const aclScalar* value
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_masked_fill_scalar = _optional_symbol(
            libs,
            "aclnnInplaceMaskedFillScalar",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnInplaceIndexCopy
        self.aclnn_inplace_index_copy_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceIndexCopyGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,      # aclTensor* selfRef
                ctypes.c_int64,       # int64_t dim
                ctypes.c_void_p,      # const aclTensor* index
                ctypes.c_void_p,      # const aclTensor* source
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_index_copy = _optional_symbol(
            libs,
            "aclnnInplaceIndexCopy",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnInplaceIndexFill (scalar value variant)
        self.aclnn_inplace_index_fill_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceIndexFillGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,      # aclTensor* selfRef
                ctypes.c_int64,       # int64_t dim
                ctypes.c_void_p,      # const aclTensor* index
                ctypes.c_void_p,      # const aclScalar* value
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_index_fill = _optional_symbol(
            libs,
            "aclnnInplaceIndexFill",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnIndexAdd
        self.aclnn_index_add_get_workspace = _optional_symbol(
            libs,
            "aclnnIndexAddGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,      # const aclTensor* self
                ctypes.c_int64,       # int64_t dim
                ctypes.c_void_p,      # const aclTensor* index
                ctypes.c_void_p,      # const aclTensor* source
                ctypes.c_void_p,      # const aclScalar* alpha
                ctypes.c_void_p,      # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_index_add = _optional_symbol(
            libs,
            "aclnnIndexAdd",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnScatterAdd
        self.aclnn_scatter_add_get_workspace = _optional_symbol(
            libs,
            "aclnnScatterAddGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,      # const aclTensor* self
                ctypes.c_int64,       # int64_t dim
                ctypes.c_void_p,      # const aclTensor* index
                ctypes.c_void_p,      # const aclTensor* src
                ctypes.c_void_p,      # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_scatter_add = _optional_symbol(
            libs,
            "aclnnScatterAdd",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnInplaceMaskedScatter
        self.aclnn_inplace_masked_scatter_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceMaskedScatterGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,      # aclTensor* selfRef
                ctypes.c_void_p,      # const aclTensor* mask
                ctypes.c_void_p,      # const aclTensor* source
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_masked_scatter = _optional_symbol(
            libs,
            "aclnnInplaceMaskedScatter",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        self.aclnn_sub_get_workspace = _optional_symbol(
            libs,
            "aclnnSubGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_sub = _optional_symbol(
            libs,
            "aclnnSub",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_div_get_workspace = _optional_symbol(
            libs,
            "aclnnDivGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_div = _optional_symbol(
            libs,
            "aclnnDiv",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_add_scalar_get_workspace = _optional_symbol(
            libs,
            "aclnnAddsGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_add_scalar = _optional_symbol(
            libs,
            "aclnnAdds",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sub_scalar_get_workspace = _optional_symbol(
            libs,
            "aclnnSubsGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_sub_scalar = _optional_symbol(
            libs,
            "aclnnSubs",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_maximum_get_workspace = _optional_symbol(
            libs,
            "aclnnMaximumGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_maximum = _optional_symbol(
            libs,
            "aclnnMaximum",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_minimum_get_workspace = _optional_symbol(
            libs,
            "aclnnMinimumGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_minimum = _optional_symbol(
            libs,
            "aclnnMinimum",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_atan_get_workspace = _optional_symbol(
            libs,
            "aclnnAtanGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_atan = _optional_symbol(
            libs,
            "aclnnAtan",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_atan2_get_workspace = _optional_symbol(
            libs,
            "aclnnAtan2GetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_atan2 = _optional_symbol(
            libs,
            "aclnnAtan2",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_asin_get_workspace = _optional_symbol(
            libs,
            "aclnnAsinGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_asin = _optional_symbol(
            libs,
            "aclnnAsin",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_acos_get_workspace = _optional_symbol(
            libs,
            "aclnnAcosGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_acos = _optional_symbol(
            libs,
            "aclnnAcos",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_asinh_get_workspace = _optional_symbol(
            libs,
            "aclnnAsinhGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_asinh = _optional_symbol(
            libs,
            "aclnnAsinh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_acosh_get_workspace = _optional_symbol(
            libs,
            "aclnnAcoshGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_acosh = _optional_symbol(
            libs,
            "aclnnAcosh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_atanh_get_workspace = _optional_symbol(
            libs,
            "aclnnAtanhGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_atanh = _optional_symbol(
            libs,
            "aclnnAtanh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_relu_get_workspace = _bind_symbol(
            libs,
            "aclnnReluGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_relu = _bind_symbol(
            libs,
            "aclnnRelu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_abs_get_workspace = _optional_symbol(
            libs,
            "aclnnAbsGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_abs = _optional_symbol(
            libs,
            "aclnnAbs",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_neg_get_workspace = _optional_symbol(
            libs,
            "aclnnNegGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_neg = _optional_symbol(
            libs,
            "aclnnNeg",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_exp_get_workspace = _optional_symbol(
            libs,
            "aclnnExpGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_exp = _optional_symbol(
            libs,
            "aclnnExp",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_log_get_workspace = _optional_symbol(
            libs,
            "aclnnLogGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_log = _optional_symbol(
            libs,
            "aclnnLog",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_expm1_get_workspace = _optional_symbol(
            libs,
            "aclnnExpm1GetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_expm1 = _optional_symbol(
            libs,
            "aclnnExpm1",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_log1p_get_workspace = _optional_symbol(
            libs,
            "aclnnLog1pGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_log1p = _optional_symbol(
            libs,
            "aclnnLog1p",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sqrt_get_workspace = _optional_symbol(
            libs,
            "aclnnSqrtGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sqrt = _optional_symbol(
            libs,
            "aclnnSqrt",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_rsqrt_get_workspace = _optional_symbol(
            libs,
            "aclnnRsqrtGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_rsqrt = _optional_symbol(
            libs,
            "aclnnRsqrt",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sin_get_workspace = _optional_symbol(
            libs,
            "aclnnSinGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sin = _optional_symbol(
            libs,
            "aclnnSin",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_cos_get_workspace = _optional_symbol(
            libs,
            "aclnnCosGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_cos = _optional_symbol(
            libs,
            "aclnnCos",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_tan_get_workspace = _optional_symbol(
            libs,
            "aclnnTanGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_tan = _optional_symbol(
            libs,
            "aclnnTan",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_tanh_get_workspace = _optional_symbol(
            libs,
            "aclnnTanhGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_tanh = _optional_symbol(
            libs,
            "aclnnTanh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sigmoid_get_workspace = _optional_symbol(
            libs,
            "aclnnSigmoidGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sigmoid = _optional_symbol(
            libs,
            "aclnnSigmoid",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sign_get_workspace = _optional_symbol(
            libs,
            "aclnnSignGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sign = _optional_symbol(
            libs,
            "aclnnSign",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_signbit_get_workspace = _optional_symbol(
            libs,
            "aclnnSignbitGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_signbit = _optional_symbol(
            libs,
            "aclnnSignbit",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        
        self.aclnn_logical_not_get_workspace = _optional_symbol(
            libs,
            "aclnnLogicalNotGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_logical_not = _optional_symbol(
            libs,
            "aclnnLogicalNot",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_logical_and_get_workspace = _optional_symbol(
            libs,
            "aclnnLogicalAndGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_logical_and = _optional_symbol(
            libs,
            "aclnnLogicalAnd",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_logical_or_get_workspace = _optional_symbol(
            libs,
            "aclnnLogicalOrGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_logical_or = _optional_symbol(
            libs,
            "aclnnLogicalOr",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_logical_xor_get_workspace = _optional_symbol(
            libs,
            "aclnnLogicalXorGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )

        self.aclnn_swhere_get_workspace = _optional_symbol(
            libs,
            "aclnnSWhereGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_swhere = _optional_symbol(
            libs,
            "aclnnSWhere",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_logical_xor = _optional_symbol(
            libs,
            "aclnnLogicalXor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Bitwise ops
        self.aclnn_bitwise_not_get_workspace = _optional_symbol(
            libs,
            "aclnnBitwiseNotGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_bitwise_not = _optional_symbol(
            libs,
            "aclnnBitwiseNot",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_bitwise_and_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnBitwiseAndTensorGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_bitwise_and_tensor = _optional_symbol(
            libs,
            "aclnnBitwiseAndTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_bitwise_or_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnBitwiseOrTensorGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_bitwise_or_tensor = _optional_symbol(
            libs,
            "aclnnBitwiseOrTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_bitwise_xor_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnBitwiseXorTensorGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_bitwise_xor_tensor = _optional_symbol(
            libs,
            "aclnnBitwiseXorTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_isfinite_get_workspace = _optional_symbol(
            libs,
            "aclnnIsFiniteGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_isfinite = _optional_symbol(
            libs,
            "aclnnIsFinite",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_isinf_get_workspace = _optional_symbol(
            libs,
            "aclnnIsInfGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_isinf = _optional_symbol(
            libs,
            "aclnnIsInf",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_isposinf_get_workspace = _optional_symbol(
            libs,
            "aclnnIsPosInfGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_isposinf = _optional_symbol(
            libs,
            "aclnnIsPosInf",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_isneginf_get_workspace = _optional_symbol(
            libs,
            "aclnnIsNegInfGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_isneginf = _optional_symbol(
            libs,
            "aclnnIsNegInf",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_ne_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnNeTensorGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_ne_tensor = _optional_symbol(
            libs,
            "aclnnNeTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_eq_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnEqTensorGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_eq_tensor = _optional_symbol(
            libs,
            "aclnnEqTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_eq_scalar_get_workspace = _optional_symbol(
            libs,
            "aclnnEqScalarGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_eq_scalar = _optional_symbol(
            libs,
            "aclnnEqScalar",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_argmax_get_workspace = _optional_symbol(
            libs,
            "aclnnArgMaxGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_argmax = _optional_symbol(
            libs,
            "aclnnArgMax",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_argmin_get_workspace = _optional_symbol(
            libs,
            "aclnnArgMinGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_argmin = _optional_symbol(
            libs,
            "aclnnArgMin",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_max_dim_get_workspace = _optional_symbol(
            libs,
            "aclnnMaxDimGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_max_dim = _optional_symbol(
            libs,
            "aclnnMaxDim",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_min_dim_get_workspace = _optional_symbol(
            libs,
            "aclnnMinDimGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_min_dim = _optional_symbol(
            libs,
            "aclnnMinDim",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_cast_get_workspace = _optional_symbol(
            libs,
            "aclnnCastGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_cast = _optional_symbol(
            libs,
            "aclnnCast",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_cosh_get_workspace = _optional_symbol(
            libs,
            "aclnnCoshGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_cosh = _optional_symbol(
            libs,
            "aclnnCosh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sinh_get_workspace = _optional_symbol(
            libs,
            "aclnnSinhGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sinh = _optional_symbol(
            libs,
            "aclnnSinh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_erf_get_workspace = _optional_symbol(
            libs,
            "aclnnErfGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_erf = _optional_symbol(
            libs,
            "aclnnErf",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_erfc_get_workspace = _optional_symbol(
            libs,
            "aclnnErfcGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_erfc = _optional_symbol(
            libs,
            "aclnnErfc",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_softplus_get_workspace = _optional_symbol(
            libs,
            "aclnnSoftplusGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_softplus = _optional_symbol(
            libs,
            "aclnnSoftplus",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_hardtanh_get_workspace = _optional_symbol(
            libs,
            "aclnnHardtanhGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_hardtanh = _optional_symbol(
            libs,
            "aclnnHardtanh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_clamp_get_workspace = _optional_symbol(
            libs,
            "aclnnClampGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_clamp = _optional_symbol(
            libs,
            "aclnnClamp",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_clamp_min_get_workspace = _optional_symbol(
            libs,
            "aclnnClampMinGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_clamp_min = _optional_symbol(
            libs,
            "aclnnClampMin",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_clamp_max_get_workspace = _optional_symbol(
            libs,
            "aclnnClampMaxGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_clamp_max = _optional_symbol(
            libs,
            "aclnnClampMax",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_clamp_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnClampTensorGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_clamp_tensor = _optional_symbol(
            libs,
            "aclnnClampTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_clamp_min_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnClampMinTensorGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_clamp_min_tensor = _optional_symbol(
            libs,
            "aclnnClampMinTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_clamp_max_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnClampMaxTensorGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_clamp_max_tensor = _optional_symbol(
            libs,
            "aclnnClampMaxTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_floor_get_workspace = _optional_symbol(
            libs,
            "aclnnFloorGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_floor = _optional_symbol(
            libs,
            "aclnnFloor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_ceil_get_workspace = _optional_symbol(
            libs,
            "aclnnCeilGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_ceil = _optional_symbol(
            libs,
            "aclnnCeil",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_round_get_workspace = _optional_symbol(
            libs,
            "aclnnRoundGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_round = _optional_symbol(
            libs,
            "aclnnRound",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_trunc_get_workspace = _optional_symbol(
            libs,
            "aclnnTruncGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_trunc = _optional_symbol(
            libs,
            "aclnnTrunc",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_frac_get_workspace = _optional_symbol(
            libs,
            "aclnnFracGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_frac = _optional_symbol(
            libs,
            "aclnnFrac",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_log2_get_workspace = _optional_symbol(
            libs,
            "aclnnLog2GetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_log2 = _optional_symbol(
            libs,
            "aclnnLog2",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_log10_get_workspace = _optional_symbol(
            libs,
            "aclnnLog10GetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_log10 = _optional_symbol(
            libs,
            "aclnnLog10",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_exp2_get_workspace = _optional_symbol(
            libs,
            "aclnnExp2GetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_exp2 = _optional_symbol(
            libs,
            "aclnnExp2",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_pow_tensor_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnPowTensorTensorGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_pow_tensor_tensor = _optional_symbol(
            libs,
            "aclnnPowTensorTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_pow_tensor_scalar_get_workspace = _optional_symbol(
            libs,
            "aclnnPowTensorScalarGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_pow_tensor_scalar = _optional_symbol(
            libs,
            "aclnnPowTensorScalar",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_reduce_sum_get_workspace = _bind_symbol(
            libs,
            "aclnnReduceSumGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_reduce_sum = _bind_symbol(
            libs,
            "aclnnReduceSum",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_matmul_get_workspace = _optional_symbol(
            libs,
            "aclnnMatmulGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int8,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_matmul = _optional_symbol(
            libs,
            "aclnnMatmul",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Dot: aclnnDotGetWorkspaceSize(self, tensor, out, workspaceSize, executor)
        self.aclnn_dot_get_workspace = _optional_symbol(
            libs,
            "aclnnDotGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_dot = _optional_symbol(
            libs,
            "aclnnDot",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Mv: aclnnMvGetWorkspaceSize(self, vec, out, cubeMathType, workspaceSize, executor)
        self.aclnn_mv_get_workspace = _optional_symbol(
            libs,
            "aclnnMvGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int8, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_mv = _optional_symbol(
            libs,
            "aclnnMv",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Ger: aclnnGerGetWorkspaceSize(self, vec2, out, workspaceSize, executor)
        self.aclnn_ger_get_workspace = _optional_symbol(
            libs,
            "aclnnGerGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_ger = _optional_symbol(
            libs,
            "aclnnGer",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Median (global): aclnnMedianGetWorkspaceSize(self, valuesOut, workspaceSize, executor)
        self.aclnn_median_get_workspace = _optional_symbol(
            libs,
            "aclnnMedianGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_median = _optional_symbol(
            libs,
            "aclnnMedian",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # MedianDim: aclnnMedianDimGetWorkspaceSize(self, dim, keepDim, valuesOut, indicesOut, workspaceSize, executor)
        self.aclnn_median_dim_get_workspace = _optional_symbol(
            libs,
            "aclnnMedianDimGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_int64, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_median_dim = _optional_symbol(
            libs,
            "aclnnMedianDim",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Kthvalue: aclnnKthvalueGetWorkspaceSize(self, k, dim, keepdim, valuesOut, indicesOut, workspaceSize, executor)
        self.aclnn_kthvalue_get_workspace = _optional_symbol(
            libs,
            "aclnnKthvalueGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_kthvalue = _optional_symbol(
            libs,
            "aclnnKthvalue",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # SearchSorted: aclnnSearchSortedGetWorkspaceSize(sortedSequence, self, outInt32, right, sorter, out, workspaceSize, executor)
        self.aclnn_search_sorted_get_workspace = _optional_symbol(
            libs,
            "aclnnSearchSortedGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_search_sorted = _optional_symbol(
            libs,
            "aclnnSearchSorted",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Unique: aclnnUniqueGetWorkspaceSize(self, sorted, returnInverse, valueOut, inverseOut, workspaceSize, executor)
        self.aclnn_unique_get_workspace = _optional_symbol(
            libs,
            "aclnnUniqueGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_unique = _optional_symbol(
            libs,
            "aclnnUnique",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Randperm: aclnnRandpermGetWorkspaceSize(n, seed, offset, out, workspaceSize, executor)
        self.aclnn_randperm_get_workspace = _optional_symbol(
            libs,
            "aclnnRandpermGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_randperm = _optional_symbol(
            libs,
            "aclnnRandperm",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Flatten: aclnnFlattenGetWorkspaceSize(self, axis, out, workspaceSize, executor)
        # NOTE: aclnnFlatten always produces 2D output, different from torch.flatten
        self.aclnn_flatten_get_workspace = _optional_symbol(
            libs,
            "aclnnFlattenGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_flatten = _optional_symbol(
            libs,
            "aclnnFlatten",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_batch_matmul_get_workspace = _optional_symbol(
            libs,
            "aclnnBatchMatMulGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int8,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_batch_matmul = _optional_symbol(
            libs,
            "aclnnBatchMatMul",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_inplace_one_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceOneGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_one = _optional_symbol(
            libs,
            "aclnnInplaceOne",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_inplace_zero_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceZeroGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_zero = _optional_symbol(
            libs,
            "aclnnInplaceZero",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # TensorList support
        self.acl_create_tensor_list = _optional_symbol(
            libs,
            "aclCreateTensorList",
            ctypes.c_void_p,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64],
        )
        self.acl_destroy_tensor_list = _optional_symbol(
            libs,
            "aclDestroyTensorList",
            ctypes.c_int32,
            [ctypes.c_void_p],
        )
        # Cat
        self.aclnn_cat_get_workspace = _optional_symbol(
            libs,
            "aclnnCatGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # tensorList
                ctypes.c_int64,   # dim
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_cat = _optional_symbol(
            libs,
            "aclnnCat",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Stack
        self.aclnn_stack_get_workspace = _optional_symbol(
            libs,
            "aclnnStackGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # tensorList
                ctypes.c_int64,   # dim
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_stack = _optional_symbol(
            libs,
            "aclnnStack",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Where (SWhere = Select Where)
        self.aclnn_s_where_get_workspace = _optional_symbol(
            libs,
            "aclnnSWhereGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # condition
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # other
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_s_where = _optional_symbol(
            libs,
            "aclnnSWhere",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Mean
        self.aclnn_mean_get_workspace = _optional_symbol(
            libs,
            "aclnnMeanGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # dim (IntArray)
                ctypes.c_bool,    # keepdim
                ctypes.c_int32,   # dtype
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_mean = _optional_symbol(
            libs,
            "aclnnMean",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Softmax
        self.aclnn_softmax_get_workspace = _optional_symbol(
            libs,
            "aclnnSoftmaxGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_int64,   # dim
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_softmax = _optional_symbol(
            libs,
            "aclnnSoftmax",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # LogSoftmax
        self.aclnn_log_softmax_get_workspace = _optional_symbol(
            libs,
            "aclnnLogSoftmaxGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_int64,   # dim
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_log_softmax = _optional_symbol(
            libs,
            "aclnnLogSoftmax",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Gelu
        self.aclnn_gelu_get_workspace = _optional_symbol(
            libs,
            "aclnnGeluGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_gelu = _optional_symbol(
            libs,
            "aclnnGelu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # LayerNorm
        self.aclnn_layer_norm_get_workspace = _optional_symbol(
            libs,
            "aclnnLayerNormGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # input
                ctypes.c_void_p,  # normalizedShape (IntArray)
                ctypes.c_void_p,  # weight (optional)
                ctypes.c_void_p,  # bias (optional)
                ctypes.c_double,  # eps
                ctypes.c_void_p,  # out
                ctypes.c_void_p,  # mean (optional)
                ctypes.c_void_p,  # rstd (optional)
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_layer_norm = _optional_symbol(
            libs,
            "aclnnLayerNorm",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Silu
        self.aclnn_silu_get_workspace = _optional_symbol(
            libs,
            "aclnnSiluGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_silu = _optional_symbol(
            libs,
            "aclnnSilu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # LeakyRelu
        self.aclnn_leaky_relu_get_workspace = _optional_symbol(
            libs,
            "aclnnLeakyReluGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # negative_slope (Scalar)
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_leaky_relu = _optional_symbol(
            libs,
            "aclnnLeakyRelu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Elu
        self.aclnn_elu_get_workspace = _optional_symbol(
            libs,
            "aclnnEluGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # alpha (Scalar)
                ctypes.c_void_p,  # scale (Scalar)
                ctypes.c_void_p,  # input_scale (Scalar)
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_elu = _optional_symbol(
            libs,
            "aclnnElu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Mish
        self.aclnn_mish_get_workspace = _optional_symbol(
            libs,
            "aclnnMishGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_mish = _optional_symbol(
            libs,
            "aclnnMish",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Prelu
        self.aclnn_prelu_get_workspace = _optional_symbol(
            libs,
            "aclnnPreluGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # weight
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_prelu = _optional_symbol(
            libs,
            "aclnnPrelu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # BatchNorm
        self.aclnn_batch_norm_get_workspace = _optional_symbol(
            libs,
            "aclnnBatchNormGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # input
                ctypes.c_void_p,  # weight (optional)
                ctypes.c_void_p,  # bias (optional)
                ctypes.c_void_p,  # running_mean (optional)
                ctypes.c_void_p,  # running_var (optional)
                ctypes.c_bool,    # training
                ctypes.c_double,  # momentum
                ctypes.c_double,  # eps
                ctypes.c_void_p,  # out
                ctypes.c_void_p,  # save_mean (optional)
                ctypes.c_void_p,  # save_invstd (optional)
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_batch_norm = _optional_symbol(
            libs,
            "aclnnBatchNorm",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # GroupNorm
        self.aclnn_group_norm_get_workspace = _optional_symbol(
            libs,
            "aclnnGroupNormGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self (input)
                ctypes.c_void_p,  # gamma (weight, optional)
                ctypes.c_void_p,  # beta (bias, optional)
                ctypes.c_int64,   # N (batch size)
                ctypes.c_int64,   # C (channels)
                ctypes.c_int64,   # HxW (spatial dimensions)
                ctypes.c_int64,   # group (num_groups)
                ctypes.c_double,  # eps
                ctypes.c_void_p,  # out
                ctypes.c_void_p,  # meanOut (optional)
                ctypes.c_void_p,  # rstdOut (optional)
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_group_norm = _optional_symbol(
            libs,
            "aclnnGroupNorm",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Gather
        self.aclnn_gather_get_workspace = _optional_symbol(
            libs,
            "aclnnGatherGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_int64,   # dim
                ctypes.c_void_p,  # index
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_gather = _optional_symbol(
            libs,
            "aclnnGather",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # ConstantPadNd
        self.aclnn_constant_pad_nd_get_workspace = _optional_symbol(
            libs,
            "aclnnConstantPadNdGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # pad (IntArray)
                ctypes.c_void_p,  # value (Scalar)
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_constant_pad_nd = _optional_symbol(
            libs,
            "aclnnConstantPadNd",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Gather
        self.aclnn_gather_get_workspace = _optional_symbol(
            libs,
            "aclnnGatherGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_int64,   # dim
                ctypes.c_void_p,  # index
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_gather = _optional_symbol(
            libs,
            "aclnnGather",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # MaskedSelect
        self.aclnn_masked_select_get_workspace = _optional_symbol(
            libs,
            "aclnnMaskedSelectGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # mask
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_masked_select = _optional_symbol(
            libs,
            "aclnnMaskedSelect",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Embedding
        self.aclnn_embedding_get_workspace = _optional_symbol(
            libs,
            "aclnnEmbeddingGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # weight
                ctypes.c_void_p,  # indices
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_embedding = _optional_symbol(
            libs,
            "aclnnEmbedding",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Dropout GenMask + DoMask (two-step dropout)
        self.aclnn_dropout_gen_mask_get_workspace = _optional_symbol(
            libs,
            "aclnnDropoutGenMaskGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # shape (IntArray)
                ctypes.c_double,  # prob
                ctypes.c_int64,   # seed
                ctypes.c_int64,   # offset
                ctypes.c_void_p,  # out (uint8 mask)
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_dropout_gen_mask = _optional_symbol(
            libs,
            "aclnnDropoutGenMask",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_dropout_do_mask_get_workspace = _optional_symbol(
            libs,
            "aclnnDropoutDoMaskGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self (input)
                ctypes.c_void_p,  # mask
                ctypes.c_double,  # prob
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_dropout_do_mask = _optional_symbol(
            libs,
            "aclnnDropoutDoMask",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # InplaceNormal (for randn)
        self.aclnn_inplace_normal_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceNormalGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_float,   # mean
                ctypes.c_float,   # std
                ctypes.c_int64,   # seed
                ctypes.c_int64,   # offset
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_normal = _optional_symbol(
            libs,
            "aclnnInplaceNormal",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # InplaceUniform (for rand)
        self.aclnn_inplace_uniform_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceUniformGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_double,  # from
                ctypes.c_double,  # to
                ctypes.c_int64,   # seed
                ctypes.c_int64,   # offset
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_uniform = _optional_symbol(
            libs,
            "aclnnInplaceUniform",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # InplaceFillScalar (for fill_)
        self.aclnn_inplace_fill_scalar_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceFillScalarGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # value (acl scalar)
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_fill_scalar = _optional_symbol(
            libs,
            "aclnnInplaceFillScalar",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Copy (for copy_)
        self.aclnn_inplace_copy_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceCopyGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # dst (self)
                ctypes.c_void_p,  # src
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_copy = _optional_symbol(
            libs,
            "aclnnInplaceCopy",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Erfinv (for erfinv_)
        self.aclnn_erfinv_get_workspace = _optional_symbol(
            libs,
            "aclnnErfinvGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_erfinv = _optional_symbol(
            libs,
            "aclnnErfinv",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # LinalgQr (for torch.linalg.qr)
        self.aclnn_linalg_qr_get_workspace = _optional_symbol(
            libs,
            "aclnnLinalgQrGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_int64,   # mode
                ctypes.c_void_p,  # Q out
                ctypes.c_void_p,  # R out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_linalg_qr = _optional_symbol(
            libs,
            "aclnnLinalgQr",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnVar: (self, dim:IntArray, unbiased:bool, keepdim:bool, out) -> status
        self.aclnn_var_get_workspace = _optional_symbol(
            libs,
            "aclnnVarGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclIntArray* dim
                ctypes.c_bool,                     # bool unbiased
                ctypes.c_bool,                     # bool keepdim
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_var = _optional_symbol(
            libs,
            "aclnnVar",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnNorm: (self, p:Scalar, dim:IntArray, keepdim:bool, out) -> status
        self.aclnn_norm_get_workspace = _optional_symbol(
            libs,
            "aclnnNormGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclScalar* pScalar
                ctypes.c_void_p,                   # const aclIntArray* dim
                ctypes.c_bool,                     # bool keepdim
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_norm = _optional_symbol(
            libs,
            "aclnnNorm",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnProd: (self, dtype:DataType, out) -> status (all-reduce)
        self.aclnn_prod_get_workspace = _optional_symbol(
            libs,
            "aclnnProdGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_int32,                    # aclDataType dtype
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_prod = _optional_symbol(
            libs,
            "aclnnProd",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnProdDim: (self, dim:int64, keepDim:bool, dtype:DataType, out) -> status
        self.aclnn_prod_dim_get_workspace = _optional_symbol(
            libs,
            "aclnnProdDimGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_int64,                    # int64_t dim
                ctypes.c_bool,                     # bool keepDim
                ctypes.c_int32,                    # aclDataType dtype
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_prod_dim = _optional_symbol(
            libs,
            "aclnnProdDim",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnFloorDivide: (self, other, out) -> status
        self.aclnn_floor_divide_get_workspace = _optional_symbol(
            libs,
            "aclnnFloorDivideGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclTensor* other
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_floor_divide = _optional_symbol(
            libs,
            "aclnnFloorDivide",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnRmsNorm: (x, gamma, epsilon:double, yOut, rstdOut) -> status
        self.aclnn_rms_norm_get_workspace = _optional_symbol(
            libs,
            "aclnnRmsNormGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* x
                ctypes.c_void_p,                   # const aclTensor* gamma
                ctypes.c_double,                   # double epsilon
                ctypes.c_void_p,                   # const aclTensor* yOut
                ctypes.c_void_p,                   # const aclTensor* rstdOut
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_rms_norm = _optional_symbol(
            libs,
            "aclnnRmsNorm",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnConvolution
        self.aclnn_convolution_get_workspace = _optional_symbol(
            libs,
            "aclnnConvolutionGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* input
                ctypes.c_void_p,                   # const aclTensor* weight
                ctypes.c_void_p,                   # const aclTensor* bias (nullable)
                ctypes.c_void_p,                   # const aclIntArray* stride
                ctypes.c_void_p,                   # const aclIntArray* padding
                ctypes.c_void_p,                   # const aclIntArray* dilation
                ctypes.c_bool,                     # bool transposed
                ctypes.c_void_p,                   # const aclIntArray* outputPadding
                ctypes.c_int64,                    # int64_t groups
                ctypes.c_void_p,                   # aclTensor* output
                ctypes.c_int8,                     # int8_t cubeMathType
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_convolution = _optional_symbol(
            libs,
            "aclnnConvolution",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnMaxPool
        self.aclnn_max_pool_get_workspace = _optional_symbol(
            libs,
            "aclnnMaxPoolGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclIntArray* kernelShape
                ctypes.c_void_p,                   # const aclIntArray* strides
                ctypes.c_int64,                    # int64_t autoPad
                ctypes.c_void_p,                   # const aclIntArray* pads
                ctypes.c_void_p,                   # const aclIntArray* dilations
                ctypes.c_int64,                    # int64_t ceilMode
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_max_pool = _optional_symbol(
            libs,
            "aclnnMaxPool",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnMaxPool2dWithIndices — supports fp32/fp16/bf16, preferred over aclnnMaxPool
        self.aclnn_max_pool2d_with_indices_get_workspace = _optional_symbol(
            libs,
            "aclnnMaxPool2dWithIndicesGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclIntArray* kernelSize
                ctypes.c_void_p,                   # const aclIntArray* stride
                ctypes.c_void_p,                   # const aclIntArray* padding
                ctypes.c_void_p,                   # const aclIntArray* dilation
                ctypes.c_bool,                     # bool ceilMode
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.c_void_p,                   # aclTensor* indices
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_max_pool2d_with_indices = _optional_symbol(
            libs,
            "aclnnMaxPool2dWithIndices",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnMaxPool2dWithMask — used on Ascend910B (pre-910C), supports fp32/fp16
        # indices are int8 mask tensors (not actual position indices)
        self.aclnn_max_pool2d_with_mask_get_workspace = _optional_symbol(
            libs,
            "aclnnMaxPool2dWithMaskGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclIntArray* kernelSize
                ctypes.c_void_p,                   # const aclIntArray* stride
                ctypes.c_void_p,                   # const aclIntArray* padding
                ctypes.c_void_p,                   # const aclIntArray* dilation
                ctypes.c_bool,                     # bool ceilMode
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.c_void_p,                   # aclTensor* indices (mask)
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_max_pool2d_with_mask = _optional_symbol(
            libs,
            "aclnnMaxPool2dWithMask",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAvgPool2d
        self.aclnn_avg_pool2d_get_workspace = _optional_symbol(
            libs,
            "aclnnAvgPool2dGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclIntArray* kernelSize
                ctypes.c_void_p,                   # const aclIntArray* strides
                ctypes.c_void_p,                   # const aclIntArray* paddings
                ctypes.c_bool,                     # bool ceilMode
                ctypes.c_bool,                     # bool countIncludePad
                ctypes.c_int64,                    # int64_t divisorOverride
                ctypes.c_int8,                     # int8_t cubeMathType
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_avg_pool2d = _optional_symbol(
            libs,
            "aclnnAvgPool2d",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAdaptiveAvgPool2d
        self.aclnn_adaptive_avg_pool2d_get_workspace = _optional_symbol(
            libs,
            "aclnnAdaptiveAvgPool2dGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclIntArray* outputSize
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_adaptive_avg_pool2d = _optional_symbol(
            libs,
            "aclnnAdaptiveAvgPool2d",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # ---- Backward kernel bindings ----

        # aclnnSoftmaxBackward(gradOutput, output, dim, gradInput)
        self.aclnn_softmax_backward_get_workspace = _optional_symbol(
            libs, "aclnnSoftmaxBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_softmax_backward = _optional_symbol(
            libs, "aclnnSoftmaxBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnLogSoftmaxBackward(gradOutput, output, dim, gradInput)
        self.aclnn_log_softmax_backward_get_workspace = _optional_symbol(
            libs, "aclnnLogSoftmaxBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_log_softmax_backward = _optional_symbol(
            libs, "aclnnLogSoftmaxBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnGeluBackward(gradOutput, self, gradInput)
        self.aclnn_gelu_backward_get_workspace = _optional_symbol(
            libs, "aclnnGeluBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_gelu_backward = _optional_symbol(
            libs, "aclnnGeluBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnLayerNormBackward(gradOut, input, normalizedShape, mean, rstd,
        #                        weight, bias, outputMask, gradInput, gradWeight, gradBias)
        self.aclnn_layer_norm_backward_get_workspace = _optional_symbol(
            libs, "aclnnLayerNormBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradOut, input, normalizedShape
             ctypes.c_void_p, ctypes.c_void_p,  # mean, rstd
             ctypes.c_void_p, ctypes.c_void_p,  # weight, bias
             ctypes.c_void_p,  # outputMask (aclBoolArray)
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradInput, gradWeight, gradBias
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_layer_norm_backward = _optional_symbol(
            libs, "aclnnLayerNormBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnThresholdBackward(gradOutput, self, threshold, gradInput) — relu backward
        self.aclnn_threshold_backward_get_workspace = _optional_symbol(
            libs, "aclnnThresholdBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_threshold_backward = _optional_symbol(
            libs, "aclnnThresholdBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnHardshrinkBackward(gradOutput, self, lambd, gradInput)
        self.aclnn_hardshrink_backward_get_workspace = _optional_symbol(
            libs, "aclnnHardshrinkBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_hardshrink_backward = _optional_symbol(
            libs, "aclnnHardshrinkBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnSoftshrinkBackward(gradOutput, self, lambd, gradInput)
        self.aclnn_softshrink_backward_get_workspace = _optional_symbol(
            libs, "aclnnSoftshrinkBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_softshrink_backward = _optional_symbol(
            libs, "aclnnSoftshrinkBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnSiluBackward(gradOutput, self, gradInput)
        self.aclnn_silu_backward_get_workspace = _optional_symbol(
            libs, "aclnnSiluBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_silu_backward = _optional_symbol(
            libs, "aclnnSiluBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnHardswishBackward(gradOutput, self, gradInput)
        self.aclnn_hardswish_backward_get_workspace = _optional_symbol(
            libs, "aclnnHardswishBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_hardswish_backward = _optional_symbol(
            libs, "aclnnHardswishBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnHardsigmoidBackward(gradOutput, self, gradInput)
        self.aclnn_hardsigmoid_backward_get_workspace = _optional_symbol(
            libs, "aclnnHardsigmoidBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_hardsigmoid_backward = _optional_symbol(
            libs, "aclnnHardsigmoidBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnMishBackward(gradOutput, self, gradInput)
        self.aclnn_mish_backward_get_workspace = _optional_symbol(
            libs, "aclnnMishBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_mish_backward = _optional_symbol(
            libs, "aclnnMishBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnSoftplusBackward(gradOutput, self, beta, threshold, gradInput)
        self.aclnn_softplus_backward_get_workspace = _optional_symbol(
            libs, "aclnnSoftplusBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_softplus_backward = _optional_symbol(
            libs, "aclnnSoftplusBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnHardtanhBackward(gradOutput, self, min, max, gradInput)
        self.aclnn_hardtanh_backward_get_workspace = _optional_symbol(
            libs, "aclnnHardtanhBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_hardtanh_backward = _optional_symbol(
            libs, "aclnnHardtanhBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnLeakyReluBackward(gradOutput, self, negativeSlope, selfIsResult, gradInput)
        self.aclnn_leaky_relu_backward_get_workspace = _optional_symbol(
            libs, "aclnnLeakyReluBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_leaky_relu_backward = _optional_symbol(
            libs, "aclnnLeakyReluBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnEluBackward(gradOutput, alpha, scale, inputScale, isResult, selfOrResult, gradInput)
        self.aclnn_elu_backward_get_workspace = _optional_symbol(
            libs, "aclnnEluBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_elu_backward = _optional_symbol(
            libs, "aclnnEluBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnPreluBackward(gradOutput, self, weight, gradInput, gradWeight)
        self.aclnn_prelu_backward_get_workspace = _optional_symbol(
            libs, "aclnnPreluBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_prelu_backward = _optional_symbol(
            libs, "aclnnPreluBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnSigmoidBackward(gradOutput, output, gradInput)
        self.aclnn_sigmoid_backward_get_workspace = _optional_symbol(
            libs, "aclnnSigmoidBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sigmoid_backward = _optional_symbol(
            libs, "aclnnSigmoidBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnTanhBackward(gradOutput, output, gradInput)
        self.aclnn_tanh_backward_get_workspace = _optional_symbol(
            libs, "aclnnTanhBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_tanh_backward = _optional_symbol(
            libs, "aclnnTanhBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnConvolutionBackward
        self.aclnn_convolution_backward_get_workspace = _optional_symbol(
            libs, "aclnnConvolutionBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradOutput, input, weight
             ctypes.c_void_p,  # biasSizes (IntArray, nullable)
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # stride, padding, dilation
             ctypes.c_bool,  # transposed
             ctypes.c_void_p,  # outputPadding
             ctypes.c_int64,  # groups
             ctypes.c_void_p,  # outputMask (aclBoolArray)
             ctypes.c_int8,  # cubeMathType
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradInput, gradWeight, gradBias
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_convolution_backward = _optional_symbol(
            libs, "aclnnConvolutionBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnMaxPool2dWithMaskBackward
        self.aclnn_max_pool2d_with_mask_backward_get_workspace = _optional_symbol(
            libs, "aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradOutput, input, mask
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # ksize, stride, pad, dilation
             ctypes.c_bool,  # ceilMode
             ctypes.c_void_p,  # gradInput
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_max_pool2d_with_mask_backward = _optional_symbol(
            libs, "aclnnMaxPool2dWithMaskBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAvgPool2dBackward
        self.aclnn_avg_pool2d_backward_get_workspace = _optional_symbol(
            libs, "aclnnAvgPool2dBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,  # gradOutput, self
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # kernelSize, stride, padding
             ctypes.c_bool, ctypes.c_bool,  # ceilMode, countIncludePad
             ctypes.c_int64,  # divisorOverride (int64_t, 0 means no override)
             ctypes.c_int8,  # cubeMathType
             ctypes.c_void_p,  # gradInput
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_avg_pool2d_backward = _optional_symbol(
            libs, "aclnnAvgPool2dBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAdaptiveAvgPool2dBackward(gradOutput, self, gradInput)
        self.aclnn_adaptive_avg_pool2d_backward_get_workspace = _optional_symbol(
            libs, "aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_adaptive_avg_pool2d_backward = _optional_symbol(
            libs, "aclnnAdaptiveAvgPool2dBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAdaptiveAvgPool3dBackward(gradOutput, self, gradInput)
        self.aclnn_adaptive_avg_pool3d_backward_get_workspace = _optional_symbol(
            libs, "aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_adaptive_avg_pool3d_backward = _optional_symbol(
            libs, "aclnnAdaptiveAvgPool3dBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAvgPool3dBackward(gradOutput, self, kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride, output)
        self.aclnn_avg_pool3d_backward_get_workspace = _optional_symbol(
            libs, "aclnnAvgPool3dBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_bool, ctypes.c_bool, ctypes.c_int64,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_avg_pool3d_backward = _optional_symbol(
            libs, "aclnnAvgPool3dBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnBatchNormBackward
        self.aclnn_batch_norm_backward_get_workspace = _optional_symbol(
            libs, "aclnnBatchNormBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradOut, input, weight
             ctypes.c_void_p, ctypes.c_void_p,  # runningMean, runningVar
             ctypes.c_void_p, ctypes.c_void_p,  # saveMean, saveInvstd
             ctypes.c_bool, ctypes.c_double,  # train, eps
             ctypes.c_void_p,  # outputMask (aclBoolArray)
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradInput, gradWeight, gradBias
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_batch_norm_backward = _optional_symbol(
            libs, "aclnnBatchNormBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnGroupNormBackward
        self.aclnn_group_norm_backward_get_workspace = _optional_symbol(
            libs, "aclnnGroupNormBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradOut, input, mean, rstd
             ctypes.c_void_p,  # gamma (weight)
             ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # N, C, HxW, group
             ctypes.c_void_p,  # outputMask (aclBoolArray)
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradInput, gradGamma, gradBeta
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_group_norm_backward = _optional_symbol(
            libs, "aclnnGroupNormBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnEmbeddingDenseBackward
        self.aclnn_embedding_dense_backward_get_workspace = _optional_symbol(
            libs, "aclnnEmbeddingDenseBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,  # gradOutput, indices
             ctypes.c_int64, ctypes.c_int64, ctypes.c_bool,  # numWeights, paddingIdx, scaleGradByFreq
             ctypes.c_void_p,  # gradWeight
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_embedding_dense_backward = _optional_symbol(
            libs, "aclnnEmbeddingDenseBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnRmsNormGrad(dy, x, rstd, gamma, dx, dgamma)
        self.aclnn_rms_norm_grad_get_workspace = _optional_symbol(
            libs, "aclnnRmsNormGradGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # dy, x, rstd, gamma
             ctypes.c_void_p, ctypes.c_void_p,  # dx, dgamma
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_rms_norm_grad = _optional_symbol(
            libs, "aclnnRmsNormGrad", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnUnfoldGrad(gradOut, inputSizes, dim, size, step, out)
        self.aclnn_unfold_grad_get_workspace = _optional_symbol(
            libs, "aclnnUnfoldGradGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
             ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_unfold_grad = _optional_symbol(
            libs, "aclnnUnfoldGrad", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnSeluBackward(gradOutput, result, gradInput)
        self.aclnn_selu_backward_get_workspace = _optional_symbol(
            libs, "aclnnSeluBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_selu_backward = _optional_symbol(
            libs, "aclnnSeluBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnMaxPool3dWithArgmax (forward) — returns output + argmax indices (int32)
        self.aclnn_max_pool3d_with_argmax_get_workspace = _optional_symbol(
            libs, "aclnnMaxPool3dWithArgmaxGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p,                   # const aclTensor* self
             ctypes.c_void_p,                   # const aclIntArray* kernelSize
             ctypes.c_void_p,                   # const aclIntArray* stride
             ctypes.c_void_p,                   # const aclIntArray* padding
             ctypes.c_void_p,                   # const aclIntArray* dilation
             ctypes.c_bool,                     # bool ceilMode
             ctypes.c_void_p,                   # aclTensor* out
             ctypes.c_void_p,                   # aclTensor* indices
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_max_pool3d_with_argmax = _optional_symbol(
            libs, "aclnnMaxPool3dWithArgmax", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAdaptiveMaxPool2d (forward) — returns output + indices (int64)
        self.aclnn_adaptive_max_pool2d_get_workspace = _optional_symbol(
            libs, "aclnnAdaptiveMaxPool2dGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p,                   # const aclTensor* self
             ctypes.c_void_p,                   # const aclIntArray* outputSize
             ctypes.c_void_p,                   # aclTensor* outputOut
             ctypes.c_void_p,                   # aclTensor* indicesOut
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_adaptive_max_pool2d = _optional_symbol(
            libs, "aclnnAdaptiveMaxPool2d", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAdaptiveMaxPool2dBackward — uses indices from forward
        self.aclnn_adaptive_max_pool2d_backward_get_workspace = _optional_symbol(
            libs, "aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradOutput, self, indices
             ctypes.c_void_p,                                      # gradInput
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_adaptive_max_pool2d_backward = _optional_symbol(
            libs, "aclnnAdaptiveMaxPool2dBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnMaxPool3dWithArgmaxBackward — uses argmax indices from forward
        self.aclnn_max_pool3d_with_argmax_backward_get_workspace = _optional_symbol(
            libs, "aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradOutput, self, indices
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # ksize, stride, pad, dilation
             ctypes.c_bool,  # ceilMode
             ctypes.c_void_p,  # gradInput
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_max_pool3d_with_argmax_backward = _optional_symbol(
            libs, "aclnnMaxPool3dWithArgmaxBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # ---------------------------------------------------------------
        # P1 missing ops
        # ---------------------------------------------------------------

        # aclnnReciprocal: (self, out) — standard unary
        self.aclnn_reciprocal_get_workspace = _optional_symbol(
            libs, "aclnnReciprocalGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_reciprocal = _optional_symbol(
            libs, "aclnnReciprocal", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAddmm: (self, mat1, mat2, beta:Scalar, alpha:Scalar, out, cubeMathType)
        self.aclnn_addmm_get_workspace = _optional_symbol(
            libs, "aclnnAddmmGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_int8,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_addmm = _optional_symbol(
            libs, "aclnnAddmm", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnEinsum: (tensorList, equation:char*, output)
        self.aclnn_einsum_get_workspace = _optional_symbol(
            libs, "aclnnEinsumGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_einsum = _optional_symbol(
            libs, "aclnnEinsum", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnUpsampleNearest2d: (self, outputSize:IntArray, out)
        self.aclnn_upsample_nearest2d_get_workspace = _optional_symbol(
            libs, "aclnnUpsampleNearest2dGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_upsample_nearest2d = _optional_symbol(
            libs, "aclnnUpsampleNearest2d", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnUpsampleBilinear2d: (self, outputSize, alignCorners, scalesH, scalesW, out)
        self.aclnn_upsample_bilinear2d_get_workspace = _optional_symbol(
            libs, "aclnnUpsampleBilinear2dGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool,
             ctypes.c_double, ctypes.c_double, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_upsample_bilinear2d = _optional_symbol(
            libs, "aclnnUpsampleBilinear2d", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnUpsampleNearest2dBackward(gradOut, outputSize, inputSize, scalesH, scalesW, gradInput)
        self.aclnn_upsample_nearest2d_backward_get_workspace = _optional_symbol(
            libs, "aclnnUpsampleNearest2dBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_double, ctypes.c_double,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_upsample_nearest2d_backward = _optional_symbol(
            libs, "aclnnUpsampleNearest2dBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnUpsampleBilinear2dBackward(gradOut, outputSize, inputSize, alignCorners, scalesH, scalesW, gradInput)
        self.aclnn_upsample_bilinear2d_backward_get_workspace = _optional_symbol(
            libs, "aclnnUpsampleBilinear2dBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_bool, ctypes.c_double, ctypes.c_double,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_upsample_bilinear2d_backward = _optional_symbol(
            libs, "aclnnUpsampleBilinear2dBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnUpsampleBicubic2dBackward(gradOut, outputSize, inputSize, alignCorners, scalesH, scalesW, gradInput)
        self.aclnn_upsample_bicubic2d_backward_get_workspace = _optional_symbol(
            libs, "aclnnUpsampleBicubic2dBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_bool, ctypes.c_double, ctypes.c_double,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_upsample_bicubic2d_backward = _optional_symbol(
            libs, "aclnnUpsampleBicubic2dBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnUpsampleNearest1dBackward(gradOut, outputSize, inputSize, scales, gradInput)
        self.aclnn_upsample_nearest1d_backward_get_workspace = _optional_symbol(
            libs, "aclnnUpsampleNearest1dBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_double,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_upsample_nearest1d_backward = _optional_symbol(
            libs, "aclnnUpsampleNearest1dBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnUpsampleLinear1dBackward(gradOut, outputSize, inputSize, alignCorners, scales, gradInput)
        self.aclnn_upsample_linear1d_backward_get_workspace = _optional_symbol(
            libs, "aclnnUpsampleLinear1dBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_bool, ctypes.c_double,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_upsample_linear1d_backward = _optional_symbol(
            libs, "aclnnUpsampleLinear1dBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnOneHot: (self, numClasses, onValue, offValue, axis, out)
        self.aclnn_one_hot_get_workspace = _optional_symbol(
            libs, "aclnnOneHotGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_int64,
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_one_hot = _optional_symbol(
            libs, "aclnnOneHot", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # --- P0: Replace composite ops with ACLNN large kernels ---

        # aclnnLerp: (self, end, weight_tensor, out)
        self.aclnn_lerp_get_workspace = _optional_symbol(
            libs, "aclnnLerpGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_lerp = _optional_symbol(
            libs, "aclnnLerp", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnLerps: (self, end, weight_scalar, out)
        self.aclnn_lerps_get_workspace = _optional_symbol(
            libs, "aclnnLerpsGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_lerps = _optional_symbol(
            libs, "aclnnLerps", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnAddcmul: (self, tensor1, tensor2, value_scalar, out)
        self.aclnn_addcmul_get_workspace = _optional_symbol(
            libs, "aclnnAddcmulGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_addcmul = _optional_symbol(
            libs, "aclnnAddcmul", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnAddcdiv: (self, tensor1, tensor2, value_scalar, out)
        self.aclnn_addcdiv_get_workspace = _optional_symbol(
            libs, "aclnnAddcdivGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_addcdiv = _optional_symbol(
            libs, "aclnnAddcdiv", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnLogAddExp: (self, other, out) — standard binary
        self.aclnn_logaddexp_get_workspace = _optional_symbol(
            libs, "aclnnLogAddExpGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_logaddexp = _optional_symbol(
            libs, "aclnnLogAddExp", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnLogAddExp2: (self, other, out) — standard binary
        self.aclnn_logaddexp2_get_workspace = _optional_symbol(
            libs, "aclnnLogAddExp2GetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_logaddexp2 = _optional_symbol(
            libs, "aclnnLogAddExp2", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnRemainderTensorTensor: (self, other, out) — standard binary
        self.aclnn_remainder_tt_get_workspace = _optional_symbol(
            libs, "aclnnRemainderTensorTensorGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_remainder_tt = _optional_symbol(
            libs, "aclnnRemainderTensorTensor", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnFmodTensor: (self, other, out) — standard binary
        self.aclnn_fmod_tensor_get_workspace = _optional_symbol(
            libs, "aclnnFmodTensorGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_fmod_tensor = _optional_symbol(
            libs, "aclnnFmodTensor", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # --- P1: New ops with ACLNN large kernels ---

        # aclnnBaddbmm: (self, batch1, batch2, beta_scalar, alpha_scalar, out, cubeMathType)
        self.aclnn_baddbmm_get_workspace = _optional_symbol(
            libs, "aclnnBaddbmmGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_int8,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_baddbmm = _optional_symbol(
            libs, "aclnnBaddbmm", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnTrace: (self, out)
        self.aclnn_trace_get_workspace = _optional_symbol(
            libs, "aclnnTraceGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_trace = _optional_symbol(
            libs, "aclnnTrace", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnCummin: (self, dim, valuesOut, indicesOut)
        self.aclnn_cummin_get_workspace = _optional_symbol(
            libs, "aclnnCumminGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_int64,
             ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_cummin = _optional_symbol(
            libs, "aclnnCummin", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnLogSumExp: (self, dim_array, keepDim, out)
        self.aclnn_logsumexp_get_workspace = _optional_symbol(
            libs, "aclnnLogSumExpGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_logsumexp = _optional_symbol(
            libs, "aclnnLogSumExp", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnRenorm: (self, p_scalar, dim, maxNorm_scalar, out)
        self.aclnn_renorm_get_workspace = _optional_symbol(
            libs, "aclnnRenormGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
             ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_renorm = _optional_symbol(
            libs, "aclnnRenorm", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnLogicalXor: (self, other, out) — standard binary
        self.aclnn_logical_xor_get_workspace = _optional_symbol(
            libs, "aclnnLogicalXorGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_logical_xor = _optional_symbol(
            libs, "aclnnLogicalXor", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        self._init_comparison_and_new_ops(libs)


    def _init_comparison_and_new_ops(self, libs):
        """Initialize comparison, isclose, instance_norm, nansum, cross bindings."""
        # --- P0: Replace comparison composites with ACLNN large kernels ---

        # aclnnLtTensor: (self, other, out) — standard binary comparison
        self.aclnn_lt_tensor_get_workspace = _optional_symbol(
            libs, "aclnnLtTensorGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_lt_tensor = _optional_symbol(
            libs, "aclnnLtTensor", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnLeTensor: (self, other, out)
        self.aclnn_le_tensor_get_workspace = _optional_symbol(
            libs, "aclnnLeTensorGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_le_tensor = _optional_symbol(
            libs, "aclnnLeTensor", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnGtTensor: (self, other, out)
        self.aclnn_gt_tensor_get_workspace = _optional_symbol(
            libs, "aclnnGtTensorGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_gt_tensor = _optional_symbol(
            libs, "aclnnGtTensor", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnGeTensor: (self, other, out)
        self.aclnn_ge_tensor_get_workspace = _optional_symbol(
            libs, "aclnnGeTensorGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_ge_tensor = _optional_symbol(
            libs, "aclnnGeTensor", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnIsClose: (self, other, rtol, atol, equal_nan, out)
        self.aclnn_isclose_get_workspace = _optional_symbol(
            libs, "aclnnIsCloseGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_double, ctypes.c_double, ctypes.c_bool,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_isclose = _optional_symbol(
            libs, "aclnnIsClose", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnInstanceNorm: (x, gamma, beta, dataFormat, eps, y, mean, variance)
        self.aclnn_instance_norm_get_workspace = _optional_symbol(
            libs, "aclnnInstanceNormGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_char_p, ctypes.c_double,
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_instance_norm = _optional_symbol(
            libs, "aclnnInstanceNorm", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # --- P1: New ops ---

        # aclnnReduceNansum: (self, dim_array, keepDim, dtype, out)
        self.aclnn_reduce_nansum_get_workspace = _optional_symbol(
            libs, "aclnnReduceNansumGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool,
             ctypes.c_int32, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_reduce_nansum = _optional_symbol(
            libs, "aclnnReduceNansum", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnLinalgCross: (self, other, dim, out)
        self.aclnn_linalg_cross_get_workspace = _optional_symbol(
            libs, "aclnnLinalgCrossGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_linalg_cross = _optional_symbol(
            libs, "aclnnLinalgCross", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnIm2col: (self, kernelSize, dilation, padding, stride, out)
        self.aclnn_im2col_get_workspace = _optional_symbol(
            libs, "aclnnIm2colGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_im2col = _optional_symbol(
            libs, "aclnnIm2col", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnGridSampler2D: (input, grid, interpolation_mode, padding_mode, align_corners, out)
        self.aclnn_grid_sampler2d_get_workspace = _optional_symbol(
            libs, "aclnnGridSampler2DGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_int64, ctypes.c_int64, ctypes.c_bool,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_grid_sampler2d = _optional_symbol(
            libs, "aclnnGridSampler2D", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnGridSampler2DBackward
        self.aclnn_grid_sampler2d_backward_get_workspace = _optional_symbol(
            libs, "aclnnGridSampler2DBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_int64, ctypes.c_int64, ctypes.c_bool,
             ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_grid_sampler2d_backward = _optional_symbol(
            libs, "aclnnGridSampler2DBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAffineGrid: (theta, size, align_corners, out)
        self.aclnn_affine_grid_get_workspace = _optional_symbol(
            libs, "aclnnAffineGridGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_affine_grid = _optional_symbol(
            libs, "aclnnAffineGrid", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnSquare (unary: out = x * x)
        self.aclnn_square_get_workspace = _optional_symbol(
            libs, "aclnnSquareGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_square = _optional_symbol(
            libs, "aclnnSquare", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # aclnnDigamma (unary: out = digamma(x))
        self.aclnn_digamma_get_workspace = _optional_symbol(
            libs, "aclnnDigammaGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_digamma = _optional_symbol(
            libs, "aclnnDigamma", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnLgamma (unary: out = lgamma(x))
        self.aclnn_lgamma_get_workspace = _optional_symbol(
            libs, "aclnnLgammaGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_lgamma = _optional_symbol(
            libs, "aclnnLgamma", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnSinc (unary: out = sinc(x))
        self.aclnn_sinc_get_workspace = _optional_symbol(
            libs, "aclnnSincGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sinc = _optional_symbol(
            libs, "aclnnSinc", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnInverse (out = inv(x))
        self.aclnn_inverse_get_workspace = _optional_symbol(
            libs, "aclnnInverseGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_inverse = _optional_symbol(
            libs, "aclnnInverse", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnLinalgVectorNorm (self, ord_scalar, dim[], keepdim, dtype, out)
        self.aclnn_linalg_vector_norm_get_workspace = _optional_symbol(
            libs, "aclnnLinalgVectorNormGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_bool, ctypes.c_int32, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_linalg_vector_norm = _optional_symbol(
            libs, "aclnnLinalgVectorNorm", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAminmax (self, dim[], keepDim, minOut, maxOut)
        self.aclnn_aminmax_get_workspace = _optional_symbol(
            libs, "aclnnAminmaxGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool,
             ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_aminmax = _optional_symbol(
            libs, "aclnnAminmax", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnBincount (self, weights_or_null, minlength, out)
        self.aclnn_bincount_get_workspace = _optional_symbol(
            libs, "aclnnBincountGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_bincount = _optional_symbol(
            libs, "aclnnBincount", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAdaptiveAvgPool3d (self, outputSize[], out)
        self.aclnn_adaptive_avg_pool3d_get_workspace = _optional_symbol(
            libs, "aclnnAdaptiveAvgPool3dGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_adaptive_avg_pool3d = _optional_symbol(
            libs, "aclnnAdaptiveAvgPool3d", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnUpsampleBicubic2d (self, outSize[], alignCorners, scalesH, scalesW, out)
        self.aclnn_upsample_bicubic2d_get_workspace = _optional_symbol(
            libs, "aclnnUpsampleBicubic2dGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool,
             ctypes.c_double, ctypes.c_double, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_upsample_bicubic2d = _optional_symbol(
            libs, "aclnnUpsampleBicubic2d", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnUpsampleLinear1d (self, outSize[], alignCorners, scales, out)
        self.aclnn_upsample_linear1d_get_workspace = _optional_symbol(
            libs, "aclnnUpsampleLinear1dGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool,
             ctypes.c_double, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_upsample_linear1d = _optional_symbol(
            libs, "aclnnUpsampleLinear1d", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnApplyAdamWV2 (var, m, v, maxGradNormOpt, grad, step, lr, beta1, beta2, wd, eps, amsgrad, maximize)
        self.aclnn_apply_adam_w_v2_get_workspace = _optional_symbol(
            libs, "aclnnApplyAdamWV2GetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_float, ctypes.c_float, ctypes.c_float,
             ctypes.c_float, ctypes.c_float,
             ctypes.c_bool, ctypes.c_bool,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_apply_adam_w_v2 = _optional_symbol(
            libs, "aclnnApplyAdamWV2", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnRepeatInterleaveGrad(yGrad, repeats, axis, out)
        self.aclnn_repeat_interleave_grad_get_workspace = _optional_symbol(
            libs, "aclnnRepeatInterleaveGradGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_repeat_interleave_grad = _optional_symbol(
            libs, "aclnnRepeatInterleaveGrad", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )



_ACL_DTYPE = {
    "float32": 0,
    "float16": 1,
    "float64": 11,
    "bfloat16": 27,
    "int8": 2,
    "int16": 6,
    "int32": 3,
    "int64": 9,
    "uint8": 4,
    "bool": 12,
    "complex64": 16,
    "complex128": 17,
}

_ACL_FORMAT_ND = 2
_ACL_FORMAT_NCHW = 0


def _normalize_dtype(dtype):
    name = getattr(dtype, "name", None)
    if name is not None:
        return name
    return str(dtype)


def _dtype_to_acl(dtype):
    dtype = _normalize_dtype(dtype)
    if dtype not in _ACL_DTYPE:
        raise ValueError(f"Unsupported dtype for ACLNN: {dtype}")
    return _ACL_DTYPE[dtype]


def _float32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _float_to_float16_bits(value):
    f32 = _float32_bits(value)
    sign = (f32 >> 31) & 0x1
    exponent = (f32 >> 23) & 0xFF
    mantissa = f32 & 0x7FFFFF
    if exponent == 0xFF:
        half_exp = 0x1F
        half_mant = 0x200 if mantissa != 0 else 0
    elif exponent > 142:
        half_exp = 0x1F
        half_mant = 0
    elif exponent < 113:
        if exponent < 103:
            half_exp = 0
            half_mant = 0
        else:
            shift = 113 - exponent
            mantissa = mantissa | 0x800000
            half_mant = mantissa >> (shift + 13)
            round_bit = (mantissa >> (shift + 12)) & 1
            sticky = mantissa & ((1 << (shift + 12)) - 1)
            if round_bit and (sticky or (half_mant & 1)):
                half_mant += 1
            half_exp = 0
            if half_mant == 0x400:
                half_exp = 1
                half_mant = 0
    else:
        half_exp = exponent - 112
        half_mant = mantissa >> 13
        round_bit = (mantissa >> 12) & 1
        sticky = mantissa & 0xFFF
        if round_bit and (sticky or (half_mant & 1)):
            half_mant += 1
            if half_mant == 0x400:
                half_mant = 0
                half_exp += 1
                if half_exp >= 0x1F:
                    half_exp = 0x1F
                    half_mant = 0
    return (sign << 15) | (half_exp << 10) | half_mant


def _float_to_bfloat16_bits(value):
    bits = _float32_bits(value)
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return (rounded >> 16) & 0xFFFF


def _scalar_bytes(value, dtype):
    dtype = _normalize_dtype(dtype)
    if dtype == "float16":
        bits = _float_to_float16_bits(float(value))
        return int(bits).to_bytes(2, byteorder="little", signed=False)
    if dtype == "bfloat16":
        bits = _float_to_bfloat16_bits(float(value))
        return int(bits).to_bytes(2, byteorder="little", signed=False)
    if dtype == "float32":
        return struct.pack("<f", float(value))
    if dtype == "float64":
        return struct.pack("<d", float(value))
    if dtype == "int8":
        return int(value).to_bytes(1, byteorder="little", signed=True)
    if dtype == "uint8":
        return int(value).to_bytes(1, byteorder="little", signed=False)
    if dtype == "int16":
        return int(value).to_bytes(2, byteorder="little", signed=True)
    if dtype == "int32":
        return int(value).to_bytes(4, byteorder="little", signed=True)
    if dtype == "int64":
        return int(value).to_bytes(8, byteorder="little", signed=True)
    if dtype == "bool":
        return (1 if bool(value) else 0).to_bytes(1, byteorder="little", signed=False)
    raise ValueError(f"Unsupported scalar dtype for ACLNN: {dtype}")


def _make_int64_array(values):
    if not values:
        return None
    data = (ctypes.c_int64 * len(values))()
    for i, v in enumerate(values):
        data[i] = int(v)
    return data


def _make_bool_array(values):
    """Create a ctypes bool array for aclCreateBoolArray."""
    if not values:
        return None
    data = (ctypes.c_bool * len(values))()
    for i, v in enumerate(values):
        data[i] = bool(v)
    return data


def _create_tensor(bindings, shape, stride, dtype, data_ptr, fmt=_ACL_FORMAT_ND):
    _ = (bindings, shape, stride, dtype, data_ptr, fmt)
    _require_ctypes_npu_path_disabled()


def _create_scalar(bindings, value, dtype):
    _ = (bindings, value, dtype)
    _require_ctypes_npu_path_disabled()


def _bind_symbol(libs, name, restype, argtypes):
    preferred = None
    if name in {
        "aclnnAdd", "aclnnAddGetWorkspaceSize",
        "aclnnSub", "aclnnSubGetWorkspaceSize",
        "aclnnMul", "aclnnMulGetWorkspaceSize",
        "aclnnDiv", "aclnnDivGetWorkspaceSize",
        "aclnnAdds", "aclnnAddsGetWorkspaceSize",
        "aclnnSubs", "aclnnSubsGetWorkspaceSize",
    }:
        for lib in libs:
            path = getattr(lib, "_name", "") or ""
            if path.endswith("libopapi.so") and hasattr(lib, name):
                preferred = lib
                break
    if preferred is not None:
        func = getattr(preferred, name)
        func.restype = restype
        func.argtypes = argtypes
        return func
    for lib in libs:
        if hasattr(lib, name):
            func = getattr(lib, name)
            func.restype = restype
            func.argtypes = argtypes
            return func
    raise AttributeError(f"ACLNN symbol not found: {name}")


def _optional_symbol(libs, name, restype, argtypes):
    try:
        return _bind_symbol(libs, name, restype, argtypes)
    except AttributeError:
        return None


def _load_libs():
    global _LIB_HANDLES
    if _LIB_HANDLES is not None:
        return _LIB_HANDLES
    lib_dirs = _get_lib_dirs()
    base_libs, preload_libs, main_libs = _get_lib_names()
    libs = []
    resolved_paths = []  # absolute paths for FFI init
    for lib_name in base_libs:
        lib_path = None
        for base in lib_dirs:
            candidate = os.path.join(base, lib_name)
            if os.path.exists(candidate):
                lib_path = candidate
                break
        if lib_path is None:
            raise FileNotFoundError(f"ACLNN base library not found: {lib_name}")
        libs.append(ctypes.CDLL(lib_path))
        resolved_paths.append(lib_path)
    for lib_name in preload_libs:
        lib_path = None
        for base in lib_dirs:
            candidate = os.path.join(base, lib_name)
            if os.path.exists(candidate):
                lib_path = candidate
                break
        if lib_path is not None:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    for lib_name in main_libs:
        lib_path = None
        for base in lib_dirs:
            candidate = os.path.join(base, lib_name)
            if os.path.exists(candidate):
                lib_path = candidate
                break
        if lib_path is None:
            raise FileNotFoundError(f"ACLNN library not found: {lib_name}")
        libs.append(ctypes.CDLL(lib_path))
        resolved_paths.append(lib_path)
    _LIB_HANDLES = libs
    # Initialize the Cython FFI layer with resolved library paths when available.
    if _ffi is not None:
        try:
            _ffi.init(resolved_paths)
        except Exception:  # pylint: disable=broad-except
            pass
    return libs


def _init_aclnn(bindings):
    global _ACLNN_INITIALIZED
    if _ACLNN_INITIALIZED:
        return
    # Write a minimal config JSON for aclnnInit.
    # Without this config, some ACLNN ops (e.g. aclnnGroupNorm) can corrupt
    # internal state and cause subsequent ops to fail with error 561103.
    # This matches the pattern used by torch_npu and MindSpore.
    import json, tempfile, os
    config = {"dump": {"dump_scene": "lite_exception"}}
    config_path = os.path.join(tempfile.gettempdir(), "candle_aclnn_config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(config, f)
    ret = bindings.aclnn_init(config_path.encode("utf-8"))
    if ret != 0:
        raise RuntimeError(f"aclnnInit failed: {ret}")
    _ACLNN_INITIALIZED = True
    _register_cleanup()




def _register_cleanup():
    global _CLEANUP_REGISTERED
    if _CLEANUP_REGISTERED:
        return
    atexit.register(_cleanup_aclnn)
    _CLEANUP_REGISTERED = True


def _executor_handle(executor):
    if executor is None:
        return 0
    value = getattr(executor, "value", executor)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _apply_deferred_cleanup(cleanup):
    if cleanup is None:
        return
    for kind, resource in cleanup:
        try:
            if _ffi is None or not _ffi.is_initialized():
                continue
            if kind == "tensor":
                _ffi.destroy_tensor(int(resource))
            elif kind == "scalar":
                _ffi.destroy_scalar(int(resource))
        except Exception:
            pass


def _run_deferred_executor_cleanup(handle):
    cleanup = _DEFERRED_EXECUTOR_CLEANUP.pop(handle, None)
    _apply_deferred_cleanup(cleanup)


def _register_deferred_executor_cleanup(executor, cleanup):
    handle = _executor_handle(executor)
    if handle == 0 or not cleanup:
        return
    _register_cleanup()
    _DEFERRED_EXECUTOR_CLEANUP.setdefault(handle, []).extend(cleanup)


def _destroy_deferred_executor(executor):
    handle = _executor_handle(executor)
    if handle == 0:
        return
    _run_deferred_executor_cleanup(handle)


def _cleanup_aclnn():
    global _ACLNN_FINALIZED, _DEFERRED_EXECUTORS
    if _ACLNN_FINALIZED:
        return
    for executor in _DEFERRED_EXECUTORS:
        try:
            _destroy_deferred_executor(executor)
        except Exception:
            pass
    _DEFERRED_EXECUTORS = []
    _ACLNN_FINALIZED = True


def _defer_executor(executor, cleanup=None):
    handle = _executor_handle(executor)
    if handle == 0:
        if cleanup:
            _apply_deferred_cleanup(cleanup)
        return
    _register_cleanup()
    if cleanup:
        _register_deferred_executor_cleanup(handle, cleanup)
    _DEFERRED_EXECUTORS.append(handle)


def flush_deferred_executors():
    """Destroy all deferred ACLNN executors to prevent pool exhaustion.

    Called by runtime.synchronize() so that executor handles are reclaimed
    at the same sync point that drains workspace memory.  Without this,
    the executor list grows unboundedly and eventually causes
    aclnnMatmulGetWorkspaceSize (and other ops) to fail with 561103.
    """
    global _DEFERRED_EXECUTORS  # pylint: disable=global-statement
    if not _DEFERRED_EXECUTORS:
        return
    executors = _DEFERRED_EXECUTORS
    _DEFERRED_EXECUTORS = []
    for executor in executors:
        try:
            _destroy_deferred_executor(executor)
        except Exception:
            pass


def get_bindings():
    global _BINDINGS
    if _BINDINGS is None:
        libs = _load_libs()
        _BINDINGS = AclnnBindings(libs)
        _init_aclnn(_BINDINGS)
    return _BINDINGS


def symbols_ok():
    try:
        bindings = get_bindings()
        required = [
            bindings.aclnn_add_get_workspace,
            bindings.aclnn_add,
            bindings.aclnn_mul_get_workspace,
            bindings.aclnn_mul,
            bindings.aclnn_relu_get_workspace,
            bindings.aclnn_relu,
            bindings.aclnn_reduce_sum_get_workspace,
            bindings.aclnn_reduce_sum,
        ]
        return all(required)
    except Exception:
        return False


def is_available():
    try:
        _load_libs()
        return True
    except Exception:
        return False


def add(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
        out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("add")
    # Create alpha=1 scalar via FFI
    alpha_bytes = _scalar_bytes(1, dtype)
    alpha_handle = _ffi.create_scalar(alpha_bytes, dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Add")
        ws_size, executor = _ffi.binary_op_with_alpha(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr),
            alpha_handle, stream_ptr)
        workspace = None
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size,
                               executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAdd failed: {ret}")
        _maybe_sync(runtime)
    finally:
        cleanup = [("scalar", int(alpha_handle))] if int(alpha_handle) != 0 else None
        _defer_executor(ctypes.c_void_p(executor), cleanup=cleanup)
        if cleanup is None:
            _ffi.destroy_scalar(alpha_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def mul(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
        out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("mul")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Mul")
        ws_size, executor = _ffi.binary_op_no_alpha(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr),
            stream_ptr)
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size,
                               executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMul failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def sub(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
        out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("sub")
    alpha_bytes = _scalar_bytes(1, dtype)
    alpha_handle = _ffi.create_scalar(alpha_bytes, dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Sub")
        ws_size, executor = _ffi.binary_op_with_alpha(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr),
            alpha_handle, stream_ptr)
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size,
                               executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSub failed: {ret}")
        _maybe_sync(runtime)
    finally:
        cleanup = [("scalar", int(alpha_handle))] if int(alpha_handle) != 0 else None
        _defer_executor(ctypes.c_void_p(executor), cleanup=cleanup)
        if cleanup is None:
            _ffi.destroy_scalar(alpha_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def div(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
        out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("div")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Div")
        ws_size, executor = _ffi.binary_op_no_alpha(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr),
            stream_ptr)
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size,
                               executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnDiv failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def add_scalar(self_ptr, scalar_value, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("add_scalar")
    scalar_handle = _ffi.create_scalar(_scalar_bytes(scalar_value, dtype), dtype_code)
    alpha_handle = _ffi.create_scalar(_scalar_bytes(1, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Adds")
        ws_size, executor = _ffi.tensor_scalar_op_with_alpha(
            getws_ptr, exec_ptr,
            shape, stride,
            shape, stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(out_ptr),
            scalar_handle, alpha_handle,
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAdds failed: {ret}")
        _maybe_sync(runtime)
    finally:
        cleanup = []
        if int(scalar_handle) != 0:
            cleanup.append(("scalar", int(scalar_handle)))
        if int(alpha_handle) != 0:
            cleanup.append(("scalar", int(alpha_handle)))
        _defer_executor(ctypes.c_void_p(executor), cleanup=cleanup or None)
        if not cleanup:
            if int(scalar_handle) != 0:
                _ffi.destroy_scalar(scalar_handle)
            if int(alpha_handle) != 0:
                _ffi.destroy_scalar(alpha_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def sub_scalar(self_ptr, scalar_value, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("sub_scalar")
    scalar_handle = _ffi.create_scalar(_scalar_bytes(scalar_value, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Subs")
        ws_size, executor = _ffi.tensor_scalar_op_no_alpha(
            getws_ptr, exec_ptr,
            shape, stride,
            shape, stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(out_ptr),
            scalar_handle,
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSubs failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(scalar_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def maximum(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride, out_shape, out_stride,
            dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("maximum")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Maximum")
        ws_size, executor = _ffi.binary_op_no_alpha(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr),
            stream_ptr)
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMaximum failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def minimum(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride, out_shape, out_stride,
            dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("minimum")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Minimum")
        ws_size, executor = _ffi.binary_op_no_alpha(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr),
            stream_ptr)
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMinimum failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def atan(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_atan_get_workspace is None or bindings.aclnn_atan is None:
        raise RuntimeError("aclnnAtan symbols not available")
    return _unary_call(bindings, "aclnnAtan", bindings.aclnn_atan_get_workspace, bindings.aclnn_atan,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def atan2(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride, out_shape, out_stride,
          dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("atan2")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Atan2")
        ws_size, executor = _ffi.binary_op_no_alpha(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr),
            stream_ptr)
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAtan2 failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def asin(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_asin_get_workspace is None or bindings.aclnn_asin is None:
        raise RuntimeError("aclnnAsin symbols not available")
    return _unary_call(bindings, "aclnnAsin", bindings.aclnn_asin_get_workspace, bindings.aclnn_asin,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def acos(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_acos_get_workspace is None or bindings.aclnn_acos is None:
        raise RuntimeError("aclnnAcos symbols not available")
    return _unary_call(bindings, "aclnnAcos", bindings.aclnn_acos_get_workspace, bindings.aclnn_acos,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def asinh(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_asinh_get_workspace is None or bindings.aclnn_asinh is None:
        raise RuntimeError("aclnnAsinh symbols not available")
    return _unary_call(bindings, "aclnnAsinh", bindings.aclnn_asinh_get_workspace, bindings.aclnn_asinh,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def acosh(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_acosh_get_workspace is None or bindings.aclnn_acosh is None:
        raise RuntimeError("aclnnAcosh symbols not available")
    return _unary_call(bindings, "aclnnAcosh", bindings.aclnn_acosh_get_workspace, bindings.aclnn_acosh,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def atanh(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_atanh_get_workspace is None or bindings.aclnn_atanh is None:
        raise RuntimeError("aclnnAtanh symbols not available")
    return _unary_call(bindings, "aclnnAtanh", bindings.aclnn_atanh_get_workspace, bindings.aclnn_atanh,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def relu(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnRelu", bindings.aclnn_relu_get_workspace, bindings.aclnn_relu,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)




def inplace_one(out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_inplace_one_get_workspace is None or bindings.aclnn_inplace_one is None:
        raise RuntimeError("aclnnInplaceOne not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("inplace_one")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("InplaceOne")
        ws_size, executor = _ffi.inplace_unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            dtype_code,
            _ACL_FORMAT_ND,
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnInplaceOne failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def inplace_zero(out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_inplace_zero_get_workspace is None or bindings.aclnn_inplace_zero is None:
        raise RuntimeError("aclnnInplaceZero not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("inplace_zero")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("InplaceZero")
        ws_size, executor = _ffi.inplace_unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            dtype_code,
            _ACL_FORMAT_ND,
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnInplaceZero failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def reduce_sum(self_ptr, out_ptr, shape, stride, dtype, dims, keepdim, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("reduce_sum")
    executor = 0
    workspace = None
    try:
        dim_values = dims["dims"]
        if dim_values is None:
            dim_values = ()
        getws_ptr, exec_ptr = _ffi.resolve_op("ReduceSum")
        ws_size, executor = _ffi.reduce_sum_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            dims["out_shape"],
            dims["out_stride"],
            tuple(dim_values),
            bool(keepdim),
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnReduceSum failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)





def argmax(self_ptr, out_ptr, shape, stride, dtype, dim, keepdim, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("argmax")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("ArgMax")
        ws_size, executor = _ffi.arg_reduce_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            int(dim),
            bool(keepdim),
            _dtype_to_acl(dtype),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnArgMax failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def argmin(self_ptr, out_ptr, shape, stride, dtype, dim, keepdim, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("argmin")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("ArgMin")
        ws_size, executor = _ffi.arg_reduce_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            int(dim),
            bool(keepdim),
            _dtype_to_acl(dtype),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnArgMin failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def max_dim(self_ptr, out_ptr, indices_ptr, shape, stride, dtype, dim, keepdim,
            out_shape, out_stride, index_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("max_dim")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("MaxDim")
        ws_size, executor = _ffi.dual_output_with_indices_op(
            "dim_reduce",
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            out_shape,
            index_stride,
            int(dim),
            bool(keepdim),
            False,
            0,
            _dtype_to_acl(dtype),
            _dtype_to_acl(dtype),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(indices_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMaxDim failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def min_dim(self_ptr, out_ptr, indices_ptr, shape, stride, dtype, dim, keepdim,
            out_shape, out_stride, index_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("min_dim")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("MinDim")
        ws_size, executor = _ffi.dual_output_with_indices_op(
            "dim_reduce",
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            out_shape,
            index_stride,
            int(dim),
            bool(keepdim),
            False,
            0,
            _dtype_to_acl(dtype),
            _dtype_to_acl(dtype),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(indices_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMinDim failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)



def cast(self_ptr, out_ptr, shape, stride, src_dtype, dst_dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("cast")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Cast")
        ws_size, executor = _ffi.cast_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            _dtype_to_acl(src_dtype),
            _dtype_to_acl(dst_dtype),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnCast failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def arange(start, end, step, out_ptr, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_arange_get_workspace is None or bindings.aclnn_arange is None:
        raise RuntimeError("aclnnArange symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("arange")
    start_scalar = _ffi.create_scalar(_scalar_bytes(start, dtype), dtype_code)
    end_scalar = _ffi.create_scalar(_scalar_bytes(end, dtype), dtype_code)
    step_scalar = _ffi.create_scalar(_scalar_bytes(step, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Arange")
        ws_size, executor = _ffi.output_tensor_three_scalars_op(
            getws_ptr,
            exec_ptr,
            out_shape,
            out_stride,
            dtype_code,
            _ACL_FORMAT_ND,
            int(out_ptr),
            int(start_scalar),
            int(end_scalar),
            int(step_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnArange failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(int(start_scalar))
        _ffi.destroy_scalar(int(end_scalar))
        _ffi.destroy_scalar(int(step_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def linspace(start, end, steps, out_ptr, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not linspace_symbols_ok():
        raise RuntimeError("aclnnLinspace symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("linspace")
    start_scalar = _ffi.create_scalar(_scalar_bytes(start, dtype), dtype_code)
    end_scalar = _ffi.create_scalar(_scalar_bytes(end, dtype), dtype_code)
    steps_scalar = _ffi.create_scalar(_scalar_bytes(int(steps), "int64"), _dtype_to_acl("int64"))
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Linspace")
        ws_size, executor = _ffi.output_tensor_three_scalars_op(
            getws_ptr,
            exec_ptr,
            out_shape,
            out_stride,
            dtype_code,
            _ACL_FORMAT_ND,
            int(out_ptr),
            int(start_scalar),
            int(end_scalar),
            int(steps_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLinspace failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(int(start_scalar))
        _ffi.destroy_scalar(int(end_scalar))
        _ffi.destroy_scalar(int(steps_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def eye(n, m, out_ptr, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not eye_symbols_ok():
        raise RuntimeError("aclnnEye symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("eye")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Eye")
        ws_size, executor = _ffi.output_tensor_two_ints_op(
            getws_ptr,
            exec_ptr,
            out_shape,
            out_stride,
            int(n),
            int(m),
            dtype_code,
            _ACL_FORMAT_ND,
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnEye failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def range_(start, end, step, out_ptr, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not range_symbols_ok():
        raise RuntimeError("aclnnRange symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("range_")
    start_scalar = _ffi.create_scalar(_scalar_bytes(start, dtype), dtype_code)
    end_scalar = _ffi.create_scalar(_scalar_bytes(end, dtype), dtype_code)
    step_scalar = _ffi.create_scalar(_scalar_bytes(step, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Range")
        ws_size, executor = _ffi.output_tensor_three_scalars_op(
            getws_ptr,
            exec_ptr,
            out_shape,
            out_stride,
            dtype_code,
            _ACL_FORMAT_ND,
            int(out_ptr),
            int(start_scalar),
            int(end_scalar),
            int(step_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnRange failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(int(start_scalar))
        _ffi.destroy_scalar(int(end_scalar))
        _ffi.destroy_scalar(int(step_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def flip(self_ptr, out_ptr, shape, stride, dtype, dims, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_flip_get_workspace is None or bindings.aclnn_flip is None:
        raise RuntimeError("aclnnFlip symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    normalized_dims = tuple(int(dim) for dim in dims)

    _require_native_npu_ffi("flip")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Flip")
        ws_size, executor = _ffi.tensor_int_array_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            normalized_dims,
            shape,
            stride,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnFlip failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def roll(self_ptr, out_ptr, shape, stride, dtype, shifts, dims, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_roll_get_workspace is None or bindings.aclnn_roll is None:
        raise RuntimeError("aclnnRoll symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    normalized_shifts = tuple(int(shift) for shift in shifts)
    normalized_dims = tuple(int(dim) for dim in dims)

    _require_native_npu_ffi("roll")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Roll")
        ws_size, executor = _ffi.tensor_two_int_arrays_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            normalized_shifts,
            normalized_dims,
            shape,
            stride,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnRoll failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def cumsum(self_ptr, out_ptr, shape, stride, self_dtype, dim, out_dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_cumsum_get_workspace is None or bindings.aclnn_cumsum is None:
        raise RuntimeError("aclnnCumsum symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    out_dtype_code = _dtype_to_acl(out_dtype)

    _require_native_npu_ffi("cumsum")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Cumsum")
        ws_size, executor = _ffi.axis_dtype_unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            int(dim),
            out_dtype_code,
            self_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnCumsum failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def cumprod(self_ptr, out_ptr, shape, stride, self_dtype, dim, out_dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_cumprod_get_workspace is None or bindings.aclnn_cumprod is None:
        raise RuntimeError("aclnnCumprod symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    out_dtype_code = _dtype_to_acl(out_dtype)

    _require_native_npu_ffi("cumprod")
    executor = 0
    workspace = None
    dim_scalar = 0
    try:
        dim_scalar = _ffi.create_scalar(_scalar_bytes(int(dim), "int32"), _dtype_to_acl("int32"))
        getws_ptr, exec_ptr = _ffi.resolve_op("Cumprod")
        ws_size, executor = _ffi.tensor_scalar_dtype_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            out_dtype_code,
            self_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(dim_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnCumprod failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if dim_scalar:
            _ffi.destroy_scalar(int(dim_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def cummax(self_ptr, values_ptr, indices_ptr, shape, stride, dtype, dim, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_cummax_get_workspace is None or bindings.aclnn_cummax is None:
        raise RuntimeError("aclnnCummax symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("cummax")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Cummax")
        ws_size, executor = _ffi.dual_output_with_indices_op(
            "cummax",
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            shape,
            stride,
            int(dim),
            False,
            False,
            0,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(values_ptr),
            int(indices_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnCummax failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def argsort(self_ptr, out_ptr, shape, stride, dim, descending, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("argsort")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Argsort")
        ws_size, executor = _ffi.argsort_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            int(dim),
            bool(descending),
            _dtype_to_acl(dtype),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnArgsort failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_free(workspace)


def sort(self_ptr, values_ptr, indices_ptr, shape, stride, dim, descending, stable, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("sort")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Sort")
        ws_size, executor = _ffi.dual_output_with_indices_op(
            "sort",
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            shape,
            stride,
            int(dim),
            bool(stable),
            bool(descending),
            0,
            _dtype_to_acl(dtype),
            _dtype_to_acl(dtype),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(values_ptr),
            int(indices_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSort failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_free(workspace)


def topk(self_ptr, values_ptr, indices_ptr, shape, stride, k, dim, largest, sorted_flag, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    out_shape = list(shape)
    out_shape[int(dim)] = int(k)
    out_shape = tuple(out_shape)
    out_stride = _contiguous_stride(out_shape)

    _require_native_npu_ffi("topk")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Topk")
        ws_size, executor = _ffi.dual_output_with_indices_op(
            "topk",
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            out_shape,
            out_stride,
            int(dim),
            bool(largest),
            bool(sorted_flag),
            int(k),
            _dtype_to_acl(dtype),
            _dtype_to_acl(dtype),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(values_ptr),
            int(indices_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnTopk failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_free(workspace)


def tril(self_ptr, out_ptr, shape, stride, diagonal, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_tril_get_workspace is None or bindings.aclnn_tril is None:
        raise RuntimeError("aclnnTril symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("tril")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Tril")
        ws_size, executor = _ffi.axis_unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            int(diagonal),
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnTril failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def triu(self_ptr, out_ptr, shape, stride, diagonal, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_triu_get_workspace is None or bindings.aclnn_triu is None:
        raise RuntimeError("aclnnTriu symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("triu")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Triu")
        ws_size, executor = _ffi.axis_unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            int(diagonal),
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnTriu failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def nonzero(self_ptr, out_ptr, shape, stride, dtype, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_nonzero_get_workspace is None or bindings.aclnn_nonzero is None:
        raise RuntimeError("aclnnNonzero symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("nonzero")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Nonzero")
        ws_size, executor = _ffi.unary_out_dtype_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            dtype_code,
            _dtype_to_acl("int64"),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnNonzero failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def repeat(self_ptr, out_ptr, shape, stride, dtype, repeats, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_repeat_get_workspace is None or bindings.aclnn_repeat is None:
        raise RuntimeError("aclnnRepeat symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    normalized_repeats = tuple(int(value) for value in repeats)

    _require_native_npu_ffi("repeat")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Repeat")
        ws_size, executor = _ffi.tensor_int_array_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            normalized_repeats,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnRepeat failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def _destroy_tensor_handles(handles):
    if _ffi is None:
        return
    for handle in handles:
        if handle:
            _ffi.destroy_tensor(int(handle))


def repeat_interleave_tensor(
    self_ptr,
    repeats_ptr,
    out_ptr,
    shape,
    stride,
    dtype,
    repeats_shape,
    repeats_stride,
    repeats_dtype,
    dim,
    output_size,
    out_shape,
    out_stride,
    runtime,
    stream=None,
):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()

    use_dim = dim is not None
    if use_dim:
        if (
            bindings.aclnn_repeat_interleave_with_dim_get_workspace is None
            or bindings.aclnn_repeat_interleave_with_dim is None
        ):
            raise RuntimeError("aclnnRepeatInterleaveWithDim symbols not available")
    else:
        if (
            bindings.aclnn_repeat_interleave_get_workspace is None
            or bindings.aclnn_repeat_interleave is None
        ):
            raise RuntimeError("aclnnRepeatInterleave symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    repeats_dtype_code = _dtype_to_acl(repeats_dtype)

    _require_native_npu_ffi("repeat_interleave_tensor")
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    self_handle = 0
    repeats_handle = 0
    out_handle = 0
    try:
        self_handle = _ffi.create_tensor(shape, stride, dtype_code, int(self_ptr), _ACL_FORMAT_ND)
        repeats_handle = _ffi.create_tensor(
            repeats_shape, repeats_stride, repeats_dtype_code, int(repeats_ptr), _ACL_FORMAT_ND
        )
        out_handle = _ffi.create_tensor(out_shape, out_stride, dtype_code, int(out_ptr), _ACL_FORMAT_ND)

        if use_dim:
            ret = bindings.aclnn_repeat_interleave_with_dim_get_workspace(
                ctypes.c_void_p(int(self_handle)),
                ctypes.c_void_p(int(repeats_handle)),
                ctypes.c_int64(int(dim)),
                ctypes.c_int64(int(output_size)),
                ctypes.c_void_p(int(out_handle)),
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
        else:
            ret = bindings.aclnn_repeat_interleave_get_workspace(
                ctypes.c_void_p(int(self_handle)),
                ctypes.c_void_p(int(repeats_handle)),
                ctypes.c_int64(int(output_size)),
                ctypes.c_void_p(int(out_handle)),
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")

        exec_handle = _executor_handle(executor)
        if exec_handle == 0:
            raise RuntimeError("aclnn repeat_interleave returned null executor")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        if use_dim:
            ret = bindings.aclnn_repeat_interleave_with_dim(
                ctypes.c_void_p(int(workspace or 0)),
                ctypes.c_uint64(int(workspace_size.value)),
                ctypes.c_void_p(exec_handle),
                ctypes.c_void_p(stream_ptr),
            )
            if ret != 0:
                raise RuntimeError(f"aclnnRepeatInterleaveWithDim failed: {ret}")
        else:
            ret = bindings.aclnn_repeat_interleave(
                ctypes.c_void_p(int(workspace or 0)),
                ctypes.c_uint64(int(workspace_size.value)),
                ctypes.c_void_p(exec_handle),
                ctypes.c_void_p(stream_ptr),
            )
            if ret != 0:
                raise RuntimeError(f"aclnnRepeatInterleave failed: {ret}")

        cleanup = [
            ("tensor", self_handle),
            ("tensor", repeats_handle),
            ("tensor", out_handle),
        ]
        self_handle = repeats_handle = out_handle = 0
        _defer_executor(executor, cleanup=cleanup)
        executor = ctypes.c_void_p()
        if runtime is not None:
            runtime.synchronize()
        _maybe_sync(runtime)
    finally:
        if _executor_handle(executor):
            cleanup = [
                ("tensor", self_handle),
                ("tensor", repeats_handle),
                ("tensor", out_handle),
            ]
            self_handle = repeats_handle = out_handle = 0
            _defer_executor(executor, cleanup=cleanup)
        _destroy_tensor_handles((self_handle, repeats_handle, out_handle))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def repeat_interleave_int(
    self_ptr,
    out_ptr,
    shape,
    stride,
    dtype,
    repeats,
    dim,
    output_size,
    out_shape,
    out_stride,
    runtime,
    stream=None,
):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()

    use_dim = dim is not None
    if use_dim:
        if (
            bindings.aclnn_repeat_interleave_int_with_dim_get_workspace is None
            or bindings.aclnn_repeat_interleave_int_with_dim is None
        ):
            raise RuntimeError("aclnnRepeatInterleaveIntWithDim symbols not available")
    else:
        if bindings.aclnn_repeat_interleave_int_get_workspace is None or bindings.aclnn_repeat_interleave_int is None:
            raise RuntimeError("aclnnRepeatInterleaveInt symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("repeat_interleave_int")
    executor = 0
    workspace = None
    try:
        if use_dim:
            getws_ptr, exec_ptr = _ffi.resolve_op("RepeatInterleaveIntWithDim")
            ws_size, executor = _ffi.tensor_three_ints_op(
                getws_ptr,
                exec_ptr,
                shape,
                stride,
                out_shape,
                out_stride,
                int(repeats),
                int(dim),
                int(output_size),
                dtype_code,
                dtype_code,
                _ACL_FORMAT_ND,
                int(self_ptr),
                int(out_ptr),
                stream_ptr,
            )
        else:
            getws_ptr, exec_ptr = _ffi.resolve_op("RepeatInterleaveInt")
            ws_size, executor = _ffi.tensor_two_ints_op(
                getws_ptr,
                exec_ptr,
                shape,
                stride,
                out_shape,
                out_stride,
                int(repeats),
                int(output_size),
                dtype_code,
                dtype_code,
                _ACL_FORMAT_ND,
                int(self_ptr),
                int(out_ptr),
                stream_ptr,
            )

        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                if use_dim:
                    raise RuntimeError(f"aclnnRepeatInterleaveIntWithDim failed: {ret}")
                raise RuntimeError(f"aclnnRepeatInterleaveInt failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def scatter(
    self_ptr,
    index_ptr,
    src_ptr,
    out_ptr,
    self_shape,
    self_stride,
    self_dtype,
    index_shape,
    index_stride,
    index_dtype,
    src_shape,
    src_stride,
    src_dtype,
    dim,
    reduce,
    runtime,
    stream=None,
):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_scatter_get_workspace is None or bindings.aclnn_scatter is None:
        raise RuntimeError("aclnnScatter symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("scatter")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Scatter")
        ws_size, executor = _ffi.ternary_two_inputs_with_dims_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            index_shape,
            index_stride,
            src_shape,
            src_stride,
            self_shape,
            self_stride,
            int(dim),
            int(reduce),
            _dtype_to_acl(self_dtype),
            _dtype_to_acl(index_dtype),
            _dtype_to_acl(src_dtype),
            _dtype_to_acl(self_dtype),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(index_ptr),
            int(src_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnScatter failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def diag(self_ptr, out_ptr, shape, stride, dtype, diagonal, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_diag_get_workspace is None or bindings.aclnn_diag is None:
        raise RuntimeError("aclnnDiag symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("diag")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Diag")
        ws_size, executor = _ffi.axis_unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            int(diagonal),
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnDiag failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def index_put_impl(self_ptr, self_shape, self_stride, self_dtype,
                   index_entries,
                   values_ptr, values_shape, values_stride, values_dtype,
                   accumulate, unsafe, runtime, stream=None,
                   self_storage_offset=0, self_storage_dims=None, self_storage_ptr=None,
                   values_storage_offset=0, values_storage_dims=None, values_storage_ptr=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_index_put_impl_get_workspace is None or bindings.aclnn_index_put_impl is None:
        raise RuntimeError("aclnnIndexPutImpl symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    values_dtype_code = _dtype_to_acl(values_dtype)

    _require_native_npu_ffi("index_put_impl")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("IndexPutImpl")
        ws_size, executor = _ffi.index_put_impl_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            self_dtype_code,
            index_entries,
            values_shape,
            values_stride,
            values_dtype_code,
            bool(accumulate),
            bool(unsafe),
            int(self_ptr),
            int(values_ptr),
            _ACL_FORMAT_ND,
            stream_ptr,
            int(self_storage_offset),
            self_storage_dims,
            0 if self_storage_ptr is None else int(self_storage_ptr),
            int(values_storage_offset),
            values_storage_dims,
            0 if values_storage_ptr is None else int(values_storage_ptr),
        )
        exec_workspace = 0
        exec_workspace_size = ws_size
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            exec_workspace = int(workspace)
        ret = _ffi.execute(exec_ptr, exec_workspace, exec_workspace_size, executor, stream_ptr)
        if ret != 0:
            raise RuntimeError(f"aclnnIndexPutImpl failed: {ret}")
        if hasattr(runtime, "synchronize_stream"):
            runtime.synchronize_stream(stream_ptr)
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def _unary_call(bindings, name, get_workspace_fn, exec_fn, self_ptr, out_ptr, shape, stride, dtype, runtime, stream, out_dtype=None):
    _ = (bindings, get_workspace_fn, exec_fn)
    if out_dtype is None:
        out_dtype = dtype
    stream_ptr = int(runtime.stream if stream is None else stream)
    _require_native_npu_ffi(name.removeprefix("aclnn").lower())
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op(name.removeprefix("aclnn"))
        ws_size, executor = _ffi.unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            _dtype_to_acl(dtype),
            _dtype_to_acl(out_dtype),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"{name} failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_free(workspace)

def abs(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnAbs", bindings.aclnn_abs_get_workspace, bindings.aclnn_abs,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def neg(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnNeg", bindings.aclnn_neg_get_workspace, bindings.aclnn_neg,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def sign(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSign", bindings.aclnn_sign_get_workspace, bindings.aclnn_sign,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)










def logical_xor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
                out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not logical_xor_symbols_ok():
        raise RuntimeError("aclnnLogicalXor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("logical_xor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LogicalXor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, _dtype_to_acl("bool"), _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLogicalXor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)

def logical_or(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
               out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not logical_or_symbols_ok():
        raise RuntimeError("aclnnLogicalOr symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("logical_or")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LogicalOr")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, _dtype_to_acl("bool"), _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLogicalOr failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)

def logical_and(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
                out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not logical_and_symbols_ok():
        raise RuntimeError("aclnnLogicalAnd symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("logical_and")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LogicalAnd")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, _dtype_to_acl("bool"), _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLogicalAnd failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)



def swhere(cond_ptr, self_ptr, other_ptr, out_ptr, cond_shape, cond_stride, self_shape, self_stride,
          other_shape, other_stride, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_swhere_get_workspace is None or bindings.aclnn_swhere is None:
        raise RuntimeError("aclnnSWhere symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("swhere")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("SWhere")
        ws_size, executor = _ffi.where_op(
            getws_ptr, exec_ptr,
            cond_shape, cond_stride,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            _dtype_to_acl("bool"), dtype_code, dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(cond_ptr), int(self_ptr), int(other_ptr), int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSWhere failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def logical_not(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnLogicalNot", bindings.aclnn_logical_not_get_workspace,
                       bindings.aclnn_logical_not, self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


# Bitwise operations
def bitwise_not(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_bitwise_not_get_workspace is None or bindings.aclnn_bitwise_not is None:
        raise RuntimeError("aclnnBitwiseNot symbols not available")
    return _unary_call(bindings, "aclnnBitwiseNot", bindings.aclnn_bitwise_not_get_workspace,
                       bindings.aclnn_bitwise_not, self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def bitwise_and(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
                out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not bitwise_and_symbols_ok():
        raise RuntimeError("aclnnBitwiseAndTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("bitwise_and")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("BitwiseAndTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnBitwiseAndTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def bitwise_or(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
               out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not bitwise_or_symbols_ok():
        raise RuntimeError("aclnnBitwiseOrTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("bitwise_or")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("BitwiseOrTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnBitwiseOrTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def bitwise_xor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
                out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not bitwise_xor_symbols_ok():
        raise RuntimeError("aclnnBitwiseXorTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("bitwise_xor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("BitwiseXorTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnBitwiseXorTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def signbit(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSignbit", bindings.aclnn_signbit_get_workspace, bindings.aclnn_signbit,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream, out_dtype="bool")


def isfinite(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnIsFinite", bindings.aclnn_isfinite_get_workspace, bindings.aclnn_isfinite,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream, out_dtype="bool")


def isinf(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnIsInf", bindings.aclnn_isinf_get_workspace, bindings.aclnn_isinf,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream, out_dtype="bool")



def isposinf(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnIsPosInf", bindings.aclnn_isposinf_get_workspace, bindings.aclnn_isposinf,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream, out_dtype="bool")


def isneginf(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnIsNegInf", bindings.aclnn_isneginf_get_workspace, bindings.aclnn_isneginf,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream, out_dtype="bool")

def cosh(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnCosh", bindings.aclnn_cosh_get_workspace, bindings.aclnn_cosh,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def sinh(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSinh", bindings.aclnn_sinh_get_workspace, bindings.aclnn_sinh,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def erf(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnErf", bindings.aclnn_erf_get_workspace, bindings.aclnn_erf,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def erfc(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnErfc", bindings.aclnn_erfc_get_workspace, bindings.aclnn_erfc,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def softplus(self_ptr, out_ptr, shape, stride, dtype, beta, threshold, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_softplus_get_workspace is None or bindings.aclnn_softplus is None:
        raise RuntimeError("aclnnSoftplus symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("softplus")
    executor = 0
    workspace = None
    beta_scalar = 0
    threshold_scalar = 0
    try:
        beta_scalar = _ffi.create_scalar(_scalar_bytes(beta, dtype), dtype_code)
        threshold_scalar = _ffi.create_scalar(_scalar_bytes(threshold, dtype), dtype_code)
        getws_ptr, exec_ptr = _ffi.resolve_op("Softplus")
        ws_size, executor = _ffi.tensor_two_scalars_op(
            getws_ptr, exec_ptr,
            shape, stride,
            shape, stride,
            dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(out_ptr),
            int(beta_scalar), int(threshold_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSoftplus failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if beta_scalar:
            _ffi.destroy_scalar(int(beta_scalar))
        if threshold_scalar:
            _ffi.destroy_scalar(int(threshold_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def hardtanh(self_ptr, out_ptr, shape, stride, dtype, min_val, max_val, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_hardtanh_get_workspace is None or bindings.aclnn_hardtanh is None:
        raise RuntimeError("aclnnHardtanh symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("hardtanh")
    executor = 0
    workspace = None
    min_scalar = 0
    max_scalar = 0
    try:
        min_scalar = _ffi.create_scalar(_scalar_bytes(min_val, dtype), dtype_code)
        max_scalar = _ffi.create_scalar(_scalar_bytes(max_val, dtype), dtype_code)
        getws_ptr, exec_ptr = _ffi.resolve_op("Hardtanh")
        ws_size, executor = _ffi.tensor_two_scalars_op(
            getws_ptr, exec_ptr,
            shape, stride,
            shape, stride,
            dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(out_ptr),
            int(min_scalar), int(max_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnHardtanh failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if min_scalar:
            _ffi.destroy_scalar(int(min_scalar))
        if max_scalar:
            _ffi.destroy_scalar(int(max_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def clamp_scalar(self_ptr, out_ptr, shape, stride, dtype, min_val, max_val, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_clamp_get_workspace is None or bindings.aclnn_clamp is None:
        raise RuntimeError("aclnnClamp symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("clamp_scalar")
    executor = 0
    workspace = None
    min_scalar = 0
    max_scalar = 0
    try:
        if min_val is not None:
            min_scalar = _ffi.create_scalar(_scalar_bytes(min_val, dtype), dtype_code)
        if max_val is not None:
            max_scalar = _ffi.create_scalar(_scalar_bytes(max_val, dtype), dtype_code)
        getws_ptr, exec_ptr = _ffi.resolve_op("Clamp")
        ws_size, executor = _ffi.clamp_optional_scalars_op(
            getws_ptr, exec_ptr,
            shape, stride,
            shape, stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(out_ptr),
            int(min_scalar), int(max_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnClamp failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if min_scalar:
            _ffi.destroy_scalar(int(min_scalar))
        if max_scalar:
            _ffi.destroy_scalar(int(max_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def clamp_min_scalar(self_ptr, out_ptr, shape, stride, dtype, min_val, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_clamp_min_get_workspace is None or bindings.aclnn_clamp_min is None:
        raise RuntimeError("aclnnClampMin symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("clamp_min_scalar")
    executor = 0
    workspace = None
    min_scalar = 0
    try:
        min_scalar = _ffi.create_scalar(_scalar_bytes(min_val, dtype), dtype_code)
        getws_ptr, exec_ptr = _ffi.resolve_op("ClampMin")
        ws_size, executor = _ffi.tensor_scalar_op_no_alpha(
            getws_ptr, exec_ptr,
            shape, stride,
            shape, stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(out_ptr),
            int(min_scalar), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnClampMin failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if min_scalar:
            _ffi.destroy_scalar(int(min_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def clamp_max_scalar(self_ptr, out_ptr, shape, stride, dtype, max_val, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_clamp_max_get_workspace is None or bindings.aclnn_clamp_max is None:
        raise RuntimeError("aclnnClampMax symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("clamp_max_scalar")
    executor = 0
    workspace = None
    max_scalar = 0
    try:
        max_scalar = _ffi.create_scalar(_scalar_bytes(max_val, dtype), dtype_code)
        getws_ptr, exec_ptr = _ffi.resolve_op("ClampMax")
        ws_size, executor = _ffi.tensor_scalar_op_no_alpha(
            getws_ptr, exec_ptr,
            shape, stride,
            shape, stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(out_ptr),
            int(max_scalar), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnClampMax failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if max_scalar:
            _ffi.destroy_scalar(int(max_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def clamp_tensor(self_ptr, min_ptr, max_ptr, out_ptr, self_shape, self_stride, min_shape, min_stride,
                 max_shape, max_stride, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_clamp_tensor_get_workspace is None or bindings.aclnn_clamp_tensor is None:
        raise RuntimeError("aclnnClampTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("clamp_tensor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("ClampTensor")
        ws_size, executor = _ffi.clamp_tensor_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            min_shape, min_stride,
            max_shape, max_stride,
            out_shape, out_stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(min_ptr), int(max_ptr), int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnClampTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def clamp_min_tensor(self_ptr, min_ptr, out_ptr, self_shape, self_stride, min_shape, min_stride,
                     out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_clamp_min_tensor_get_workspace is None or bindings.aclnn_clamp_min_tensor is None:
        raise RuntimeError("aclnnClampMinTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("clamp_min_tensor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("ClampMinTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            min_shape, min_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(min_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnClampMinTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def clamp_max_tensor(self_ptr, max_ptr, out_ptr, self_shape, self_stride, max_shape, max_stride,
                     out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_clamp_max_tensor_get_workspace is None or bindings.aclnn_clamp_max_tensor is None:
        raise RuntimeError("aclnnClampMaxTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("clamp_max_tensor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("ClampMaxTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            max_shape, max_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(max_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnClampMaxTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)




def eq_scalar(self_ptr, scalar_value, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    _require_native_npu_ffi("eq_scalar")
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_eq_scalar_get_workspace is None or bindings.aclnn_eq_scalar is None:
        raise RuntimeError("aclnnEqScalar symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    executor = 0
    workspace = None
    scalar = 0
    try:
        scalar = _ffi.create_scalar(_scalar_bytes(scalar_value, dtype), dtype_code)
        getws_ptr, exec_ptr = _ffi.resolve_op("EqScalar")
        ws_size, executor = _ffi.tensor_scalar_bool_out_op(
            getws_ptr, exec_ptr,
            shape, stride,
            shape, stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(out_ptr),
            int(scalar), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnEqScalar failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if scalar:
            _ffi.destroy_scalar(int(scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)



def eq_tensor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
              out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not eq_tensor_symbols_ok():
        raise RuntimeError("aclnnEqTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("eq_tensor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("EqTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, _dtype_to_acl("bool"), _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnEqTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)

def ne_tensor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
              out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not ne_tensor_symbols_ok():
        raise RuntimeError("aclnnNeTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("ne_tensor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("NeTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, _dtype_to_acl("bool"), _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnNeTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def exp(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnExp", bindings.aclnn_exp_get_workspace, bindings.aclnn_exp,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def log(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnLog", bindings.aclnn_log_get_workspace, bindings.aclnn_log,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def expm1(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_expm1_get_workspace is None or bindings.aclnn_expm1 is None:
        raise RuntimeError("aclnnExpm1 symbols not available")
    return _unary_call(bindings, "aclnnExpm1", bindings.aclnn_expm1_get_workspace, bindings.aclnn_expm1,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def log1p(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_log1p_get_workspace is None or bindings.aclnn_log1p is None:
        raise RuntimeError("aclnnLog1p symbols not available")
    return _unary_call(bindings, "aclnnLog1p", bindings.aclnn_log1p_get_workspace, bindings.aclnn_log1p,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def sqrt(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSqrt", bindings.aclnn_sqrt_get_workspace, bindings.aclnn_sqrt,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def rsqrt(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnRsqrt", bindings.aclnn_rsqrt_get_workspace, bindings.aclnn_rsqrt,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def sin(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSin", bindings.aclnn_sin_get_workspace, bindings.aclnn_sin,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def cos(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnCos", bindings.aclnn_cos_get_workspace, bindings.aclnn_cos,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def tan(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnTan", bindings.aclnn_tan_get_workspace, bindings.aclnn_tan,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def tanh(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnTanh", bindings.aclnn_tanh_get_workspace, bindings.aclnn_tanh,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def sigmoid(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSigmoid", bindings.aclnn_sigmoid_get_workspace, bindings.aclnn_sigmoid,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def floor(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnFloor", bindings.aclnn_floor_get_workspace, bindings.aclnn_floor,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def ceil(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnCeil", bindings.aclnn_ceil_get_workspace, bindings.aclnn_ceil,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def round(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnRound", bindings.aclnn_round_get_workspace, bindings.aclnn_round,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def trunc(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnTrunc", bindings.aclnn_trunc_get_workspace, bindings.aclnn_trunc,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def frac(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnFrac", bindings.aclnn_frac_get_workspace, bindings.aclnn_frac,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def log2(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnLog2", bindings.aclnn_log2_get_workspace, bindings.aclnn_log2,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def log10(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnLog10", bindings.aclnn_log10_get_workspace, bindings.aclnn_log10,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def exp2(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnExp2", bindings.aclnn_exp2_get_workspace, bindings.aclnn_exp2,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def pow_tensor_tensor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
                      out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_pow_tensor_tensor_get_workspace is None or bindings.aclnn_pow_tensor_tensor is None:
        raise RuntimeError("aclnnPowTensorTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("pow_tensor_tensor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("PowTensorTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnPowTensorTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def pow_tensor_scalar(self_ptr, scalar_value, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_pow_tensor_scalar_get_workspace is None or bindings.aclnn_pow_tensor_scalar is None:
        raise RuntimeError("aclnnPowTensorScalar symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("pow_tensor_scalar")
    executor = 0
    workspace = None
    scalar = 0
    try:
        scalar = _ffi.create_scalar(_scalar_bytes(scalar_value, dtype), dtype_code)
        getws_ptr, exec_ptr = _ffi.resolve_op("PowTensorScalar")
        ws_size, executor = _ffi.tensor_scalar_op_no_alpha(
            getws_ptr, exec_ptr,
            shape, stride,
            shape, stride,
            dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(out_ptr),
            int(scalar), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnPowTensorScalar failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if scalar:
            _ffi.destroy_scalar(int(scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def matmul(a_ptr, b_ptr, out_ptr, a_shape, a_stride, b_shape, b_stride, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not bindings.aclnn_matmul_get_workspace or not bindings.aclnn_matmul:
        raise RuntimeError("aclnnMatmul symbols not available")
    is_batched = len(a_shape) > 2 or len(b_shape) > 2
    if is_batched:
        if not bindings.aclnn_batch_matmul_get_workspace or not bindings.aclnn_batch_matmul:
            raise RuntimeError("aclnnBatchMatMul symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("matmul")
    executor = 0
    workspace = None
    try:
        op_name = "BatchMatMul" if is_batched else "Matmul"
        getws_ptr, exec_ptr = _ffi.resolve_op(op_name)
        ws_size, executor = _ffi.binary_two_inputs_with_int8_op(
            getws_ptr, exec_ptr,
            a_shape, a_stride,
            b_shape, b_stride,
            out_shape, out_stride,
            1,
            dtype_code, dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(a_ptr), int(b_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                if is_batched:
                    raise RuntimeError(f"aclnnBatchMatMul failed: {ret}")
                raise RuntimeError(f"aclnnMatmul failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def fill_scalar_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_inplace_fill_scalar_get_workspace,
            bindings.aclnn_inplace_fill_scalar,
        ])
    except Exception:
        return False



def ones_zero_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_inplace_one_get_workspace,
            bindings.aclnn_inplace_one,
            bindings.aclnn_inplace_zero_get_workspace,
            bindings.aclnn_inplace_zero,
        ])
    except Exception:
        return False


def trunc_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_trunc_get_workspace, bindings.aclnn_trunc])
    except Exception:
        return False


def add_scalar_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_add_scalar_get_workspace, bindings.aclnn_add_scalar])
    except Exception:
        return False


def frac_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_frac_get_workspace, bindings.aclnn_frac])
    except Exception:
        return False


def sign_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_sign_get_workspace, bindings.aclnn_sign])
    except Exception:
        return False


def signbit_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_signbit_get_workspace, bindings.aclnn_signbit])
    except Exception:
        return False


def ne_tensor_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_ne_tensor_get_workspace, bindings.aclnn_ne_tensor])
    except Exception:
        return False

def logical_not_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_logical_not_get_workspace, bindings.aclnn_logical_not])
    except Exception:
        return False

def logical_and_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_logical_and_get_workspace, bindings.aclnn_logical_and])
    except Exception:
        return False

def eq_scalar_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_eq_scalar_get_workspace, bindings.aclnn_eq_scalar])
    except Exception:
        return False

def logical_or_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_logical_or_get_workspace, bindings.aclnn_logical_or])
    except Exception:
        return False

def logical_xor_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_logical_xor_get_workspace, bindings.aclnn_logical_xor])
    except Exception:
        return False


def bitwise_not_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_bitwise_not_get_workspace, bindings.aclnn_bitwise_not])
    except Exception:
        return False


def bitwise_and_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_bitwise_and_tensor_get_workspace, bindings.aclnn_bitwise_and_tensor])
    except Exception:
        return False


def bitwise_or_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_bitwise_or_tensor_get_workspace, bindings.aclnn_bitwise_or_tensor])
    except Exception:
        return False


def bitwise_xor_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_bitwise_xor_tensor_get_workspace, bindings.aclnn_bitwise_xor_tensor])
    except Exception:
        return False


def expm1_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_expm1_get_workspace, bindings.aclnn_expm1])
    except Exception:
        return False


def log1p_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_log1p_get_workspace, bindings.aclnn_log1p])
    except Exception:
        return False


def dot_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_dot_get_workspace, bindings.aclnn_dot])
    except Exception:
        return False


def mv_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_mv_get_workspace, bindings.aclnn_mv])
    except Exception:
        return False


def ger_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_ger_get_workspace, bindings.aclnn_ger])
    except Exception:
        return False


def median_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_median_get_workspace, bindings.aclnn_median])
    except Exception:
        return False


def median_dim_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_median_dim_get_workspace, bindings.aclnn_median_dim])
    except Exception:
        return False


def kthvalue_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_kthvalue_get_workspace, bindings.aclnn_kthvalue])
    except Exception:
        return False


def search_sorted_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_search_sorted_get_workspace, bindings.aclnn_search_sorted])
    except Exception:
        return False


def unique_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_unique_get_workspace, bindings.aclnn_unique])
    except Exception:
        return False


def randperm_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_randperm_get_workspace, bindings.aclnn_randperm])
    except Exception:
        return False


def flatten_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_flatten_get_workspace, bindings.aclnn_flatten])
    except Exception:
        return False


def eq_tensor_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_eq_tensor_get_workspace, bindings.aclnn_eq_tensor])
    except Exception:
        return False


def argmax_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_argmax_get_workspace, bindings.aclnn_argmax])
    except Exception:
        return False


def argmin_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_argmin_get_workspace, bindings.aclnn_argmin])
    except Exception:
        return False


def max_dim_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_max_dim_get_workspace, bindings.aclnn_max_dim])
    except Exception:
        return False


def min_dim_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_min_dim_get_workspace, bindings.aclnn_min_dim])
    except Exception:
        return False

def cast_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_cast_get_workspace, bindings.aclnn_cast])
    except Exception:
        return False

def isposinf_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_isposinf_get_workspace, bindings.aclnn_isposinf])
    except Exception:
        return False


def isneginf_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_isneginf_get_workspace, bindings.aclnn_isneginf])
    except Exception:
        return False


def arange_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_arange_get_workspace, bindings.aclnn_arange])
    except Exception:
        return False


def linspace_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_linspace_get_workspace, bindings.aclnn_linspace])
    except Exception:
        return False


def eye_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_eye_get_workspace, bindings.aclnn_eye])
    except Exception:
        return False


def range_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_range_get_workspace, bindings.aclnn_range])
    except Exception:
        return False


def flip_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_flip_get_workspace, bindings.aclnn_flip])
    except Exception:
        return False


def roll_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_roll_get_workspace, bindings.aclnn_roll])
    except Exception:
        return False


def cumsum_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_cumsum_get_workspace, bindings.aclnn_cumsum])
    except Exception:
        return False


def cumprod_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_cumprod_get_workspace, bindings.aclnn_cumprod])
    except Exception:
        return False


def cummax_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_cummax_get_workspace, bindings.aclnn_cummax])
    except Exception:
        return False


def argsort_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_argsort_get_workspace, bindings.aclnn_argsort])
    except Exception:
        return False


def sort_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_sort_get_workspace, bindings.aclnn_sort])
    except Exception:
        return False


def topk_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_topk_get_workspace, bindings.aclnn_topk])
    except Exception:
        return False


def tril_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_tril_get_workspace, bindings.aclnn_tril])
    except Exception:
        return False


def triu_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_triu_get_workspace, bindings.aclnn_triu])
    except Exception:
        return False


def nonzero_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_nonzero_get_workspace, bindings.aclnn_nonzero])
    except Exception:
        return False


def repeat_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_repeat_get_workspace, bindings.aclnn_repeat])
    except Exception:
        return False


def repeat_interleave_int_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_repeat_interleave_int_get_workspace,
            bindings.aclnn_repeat_interleave_int,
            bindings.aclnn_repeat_interleave_int_with_dim_get_workspace,
            bindings.aclnn_repeat_interleave_int_with_dim,
        ])
    except Exception:
        return False


def repeat_interleave_tensor_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_repeat_interleave_get_workspace,
            bindings.aclnn_repeat_interleave,
            bindings.aclnn_repeat_interleave_with_dim_get_workspace,
            bindings.aclnn_repeat_interleave_with_dim,
        ])
    except Exception:
        return False


def scatter_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_scatter_get_workspace, bindings.aclnn_scatter])
    except Exception:
        return False


def diag_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_diag_get_workspace, bindings.aclnn_diag])
    except Exception:
        return False


def index_put_impl_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_index_put_impl_get_workspace,
            bindings.aclnn_index_put_impl,
        ])
    except Exception:
        return False


def _create_tensor_list(bindings, tensor_ptrs, shapes, strides, dtypes):
    """Create aclTensorList from multiple tensors."""
    if bindings.acl_create_tensor_list is None:
        raise RuntimeError("aclCreateTensorList not available")

    num_tensors = len(tensor_ptrs)
    tensor_array = (ctypes.c_void_p * num_tensors)()
    tensor_keeps = []

    for i in range(num_tensors):
        created = _create_tensor(bindings, shapes[i], strides[i], dtypes[i], tensor_ptrs[i])  # pylint: disable=assignment-from-no-return
        tensor, keep = created  # pylint: disable=unpacking-non-sequence
        tensor_array[i] = tensor
        tensor_keeps.append((tensor, keep))

    tensor_list = bindings.acl_create_tensor_list(tensor_array, ctypes.c_uint64(num_tensors))
    if not tensor_list:
        raise RuntimeError("aclCreateTensorList failed")

    return tensor_list, tensor_keeps


def _create_tensor_list_with_nones(bindings, entries):
    """Create aclTensorList where entries may be None (null pointer in the list).

    *entries* is a list of either ``None`` or a tuple
    ``(data_ptr, shape, stride, dtype)`` for each dimension.
    """
    if bindings.acl_create_tensor_list is None:
        raise RuntimeError("aclCreateTensorList not available")

    num = len(entries)
    tensor_array = (ctypes.c_void_p * num)()
    tensor_keeps = []

    for i, entry in enumerate(entries):
        if entry is None:
            tensor_array[i] = ctypes.c_void_p(0)
            tensor_keeps.append(None)
        else:
            data_ptr, shape, stride, dtype = entry
            created = _create_tensor(bindings, shape, stride, dtype, data_ptr)  # pylint: disable=assignment-from-no-return
            tensor, keep = created  # pylint: disable=unpacking-non-sequence
            tensor_array[i] = tensor
            tensor_keeps.append((tensor, keep))

    tensor_list = bindings.acl_create_tensor_list(tensor_array, ctypes.c_uint64(num))
    if not tensor_list:
        raise RuntimeError("aclCreateTensorList failed")

    return tensor_list, tensor_keeps


def index_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_index_get_workspace,
            bindings.aclnn_index,
        ])
    except Exception:
        return False


def slice_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_slice_get_workspace,
            bindings.aclnn_slice,
        ])
    except Exception:
        return False


def index(self_ptr, self_shape, self_stride, self_dtype,
          index_entries, out_ptr, out_shape, out_stride, out_dtype,
          runtime, stream=None,
          self_storage_offset=0, self_storage_dims=None, self_storage_ptr=None):
    """aclnnIndex — advanced indexing getitem.

    *index_entries* is a list (length == ndim of self) where each element is
    either ``None`` (dimension not indexed) or a tuple
    ``(data_ptr, shape, stride, dtype)`` for an index tensor.
    """
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_index_get_workspace is None or bindings.aclnn_index is None:
        raise RuntimeError("aclnnIndex symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    out_dtype_code = _dtype_to_acl(out_dtype)

    _require_native_npu_ffi("index")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Index")
        ws_size, executor = _ffi.index_with_optional_tensor_list_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            self_dtype_code,
            index_entries,
            out_shape,
            out_stride,
            out_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
            int(self_storage_offset),
            self_storage_dims,
            0 if self_storage_ptr is None else int(self_storage_ptr),
        )
        exec_workspace = 0
        exec_workspace_size = ws_size
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            exec_workspace = int(workspace)
        ret = _ffi.execute(exec_ptr, exec_workspace, exec_workspace_size, executor, stream_ptr)
        if ret != 0:
            raise RuntimeError(f"aclnnIndex failed: {ret}")
        if hasattr(runtime, "synchronize_stream"):
            runtime.synchronize_stream(stream_ptr)
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def slice_op(self_ptr, self_shape, self_stride, self_dtype,
             dim, start, end, step,
             out_ptr, out_shape, out_stride, out_dtype,
             runtime, stream=None):
    """aclnnSlice — strided slicing on a single dimension."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_slice_get_workspace is None or bindings.aclnn_slice is None:
        raise RuntimeError("aclnnSlice symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("slice_op")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Slice")
        ws_size, executor = _ffi.slice_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            out_shape, out_stride,
            int(dim), int(start), int(end), int(step),
            _dtype_to_acl(self_dtype), _dtype_to_acl(out_dtype), _ACL_FORMAT_ND,
            int(self_ptr), int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSlice failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def cat(tensor_ptrs, shapes, strides, dtypes, dim, out_ptr, out_shape, out_stride, out_dtype, runtime, stream=None):
    """Concatenate tensors along an existing dimension using aclnnCat."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_cat_get_workspace is None or bindings.aclnn_cat is None:
        raise RuntimeError("aclnnCat symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("cat")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Cat")
        ws_size, executor = _ffi.tensor_list_axis_op(
            getws_ptr, exec_ptr,
            tuple(tensor_ptrs), tuple(shapes), tuple(strides), tuple(dtypes),
            int(dim),
            out_shape, out_stride,
            _dtype_to_acl(out_dtype), _ACL_FORMAT_ND,
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnCat failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def stack(tensor_ptrs, shapes, strides, dtypes, dim, out_ptr, out_shape, out_stride, out_dtype, runtime, stream=None):
    """Stack tensors along a new dimension using aclnnStack."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_stack_get_workspace is None or bindings.aclnn_stack is None:
        raise RuntimeError("aclnnStack symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("stack")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Stack")
        ws_size, executor = _ffi.tensor_list_axis_op(
            getws_ptr, exec_ptr,
            tuple(tensor_ptrs), tuple(shapes), tuple(strides), tuple(dtypes),
            int(dim),
            out_shape, out_stride,
            _dtype_to_acl(out_dtype), _ACL_FORMAT_ND,
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnStack failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def s_where(condition_ptr, self_ptr, other_ptr, out_ptr,
            condition_shape, condition_stride, condition_dtype,
            self_shape, self_stride, self_dtype,
            other_shape, other_stride, other_dtype,
            out_shape, out_stride, out_dtype,
            runtime, stream=None):
    """Element-wise where using aclnnSWhere."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_s_where_get_workspace is None or bindings.aclnn_s_where is None:
        raise RuntimeError("aclnnSWhere symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("s_where")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("SWhere")
        ws_size, executor = _ffi.where_op(
            getws_ptr, exec_ptr,
            condition_shape, condition_stride,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            _dtype_to_acl(condition_dtype), _dtype_to_acl(self_dtype), _dtype_to_acl(other_dtype), _dtype_to_acl(out_dtype), _ACL_FORMAT_ND,
            int(condition_ptr), int(self_ptr), int(other_ptr), int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSWhere failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def cat_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.acl_create_tensor_list,
            bindings.acl_destroy_tensor_list,
            bindings.aclnn_cat_get_workspace,
            bindings.aclnn_cat
        ])
    except Exception:
        return False


def stack_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.acl_create_tensor_list,
            bindings.acl_destroy_tensor_list,
            bindings.aclnn_stack_get_workspace,
            bindings.aclnn_stack
        ])
    except Exception:
        return False


def s_where_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_s_where_get_workspace, bindings.aclnn_s_where])
    except Exception:
        return False


def mean(self_ptr, out_ptr, shape, stride, dtype, dims, keepdim, out_shape, out_stride, runtime, stream=None):
    """Compute mean along dimensions using aclnnMean."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_mean_get_workspace is None or bindings.aclnn_mean is None:
        raise RuntimeError("aclnnMean symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("mean")
    executor = 0
    workspace = None
    try:
        dim_values = dims if dims is not None else ()
        getws_ptr, exec_ptr = _ffi.resolve_op("Mean")
        ws_size, executor = _ffi.reduce_sum_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            tuple(dim_values),
            bool(keepdim),
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMean failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def softmax(self_ptr, out_ptr, shape, stride, dtype, dim, runtime, stream=None):
    """Compute softmax along a dimension using aclnnSoftmax."""
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("softmax")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Softmax")
        ws_size, executor = _ffi.axis_unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            int(dim),
            _dtype_to_acl(dtype),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSoftmax failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def log_softmax(self_ptr, out_ptr, shape, stride, dtype, dim, runtime, stream=None):
    """Compute log_softmax along a dimension using aclnnLogSoftmax."""
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("log_softmax")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LogSoftmax")
        ws_size, executor = _ffi.axis_unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            int(dim),
            _dtype_to_acl(dtype),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLogSoftmax failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def gelu(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    """Compute GELU activation using aclnnGelu."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnGelu", bindings.aclnn_gelu_get_workspace, bindings.aclnn_gelu,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def layer_norm(input_ptr, weight_ptr, bias_ptr, out_ptr, mean_ptr, rstd_ptr,
               input_shape, input_stride, weight_shape, weight_stride,
               bias_shape, bias_stride, out_shape, out_stride,
               stats_shape, stats_stride, normalized_shape, eps, dtype, runtime, stream=None):
    """Compute layer normalization using aclnnLayerNorm."""
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("layer_norm")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LayerNorm")
        ws_size, executor = _ffi.layer_norm_op(
            getws_ptr,
            exec_ptr,
            input_shape,
            input_stride,
            out_shape,
            out_stride,
            stats_shape,
            stats_stride,
            weight_shape,
            weight_stride,
            bias_shape,
            bias_stride,
            normalized_shape,
            float(eps),
            _dtype_to_acl(dtype),
            _ACL_FORMAT_ND,
            int(input_ptr),
            0 if weight_ptr is None else int(weight_ptr),
            0 if bias_ptr is None else int(bias_ptr),
            int(out_ptr),
            0 if mean_ptr is None else int(mean_ptr),
            0 if rstd_ptr is None else int(rstd_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLayerNorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def embedding(weight_ptr, indices_ptr, out_ptr, weight_shape, weight_stride,
              indices_shape, indices_stride, out_shape, out_stride,
              weight_dtype, indices_dtype, runtime, stream=None):
    """Compute embedding lookup using aclnnEmbedding."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_embedding_get_workspace is None or bindings.aclnn_embedding is None:
        raise RuntimeError("aclnnEmbedding symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("embedding")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Embedding")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            weight_shape, weight_stride,
            indices_shape, indices_stride,
            out_shape, out_stride,
            _dtype_to_acl(weight_dtype), _dtype_to_acl(indices_dtype), _dtype_to_acl(weight_dtype), _ACL_FORMAT_ND,
            int(weight_ptr), int(indices_ptr), int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnEmbedding failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)




def gather(self_ptr, index_ptr, out_ptr,
           self_shape, self_stride, self_dtype,
           index_shape, index_stride, index_dtype,
           out_shape, out_stride, out_dtype,
           dim, runtime, stream=None):
    """Gather elements along dim using "aclnnGather"."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_gather_get_workspace is None or bindings.aclnn_gather is None:
        raise RuntimeError("aclnnGather symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("gather")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Gather")
        ws_size, executor = _ffi.binary_two_inputs_with_dim_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            index_shape, index_stride,
            out_shape, out_stride,
            int(dim),
            _dtype_to_acl(self_dtype), _dtype_to_acl(index_dtype), _dtype_to_acl(out_dtype), _ACL_FORMAT_ND,
            int(self_ptr), int(index_ptr), int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnGather failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def masked_select(self_ptr, mask_ptr, out_ptr,
                  self_shape, self_stride, self_dtype,
                  mask_shape, mask_stride, mask_dtype,
                  out_shape, out_stride, out_dtype,
                  runtime, stream=None):
    """Masked select using "aclnnMaskedSelect"."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_masked_select_get_workspace is None or bindings.aclnn_masked_select is None:
        raise RuntimeError("aclnnMaskedSelect symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("masked_select")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("MaskedSelect")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            mask_shape, mask_stride,
            out_shape, out_stride,
            _dtype_to_acl(self_dtype), _dtype_to_acl(mask_dtype), _dtype_to_acl(out_dtype), _ACL_FORMAT_ND,
            int(self_ptr), int(mask_ptr), int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMaskedSelect failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def constant_pad_nd(self_ptr, out_ptr,
                    self_shape, self_stride, self_dtype,
                    pad_widths, value,
                    out_shape, out_stride, out_dtype,
                    runtime, stream=None):
    """Pad tensor with constant value using aclnnConstantPadNd."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_constant_pad_nd_get_workspace is None or bindings.aclnn_constant_pad_nd is None:
        raise RuntimeError("aclnnConstantPadNd symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    out_dtype_code = _dtype_to_acl(out_dtype)
    normalized_pad_widths = tuple(int(x) for x in pad_widths)

    _require_native_npu_ffi("constant_pad_nd")
    executor = 0
    workspace = None
    value_scalar = 0
    try:
        value_scalar = _ffi.create_scalar(_scalar_bytes(value, self_dtype), self_dtype_code)
        getws_ptr, exec_ptr = _ffi.resolve_op("ConstantPadNd")
        ws_size, executor = _ffi.tensor_int_array_scalar_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            normalized_pad_widths,
            out_shape,
            out_stride,
            self_dtype_code,
            out_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(value_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnConstantPadNd failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if value_scalar:
            _ffi.destroy_scalar(int(value_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def gather_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_gather_get_workspace, bindings.aclnn_gather])
    except Exception:
        return False


def masked_select_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_masked_select_get_workspace, bindings.aclnn_masked_select])
    except Exception:
        return False

def mean_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_mean_get_workspace, bindings.aclnn_mean])
    except Exception:
        return False


def softmax_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_softmax_get_workspace, bindings.aclnn_softmax])
    except Exception:
        return False


def log_softmax_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_log_softmax_get_workspace, bindings.aclnn_log_softmax])
    except Exception:
        return False


def gelu_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_gelu_get_workspace, bindings.aclnn_gelu])
    except Exception:
        return False


def layer_norm_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_layer_norm_get_workspace, bindings.aclnn_layer_norm])
    except Exception:
        return False


def embedding_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_embedding_get_workspace, bindings.aclnn_embedding])
    except Exception:
        return False


def silu_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_silu_get_workspace, bindings.aclnn_silu])
    except Exception:
        return False


def leaky_relu_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_leaky_relu_get_workspace, bindings.aclnn_leaky_relu])
    except Exception:
        return False


def elu_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_elu_get_workspace, bindings.aclnn_elu])
    except Exception:
        return False


def mish_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_mish_get_workspace, bindings.aclnn_mish])
    except Exception:
        return False


def prelu_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_prelu_get_workspace, bindings.aclnn_prelu])
    except Exception:
        return False


def batch_norm_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_batch_norm_get_workspace, bindings.aclnn_batch_norm])
    except Exception:
        return False


def group_norm_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_group_norm_get_workspace, bindings.aclnn_group_norm])
    except Exception:
        return False


def gather_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_gather_get_workspace, bindings.aclnn_gather])
    except Exception:
        return False


def constant_pad_nd_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_constant_pad_nd_get_workspace, bindings.aclnn_constant_pad_nd])
    except Exception:
        return False


def silu(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    """Compute SiLU activation using aclnnSilu."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSilu", bindings.aclnn_silu_get_workspace, bindings.aclnn_silu,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def leaky_relu(self_ptr, out_ptr, shape, stride, dtype, negative_slope, runtime, stream=None):
    """Compute LeakyReLU activation using aclnnLeakyRelu."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_leaky_relu_get_workspace is None or bindings.aclnn_leaky_relu is None:
        raise RuntimeError("aclnnLeakyRelu symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("leaky_relu")
    executor = 0
    workspace = None
    slope_scalar = 0
    try:
        slope_scalar = _ffi.create_scalar(_scalar_bytes(negative_slope, dtype), dtype_code)
        getws_ptr, exec_ptr = _ffi.resolve_op("LeakyRelu")
        ws_size, executor = _ffi.leaky_relu_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(slope_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLeakyRelu failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if slope_scalar:
            _ffi.destroy_scalar(int(slope_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def elu(self_ptr, out_ptr, shape, stride, dtype, alpha, runtime, stream=None,
        scale=1.0, input_scale=1.0):
    """Compute ELU activation using aclnnElu."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_elu_get_workspace is None or bindings.aclnn_elu is None:
        raise RuntimeError("aclnnElu symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("elu")
    executor = 0
    workspace = None
    alpha_scalar = 0
    scale_scalar = 0
    input_scale_scalar = 0
    try:
        alpha_scalar = _ffi.create_scalar(_scalar_bytes(alpha, dtype), dtype_code)
        scale_scalar = _ffi.create_scalar(_scalar_bytes(scale, dtype), dtype_code)
        input_scale_scalar = _ffi.create_scalar(_scalar_bytes(input_scale, dtype), dtype_code)
        getws_ptr, exec_ptr = _ffi.resolve_op("Elu")
        ws_size, executor = _ffi.tensor_three_scalars_op(
            getws_ptr, exec_ptr,
            shape, stride,
            shape, stride,
            dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(out_ptr),
            int(alpha_scalar), int(scale_scalar), int(input_scale_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnElu failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if alpha_scalar:
            _ffi.destroy_scalar(int(alpha_scalar))
        if scale_scalar:
            _ffi.destroy_scalar(int(scale_scalar))
        if input_scale_scalar:
            _ffi.destroy_scalar(int(input_scale_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def mish(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    """Compute Mish activation using aclnnMish."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_mish_get_workspace is None or bindings.aclnn_mish is None:
        raise RuntimeError("aclnnMish symbols not available")
    return _unary_call(bindings, "aclnnMish", bindings.aclnn_mish_get_workspace, bindings.aclnn_mish,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def prelu(self_ptr, weight_ptr, out_ptr, shape, stride, weight_shape, weight_stride, dtype, runtime, stream=None):
    """Compute PReLU activation using aclnnPrelu."""
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not prelu_symbols_ok():
        raise RuntimeError("aclnnPrelu symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("prelu")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Prelu")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            weight_shape,
            weight_stride,
            shape,
            stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(weight_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnPrelu failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def batch_norm(input_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr, out_ptr,
               input_shape, input_stride, weight_shape, weight_stride, bias_shape, bias_stride,
               running_mean_shape, running_mean_stride, running_var_shape, running_var_stride,
               out_shape, out_stride, training, momentum, eps, dtype, runtime, stream=None,
               ext_save_mean_ptr=None, ext_save_invstd_ptr=None):
    """Compute batch normalization using aclnnBatchNorm."""
    global acl
    _require_native_npu_ffi("batch_norm")
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_batch_norm_get_workspace is None or bindings.aclnn_batch_norm is None:
        raise RuntimeError("aclnnBatchNorm symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    stats_dtype_code = _dtype_to_acl("float32")

    C = input_shape[1] if len(input_shape) >= 2 else 1
    aux_shape = (C,)
    aux_stride = (1,)
    aux_itemsize = 4

    _own_save_ptrs = ext_save_mean_ptr is None
    save_mean_ptr = None
    save_invstd_ptr = None
    if _own_save_ptrs:
        save_mean_ptr = _npu_runtime_alloc_device(C * aux_itemsize, runtime=runtime)
        save_invstd_ptr = _npu_runtime_alloc_device(C * aux_itemsize, runtime=runtime)
    else:
        save_mean_ptr = ext_save_mean_ptr
        save_invstd_ptr = ext_save_invstd_ptr

    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("BatchNorm")
        ws_size, executor = _ffi.batch_norm_op(
            getws_ptr,
            exec_ptr,
            input_shape,
            input_stride,
            weight_shape,
            weight_stride,
            bias_shape,
            bias_stride,
            running_mean_shape,
            running_mean_stride,
            running_var_shape,
            running_var_stride,
            out_shape,
            out_stride,
            aux_shape,
            aux_stride,
            aux_shape,
            aux_stride,
            bool(training),
            float(momentum),
            float(eps),
            dtype_code,
            stats_dtype_code,
            _ACL_FORMAT_NCHW,
            _ACL_FORMAT_ND,
            _ACL_FORMAT_ND,
            int(input_ptr),
            0 if weight_ptr is None else int(weight_ptr),
            0 if bias_ptr is None else int(bias_ptr),
            0 if running_mean_ptr is None else int(running_mean_ptr),
            0 if running_var_ptr is None else int(running_var_ptr),
            int(out_ptr),
            int(save_mean_ptr),
            int(save_invstd_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnBatchNorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if save_mean_ptr is not None and _own_save_ptrs:
            runtime.defer_free(save_mean_ptr)
        if save_invstd_ptr is not None and _own_save_ptrs:
            runtime.defer_free(save_invstd_ptr)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def group_norm(input_ptr, weight_ptr, bias_ptr, out_ptr,
               input_shape, input_stride, weight_shape, weight_stride, bias_shape, bias_stride,
               out_shape, out_stride, num_groups, eps, dtype, runtime, stream=None):
    """Compute group normalization using aclnnGroupNorm."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_group_norm_get_workspace is None or bindings.aclnn_group_norm is None:
        raise RuntimeError("aclnnGroupNorm symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    stats_dtype_code = _dtype_to_acl("float32")

    _require_native_npu_ffi("group_norm")
    N = input_shape[0]
    C = input_shape[1]
    HxW = 1
    for dim in input_shape[2:]:
        HxW *= dim

    input_fmt = _ACL_FORMAT_NCHW if len(input_shape) >= 4 else _ACL_FORMAT_ND
    aux_shape = (N, num_groups)
    aux_stride = (num_groups, 1)
    aux_itemsize = 4
    aux_numel = N * num_groups
    mean_out_ptr = None
    rstd_out_ptr = None
    executor = 0
    workspace = None
    try:
        mean_out_ptr = _npu_runtime_alloc_device(aux_numel * aux_itemsize, runtime=runtime)
        rstd_out_ptr = _npu_runtime_alloc_device(aux_numel * aux_itemsize, runtime=runtime)
        getws_ptr, exec_ptr = _ffi.resolve_op("GroupNorm")
        ws_size, executor = _ffi.group_norm_op(
            getws_ptr,
            exec_ptr,
            input_shape,
            input_stride,
            weight_shape,
            weight_stride,
            bias_shape,
            bias_stride,
            out_shape,
            out_stride,
            aux_shape,
            aux_stride,
            aux_shape,
            aux_stride,
            int(N),
            int(C),
            int(HxW),
            int(num_groups),
            float(eps),
            dtype_code,
            stats_dtype_code,
            input_fmt,
            _ACL_FORMAT_ND,
            _ACL_FORMAT_ND,
            int(input_ptr),
            0 if weight_ptr is None else int(weight_ptr),
            0 if bias_ptr is None else int(bias_ptr),
            int(out_ptr),
            int(mean_out_ptr),
            int(rstd_out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnGroupNorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if mean_out_ptr is not None:
            runtime.defer_free(mean_out_ptr)
        if rstd_out_ptr is not None:
            runtime.defer_free(rstd_out_ptr)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def dropout_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_dropout_gen_mask_get_workspace,
            bindings.aclnn_dropout_gen_mask,
            bindings.aclnn_dropout_do_mask_get_workspace,
            bindings.aclnn_dropout_do_mask,
        ])
    except Exception:
        return False


def inplace_normal_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_inplace_normal_get_workspace, bindings.aclnn_inplace_normal])
    except Exception:
        return False


def _align_up(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def dropout_gen_mask(shape, p, seed, offset, mask_ptr, mask_numel, runtime, stream=None):
    """Generate dropout mask using aclnnDropoutGenMask.

    The mask is bit-packed: output shape is (align(numel, 128) / 8,) with dtype uint8.
    """
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_dropout_gen_mask_get_workspace is None or bindings.aclnn_dropout_gen_mask is None:
        raise RuntimeError("aclnnDropoutGenMask symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    mask_shape = (mask_numel,)
    mask_stride = (1,)
    mask_dtype_code = _dtype_to_acl("uint8")

    _require_native_npu_ffi("dropout_gen_mask")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("DropoutGenMask")
        ws_size, executor = _ffi.output_tensor_int_array_double_two_ints_op(
            getws_ptr,
            exec_ptr,
            mask_shape,
            mask_stride,
            tuple(shape),
            float(p),
            int(seed),
            int(offset),
            mask_dtype_code,
            _ACL_FORMAT_ND,
            int(mask_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnDropoutGenMask failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def dropout_do_mask(input_ptr, mask_ptr, out_ptr, shape, stride, dtype, mask_numel, p, runtime, stream=None):
    """Apply dropout mask using aclnnDropoutDoMask."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_dropout_do_mask_get_workspace is None or bindings.aclnn_dropout_do_mask is None:
        raise RuntimeError("aclnnDropoutDoMask symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    mask_dtype_code = _dtype_to_acl("uint8")
    mask_shape = (mask_numel,)
    mask_stride = (1,)

    _require_native_npu_ffi("dropout_do_mask")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("DropoutDoMask")
        ws_size, executor = _ffi.two_tensor_one_double_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            mask_shape,
            mask_stride,
            shape,
            stride,
            float(p),
            dtype_code,
            mask_dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(input_ptr),
            int(mask_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnDropoutDoMask failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def inplace_normal(self_ptr, shape, stride, dtype, mean, std, seed, offset, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_inplace_normal_get_workspace is None or bindings.aclnn_inplace_normal is None:
        raise RuntimeError("aclnnInplaceNormal symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("inplace_normal")
    executor = 0
    workspace = None

    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("InplaceNormal")
        ws_size, executor = _ffi.inplace_normal_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            float(mean),
            float(std),
            int(seed),
            int(offset),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnInplaceNormal failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def inplace_uniform(self_ptr, shape, stride, dtype, low, high, seed, offset, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_inplace_uniform_get_workspace is None or bindings.aclnn_inplace_uniform is None:
        raise RuntimeError("aclnnInplaceUniform symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("inplace_uniform")
    executor = 0
    workspace = None

    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("InplaceUniform")
        ws_size, executor = _ffi.inplace_uniform_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            float(low),
            float(high),
            int(seed),
            int(offset),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnInplaceUniform failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)




def inplace_fill_scalar(self_ptr, shape, stride, dtype, value, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_inplace_fill_scalar_get_workspace is None or bindings.aclnn_inplace_fill_scalar is None:
        raise RuntimeError("aclnnInplaceFillScalar symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("inplace_fill_scalar")
    value_scalar = _ffi.create_scalar(_scalar_bytes(value, dtype), dtype_code)
    executor = 0
    workspace = None

    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("InplaceFillScalar")
        ws_size, executor = _ffi.inplace_fill_scalar_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(value_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnInplaceFillScalar failed: {ret}")
        _maybe_sync(runtime)
    finally:
        cleanup = [("scalar", int(value_scalar))] if int(value_scalar) != 0 else None
        _defer_executor(ctypes.c_void_p(executor), cleanup=cleanup)
        if cleanup is None:
            _ffi.destroy_scalar(int(value_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def inplace_copy(dst_ptr, src_ptr, dst_shape, dst_stride, dst_dtype, src_shape, src_stride, src_dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_inplace_copy_get_workspace is None or bindings.aclnn_inplace_copy is None:
        raise RuntimeError("aclnnInplaceCopy symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dst_dtype_code = _dtype_to_acl(dst_dtype)
    src_dtype_code = _dtype_to_acl(src_dtype)

    _require_native_npu_ffi("inplace_copy")
    executor = 0
    workspace = None

    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("InplaceCopy")
        ws_size, executor = _ffi.inplace_copy_op(
            getws_ptr,
            exec_ptr,
            dst_shape,
            dst_stride,
            src_shape,
            src_stride,
            dst_dtype_code,
            src_dtype_code,
            _ACL_FORMAT_ND,
            int(dst_ptr),
            int(src_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnInplaceCopy failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def erfinv(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    bindings = get_bindings()
    if bindings.aclnn_erfinv_get_workspace is None or bindings.aclnn_erfinv is None:
        raise RuntimeError("aclnnErfinv symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("erfinv")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Erfinv")
        ws_size, executor = _ffi.unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnErfinv failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def linalg_qr(self_ptr, q_ptr, r_ptr, self_shape, self_stride, q_shape, q_stride, r_shape, r_stride, dtype, mode, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    bindings = get_bindings()
    if bindings.aclnn_linalg_qr_get_workspace is None or bindings.aclnn_linalg_qr is None:
        raise RuntimeError("aclnnLinalgQr symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("linalg_qr")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LinalgQr")
        ws_size, executor = _ffi.three_tensor_one_int_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            q_shape,
            q_stride,
            r_shape,
            r_stride,
            int(mode),
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(q_ptr),
            int(r_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLinalgQr failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------------------
# Symbol checkers for new indexing ops
# ---------------------------------------------------------------------------

def masked_fill_scalar_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_inplace_masked_fill_scalar_get_workspace,
            bindings.aclnn_inplace_masked_fill_scalar,
        ])
    except Exception:
        return False


def index_copy_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_inplace_index_copy_get_workspace,
            bindings.aclnn_inplace_index_copy,
        ])
    except Exception:
        return False


def index_fill_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_inplace_index_fill_get_workspace,
            bindings.aclnn_inplace_index_fill,
        ])
    except Exception:
        return False


def index_add_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_index_add_get_workspace,
            bindings.aclnn_index_add,
        ])
    except Exception:
        return False


def scatter_add_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_scatter_add_get_workspace,
            bindings.aclnn_scatter_add,
        ])
    except Exception:
        return False


def masked_scatter_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_inplace_masked_scatter_get_workspace,
            bindings.aclnn_inplace_masked_scatter,
        ])
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Wrapper functions for new indexing ops
# ---------------------------------------------------------------------------

def inplace_masked_fill_scalar(self_ptr, self_shape, self_stride, self_dtype,
                               mask_ptr, mask_shape, mask_stride, mask_dtype,
                               value, runtime, stream=None):
    """aclnnInplaceMaskedFillScalar — in-place masked fill with scalar."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not masked_fill_scalar_symbols_ok():
        raise RuntimeError("aclnnInplaceMaskedFillScalar symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    mask_dtype_code = _dtype_to_acl(mask_dtype)

    _require_native_npu_ffi("inplace_masked_fill_scalar")
    scalar = _ffi.create_scalar(_scalar_bytes(value, self_dtype), self_dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("InplaceMaskedFillScalar")
        ws_size, executor = _ffi.inplace_masked_fill_scalar_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            mask_shape,
            mask_stride,
            self_dtype_code,
            mask_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(mask_ptr),
            int(scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnInplaceMaskedFillScalar failed: {ret}")
        _maybe_sync(runtime)
    finally:
        cleanup = [("scalar", int(scalar))] if int(scalar) != 0 else None
        _defer_executor(ctypes.c_void_p(executor), cleanup=cleanup)
        if cleanup is None:
            _ffi.destroy_scalar(int(scalar))
        if workspace is not None:
            runtime.defer_free(workspace)


def inplace_index_copy(self_ptr, self_shape, self_stride, self_dtype,
                       dim, index_ptr, index_shape, index_stride, index_dtype,
                       source_ptr, source_shape, source_stride, source_dtype,
                       runtime, stream=None):
    """aclnnInplaceIndexCopy — in-place index copy along dim."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not index_copy_symbols_ok():
        raise RuntimeError("aclnnInplaceIndexCopy symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    index_dtype_code = _dtype_to_acl(index_dtype)
    source_dtype_code = _dtype_to_acl(source_dtype)

    _require_native_npu_ffi("inplace_index_copy")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("InplaceIndexCopy")
        ws_size, executor = _ffi.inplace_index_copy_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            index_shape,
            index_stride,
            source_shape,
            source_stride,
            int(dim),
            self_dtype_code,
            index_dtype_code,
            source_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(index_ptr),
            int(source_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnInplaceIndexCopy failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_free(workspace)


def inplace_index_fill(self_ptr, self_shape, self_stride, self_dtype,
                       dim, index_ptr, index_shape, index_stride, index_dtype,
                       value, runtime, stream=None):
    """aclnnInplaceIndexFill — in-place index fill with scalar."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not index_fill_symbols_ok():
        raise RuntimeError("aclnnInplaceIndexFill symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    index_dtype_code = _dtype_to_acl(index_dtype)

    _require_native_npu_ffi("inplace_index_fill")
    scalar = _ffi.create_scalar(_scalar_bytes(value, self_dtype), self_dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("InplaceIndexFill")
        ws_size, executor = _ffi.inplace_index_fill_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            index_shape,
            index_stride,
            int(dim),
            self_dtype_code,
            index_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(index_ptr),
            int(scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnInplaceIndexFill failed: {ret}")
        _maybe_sync(runtime)
    finally:
        cleanup = [("scalar", int(scalar))] if int(scalar) != 0 else None
        _defer_executor(ctypes.c_void_p(executor), cleanup=cleanup)
        if cleanup is None:
            _ffi.destroy_scalar(int(scalar))
        if workspace is not None:
            runtime.defer_free(workspace)


def index_add(self_ptr, self_shape, self_stride, self_dtype,
              dim, index_ptr, index_shape, index_stride, index_dtype,
              source_ptr, source_shape, source_stride, source_dtype,
              alpha, out_ptr, out_shape, out_stride, out_dtype,
              runtime, stream=None):
    """aclnnIndexAdd — index add along dim with alpha."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not index_add_symbols_ok():
        raise RuntimeError("aclnnIndexAdd symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    index_dtype_code = _dtype_to_acl(index_dtype)
    source_dtype_code = _dtype_to_acl(source_dtype)
    out_dtype_code = _dtype_to_acl(out_dtype)

    _require_native_npu_ffi("index_add")
    alpha_scalar = _ffi.create_scalar(_scalar_bytes(alpha, self_dtype), self_dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("IndexAdd")
        ws_size, executor = _ffi.index_add_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            index_shape,
            index_stride,
            source_shape,
            source_stride,
            out_shape,
            out_stride,
            int(dim),
            self_dtype_code,
            index_dtype_code,
            source_dtype_code,
            out_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(index_ptr),
            int(source_ptr),
            int(alpha_scalar),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnIndexAdd failed: {ret}")
        _maybe_sync(runtime)
    finally:
        cleanup = [("scalar", int(alpha_scalar))] if int(alpha_scalar) != 0 else None
        _defer_executor(ctypes.c_void_p(executor), cleanup=cleanup)
        if cleanup is None:
            _ffi.destroy_scalar(int(alpha_scalar))
        if workspace is not None:
            runtime.defer_free(workspace)


def scatter_add_op(self_ptr, self_shape, self_stride, self_dtype,
                   dim, index_ptr, index_shape, index_stride, index_dtype,
                   src_ptr, src_shape, src_stride, src_dtype,
                   out_ptr, out_shape, out_stride, out_dtype,
                   runtime, stream=None):
    """aclnnScatterAdd — scatter add along dim."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not scatter_add_symbols_ok():
        raise RuntimeError("aclnnScatterAdd symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    index_dtype_code = _dtype_to_acl(index_dtype)
    src_dtype_code = _dtype_to_acl(src_dtype)
    out_dtype_code = _dtype_to_acl(out_dtype)

    _require_native_npu_ffi("scatter_add")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("ScatterAdd")
        ws_size, executor = _ffi.scatter_add_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            index_shape,
            index_stride,
            src_shape,
            src_stride,
            out_shape,
            out_stride,
            int(dim),
            self_dtype_code,
            index_dtype_code,
            src_dtype_code,
            out_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(index_ptr),
            int(src_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnScatterAdd failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_free(workspace)


def inplace_masked_scatter(self_ptr, self_shape, self_stride, self_dtype,
                           mask_ptr, mask_shape, mask_stride, mask_dtype,
                           source_ptr, source_shape, source_stride, source_dtype,
                           runtime, stream=None):
    """aclnnInplaceMaskedScatter — in-place masked scatter."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not masked_scatter_symbols_ok():
        raise RuntimeError("aclnnInplaceMaskedScatter symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    mask_dtype_code = _dtype_to_acl(mask_dtype)
    source_dtype_code = _dtype_to_acl(source_dtype)

    _require_native_npu_ffi("inplace_masked_scatter")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("InplaceMaskedScatter")
        ws_size, executor = _ffi.inplace_masked_scatter_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            mask_shape,
            mask_stride,
            source_shape,
            source_stride,
            self_dtype_code,
            mask_dtype_code,
            source_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(mask_ptr),
            int(source_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnInplaceMaskedScatter failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_free(workspace)


# ---------------------------------------------------------------------------
# aclnnVar
# ---------------------------------------------------------------------------
def var_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_var_get_workspace, bindings.aclnn_var])
    except Exception:
        return False


def var(self_ptr, out_ptr, shape, stride, dtype, dims, unbiased, keepdim,
        out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    if not var_symbols_ok():
        raise RuntimeError("aclnnVar symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    dims_tuple = tuple(dims) if dims else tuple(range(len(shape)))

    _require_native_npu_ffi("var")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Var")
        ws_size, executor = _ffi.tensor_int_array_two_bools_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            dims_tuple,
            bool(unbiased),
            bool(keepdim),
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnVar failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------------------
# aclnnNorm
# ---------------------------------------------------------------------------
def norm_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_norm_get_workspace, bindings.aclnn_norm])
    except Exception:
        return False


def norm(self_ptr, out_ptr, shape, stride, dtype, p, dims, keepdim,
         out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    if not norm_symbols_ok():
        raise RuntimeError("aclnnNorm symbols not available")

    from ..._dtype import float32 as f32
    out_dtype = dtype if getattr(dtype, 'is_floating_point', True) else f32

    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(dtype)
    out_dtype_code = _dtype_to_acl(out_dtype)
    dims_tuple = (int(dims),) if isinstance(dims, int) else (tuple(dims) if dims is not None else tuple(range(len(shape))))

    _require_native_npu_ffi("norm")
    p_scalar = _ffi.create_scalar(_scalar_bytes(float(p), dtype), self_dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Norm")
        ws_size, executor = _ffi.tensor_scalar_int_array_bool_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            dims_tuple,
            bool(keepdim),
            self_dtype_code,
            out_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(p_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnNorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(int(p_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------------------
# aclnnProd / aclnnProdDim
# ---------------------------------------------------------------------------
def prod_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_prod_get_workspace, bindings.aclnn_prod])
    except Exception:
        return False


def prod_dim_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_prod_dim_get_workspace, bindings.aclnn_prod_dim])
    except Exception:
        return False


def prod(self_ptr, out_ptr, shape, stride, dtype, dim, keepdim,
         out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("prod")
    executor = 0
    workspace = None
    try:
        if dim is not None:
            if not prod_dim_symbols_ok():
                raise RuntimeError("aclnnProdDim symbols not available")
            d = dim if dim >= 0 else dim + len(shape)
            getws_ptr, exec_ptr = _ffi.resolve_op("ProdDim")
            ws_size, executor = _ffi.axis_keepdim_dtype_op(
                getws_ptr,
                exec_ptr,
                shape,
                stride,
                out_shape,
                out_stride,
                int(d),
                bool(keepdim),
                dtype_code,
                dtype_code,
                _ACL_FORMAT_ND,
                int(self_ptr),
                int(out_ptr),
                stream_ptr,
            )
            op_name = "aclnnProdDim"
        else:
            if not prod_symbols_ok():
                raise RuntimeError("aclnnProd symbols not available")
            getws_ptr, exec_ptr = _ffi.resolve_op("Prod")
            ws_size, executor = _ffi.reduce_all_dtype_op(
                getws_ptr,
                exec_ptr,
                shape,
                stride,
                out_shape,
                out_stride,
                dtype_code,
                dtype_code,
                _ACL_FORMAT_ND,
                int(self_ptr),
                int(out_ptr),
                stream_ptr,
            )
            op_name = "aclnnProd"

        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"{op_name} failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------------------
# aclnnFloorDivide
# ---------------------------------------------------------------------------
def floor_divide_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_floor_divide_get_workspace, bindings.aclnn_floor_divide])
    except Exception:
        return False


def floor_divide(self_ptr, other_ptr, out_ptr,
                 self_shape, self_stride, other_shape, other_stride,
                 out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    if not floor_divide_symbols_ok():
        raise RuntimeError("aclnnFloorDivide symbols not available")

    dtype_code = _dtype_to_acl(dtype)
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("floor_divide")
    getws_ptr, exec_ptr = _ffi.resolve_op("FloorDivide")
    executor = 0
    workspace = None
    try:
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            other_shape,
            other_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(other_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnFloorDivide failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------------------
# aclnnRmsNorm
# ---------------------------------------------------------------------------
def rms_norm_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_rms_norm_get_workspace, bindings.aclnn_rms_norm])
    except Exception:
        return False


def rms_norm(x_ptr, gamma_ptr, eps, y_ptr, rstd_ptr,
             x_shape, x_stride, gamma_shape, gamma_stride,
             y_shape, y_stride, rstd_shape, rstd_stride,
             dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not rms_norm_symbols_ok():
        raise RuntimeError("aclnnRmsNorm symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("rms_norm")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("RmsNorm")
        ws_size, executor = _ffi.rms_norm_op(
            getws_ptr,
            exec_ptr,
            x_shape,
            x_stride,
            gamma_shape,
            gamma_stride,
            y_shape,
            y_stride,
            rstd_shape,
            rstd_stride,
            float(eps),
            dtype_code,
            _ACL_FORMAT_ND,
            int(x_ptr),
            int(gamma_ptr),
            int(y_ptr),
            int(rstd_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnRmsNorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------------------
# aclnnConvolution
# ---------------------------------------------------------------------------
def convolution_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_convolution_get_workspace, bindings.aclnn_convolution])
    except Exception:
        return False


def convolution(input_ptr, weight_ptr, bias_ptr,
                input_shape, input_stride, weight_shape, weight_stride,
                bias_shape, bias_stride, dtype,
                stride, padding, dilation, transposed, output_padding, groups,
                out_ptr, out_shape, out_stride,
                runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not convolution_symbols_ok():
        raise RuntimeError("aclnnConvolution symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("convolution")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Convolution")
        ws_size, executor = _ffi.convolution_op(
            getws_ptr,
            exec_ptr,
            input_shape,
            input_stride,
            weight_shape,
            weight_stride,
            bias_shape,
            bias_stride,
            out_shape,
            out_stride,
            tuple(stride),
            tuple(padding),
            tuple(dilation),
            tuple(output_padding),
            bool(transposed),
            int(groups),
            1,
            dtype_code,
            _ACL_FORMAT_NCHW,
            int(input_ptr),
            int(weight_ptr),
            0 if bias_ptr is None else int(bias_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnConvolution failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_free(workspace)


# ---------------------------------------------------------------------------
# aclnnMaxPool
# ---------------------------------------------------------------------------
def max_pool_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_max_pool_get_workspace, bindings.aclnn_max_pool])
    except Exception:
        return False


def max_pool(self_ptr, out_ptr, shape, stride_t, dtype,
             kernel_shape, strides, pads, dilations, ceil_mode,
             out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not max_pool_symbols_ok():
        raise RuntimeError("aclnnMaxPool symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("max_pool")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("MaxPool")
        ws_size, executor = _ffi.tensor_four_int_arrays_two_ints_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride_t,
            tuple(kernel_shape),
            tuple(strides),
            tuple(pads),
            tuple(dilations),
            out_shape,
            out_stride,
            0,
            1 if ceil_mode else 0,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_NCHW,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMaxPool failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------------------
# aclnnMaxPool2dWithMask — fp32/fp16-capable, used on Ascend910B
# ---------------------------------------------------------------------------
def max_pool2d_with_mask_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_max_pool2d_with_mask_get_workspace,
                    bindings.aclnn_max_pool2d_with_mask])
    except Exception:
        return False


def max_pool2d_with_mask(self_ptr, out_ptr, mask_ptr,
                         shape, stride_t, dtype,
                         kernel_size, strides, padding, dilations, ceil_mode,
                         out_shape, out_stride, mask_shape, mask_stride,
                         runtime, stream=None):
    """MaxPool2d via aclnnMaxPool2dWithMask, which supports fp32/fp16 on Ascend910B.

    The mask output is an int8 tensor used internally for backward; callers
    typically discard it for forward-only use.
    """
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not max_pool2d_with_mask_symbols_ok():
        raise RuntimeError("aclnnMaxPool2dWithMask symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    mask_dtype_code = _dtype_to_acl("int8")

    _require_native_npu_ffi("max_pool2d_with_mask")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("MaxPool2dWithMask")
        ws_size, executor = _ffi.tensor_four_int_arrays_bool_two_outputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride_t,
            tuple(kernel_size),
            tuple(strides),
            tuple(padding),
            tuple(dilations),
            out_shape,
            out_stride,
            mask_shape,
            mask_stride,
            bool(ceil_mode),
            dtype_code,
            dtype_code,
            mask_dtype_code,
            _ACL_FORMAT_NCHW,
            int(self_ptr),
            int(out_ptr),
            int(mask_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMaxPool2dWithMask failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------------------
# aclnnAvgPool2d
# ---------------------------------------------------------------------------
def avg_pool2d_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_avg_pool2d_get_workspace, bindings.aclnn_avg_pool2d])
    except Exception:
        return False


def avg_pool2d(self_ptr, out_ptr, shape, stride_t, dtype,
               kernel_size, strides, paddings, ceil_mode, count_include_pad,
               divisor_override, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not avg_pool2d_symbols_ok():
        raise RuntimeError("aclnnAvgPool2d symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("avg_pool2d")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("AvgPool2d")
        ws_size, executor = _ffi.tensor_three_int_arrays_two_bools_int64_int8_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride_t,
            tuple(kernel_size),
            tuple(strides),
            tuple(paddings),
            out_shape,
            out_stride,
            bool(ceil_mode),
            bool(count_include_pad),
            0 if divisor_override is None else int(divisor_override),
            1,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_NCHW,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAvgPool2d failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------------------
# aclnnAdaptiveAvgPool2d
# ---------------------------------------------------------------------------
def adaptive_avg_pool2d_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_adaptive_avg_pool2d_get_workspace, bindings.aclnn_adaptive_avg_pool2d])
    except Exception:
        return False


def adaptive_avg_pool2d(self_ptr, out_ptr, shape, stride_t, dtype,
                        output_size, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not adaptive_avg_pool2d_symbols_ok():
        raise RuntimeError("aclnnAdaptiveAvgPool2d symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("adaptive_avg_pool2d")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("AdaptiveAvgPool2d")
        ws_size, executor = _ffi.tensor_int_array_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride_t,
            tuple(output_size),
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_NCHW,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAdaptiveAvgPool2d failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ===========================================================================
# Backward wrapper functions
# ===========================================================================


def softmax_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_softmax_backward_get_workspace, b.aclnn_softmax_backward])
    except Exception:
        return False


def softmax_backward(grad_ptr, output_ptr, out_ptr, shape, grad_stride, output_stride, out_stride,
                     dtype, dim, runtime, stream=None):
    """aclnnSoftmaxBackward(gradOutput, output, dim, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not softmax_backward_symbols_ok():
        raise RuntimeError("aclnnSoftmaxBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("softmax_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("SoftmaxBackward")
        ws_size, executor = _ffi.binary_two_inputs_with_dim_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            output_stride,
            shape,
            out_stride,
            int(dim),
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(output_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSoftmaxBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def gelu_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_gelu_backward_get_workspace, b.aclnn_gelu_backward])
    except Exception:
        return False


def gelu_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                  dtype, runtime, stream=None):
    """aclnnGeluBackward(gradOutput, self, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not gelu_backward_symbols_ok():
        raise RuntimeError("aclnnGeluBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("gelu_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("GeluBackward")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnGeluBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def layer_norm_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_layer_norm_backward_get_workspace, b.aclnn_layer_norm_backward])
    except Exception:
        return False


def layer_norm_backward(grad_ptr, input_ptr, mean_ptr, rstd_ptr, weight_ptr, bias_ptr,
                        grad_input_ptr, grad_weight_ptr, grad_bias_ptr,
                        input_shape, input_stride, stats_shape, stats_stride,
                        weight_shape, weight_stride, bias_shape, bias_stride,
                        normalized_shape, dtype, runtime, stream=None):
    """aclnnLayerNormBackward with outputMask (aclBoolArray)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not layer_norm_backward_symbols_ok():
        raise RuntimeError("aclnnLayerNormBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    stats_dtype_code = _dtype_to_acl("float32")
    out_stride = _contiguous_stride(input_shape)
    output_mask = (True, grad_weight_ptr is not None, grad_bias_ptr is not None)

    _require_native_npu_ffi("layer_norm_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LayerNormBackward")
        ws_size, executor = _ffi.layer_norm_backward_op(
            getws_ptr,
            exec_ptr,
            input_shape,
            input_stride,
            input_shape,
            input_stride,
            input_shape,
            out_stride,
            stats_shape,
            stats_stride,
            weight_shape,
            weight_stride,
            bias_shape,
            bias_stride,
            tuple(normalized_shape),
            output_mask,
            dtype_code,
            stats_dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(input_ptr),
            int(mean_ptr),
            int(rstd_ptr),
            0 if weight_ptr is None else int(weight_ptr),
            0 if bias_ptr is None else int(bias_ptr),
            int(grad_input_ptr),
            0 if grad_weight_ptr is None else int(grad_weight_ptr),
            0 if grad_bias_ptr is None else int(grad_bias_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLayerNormBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)

def convolution_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_convolution_backward_get_workspace, b.aclnn_convolution_backward])
    except Exception:
        return False


def convolution_backward(grad_ptr, input_ptr, weight_ptr,
                         grad_shape, grad_stride, input_shape, input_stride,
                         weight_shape, weight_stride, dtype,
                         bias_sizes, stride, padding, dilation, transposed, output_padding, groups,
                         output_mask,
                         grad_input_ptr, grad_weight_ptr, grad_bias_ptr,
                         gi_shape, gi_stride, gw_shape, gw_stride, gb_shape, gb_stride,
                         runtime, stream=None):
    """aclnnConvolutionBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not convolution_backward_symbols_ok():
        raise RuntimeError("aclnnConvolutionBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("convolution_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("ConvolutionBackward")
        ws_size, executor = _ffi.convolution_backward_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            input_shape,
            input_stride,
            weight_shape,
            weight_stride,
            gi_shape,
            gi_stride,
            gw_shape,
            gw_stride,
            gb_shape,
            gb_stride,
            None if bias_sizes is None else tuple(bias_sizes),
            tuple(stride),
            tuple(padding),
            tuple(dilation),
            tuple(output_padding),
            tuple(bool(v) for v in output_mask),
            bool(transposed),
            int(groups),
            1,
            dtype_code,
            _ACL_FORMAT_NCHW,
            int(grad_ptr),
            int(input_ptr),
            int(weight_ptr),
            0 if grad_input_ptr is None else int(grad_input_ptr),
            0 if grad_weight_ptr is None else int(grad_weight_ptr),
            0 if grad_bias_ptr is None else int(grad_bias_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnConvolutionBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)

def batch_norm_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_batch_norm_backward_get_workspace, b.aclnn_batch_norm_backward])
    except Exception:
        return False


def batch_norm_backward(grad_ptr, input_ptr, weight_ptr,
                        running_mean_ptr, running_var_ptr,
                        save_mean_ptr, save_invstd_ptr,
                        grad_input_ptr, grad_weight_ptr, grad_bias_ptr,
                        grad_shape, grad_stride, input_shape, input_stride,
                        weight_shape, weight_stride,
                        rm_shape, rm_stride, rv_shape, rv_stride,
                        sm_shape, sm_stride, si_shape, si_stride,
                        gi_shape, gi_stride, gw_shape, gw_stride, gb_shape, gb_stride,
                        training, eps, output_mask, dtype, runtime, stream=None):
    """aclnnBatchNormBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not batch_norm_backward_symbols_ok():
        raise RuntimeError("aclnnBatchNormBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    stats_dtype_code = _dtype_to_acl("float32")

    _require_native_npu_ffi("batch_norm_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("BatchNormBackward")
        ws_size, executor = _ffi.batch_norm_backward_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            input_shape,
            input_stride,
            weight_shape,
            weight_stride,
            rm_shape,
            rm_stride,
            rv_shape,
            rv_stride,
            sm_shape,
            sm_stride,
            si_shape,
            si_stride,
            gi_shape,
            gi_stride,
            gw_shape,
            gw_stride,
            gb_shape,
            gb_stride,
            tuple(bool(v) for v in output_mask),
            bool(training),
            float(eps),
            dtype_code,
            stats_dtype_code,
            _ACL_FORMAT_NCHW,
            _ACL_FORMAT_ND,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(input_ptr),
            0 if weight_ptr is None else int(weight_ptr),
            0 if running_mean_ptr is None else int(running_mean_ptr),
            0 if running_var_ptr is None else int(running_var_ptr),
            int(save_mean_ptr),
            int(save_invstd_ptr),
            int(grad_input_ptr),
            0 if grad_weight_ptr is None else int(grad_weight_ptr),
            0 if grad_bias_ptr is None else int(grad_bias_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnBatchNormBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)

def embedding_dense_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_embedding_dense_backward_get_workspace, b.aclnn_embedding_dense_backward])
    except Exception:
        return False


def embedding_dense_backward(grad_ptr, indices_ptr, grad_weight_ptr,
                             grad_shape, grad_stride, indices_shape, indices_stride,
                             gw_shape, gw_stride, grad_dtype, indices_dtype,
                             num_weights, padding_idx, scale_grad_by_freq,
                             runtime, stream=None):
    """aclnnEmbeddingDenseBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not embedding_dense_backward_symbols_ok():
        raise RuntimeError("aclnnEmbeddingDenseBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    grad_dtype_code = _dtype_to_acl(grad_dtype)
    indices_dtype_code = _dtype_to_acl(indices_dtype)

    _require_native_npu_ffi("embedding_dense_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("EmbeddingDenseBackward")
        ws_size, executor = _ffi.two_tensor_two_ints_bool_mixed_fmt_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            indices_shape,
            indices_stride,
            gw_shape,
            gw_stride,
            int(num_weights),
            int(padding_idx if padding_idx is not None else -1),
            bool(scale_grad_by_freq),
            grad_dtype_code,
            indices_dtype_code,
            grad_dtype_code,
            _ACL_FORMAT_ND,
            _ACL_FORMAT_ND,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(indices_ptr),
            int(grad_weight_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnEmbeddingDenseBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------
# max_pool2d backward
# ---------------------------------------------------------------

def max_pool2d_with_mask_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_max_pool2d_with_mask_backward_get_workspace,
                    b.aclnn_max_pool2d_with_mask_backward])
    except Exception:
        return False


def max_pool2d_with_mask_backward(grad_ptr, input_ptr, mask_ptr, grad_input_ptr,
                                   grad_shape, grad_stride, input_shape, input_stride,
                                   mask_shape, mask_stride,
                                   gi_shape, gi_stride,
                                   kernel_size, strides, padding, dilation, ceil_mode,
                                   dtype, runtime, stream=None):
    """aclnnMaxPool2dWithMaskBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not max_pool2d_with_mask_backward_symbols_ok():
        raise RuntimeError("aclnnMaxPool2dWithMaskBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    mask_dtype_code = _dtype_to_acl("int8")

    _require_native_npu_ffi("max_pool2d_with_mask_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("MaxPool2dWithMaskBackward")
        ws_size, executor = _ffi.four_tensor_four_int_arrays_bool_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            input_shape,
            input_stride,
            mask_shape,
            mask_stride,
            gi_shape,
            gi_stride,
            tuple(kernel_size),
            tuple(strides),
            tuple(padding),
            tuple(dilation),
            bool(ceil_mode),
            dtype_code,
            dtype_code,
            mask_dtype_code,
            dtype_code,
            _ACL_FORMAT_NCHW,
            int(grad_ptr),
            int(input_ptr),
            int(mask_ptr),
            int(grad_input_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMaxPool2dWithMaskBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------
# avg_pool2d backward
# ---------------------------------------------------------------

def avg_pool2d_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_avg_pool2d_backward_get_workspace, b.aclnn_avg_pool2d_backward])
    except Exception:
        return False


def avg_pool2d_backward(grad_ptr, input_ptr, grad_input_ptr,
                         grad_shape, grad_stride, input_shape, input_stride,
                         gi_shape, gi_stride,
                         kernel_size, strides, padding,
                         ceil_mode, count_include_pad, divisor_override,
                         dtype, runtime, stream=None):
    """aclnnAvgPool2dBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not avg_pool2d_backward_symbols_ok():
        raise RuntimeError("aclnnAvgPool2dBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("avg_pool2d_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("AvgPool2dBackward")
        ws_size, executor = _ffi.four_tensor_three_int_arrays_two_bools_int64_int8_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            input_shape,
            input_stride,
            gi_shape,
            gi_stride,
            tuple(kernel_size),
            tuple(strides),
            tuple(padding),
            bool(ceil_mode),
            bool(count_include_pad),
            0 if divisor_override is None else int(divisor_override),
            1,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_NCHW,
            int(grad_ptr),
            int(input_ptr),
            int(grad_input_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAvgPool2dBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------
# rms_norm backward (grad)
# ---------------------------------------------------------------

def rms_norm_grad_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_rms_norm_grad_get_workspace, b.aclnn_rms_norm_grad])
    except Exception:
        return False


def rms_norm_grad(dy_ptr, x_ptr, rstd_ptr, gamma_ptr,
                  dx_ptr, dgamma_ptr,
                  dy_shape, dy_stride, x_shape, x_stride,
                  rstd_shape, rstd_stride,
                  gamma_shape, gamma_stride,
                  dx_shape, dx_stride, dgamma_shape, dgamma_stride,
                  dtype, runtime, stream=None):
    """aclnnRmsNormGrad(dy, x, rstd, gamma, dx, dgamma)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not all([bindings.aclnn_rms_norm_grad_get_workspace, bindings.aclnn_rms_norm_grad]):
        raise RuntimeError("aclnnRmsNormGrad symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("rms_norm_grad")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("RmsNormGrad")
        ws_size, executor = _ffi.rms_norm_grad_op(
            getws_ptr,
            exec_ptr,
            dy_shape,
            dy_stride,
            x_shape,
            x_stride,
            rstd_shape,
            rstd_stride,
            gamma_shape,
            gamma_stride,
            dx_shape,
            dx_stride,
            dgamma_shape,
            dgamma_stride,
            dtype_code,
            _ACL_FORMAT_ND,
            int(dy_ptr),
            int(x_ptr),
            int(rstd_ptr),
            int(gamma_ptr),
            int(dx_ptr),
            int(dgamma_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnRmsNormGrad failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)

# ---------------------------------------------------------------
# P1 ops: reciprocal, addmm, einsum, upsample_nearest2d,
#          upsample_bilinear2d, one_hot
# ---------------------------------------------------------------

def reciprocal_symbols_ok():
    b = get_bindings()
    return b.aclnn_reciprocal_get_workspace is not None and b.aclnn_reciprocal is not None

def reciprocal(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnReciprocal",
                       bindings.aclnn_reciprocal_get_workspace, bindings.aclnn_reciprocal,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def addmm_symbols_ok():
    b = get_bindings()
    return b.aclnn_addmm_get_workspace is not None and b.aclnn_addmm is not None

def addmm(self_ptr, mat1_ptr, mat2_ptr, out_ptr,
          self_shape, self_stride, self_dtype,
          mat1_shape, mat1_stride,
          mat2_shape, mat2_stride,
          out_shape, out_stride,
          beta, alpha, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not addmm_symbols_ok():
        raise RuntimeError("aclnnAddmm symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(self_dtype)

    _require_native_npu_ffi("addmm")
    beta_scalar = _ffi.create_scalar(_scalar_bytes(beta, self_dtype), dtype_code)
    alpha_scalar = _ffi.create_scalar(_scalar_bytes(alpha, self_dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Addmm")
        ws_size, executor = _ffi.four_tensor_two_scalars_one_int8_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            mat1_shape,
            mat1_stride,
            mat2_shape,
            mat2_stride,
            out_shape,
            out_stride,
            1,
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(mat1_ptr),
            int(mat2_ptr),
            int(out_ptr),
            int(beta_scalar),
            int(alpha_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAddmm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(int(beta_scalar))
        _ffi.destroy_scalar(int(alpha_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def einsum_symbols_ok():
    b = get_bindings()
    return b.aclnn_einsum_get_workspace is not None and b.aclnn_einsum is not None

def einsum(tensor_ptrs, shapes, strides, dtypes, equation,
           out_ptr, out_shape, out_stride, out_dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not einsum_symbols_ok():
        raise RuntimeError("aclnnEinsum symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    out_dtype_code = _dtype_to_acl(out_dtype)

    _require_native_npu_ffi("einsum")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Einsum")
        ws_size, executor = _ffi.tensor_list_string_op(
            getws_ptr,
            exec_ptr,
            tuple(tensor_ptrs),
            tuple(shapes),
            tuple(strides),
            tuple(dtypes),
            equation,
            out_shape,
            out_stride,
            out_dtype_code,
            _ACL_FORMAT_ND,
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnEinsum failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def upsample_nearest2d_symbols_ok():
    b = get_bindings()
    return b.aclnn_upsample_nearest2d_get_workspace is not None and b.aclnn_upsample_nearest2d is not None

def upsample_nearest2d(input_ptr, out_ptr, input_shape, input_stride, dtype,
                       output_size, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not upsample_nearest2d_symbols_ok():
        raise RuntimeError("aclnnUpsampleNearest2d symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("upsample_nearest2d")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("UpsampleNearest2d")
        ws_size, executor = _ffi.tensor_int_array_op(
            getws_ptr,
            exec_ptr,
            input_shape,
            input_stride,
            tuple(output_size),
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_NCHW,
            int(input_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnUpsampleNearest2d failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def upsample_bilinear2d_symbols_ok():
    b = get_bindings()
    return b.aclnn_upsample_bilinear2d_get_workspace is not None and b.aclnn_upsample_bilinear2d is not None

def upsample_bilinear2d(input_ptr, out_ptr, input_shape, input_stride, dtype,
                        output_size, align_corners, scales_h, scales_w,
                        out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not upsample_bilinear2d_symbols_ok():
        raise RuntimeError("aclnnUpsampleBilinear2d symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("upsample_bilinear2d")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("UpsampleBilinear2d")
        ws_size, executor = _ffi.tensor_int_array_bool_two_doubles_op(
            getws_ptr,
            exec_ptr,
            input_shape,
            input_stride,
            out_shape,
            out_stride,
            tuple(output_size),
            bool(align_corners),
            0.0 if scales_h is None else float(scales_h),
            0.0 if scales_w is None else float(scales_w),
            dtype_code,
            dtype_code,
            _ACL_FORMAT_NCHW,
            int(input_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnUpsampleBilinear2d failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def one_hot_symbols_ok():
    b = get_bindings()
    return b.aclnn_one_hot_get_workspace is not None and b.aclnn_one_hot is not None

def one_hot(self_ptr, on_ptr, off_ptr, out_ptr,
            self_shape, self_stride, self_dtype,
            on_shape, on_stride, on_dtype,
            off_shape, off_stride, off_dtype,
            out_shape, out_stride, out_dtype,
            num_classes, axis, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    get_bindings()
    if not one_hot_symbols_ok():
        raise RuntimeError("aclnnOneHot symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    on_dtype_code = _dtype_to_acl(on_dtype)
    off_dtype_code = _dtype_to_acl(off_dtype)
    out_dtype_code = _dtype_to_acl(out_dtype)

    _require_native_npu_ffi("one_hot")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("OneHot")
        ws_size, executor = _ffi.four_tensor_two_ints_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            on_shape,
            on_stride,
            off_shape,
            off_stride,
            out_shape,
            out_stride,
            int(num_classes),
            int(axis),
            self_dtype_code,
            on_dtype_code,
            off_dtype_code,
            out_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(on_ptr),
            int(off_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnOneHot failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# Dot product (vector dot vector -> scalar)
def dot(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
        out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_dot_get_workspace is None or bindings.aclnn_dot is None:
        raise RuntimeError("aclnnDot symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("dot")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Dot")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnDot failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# Matrix-vector multiplication
def mv(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
       out_shape, out_stride, dtype, cube_math_type, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_mv_get_workspace is None or bindings.aclnn_mv is None:
        raise RuntimeError("aclnnMv symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("mv")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Mv")
        ws_size, executor = _ffi.binary_two_inputs_with_int8_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            int(cube_math_type),
            dtype_code, dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMv failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# Ger (outer product): vector outer vector -> matrix
def ger(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
        out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_ger_get_workspace is None or bindings.aclnn_ger is None:
        raise RuntimeError("aclnnGer symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("ger")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Ger")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr, exec_ptr,
            self_shape, self_stride,
            other_shape, other_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, dtype_code, _ACL_FORMAT_ND,
            int(self_ptr), int(other_ptr), int(out_ptr), stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnGer failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# Global median (reduces all elements to scalar)
def median(self_ptr, out_ptr, shape, stride, dtype, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_median_get_workspace is None or bindings.aclnn_median is None:
        raise RuntimeError("aclnnMedian symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("median")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Median")
        ws_size, executor = _ffi.unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMedian failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# Median along a dimension
def median_dim(self_ptr, out_ptr, indices_ptr,
               shape, stride, dtype,
               out_shape, out_stride,
               dim, keepdim, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_median_dim_get_workspace is None or bindings.aclnn_median_dim is None:
        raise RuntimeError("aclnnMedianDim symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    indices_dtype_code = _dtype_to_acl("int64")

    _require_native_npu_ffi("median_dim")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("MedianDim")
        ws_size, executor = _ffi.three_tensor_two_ints_bool_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            out_shape,
            out_stride,
            int(dim),
            0,
            bool(keepdim),
            dtype_code,
            dtype_code,
            indices_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(indices_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMedianDim failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# Kthvalue
def kthvalue(self_ptr, out_ptr, indices_ptr,
             shape, stride, dtype,
             out_shape, out_stride,
             k, dim, keepdim, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_kthvalue_get_workspace is None or bindings.aclnn_kthvalue is None:
        raise RuntimeError("aclnnKthvalue symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    indices_dtype_code = _dtype_to_acl("int64")

    _require_native_npu_ffi("kthvalue")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Kthvalue")
        ws_size, executor = _ffi.three_tensor_two_ints_bool_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            out_shape,
            out_stride,
            int(k),
            int(dim),
            bool(keepdim),
            dtype_code,
            dtype_code,
            indices_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(indices_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnKthvalue failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# SearchSorted
def search_sorted(sorted_sequence_ptr, values_ptr, out_ptr,
                  sorted_sequence_shape, sorted_sequence_stride,
                  values_shape, values_stride, out_shape, out_stride,
                  dtype, out_int32, right, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_search_sorted_get_workspace is None or bindings.aclnn_search_sorted is None:
        raise RuntimeError("aclnnSearchSorted symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    input_dtype_code = _dtype_to_acl(dtype)
    out_dtype = "int32" if out_int32 else "int64"
    out_dtype_code = _dtype_to_acl(out_dtype)

    _require_native_npu_ffi("search_sorted")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("SearchSorted")
        ws_size, executor = _ffi.two_tensor_two_bools_op(
            getws_ptr,
            exec_ptr,
            sorted_sequence_shape,
            sorted_sequence_stride,
            values_shape,
            values_stride,
            out_shape,
            out_stride,
            bool(out_int32),
            bool(right),
            input_dtype_code,
            input_dtype_code,
            out_dtype_code,
            _ACL_FORMAT_ND,
            int(sorted_sequence_ptr),
            int(values_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSearchSorted failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# Unique
def unique(self_ptr, out_ptr, inverse_indices_ptr,
           shape, stride, dtype,
           out_shape, out_stride,
           inverse_shape, inverse_stride,
           sorted, return_inverse,
           runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_unique_get_workspace is None or bindings.aclnn_unique is None:
        raise RuntimeError("aclnnUnique symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(dtype)
    inverse_dtype_code = _dtype_to_acl("int64")

    _require_native_npu_ffi("unique")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Unique")
        ws_size, executor = _ffi.unary_two_bools_two_outputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            inverse_shape,
            inverse_stride,
            bool(sorted),
            bool(return_inverse),
            self_dtype_code,
            self_dtype_code,
            inverse_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(inverse_indices_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnUnique failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# Randperm
def randperm(n, out_ptr, dtype, runtime, stream=None, seed=None, offset=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_randperm_get_workspace is None or bindings.aclnn_randperm is None:
        raise RuntimeError("aclnnRandperm symbols not available")
    shape = (n,)
    stride = (1,)
    stream_ptr = int(runtime.stream if stream is None else stream)
    out_dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("randperm")
    executor = 0
    workspace = None
    try:
        if seed is None:
            import random
            seed = random.randint(0, 2**31 - 1)
        if offset is None:
            offset = 0
        getws_ptr, exec_ptr = _ffi.resolve_op("Randperm")
        ws_size, executor = _ffi.output_tensor_three_ints_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            int(n),
            int(seed),
            int(offset),
            out_dtype_code,
            _ACL_FORMAT_ND,
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnRandperm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# Flatten (ACLNN version always produces 2D output based on axis)
def flatten(self_ptr, out_ptr, shape, stride, dtype, axis, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_flatten_get_workspace is None or bindings.aclnn_flatten is None:
        raise RuntimeError("aclnnFlatten symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("flatten")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Flatten")
        ws_size, executor = _ffi.axis_unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            int(axis),
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnFlatten failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


# --- P0+P1 symbols_ok functions ---

def lerp_symbols_ok():
    b = get_bindings()
    return b.aclnn_lerp_get_workspace is not None and b.aclnn_lerp is not None

def addcmul_symbols_ok():
    b = get_bindings()
    return b.aclnn_addcmul_get_workspace is not None and b.aclnn_addcmul is not None

def addcdiv_symbols_ok():
    b = get_bindings()
    return b.aclnn_addcdiv_get_workspace is not None and b.aclnn_addcdiv is not None

def logaddexp_symbols_ok():
    b = get_bindings()
    return b.aclnn_logaddexp_get_workspace is not None and b.aclnn_logaddexp is not None

def logaddexp2_symbols_ok():
    b = get_bindings()
    return b.aclnn_logaddexp2_get_workspace is not None and b.aclnn_logaddexp2 is not None

def remainder_symbols_ok():
    b = get_bindings()
    return b.aclnn_remainder_tt_get_workspace is not None and b.aclnn_remainder_tt is not None

def fmod_symbols_ok():
    b = get_bindings()
    return b.aclnn_fmod_tensor_get_workspace is not None and b.aclnn_fmod_tensor is not None

def baddbmm_symbols_ok():
    b = get_bindings()
    return b.aclnn_baddbmm_get_workspace is not None and b.aclnn_baddbmm is not None

def trace_symbols_ok():
    b = get_bindings()
    return b.aclnn_trace_get_workspace is not None and b.aclnn_trace is not None

def cummin_symbols_ok():
    b = get_bindings()
    return b.aclnn_cummin_get_workspace is not None and b.aclnn_cummin is not None

def logsumexp_symbols_ok():
    b = get_bindings()
    return b.aclnn_logsumexp_get_workspace is not None and b.aclnn_logsumexp is not None

def renorm_symbols_ok():
    b = get_bindings()
    return b.aclnn_renorm_get_workspace is not None and b.aclnn_renorm is not None

def logical_xor_symbols_ok():
    b = get_bindings()
    return b.aclnn_logical_xor_get_workspace is not None and b.aclnn_logical_xor is not None


# --- P0 wrapper functions (replace composite ops) ---

def lerp_tensor(self_ptr, end_ptr, weight_ptr, out_ptr,
                self_shape, self_stride, end_shape, end_stride,
                weight_shape, weight_stride, out_shape, out_stride,
                dtype, runtime, stream=None):
    """lerp with tensor weight: self + weight * (end - self)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    if not lerp_symbols_ok():
        raise RuntimeError("aclnnLerp symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("lerp_tensor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Lerp")
        ws_size, executor = _ffi.four_tensor_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            end_shape,
            end_stride,
            weight_shape,
            weight_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(end_ptr),
            int(weight_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLerp failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def lerp_scalar(self_ptr, end_ptr, out_ptr,
                self_shape, self_stride, end_shape, end_stride,
                out_shape, out_stride, dtype, weight_value, runtime, stream=None):
    """lerp with scalar weight: self + weight * (end - self)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_lerps_get_workspace is None or bindings.aclnn_lerps is None:
        raise RuntimeError("aclnnLerps symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("lerp_scalar")
    weight_scalar = _ffi.create_scalar(_scalar_bytes(weight_value, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Lerps")
        ws_size, executor = _ffi.two_tensor_scalar_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            end_shape,
            end_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(end_ptr),
            int(out_ptr),
            int(weight_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLerps failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(int(weight_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def addcmul(self_ptr, t1_ptr, t2_ptr, out_ptr,
            self_shape, self_stride, t1_shape, t1_stride,
            t2_shape, t2_stride, out_shape, out_stride,
            dtype, value, runtime, stream=None):
    """self + value * (tensor1 * tensor2)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    if not addcmul_symbols_ok():
        raise RuntimeError("aclnnAddcmul symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("addcmul")
    value_scalar = _ffi.create_scalar(_scalar_bytes(value, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Addcmul")
        ws_size, executor = _ffi.three_tensor_scalar_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            t1_shape,
            t1_stride,
            t2_shape,
            t2_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(t1_ptr),
            int(t2_ptr),
            int(out_ptr),
            int(value_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAddcmul failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(int(value_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def addcdiv(self_ptr, t1_ptr, t2_ptr, out_ptr,
            self_shape, self_stride, t1_shape, t1_stride,
            t2_shape, t2_stride, out_shape, out_stride,
            dtype, value, runtime, stream=None):
    """self + value * (tensor1 / tensor2)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    if not addcdiv_symbols_ok():
        raise RuntimeError("aclnnAddcdiv symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("addcdiv")
    value_scalar = _ffi.create_scalar(_scalar_bytes(value, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Addcdiv")
        ws_size, executor = _ffi.three_tensor_scalar_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            t1_shape,
            t1_stride,
            t2_shape,
            t2_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(t1_ptr),
            int(t2_ptr),
            int(out_ptr),
            int(value_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAddcdiv failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(int(value_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def slogaddexp(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
               out_shape, out_stride, dtype, runtime, stream=None):
    """logaddexp: log(exp(a) + exp(b))"""
    global acl
    if acl is None:
        acl = ensure_acl()
    if not logaddexp_symbols_ok():
        raise RuntimeError("aclnnLogAddExp symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("slogaddexp")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LogAddExp")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            other_shape,
            other_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(other_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLogAddExp failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def slogaddexp2(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
                out_shape, out_stride, dtype, runtime, stream=None):
    """logaddexp2: log2(2^a + 2^b)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    if not logaddexp2_symbols_ok():
        raise RuntimeError("aclnnLogAddExp2 symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("slogaddexp2")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LogAddExp2")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            other_shape,
            other_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(other_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLogAddExp2 failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def sremainder(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
               out_shape, out_stride, dtype, runtime, stream=None):
    """remainder (Python-style modulo)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    if not remainder_symbols_ok():
        raise RuntimeError("aclnnRemainderTensorTensor symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("sremainder")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("RemainderTensorTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            other_shape,
            other_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(other_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnRemainderTensorTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def sfmod(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
          out_shape, out_stride, dtype, runtime, stream=None):
    """fmod (C-style modulo)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    if not fmod_symbols_ok():
        raise RuntimeError("aclnnFmodTensor symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("sfmod")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("FmodTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            other_shape,
            other_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(other_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnFmodTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# --- P1 wrapper functions (new ops) ---

def baddbmm(self_ptr, b1_ptr, b2_ptr, out_ptr,
            self_shape, self_stride, b1_shape, b1_stride,
            b2_shape, b2_stride, out_shape, out_stride,
            dtype, beta, alpha, runtime, stream=None):
    """beta * self + alpha * (batch1 @ batch2)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not baddbmm_symbols_ok():
        raise RuntimeError("aclnnBaddbmm symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("baddbmm")
    beta_scalar = _ffi.create_scalar(_scalar_bytes(beta, dtype), dtype_code)
    alpha_scalar = _ffi.create_scalar(_scalar_bytes(alpha, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Baddbmm")
        ws_size, executor = _ffi.four_tensor_two_scalars_one_int8_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            b1_shape,
            b1_stride,
            b2_shape,
            b2_stride,
            out_shape,
            out_stride,
            1,
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(b1_ptr),
            int(b2_ptr),
            int(out_ptr),
            int(beta_scalar),
            int(alpha_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnBaddbmm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(int(beta_scalar))
        _ffi.destroy_scalar(int(alpha_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def strace(self_ptr, out_ptr, shape, stride, dtype, out_shape, out_stride, runtime, stream=None):
    """Sum of diagonal elements of a 2D matrix."""
    global acl
    if acl is None:
        acl = ensure_acl()
    if not trace_symbols_ok():
        raise RuntimeError("aclnnTrace symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("strace")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Trace")
        ws_size, executor = _ffi.unary_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnTrace failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def cummin(self_ptr, values_ptr, indices_ptr, shape, stride, dtype,
           dim, out_shape, out_stride, runtime, stream=None):
    """Cumulative minimum along a dimension. Returns (values, indices)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not cummin_symbols_ok():
        raise RuntimeError("aclnnCummin symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("cummin")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Cummin")
        ws_size, executor = _ffi.dual_output_with_indices_op(
            "cummin",
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            out_shape,
            out_stride,
            int(dim),
            False,
            False,
            0,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(values_ptr),
            int(indices_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnCummin failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def logsumexp(self_ptr, out_ptr, shape, stride, dtype,
              dims, keepdim, out_shape, out_stride, runtime, stream=None):
    """LogSumExp reduction along dims."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not logsumexp_symbols_ok():
        raise RuntimeError("aclnnLogSumExp symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dims_tuple = (dims,) if isinstance(dims, int) else tuple(dims)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("logsumexp")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LogSumExp")
        ws_size, executor = _ffi.reduce_sum_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            dims_tuple,
            bool(keepdim),
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLogSumExp failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)

def renorm(self_ptr, out_ptr, shape, stride, dtype,
           p, dim, max_norm, runtime, stream=None):
    """Renormalize sub-tensors along dim so that p-norm <= max_norm."""
    global acl
    if acl is None:
        acl = ensure_acl()
    if not renorm_symbols_ok():
        raise RuntimeError("aclnnRenorm symbols not available")

    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("renorm")
    p_scalar = _ffi.create_scalar(_scalar_bytes(p, dtype), dtype_code)
    max_norm_scalar = _ffi.create_scalar(_scalar_bytes(max_norm, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Renorm")
        ws_size, executor = _ffi.tensor_two_scalars_dim_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            shape,
            stride,
            int(dim),
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(p_scalar),
            int(max_norm_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnRenorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(int(p_scalar))
        _ffi.destroy_scalar(int(max_norm_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# --- P0+P1 batch 3: symbols_ok + wrappers ---

def lt_tensor_symbols_ok():
    b = get_bindings()
    return b.aclnn_lt_tensor_get_workspace is not None and b.aclnn_lt_tensor is not None

def le_tensor_symbols_ok():
    b = get_bindings()
    return b.aclnn_le_tensor_get_workspace is not None and b.aclnn_le_tensor is not None

def gt_tensor_symbols_ok():
    b = get_bindings()
    return b.aclnn_gt_tensor_get_workspace is not None and b.aclnn_gt_tensor is not None

def ge_tensor_symbols_ok():
    b = get_bindings()
    return b.aclnn_ge_tensor_get_workspace is not None and b.aclnn_ge_tensor is not None

def isclose_symbols_ok():
    b = get_bindings()
    return b.aclnn_isclose_get_workspace is not None and b.aclnn_isclose is not None

def instance_norm_symbols_ok():
    b = get_bindings()
    return b.aclnn_instance_norm_get_workspace is not None and b.aclnn_instance_norm is not None

def nansum_symbols_ok():
    b = get_bindings()
    return b.aclnn_reduce_nansum_get_workspace is not None and b.aclnn_reduce_nansum is not None

def linalg_cross_symbols_ok():
    b = get_bindings()
    return b.aclnn_linalg_cross_get_workspace is not None and b.aclnn_linalg_cross is not None


def bincount_symbols_ok():
    b = get_bindings()
    return b.aclnn_bincount_get_workspace is not None and b.aclnn_bincount is not None


def adaptive_avg_pool3d_symbols_ok():
    b = get_bindings()
    return b.aclnn_adaptive_avg_pool3d_get_workspace is not None and b.aclnn_adaptive_avg_pool3d is not None


# Comparison ops: lt, le, gt, ge
def lt_tensor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
              out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not lt_tensor_symbols_ok():
        raise RuntimeError("aclnnLtTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("lt_tensor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LtTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            other_shape,
            other_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            _dtype_to_acl("bool"),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(other_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLtTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def le_tensor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
              out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not le_tensor_symbols_ok():
        raise RuntimeError("aclnnLeTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("le_tensor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LeTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            other_shape,
            other_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            _dtype_to_acl("bool"),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(other_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLeTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def gt_tensor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
              out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not gt_tensor_symbols_ok():
        raise RuntimeError("aclnnGtTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("gt_tensor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("GtTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            other_shape,
            other_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            _dtype_to_acl("bool"),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(other_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnGtTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def ge_tensor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
              out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not ge_tensor_symbols_ok():
        raise RuntimeError("aclnnGeTensor symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("ge_tensor")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("GeTensor")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            other_shape,
            other_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            _dtype_to_acl("bool"),
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(other_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnGeTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def sisclose(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
             out_shape, out_stride, dtype, rtol, atol, equal_nan, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    if not isclose_symbols_ok():
        raise RuntimeError("aclnnIsClose symbols not available")

    dtype_code = _dtype_to_acl(dtype)
    bool_code = _dtype_to_acl("bool")
    stream_ptr = int(runtime.stream if stream is None else stream)

    _require_native_npu_ffi("sisclose")
    getws_ptr, exec_ptr = _ffi.resolve_op("IsClose")
    executor = 0
    workspace = None
    try:
        ws_size, executor = _ffi.binary_two_inputs_three_attrs_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            other_shape,
            other_stride,
            out_shape,
            out_stride,
            float(rtol),
            float(atol),
            bool(equal_nan),
            dtype_code,
            dtype_code,
            bool_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(other_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnIsClose failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def sinstance_norm(x_ptr, gamma_ptr, beta_ptr, out_ptr, mean_ptr, var_ptr,
                   x_shape, x_stride, gamma_shape, gamma_stride,
                   beta_shape, beta_stride, out_shape, out_stride,
                   mean_shape, mean_stride, var_shape, var_stride,
                   dtype, eps, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not instance_norm_symbols_ok():
        raise RuntimeError("aclnnInstanceNorm symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    stats_dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("sinstance_norm")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("InstanceNorm")
        ws_size, executor = _ffi.six_tensor_string_double_op(
            getws_ptr,
            exec_ptr,
            x_shape,
            x_stride,
            gamma_shape,
            gamma_stride,
            beta_shape,
            beta_stride,
            out_shape,
            out_stride,
            mean_shape,
            mean_stride,
            var_shape,
            var_stride,
            b"NCHW\x00",
            float(eps),
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            stats_dtype_code,
            stats_dtype_code,
            _ACL_FORMAT_NCHW,
            _ACL_FORMAT_ND,
            _ACL_FORMAT_NCHW,
            int(x_ptr),
            int(gamma_ptr),
            int(beta_ptr),
            int(out_ptr),
            int(mean_ptr),
            int(var_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnInstanceNorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def reduce_nansum(self_ptr, out_ptr, shape, stride, dtype,
                  dims, keepdim, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not nansum_symbols_ok():
        raise RuntimeError("aclnnReduceNansum symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dims_tuple = (dims,) if isinstance(dims, int) else tuple(dims)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("reduce_nansum")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("ReduceNansum")
        ws_size, executor = _ffi.reduce_dims_dtype_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride,
            out_shape,
            out_stride,
            dims_tuple,
            bool(keepdim),
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnReduceNansum failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)

def linalg_cross(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
                 out_shape, out_stride, dtype, dim, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not linalg_cross_symbols_ok():
        raise RuntimeError("aclnnLinalgCross symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("linalg_cross")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LinalgCross")
        ws_size, executor = _ffi.binary_two_inputs_with_dim_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            other_shape,
            other_stride,
            out_shape,
            out_stride,
            int(dim),
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(other_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLinalgCross failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def im2col_symbols_ok():
    b = get_bindings()
    return b.aclnn_im2col_get_workspace is not None and b.aclnn_im2col is not None

def grid_sampler2d_symbols_ok():
    b = get_bindings()
    return b.aclnn_grid_sampler2d_get_workspace is not None and b.aclnn_grid_sampler2d is not None

def affine_grid_symbols_ok():
    b = get_bindings()
    return b.aclnn_affine_grid_get_workspace is not None and b.aclnn_affine_grid is not None


def sgrid_sampler2d(input_ptr, grid_ptr, out_ptr,
                    input_shape, input_stride, grid_shape, grid_stride,
                    out_shape, out_stride, dtype,
                    interpolation_mode, padding_mode, align_corners,
                    runtime, stream=None):
    """Grid sample 2D via aclnnGridSampler2D."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not grid_sampler2d_symbols_ok():
        raise RuntimeError("aclnnGridSampler2D symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("sgrid_sampler2d")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("GridSampler2D")
        ws_size, executor = _ffi.two_tensor_two_ints_bool_mixed_fmt_op(
            getws_ptr,
            exec_ptr,
            input_shape,
            input_stride,
            grid_shape,
            grid_stride,
            out_shape,
            out_stride,
            int(interpolation_mode),
            int(padding_mode),
            bool(align_corners),
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_NCHW,
            _ACL_FORMAT_ND,
            _ACL_FORMAT_NCHW,
            int(input_ptr),
            int(grid_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnGridSampler2D failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def saffine_grid(theta_ptr, out_ptr,
                 theta_shape, theta_stride, dtype,
                 size, align_corners,
                 out_shape, out_stride,
                 runtime, stream=None):
    """Affine grid generator via aclnnAffineGrid."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not affine_grid_symbols_ok():
        raise RuntimeError("aclnnAffineGrid symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("saffine_grid")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("AffineGrid")
        ws_size, executor = _ffi.tensor_int_array_bool_op(
            getws_ptr,
            exec_ptr,
            theta_shape,
            theta_stride,
            out_shape,
            out_stride,
            tuple(size),
            bool(align_corners),
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(theta_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAffineGrid failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)

# ===========================================================================
# Activation / softmax backward wrapper functions
# ===========================================================================


def threshold_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_threshold_backward_get_workspace, b.aclnn_threshold_backward])
    except Exception:
        return False


def threshold_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                       dtype, threshold, runtime, stream=None):
    """aclnnThresholdBackward(gradOutput, self, threshold, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not threshold_backward_symbols_ok():
        raise RuntimeError("aclnnThresholdBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("threshold_backward")
    scalar_handle = _ffi.create_scalar(_scalar_bytes(threshold, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("ThresholdBackward")
        ws_size, executor = _ffi.two_tensor_scalar_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            scalar_handle,
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnThresholdBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(scalar_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def hardshrink_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_hardshrink_backward_get_workspace, b.aclnn_hardshrink_backward])
    except Exception:
        return False


def hardshrink_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                        dtype, lambd, runtime, stream=None):
    """aclnnHardshrinkBackward(gradOutput, self, lambd, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not hardshrink_backward_symbols_ok():
        raise RuntimeError("aclnnHardshrinkBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("hardshrink_backward")
    scalar_handle = _ffi.create_scalar(_scalar_bytes(lambd, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("HardshrinkBackward")
        ws_size, executor = _ffi.two_tensor_scalar_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            scalar_handle,
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnHardshrinkBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(scalar_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def softshrink_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_softshrink_backward_get_workspace, b.aclnn_softshrink_backward])
    except Exception:
        return False


def softshrink_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                        dtype, lambd, runtime, stream=None):
    """aclnnSoftshrinkBackward(gradOutput, self, lambd, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not softshrink_backward_symbols_ok():
        raise RuntimeError("aclnnSoftshrinkBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("softshrink_backward")
    scalar_handle = _ffi.create_scalar(_scalar_bytes(lambd, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("SoftshrinkBackward")
        ws_size, executor = _ffi.two_tensor_scalar_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            scalar_handle,
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSoftshrinkBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(scalar_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def sigmoid_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_sigmoid_backward_get_workspace, b.aclnn_sigmoid_backward])
    except Exception:
        return False


def sigmoid_backward(grad_ptr, output_ptr, out_ptr, shape, grad_stride, output_stride, out_stride,
                     dtype, runtime, stream=None):
    """aclnnSigmoidBackward(gradOutput, output, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not sigmoid_backward_symbols_ok():
        raise RuntimeError("aclnnSigmoidBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("sigmoid_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("SigmoidBackward")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            output_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(output_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSigmoidBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def tanh_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_tanh_backward_get_workspace, b.aclnn_tanh_backward])
    except Exception:
        return False


def tanh_backward(grad_ptr, output_ptr, out_ptr, shape, grad_stride, output_stride, out_stride,
                  dtype, runtime, stream=None):
    """aclnnTanhBackward(gradOutput, output, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not tanh_backward_symbols_ok():
        raise RuntimeError("aclnnTanhBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("tanh_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("TanhBackward")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            output_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(output_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnTanhBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def silu_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_silu_backward_get_workspace, b.aclnn_silu_backward])
    except Exception:
        return False


def silu_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                  dtype, runtime, stream=None):
    """aclnnSiluBackward(gradOutput, self, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not silu_backward_symbols_ok():
        raise RuntimeError("aclnnSiluBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("silu_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("SiluBackward")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSiluBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def log_softmax_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_log_softmax_backward_get_workspace, b.aclnn_log_softmax_backward])
    except Exception:
        return False


def log_softmax_backward(grad_ptr, output_ptr, out_ptr, shape, grad_stride, output_stride, out_stride,
                         dtype, dim, runtime, stream=None):
    """aclnnLogSoftmaxBackward(gradOutput, output, dim, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not log_softmax_backward_symbols_ok():
        raise RuntimeError("aclnnLogSoftmaxBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("log_softmax_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LogSoftmaxBackward")
        ws_size, executor = _ffi.binary_two_inputs_with_dim_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            output_stride,
            shape,
            out_stride,
            int(dim),
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(output_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLogSoftmaxBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def square_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_square_get_workspace, bindings.aclnn_square])
    except Exception:
        return False


def square(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSquare", bindings.aclnn_square_get_workspace, bindings.aclnn_square,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def hardswish_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_hardswish_backward_get_workspace, b.aclnn_hardswish_backward])
    except Exception:
        return False


def hardswish_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                       dtype, runtime, stream=None):
    """aclnnHardswishBackward(gradOutput, self, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not hardswish_backward_symbols_ok():
        raise RuntimeError("aclnnHardswishBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("hardswish_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("HardswishBackward")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnHardswishBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def hardsigmoid_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_hardsigmoid_backward_get_workspace, b.aclnn_hardsigmoid_backward])
    except Exception:
        return False


def hardsigmoid_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                         dtype, runtime, stream=None):
    """aclnnHardsigmoidBackward(gradOutput, self, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not hardsigmoid_backward_symbols_ok():
        raise RuntimeError("aclnnHardsigmoidBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("hardsigmoid_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("HardsigmoidBackward")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnHardsigmoidBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def mish_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_mish_backward_get_workspace, b.aclnn_mish_backward])
    except Exception:
        return False


def mish_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                  dtype, runtime, stream=None):
    """aclnnMishBackward(gradOutput, self, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not mish_backward_symbols_ok():
        raise RuntimeError("aclnnMishBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("mish_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("MishBackward")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMishBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def softplus_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_softplus_backward_get_workspace, b.aclnn_softplus_backward])
    except Exception:
        return False


def softplus_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                      dtype, beta, threshold, runtime, stream=None):
    """aclnnSoftplusBackward(gradOutput, self, beta, threshold, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not softplus_backward_symbols_ok():
        raise RuntimeError("aclnnSoftplusBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("softplus_backward")
    beta_handle = _ffi.create_scalar(_scalar_bytes(beta, dtype), dtype_code)
    threshold_handle = _ffi.create_scalar(_scalar_bytes(threshold, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("SoftplusBackward")
        ws_size, executor = _ffi.two_tensor_two_scalars_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            beta_handle,
            threshold_handle,
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSoftplusBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(beta_handle)
        _ffi.destroy_scalar(threshold_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def hardtanh_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_hardtanh_backward_get_workspace, b.aclnn_hardtanh_backward])
    except Exception:
        return False


def hardtanh_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                      dtype, min_val, max_val, runtime, stream=None):
    """aclnnHardtanhBackward(gradOutput, self, min, max, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not hardtanh_backward_symbols_ok():
        raise RuntimeError("aclnnHardtanhBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("hardtanh_backward")
    min_handle = _ffi.create_scalar(_scalar_bytes(min_val, dtype), dtype_code)
    max_handle = _ffi.create_scalar(_scalar_bytes(max_val, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("HardtanhBackward")
        ws_size, executor = _ffi.two_tensor_two_scalars_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            min_handle,
            max_handle,
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnHardtanhBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(min_handle)
        _ffi.destroy_scalar(max_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def leaky_relu_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_leaky_relu_backward_get_workspace, b.aclnn_leaky_relu_backward])
    except Exception:
        return False


def leaky_relu_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                         dtype, negative_slope, runtime, stream=None):
    """aclnnLeakyReluBackward(gradOutput, self, negativeSlope, selfIsResult=false, gradInput)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not leaky_relu_backward_symbols_ok():
        raise RuntimeError("aclnnLeakyReluBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("leaky_relu_backward")
    slope_handle = _ffi.create_scalar(_scalar_bytes(negative_slope, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LeakyReluBackward")
        ws_size, executor = _ffi.two_tensor_scalar_bool_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            slope_handle,
            False,
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLeakyReluBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(slope_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def elu_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_elu_backward_get_workspace, b.aclnn_elu_backward])
    except Exception:
        return False


def elu_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                  dtype, alpha, scale, input_scale, runtime, stream=None):
    """aclnnEluBackward(gradOutput, alpha, scale, inputScale, isResult=false, self, gradInput)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not elu_backward_symbols_ok():
        raise RuntimeError("aclnnEluBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("elu_backward")
    alpha_handle = _ffi.create_scalar(_scalar_bytes(alpha, dtype), dtype_code)
    scale_handle = _ffi.create_scalar(_scalar_bytes(scale, dtype), dtype_code)
    input_scale_handle = _ffi.create_scalar(_scalar_bytes(input_scale, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("EluBackward")
        ws_size, executor = _ffi.two_tensor_three_scalars_bool_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            alpha_handle,
            scale_handle,
            input_scale_handle,
            False,
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnEluBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(alpha_handle)
        _ffi.destroy_scalar(scale_handle)
        _ffi.destroy_scalar(input_scale_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def prelu_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_prelu_backward_get_workspace, b.aclnn_prelu_backward])
    except Exception:
        return False


def prelu_backward(grad_ptr, self_ptr, weight_ptr, grad_input_ptr, grad_weight_ptr,
                    shape, grad_stride, self_stride, weight_shape, weight_stride,
                    gi_stride, gw_stride, dtype, runtime, stream=None):
    """aclnnPreluBackward(gradOutput, self, weight, gradInput, gradWeight)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not prelu_backward_symbols_ok():
        raise RuntimeError("aclnnPreluBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("prelu_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("PreluBackward")
        ws_size, executor = _ffi.three_tensor_two_outputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            self_stride,
            weight_shape,
            weight_stride,
            shape,
            gi_stride,
            weight_shape,
            gw_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(weight_ptr),
            int(grad_input_ptr),
            int(grad_weight_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnPreluBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------
# Upsample backward (5 ops)
# ---------------------------------------------------------------

def upsample_nearest2d_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_upsample_nearest2d_backward_get_workspace, b.aclnn_upsample_nearest2d_backward])
    except Exception:
        return False


def upsample_nearest2d_backward(grad_ptr, out_ptr,
                                 grad_shape, grad_stride, out_shape, out_stride,
                                 output_size, input_size,
                                 scales_h, scales_w,
                                 dtype, runtime, stream=None):
    """aclnnUpsampleNearest2dBackward(gradOut, outputSize, inputSize, scalesH, scalesW, gradInput)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not upsample_nearest2d_backward_symbols_ok():
        raise RuntimeError("aclnnUpsampleNearest2dBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("upsample_nearest2d_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("UpsampleNearest2dBackward")
        ws_size, executor = _ffi.tensor_two_int_arrays_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            tuple(output_size),
            tuple(input_size),
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnUpsampleNearest2dBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def upsample_bilinear2d_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_upsample_bilinear2d_backward_get_workspace, b.aclnn_upsample_bilinear2d_backward])
    except Exception:
        return False


def upsample_bilinear2d_backward(grad_ptr, out_ptr,
                                  grad_shape, grad_stride, out_shape, out_stride,
                                  output_size, input_size,
                                  align_corners, scales_h, scales_w,
                                  dtype, runtime, stream=None):
    """aclnnUpsampleBilinear2dBackward(gradOut, outputSize, inputSize, alignCorners, scalesH, scalesW, gradInput)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not upsample_bilinear2d_backward_symbols_ok():
        raise RuntimeError("aclnnUpsampleBilinear2dBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("upsample_bilinear2d_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("UpsampleBilinear2dBackward")
        ws_size, executor = _ffi.tensor_two_int_arrays_bool_two_doubles_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            tuple(output_size),
            tuple(input_size),
            out_shape,
            out_stride,
            bool(align_corners),
            0.0 if scales_h is None else float(scales_h),
            0.0 if scales_w is None else float(scales_w),
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnUpsampleBilinear2dBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def upsample_bicubic2d_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_upsample_bicubic2d_backward_get_workspace, b.aclnn_upsample_bicubic2d_backward])
    except Exception:
        return False


def upsample_bicubic2d_backward(grad_ptr, out_ptr,
                                 grad_shape, grad_stride, out_shape, out_stride,
                                 output_size, input_size,
                                 align_corners, scales_h, scales_w,
                                 dtype, runtime, stream=None):
    """aclnnUpsampleBicubic2dBackward(gradOut, outputSize, inputSize, alignCorners, scalesH, scalesW, gradInput)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not upsample_bicubic2d_backward_symbols_ok():
        raise RuntimeError("aclnnUpsampleBicubic2dBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("upsample_bicubic2d_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("UpsampleBicubic2dBackward")
        ws_size, executor = _ffi.tensor_two_int_arrays_bool_two_doubles_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            tuple(output_size),
            tuple(input_size),
            out_shape,
            out_stride,
            bool(align_corners),
            0.0 if scales_h is None else float(scales_h),
            0.0 if scales_w is None else float(scales_w),
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnUpsampleBicubic2dBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def upsample_nearest1d_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_upsample_nearest1d_backward_get_workspace, b.aclnn_upsample_nearest1d_backward])
    except Exception:
        return False


def upsample_nearest1d_backward(grad_ptr, out_ptr,
                                 grad_shape, grad_stride, out_shape, out_stride,
                                 output_size, input_size, scales,
                                 dtype, runtime, stream=None):
    """aclnnUpsampleNearest1dBackward(gradOut, outputSize, inputSize, scales, gradInput)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not upsample_nearest1d_backward_symbols_ok():
        raise RuntimeError("aclnnUpsampleNearest1dBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("upsample_nearest1d_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("UpsampleNearest1dBackward")
        ws_size, executor = _ffi.tensor_two_int_arrays_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            tuple(output_size),
            tuple(input_size),
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnUpsampleNearest1dBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def upsample_linear1d_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_upsample_linear1d_backward_get_workspace, b.aclnn_upsample_linear1d_backward])
    except Exception:
        return False


def upsample_linear1d_backward(grad_ptr, out_ptr,
                                grad_shape, grad_stride, out_shape, out_stride,
                                output_size, input_size,
                                align_corners, scales,
                                dtype, runtime, stream=None):
    """aclnnUpsampleLinear1dBackward(gradOut, outputSize, inputSize, alignCorners, scales, gradInput)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not upsample_linear1d_backward_symbols_ok():
        raise RuntimeError("aclnnUpsampleLinear1dBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("upsample_linear1d_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("UpsampleLinear1dBackward")
        ws_size, executor = _ffi.tensor_two_int_arrays_bool_double_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            tuple(output_size),
            tuple(input_size),
            out_shape,
            out_stride,
            bool(align_corners),
            -1.0 if scales is None else float(scales),
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnUpsampleLinear1dBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------
# adaptive_avg_pool2d backward
# ---------------------------------------------------------------

def adaptive_avg_pool2d_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_adaptive_avg_pool2d_backward_get_workspace, b.aclnn_adaptive_avg_pool2d_backward])
    except Exception:
        return False


def adaptive_avg_pool2d_backward(grad_ptr, self_ptr, out_ptr,
                                  grad_shape, grad_stride,
                                  self_shape, self_stride,
                                  out_shape, out_stride,
                                  dtype, runtime, stream=None):
    """aclnnAdaptiveAvgPool2dBackward(gradOutput, self, gradInput)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not adaptive_avg_pool2d_backward_symbols_ok():
        raise RuntimeError("aclnnAdaptiveAvgPool2dBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("adaptive_avg_pool2d_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("AdaptiveAvgPool2dBackward")
        ws_size, executor = _ffi.four_tensor_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            self_shape,
            self_stride,
            out_shape,
            out_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_NCHW,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAdaptiveAvgPool2dBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------
# adaptive_avg_pool3d backward
# ---------------------------------------------------------------

def adaptive_avg_pool3d_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_adaptive_avg_pool3d_backward_get_workspace, b.aclnn_adaptive_avg_pool3d_backward])
    except Exception:
        return False


def adaptive_avg_pool3d_backward(grad_ptr, self_ptr, out_ptr,
                                  grad_shape, grad_stride,
                                  self_shape, self_stride,
                                  out_shape, out_stride,
                                  dtype, runtime, stream=None):
    """aclnnAdaptiveAvgPool3dBackward(gradOutput, self, gradInput)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not adaptive_avg_pool3d_backward_symbols_ok():
        raise RuntimeError("aclnnAdaptiveAvgPool3dBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("adaptive_avg_pool3d_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("AdaptiveAvgPool3dBackward")
        ws_size, executor = _ffi.four_tensor_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            self_shape,
            self_stride,
            out_shape,
            out_stride,
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(out_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAdaptiveAvgPool3dBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------
# avg_pool3d backward
# ---------------------------------------------------------------

def avg_pool3d_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_avg_pool3d_backward_get_workspace, b.aclnn_avg_pool3d_backward])
    except Exception:
        return False


def avg_pool3d_backward(grad_ptr, input_ptr, grad_input_ptr,
                         grad_shape, grad_stride, input_shape, input_stride,
                         gi_shape, gi_stride,
                         kernel_size, strides, padding,
                         ceil_mode, count_include_pad, divisor_override,
                         dtype, runtime, stream=None):
    """aclnnAvgPool3dBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not avg_pool3d_backward_symbols_ok():
        raise RuntimeError("aclnnAvgPool3dBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("avg_pool3d_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("AvgPool3dBackward")
        ws_size, executor = _ffi.four_tensor_three_int_arrays_two_bools_int64_int8_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            input_shape,
            input_stride,
            gi_shape,
            gi_stride,
            tuple(kernel_size),
            tuple(strides),
            tuple(padding),
            bool(ceil_mode),
            bool(count_include_pad),
            0 if divisor_override is None else int(divisor_override),
            0,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(input_ptr),
            int(grad_input_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAvgPool3dBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------
# grid_sampler2d backward
# ---------------------------------------------------------------

def grid_sampler2d_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_grid_sampler2d_backward_get_workspace, b.aclnn_grid_sampler2d_backward])
    except Exception:
        return False


def grid_sampler2d_backward(grad_ptr, input_ptr, grid_ptr,
                            input_grad_ptr, grid_grad_ptr,
                            grad_shape, grad_stride,
                            input_shape, input_stride,
                            grid_shape, grid_stride,
                            ig_shape, ig_stride,
                            gg_shape, gg_stride,
                            interpolation_mode, padding_mode, align_corners,
                            compute_input_grad, compute_grid_grad,
                            dtype, runtime, stream=None):
    """aclnnGridSampler2DBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not grid_sampler2d_backward_symbols_ok():
        raise RuntimeError("aclnnGridSampler2DBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("grid_sampler2d_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("GridSampler2DBackward")
        ws_size, executor = _ffi.grid_sampler2d_backward_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            input_shape,
            input_stride,
            grid_shape,
            grid_stride,
            ig_shape,
            ig_stride,
            gg_shape,
            gg_stride,
            (bool(compute_input_grad), bool(compute_grid_grad)),
            int(interpolation_mode),
            int(padding_mode),
            bool(align_corners),
            dtype_code,
            int(grad_ptr),
            int(input_ptr),
            int(grid_ptr),
            int(input_grad_ptr),
            int(grid_grad_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnGridSampler2DBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)

def group_norm_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_group_norm_backward_get_workspace, b.aclnn_group_norm_backward])
    except Exception:
        return False


def group_norm_backward(grad_ptr, input_ptr, mean_ptr, rstd_ptr, gamma_ptr,
                        grad_input_ptr, grad_gamma_ptr, grad_beta_ptr,
                        grad_shape, grad_stride, input_shape, input_stride,
                        mean_shape, mean_stride, rstd_shape, rstd_stride,
                        gamma_shape, gamma_stride,
                        gi_shape, gi_stride,
                        gg_shape, gg_stride,
                        gb_shape, gb_stride,
                        N, C, HxW, group,
                        output_mask,
                        dtype, runtime, stream=None):
    """aclnnGroupNormBackward with outputMask (aclBoolArray)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not group_norm_backward_symbols_ok():
        raise RuntimeError("aclnnGroupNormBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    stats_dtype_code = _dtype_to_acl("float32")
    input_fmt = _ACL_FORMAT_NCHW if len(input_shape) >= 4 else _ACL_FORMAT_ND

    _require_native_npu_ffi("group_norm_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("GroupNormBackward")
        ws_size, executor = _ffi.group_norm_backward_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            input_shape,
            input_stride,
            mean_shape,
            mean_stride,
            rstd_shape,
            rstd_stride,
            gamma_shape,
            gamma_stride,
            gi_shape,
            gi_stride,
            gg_shape,
            gg_stride,
            gb_shape,
            gb_stride,
            tuple(bool(v) for v in output_mask),
            int(N),
            int(C),
            int(HxW),
            int(group),
            dtype_code,
            stats_dtype_code,
            input_fmt,
            _ACL_FORMAT_ND,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(input_ptr),
            int(mean_ptr),
            int(rstd_ptr),
            0 if gamma_ptr is None else int(gamma_ptr),
            int(grad_input_ptr),
            0 if grad_gamma_ptr is None else int(grad_gamma_ptr),
            0 if grad_beta_ptr is None else int(grad_beta_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnGroupNormBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)

def digamma(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnDigamma", bindings.aclnn_digamma_get_workspace, bindings.aclnn_digamma,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def lgamma(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnLgamma", bindings.aclnn_lgamma_get_workspace, bindings.aclnn_lgamma,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def sinc(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSinc", bindings.aclnn_sinc_get_workspace, bindings.aclnn_sinc,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def inverse(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnInverse", bindings.aclnn_inverse_get_workspace, bindings.aclnn_inverse,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def linalg_vector_norm(self_ptr, out_ptr, self_shape, self_stride,
                       out_shape, out_stride, dtype, ord_val, dim, keepdim,
                       runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("linalg_vector_norm")
    ord_scalar = _ffi.create_scalar(_scalar_bytes(ord_val, dtype), dtype_code)
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("LinalgVectorNorm")
        ws_size, executor = _ffi.tensor_scalar_int_array_bool_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            out_shape,
            out_stride,
            tuple(dim),
            bool(keepdim),
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(ord_scalar),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnLinalgVectorNorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        _ffi.destroy_scalar(int(ord_scalar))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def aminmax(self_ptr, min_out_ptr, max_out_ptr, self_shape, self_stride,
            out_shape, out_stride, dtype, dim, keepdim, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("aminmax")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Aminmax")
        ws_size, executor = _ffi.tensor_int_array_bool_two_outputs_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            out_shape,
            out_stride,
            out_shape,
            out_stride,
            tuple(dim),
            bool(keepdim),
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(min_out_ptr),
            int(max_out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAminmax failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def bincount(self_ptr, weights_ptr, out_ptr, self_shape, self_stride,
             out_shape, out_stride, self_dtype, out_dtype, minlength,
             weights_shape=None, weights_stride=None, weights_dtype=None,
             runtime=None, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not bincount_symbols_ok():
        raise RuntimeError("aclnnBincount symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    self_dtype_code = _dtype_to_acl(self_dtype)
    weights_dtype_code = self_dtype_code if weights_dtype is None else _dtype_to_acl(weights_dtype)
    out_dtype_code = _dtype_to_acl(out_dtype)

    _require_native_npu_ffi("bincount")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("Bincount")
        ws_size, executor = _ffi.optional_tensor_int_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            None if weights_ptr is None else tuple(weights_shape),
            None if weights_ptr is None else tuple(weights_stride),
            out_shape,
            out_stride,
            int(minlength),
            self_dtype_code,
            weights_dtype_code,
            out_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            0 if weights_ptr is None else int(weights_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnBincount failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def adaptive_avg_pool3d(self_ptr, out_ptr, self_shape, self_stride,
                        out_shape, out_stride, dtype, output_size,
                        runtime=None, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not adaptive_avg_pool3d_symbols_ok():
        raise RuntimeError("aclnnAdaptiveAvgPool3d symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("adaptive_avg_pool3d")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("AdaptiveAvgPool3d")
        ws_size, executor = _ffi.tensor_int_array_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            tuple(output_size),
            out_shape,
            out_stride,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAdaptiveAvgPool3d failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def upsample_bicubic2d(self_ptr, out_ptr, self_shape, self_stride,
                       out_shape, out_stride, dtype, output_size,
                       align_corners, scales_h, scales_w,
                       runtime=None, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("upsample_bicubic2d")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("UpsampleBicubic2d")
        ws_size, executor = _ffi.tensor_int_array_bool_two_doubles_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            out_shape,
            out_stride,
            tuple(output_size),
            bool(align_corners),
            0.0 if scales_h is None else float(scales_h),
            0.0 if scales_w is None else float(scales_w),
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnUpsampleBicubic2d failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def upsample_linear1d(self_ptr, out_ptr, self_shape, self_stride,
                      out_shape, out_stride, dtype, output_size,
                      align_corners, scales,
                      runtime=None, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("upsample_linear1d")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("UpsampleLinear1d")
        ws_size, executor = _ffi.tensor_int_array_bool_double_op(
            getws_ptr,
            exec_ptr,
            self_shape,
            self_stride,
            out_shape,
            out_stride,
            tuple(output_size),
            bool(align_corners),
            -1.0 if scales is None else float(scales),
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnUpsampleLinear1d failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def apply_adam_w_v2(var_ptr, m_ptr, v_ptr, max_v_ptr, grad_ptr, step_ptr,
                    var_shape, var_stride, step_shape, step_stride,
                    dtype, lr, beta1, beta2, weight_decay, eps,
                    amsgrad, maximize,
                    runtime=None, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_apply_adam_w_v2_get_workspace is None or bindings.aclnn_apply_adam_w_v2 is None:
        raise RuntimeError("aclnnApplyAdamWV2 symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("apply_adam_w_v2")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("ApplyAdamWV2")
        ws_size, executor = _ffi.six_tensor_five_floats_two_bools_op(
            getws_ptr,
            exec_ptr,
            var_shape,
            var_stride,
            var_shape,
            var_stride,
            var_shape,
            var_stride,
            var_shape,
            var_stride,
            var_shape,
            var_stride,
            step_shape,
            step_stride,
            float(lr),
            float(beta1),
            float(beta2),
            float(weight_decay),
            float(eps),
            bool(amsgrad),
            bool(maximize),
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(var_ptr),
            int(m_ptr),
            int(v_ptr),
            0 if max_v_ptr is None else int(max_v_ptr),
            int(grad_ptr),
            int(step_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnApplyAdamWV2 failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def unfold_grad_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_unfold_grad_get_workspace, b.aclnn_unfold_grad])
    except Exception:
        return False


def unfold_grad(grad_ptr, out_ptr, grad_shape, grad_stride, out_shape, out_stride,
                input_sizes, dim, size, step, dtype, runtime, stream=None):
    """aclnnUnfoldGrad(gradOut, inputSizes, dim, size, step, out)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not unfold_grad_symbols_ok():
        raise RuntimeError("aclnnUnfoldGrad symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("unfold_grad")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("UnfoldGrad")
        ws_size, executor = _ffi.tensor_int_array_three_ints_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            tuple(input_sizes),
            out_shape,
            out_stride,
            int(dim),
            int(size),
            int(step),
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnUnfoldGrad failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def selu_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_selu_backward_get_workspace, b.aclnn_selu_backward])
    except Exception:
        return False


def selu_backward(grad_ptr, result_ptr, out_ptr, shape, grad_stride, result_stride, out_stride,
                  dtype, runtime, stream=None):
    """aclnnSeluBackward(gradOutput, result, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not selu_backward_symbols_ok():
        raise RuntimeError("aclnnSeluBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)

    _require_native_npu_ffi("selu_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("SeluBackward")
        ws_size, executor = _ffi.binary_two_inputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            grad_stride,
            shape,
            result_stride,
            shape,
            out_stride,
            dtype_code,
            dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(result_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnSeluBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------
# max_pool3d_with_argmax (forward)
# ---------------------------------------------------------------

def max_pool3d_with_argmax_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_max_pool3d_with_argmax_get_workspace,
                    b.aclnn_max_pool3d_with_argmax])
    except Exception:
        return False


def max_pool3d_with_argmax(self_ptr, out_ptr, indices_ptr,
                            shape, stride_t, dtype,
                            kernel_size, strides, padding, dilation, ceil_mode,
                            out_shape, out_stride,
                            indices_shape, indices_stride,
                            runtime, stream=None):
    """MaxPool3d via aclnnMaxPool3dWithArgmax.

    Returns output + argmax indices (int32).
    """
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not max_pool3d_with_argmax_symbols_ok():
        raise RuntimeError("aclnnMaxPool3dWithArgmax symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    idx_dtype_code = _dtype_to_acl("int32")

    _require_native_npu_ffi("max_pool3d_with_argmax")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("MaxPool3dWithArgmax")
        ws_size, executor = _ffi.tensor_four_int_arrays_bool_two_outputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride_t,
            tuple(kernel_size),
            tuple(strides),
            tuple(padding),
            tuple(dilation),
            out_shape,
            out_stride,
            indices_shape,
            indices_stride,
            bool(ceil_mode),
            dtype_code,
            dtype_code,
            idx_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(indices_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMaxPool3dWithArgmax failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------
# max_pool3d_with_argmax backward
# ---------------------------------------------------------------

def max_pool3d_with_argmax_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_max_pool3d_with_argmax_backward_get_workspace,
                    b.aclnn_max_pool3d_with_argmax_backward])
    except Exception:
        return False


def max_pool3d_with_argmax_backward(grad_ptr, input_ptr, indices_ptr, grad_input_ptr,
                                      grad_shape, grad_stride,
                                      input_shape, input_stride,
                                      indices_shape, indices_stride,
                                      gi_shape, gi_stride,
                                      dtype,
                                      kernel_size, strides, padding, dilation, ceil_mode,
                                      runtime, stream=None):
    """aclnnMaxPool3dWithArgmaxBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not max_pool3d_with_argmax_backward_symbols_ok():
        raise RuntimeError("aclnnMaxPool3dWithArgmaxBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    idx_dtype_code = _dtype_to_acl("int32")

    _require_native_npu_ffi("max_pool3d_with_argmax_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("MaxPool3dWithArgmaxBackward")
        ws_size, executor = _ffi.four_tensor_four_int_arrays_bool_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            input_shape,
            input_stride,
            indices_shape,
            indices_stride,
            gi_shape,
            gi_stride,
            tuple(kernel_size),
            tuple(strides),
            tuple(padding),
            tuple(dilation),
            bool(ceil_mode),
            dtype_code,
            dtype_code,
            idx_dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(input_ptr),
            int(indices_ptr),
            int(grad_input_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnMaxPool3dWithArgmaxBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------
# adaptive_max_pool2d (forward)
# ---------------------------------------------------------------

def adaptive_max_pool2d_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_adaptive_max_pool2d_get_workspace,
                    b.aclnn_adaptive_max_pool2d])
    except Exception:
        return False


def adaptive_max_pool2d(self_ptr, out_ptr, indices_ptr,
                         shape, stride_t, dtype,
                         output_size,
                         out_shape, out_stride,
                         indices_shape, indices_stride,
                         runtime, stream=None):
    """AdaptiveMaxPool2d via aclnnAdaptiveMaxPool2d.

    Returns output + indices (int64).
    """
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not adaptive_max_pool2d_symbols_ok():
        raise RuntimeError("aclnnAdaptiveMaxPool2d symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    idx_dtype_code = _dtype_to_acl("int64")

    _require_native_npu_ffi("adaptive_max_pool2d")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("AdaptiveMaxPool2d")
        ws_size, executor = _ffi.tensor_int_array_two_outputs_op(
            getws_ptr,
            exec_ptr,
            shape,
            stride_t,
            out_shape,
            out_stride,
            indices_shape,
            indices_stride,
            tuple(output_size),
            dtype_code,
            dtype_code,
            idx_dtype_code,
            _ACL_FORMAT_ND,
            int(self_ptr),
            int(out_ptr),
            int(indices_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAdaptiveMaxPool2d failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


# ---------------------------------------------------------------
# adaptive_max_pool2d backward
# ---------------------------------------------------------------

def adaptive_max_pool2d_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_adaptive_max_pool2d_backward_get_workspace,
                    b.aclnn_adaptive_max_pool2d_backward])
    except Exception:
        return False


def adaptive_max_pool2d_backward(grad_ptr, self_ptr, indices_ptr, grad_input_ptr,
                                   grad_shape, grad_stride,
                                   self_shape, self_stride,
                                   indices_shape, indices_stride,
                                   gi_shape, gi_stride,
                                   dtype,
                                   runtime, stream=None):
    """aclnnAdaptiveMaxPool2dBackward(gradOutput, self, indices, gradInput)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not adaptive_max_pool2d_backward_symbols_ok():
        raise RuntimeError("aclnnAdaptiveMaxPool2dBackward symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    dtype_code = _dtype_to_acl(dtype)
    idx_dtype_code = _dtype_to_acl("int64")

    _require_native_npu_ffi("adaptive_max_pool2d_backward")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("AdaptiveMaxPool2dBackward")
        ws_size, executor = _ffi.four_tensor_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            self_shape,
            self_stride,
            indices_shape,
            indices_stride,
            gi_shape,
            gi_stride,
            dtype_code,
            dtype_code,
            idx_dtype_code,
            dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(self_ptr),
            int(indices_ptr),
            int(grad_input_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnAdaptiveMaxPool2dBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def repeat_interleave_grad_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_repeat_interleave_grad_get_workspace,
                    b.aclnn_repeat_interleave_grad])
    except Exception:
        return False


def repeat_interleave_grad(grad_ptr, repeats_ptr, out_ptr,
                           grad_shape, grad_stride, grad_dtype,
                           repeats_shape, repeats_stride,
                           out_shape, out_stride, out_dtype,
                           axis, runtime, stream=None):
    """aclnnRepeatInterleaveGrad(yGrad, repeats, axis, out)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not repeat_interleave_grad_symbols_ok():
        raise RuntimeError("aclnnRepeatInterleaveGrad symbols not available")
    stream_ptr = int(runtime.stream if stream is None else stream)
    grad_dtype_code = _dtype_to_acl(grad_dtype)
    out_dtype_code = _dtype_to_acl(out_dtype)
    repeats_dtype_code = _dtype_to_acl("int64")

    _require_native_npu_ffi("repeat_interleave_grad")
    executor = 0
    workspace = None
    try:
        getws_ptr, exec_ptr = _ffi.resolve_op("RepeatInterleaveGrad")
        ws_size, executor = _ffi.two_tensor_ints_bool_op(
            getws_ptr,
            exec_ptr,
            grad_shape,
            grad_stride,
            repeats_shape,
            repeats_stride,
            out_shape,
            out_stride,
            int(axis),
            False,
            grad_dtype_code,
            repeats_dtype_code,
            out_dtype_code,
            _ACL_FORMAT_ND,
            int(grad_ptr),
            int(repeats_ptr),
            int(out_ptr),
            stream_ptr,
        )
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(int(ws_size), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
            ret = _ffi.execute(exec_ptr, int(workspace), ws_size, executor, stream_ptr)
            if ret != 0:
                raise RuntimeError(f"aclnnRepeatInterleaveGrad failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(ctypes.c_void_p(executor))
        if workspace is not None:
            runtime.defer_raw_free(workspace)
