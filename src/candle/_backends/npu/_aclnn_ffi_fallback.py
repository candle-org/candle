"""Pure-Python fallback for _aclnn_ffi when Cython extension is not compiled.

Provides the same public API as _aclnn_ffi.pyx using ctypes.  This is used
automatically when the Cython .so is unavailable (e.g. pip install without
Cython, or development on a non-NPU machine).

Performance is identical to the original ctypes path — the Cython extension
is the fast path.
"""

import ctypes
import ctypes.util
import os

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_initialized = False
_lib_handles = []   # list of ctypes.CDLL
_opapi_handle = None  # ctypes.CDLL for libopapi.so, preferred for arithmetic ops
_op_cache = {}      # op_name -> (getws_ctypes_func, exec_ctypes_func)

# Ops that must prefer libopapi.so (matches _bind_symbol logic in aclnn.py)
_PREFER_OPAPI = frozenset({
    "Add", "Sub", "Mul", "Div", "Adds", "Subs",
})

# Core function bindings (set by init())
_fn_create_tensor = None
_fn_destroy_tensor = None
_fn_create_scalar = None
_fn_destroy_scalar = None
_fn_create_int_array = None
_fn_destroy_int_array = None
_fn_destroy_executor = None

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init(lib_paths):
    """Load CANN libraries and resolve core symbols."""
    global _fn_create_tensor, _fn_destroy_tensor
    global _fn_create_scalar, _fn_destroy_scalar
    global _fn_create_int_array, _fn_destroy_int_array
    global _fn_destroy_executor, _initialized

    if _initialized:
        return

    handles = []
    for path in lib_paths:
        handles.append(ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL))

    _lib_handles.clear()
    _lib_handles.extend(handles)

    # Identify the libopapi.so handle for arithmetic op preference
    global _opapi_handle
    for i, path in enumerate(lib_paths):
        if path.endswith("libopapi.so"):
            _opapi_handle = handles[i]
            break

    _fn_create_tensor = _bind(handles, "aclCreateTensor", ctypes.c_void_p, [
        ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_void_p,
    ])
    _fn_destroy_tensor = _bind(handles, "aclDestroyTensor",
                               ctypes.c_int32, [ctypes.c_void_p])
    _fn_create_scalar = _bind(handles, "aclCreateScalar",
                              ctypes.c_void_p,
                              [ctypes.c_void_p, ctypes.c_int32])
    _fn_destroy_scalar = _bind(handles, "aclDestroyScalar",
                               ctypes.c_int32, [ctypes.c_void_p])
    _fn_create_int_array = _bind(handles, "aclCreateIntArray",
                                 ctypes.c_void_p,
                                 [ctypes.POINTER(ctypes.c_int64),
                                  ctypes.c_uint64])
    _fn_destroy_int_array = _bind(handles, "aclDestroyIntArray",
                                  ctypes.c_int32, [ctypes.c_void_p])
    _fn_destroy_executor = _bind(handles, "aclDestroyAclOpExecutor",
                                 ctypes.c_int32, [ctypes.c_void_p])
    _initialized = True


def _bind(handles, name, restype, argtypes):
    """Find *name* in any of *handles* and set its signature."""
    for lib in handles:
        if hasattr(lib, name):
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
            return fn
    raise RuntimeError(f"Symbol not found in any library: {name}")


def is_initialized():
    return _initialized


# ---------------------------------------------------------------------------
# Tensor creation / destruction
# ---------------------------------------------------------------------------

def _make_int64_array(values):
    n = len(values)
    arr = (ctypes.c_int64 * n)()
    for i in range(n):
        arr[i] = int(values[i])
    return arr


def create_tensor(shape_tuple, stride_tuple, dtype_code, data_ptr, fmt=2):
    shape_arr = _make_int64_array(shape_tuple)
    stride_arr = _make_int64_array(stride_tuple)
    ndim = len(shape_tuple)
    tensor = _fn_create_tensor(
        shape_arr, ctypes.c_uint64(ndim), ctypes.c_int32(dtype_code),
        stride_arr, ctypes.c_int64(0), ctypes.c_int32(fmt),
        shape_arr, ctypes.c_uint64(ndim),
        ctypes.c_void_p(int(data_ptr)),
    )
    if not tensor:
        raise RuntimeError("aclCreateTensor returned null")
    return int(tensor)


def destroy_tensor(handle):
    return _fn_destroy_tensor(ctypes.c_void_p(handle))


# ---------------------------------------------------------------------------
# Scalar creation / destruction
# ---------------------------------------------------------------------------

def create_scalar(scalar_bytes, dtype_code):
    buf = (ctypes.c_uint8 * len(scalar_bytes)).from_buffer_copy(scalar_bytes)
    ptr = ctypes.c_void_p(ctypes.addressof(buf))
    scalar = _fn_create_scalar(ptr, ctypes.c_int32(dtype_code))
    if not scalar:
        raise RuntimeError("aclCreateScalar returned null")
    return int(scalar)


def destroy_scalar(handle):
    return _fn_destroy_scalar(ctypes.c_void_p(handle))

# ---------------------------------------------------------------------------
# IntArray creation / destruction
# ---------------------------------------------------------------------------

def create_int_array(values_tuple):
    n = len(values_tuple)
    if n == 0:
        return 0
    arr = _make_int64_array(values_tuple)
    result = _fn_create_int_array(arr, ctypes.c_uint64(n))
    if not result:
        raise RuntimeError("aclCreateIntArray returned null")
    return int(result)


def destroy_int_array(handle):
    return _fn_destroy_int_array(ctypes.c_void_p(handle))


# ---------------------------------------------------------------------------
# Executor destruction
# ---------------------------------------------------------------------------

def destroy_executor(handle):
    if handle == 0:
        return 0
    return _fn_destroy_executor(ctypes.c_void_p(handle))


# ---------------------------------------------------------------------------
# Op symbol resolution
# ---------------------------------------------------------------------------

def resolve_op(op_name):
    """Resolve GetWorkspaceSize and Execute function pointers for an op.

    Returns (getws_func, exec_func) as ctypes function objects.
    For arithmetic ops (Add/Sub/Mul/Div), prefers libopapi.so.
    """
    cached = _op_cache.get(op_name)
    if cached is not None:
        return cached

    ws_name = f"aclnn{op_name}GetWorkspaceSize"
    exec_name = f"aclnn{op_name}"

    ws_func = None
    exec_func = None

    # For arithmetic ops, try libopapi.so first
    if op_name in _PREFER_OPAPI and _opapi_handle is not None:
        if hasattr(_opapi_handle, ws_name):
            ws_func = getattr(_opapi_handle, ws_name)
        if hasattr(_opapi_handle, exec_name):
            exec_func = getattr(_opapi_handle, exec_name)

    # Fall back to searching all handles
    if ws_func is None or exec_func is None:
        for lib in _lib_handles:
            if ws_func is None and hasattr(lib, ws_name):
                ws_func = getattr(lib, ws_name)
            if exec_func is None and hasattr(lib, exec_name):
                exec_func = getattr(lib, exec_name)
            if ws_func is not None and exec_func is not None:
                break

    if ws_func is None or exec_func is None:
        raise RuntimeError(
            f"Symbol not found: {ws_name} or {exec_name}")

    # Set generic signatures — callers cast as needed
    exec_func.restype = ctypes.c_int32
    exec_func.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                          ctypes.c_void_p, ctypes.c_void_p]

    result = (ws_func, exec_func)
    _op_cache[op_name] = result
    return result


def resolve_op_optional(op_name):
    try:
        return resolve_op(op_name)
    except RuntimeError:
        return None


# ---------------------------------------------------------------------------
# Generic execute
# ---------------------------------------------------------------------------

def execute(exec_ptr, workspace_ptr, workspace_size, executor, stream):
    """Call aclnn<Op>(workspace, wsSize, executor, stream).

    In fallback mode, exec_ptr is a ctypes function object (not an int).
    """
    if isinstance(exec_ptr, int):
        # Should not happen in fallback mode, but handle gracefully
        raise RuntimeError("execute() in fallback mode requires ctypes func")
    return exec_ptr(
        ctypes.c_void_p(workspace_ptr),
        ctypes.c_uint64(workspace_size),
        ctypes.c_void_p(executor),
        ctypes.c_void_p(stream),
    )

# ---------------------------------------------------------------------------
# Binary op helpers (fallback — uses ctypes, same API as Cython version)
# ---------------------------------------------------------------------------

def _create_tensor_ct(shape, stride, dtype_code, data_ptr, fmt=2):
    """Internal: create tensor and return (handle_int, keepalive_tuple)."""
    shape_arr = _make_int64_array(shape)
    stride_arr = _make_int64_array(stride)
    ndim = len(shape)
    tensor = _fn_create_tensor(
        shape_arr, ctypes.c_uint64(ndim), ctypes.c_int32(dtype_code),
        stride_arr, ctypes.c_int64(0), ctypes.c_int32(fmt),
        shape_arr, ctypes.c_uint64(ndim),
        ctypes.c_void_p(int(data_ptr)),
    )
    if not tensor:
        raise RuntimeError("aclCreateTensor returned null")
    return tensor, (shape_arr, stride_arr)


def binary_op_with_alpha(
        getws_ptr, exec_ptr,
        self_shape, self_stride,
        other_shape, other_stride,
        out_shape, out_stride,
        dtype_code, fmt,
        self_ptr, other_ptr, out_ptr,
        alpha_scalar,
        stream):
    """Binary op with alpha (add, sub).

    In fallback mode, getws_ptr/exec_ptr are ctypes function objects.
    alpha_scalar is an int handle.

    Returns (ws_size: int, executor_ptr: int).
    """
    self_t, self_keep = _create_tensor_ct(
        self_shape, self_stride, dtype_code, self_ptr, fmt)
    other_t, other_keep = _create_tensor_ct(
        other_shape, other_stride, dtype_code, other_ptr, fmt)
    out_t, out_keep = _create_tensor_ct(
        out_shape, out_stride, dtype_code, out_ptr, fmt)

    ws_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p()

    try:
        # Set signature for this specific GetWorkspaceSize variant
        getws_ptr.restype = ctypes.c_int32
        getws_ptr.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p),
        ]
        ret = getws_ptr(
            self_t, other_t, ctypes.c_void_p(alpha_scalar), out_t,
            ctypes.byref(ws_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")

        if ws_size.value == 0:
            exec_ptr.restype = ctypes.c_int32
            exec_ptr.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                 ctypes.c_void_p, ctypes.c_void_p]
            ret = exec_ptr(
                ctypes.c_void_p(0), ctypes.c_uint64(0),
                executor, ctypes.c_void_p(stream),
            )
            if ret != 0:
                raise RuntimeError(f"Execute failed: {ret}")

        exec_int = int(executor.value) if executor.value else 0  # pylint: disable=using-constant-test
        return (ws_size.value, exec_int)
    finally:
        _fn_destroy_tensor(self_t)
        _fn_destroy_tensor(other_t)
        _fn_destroy_tensor(out_t)
        _ = (self_keep, other_keep, out_keep)


def binary_op_no_alpha(
        getws_ptr, exec_ptr,
        self_shape, self_stride,
        other_shape, other_stride,
        out_shape, out_stride,
        dtype_code, fmt,
        self_ptr, other_ptr, out_ptr,
        stream):
    """Binary op without alpha (mul, div).

    Returns (ws_size: int, executor_ptr: int).
    """
    self_t, self_keep = _create_tensor_ct(
        self_shape, self_stride, dtype_code, self_ptr, fmt)
    other_t, other_keep = _create_tensor_ct(
        other_shape, other_stride, dtype_code, other_ptr, fmt)
    out_t, out_keep = _create_tensor_ct(
        out_shape, out_stride, dtype_code, out_ptr, fmt)

    ws_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p()

    try:
        getws_ptr.restype = ctypes.c_int32
        getws_ptr.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p),
        ]
        ret = getws_ptr(
            self_t, other_t, out_t,
            ctypes.byref(ws_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")

        if ws_size.value == 0:
            exec_ptr.restype = ctypes.c_int32
            exec_ptr.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                 ctypes.c_void_p, ctypes.c_void_p]
            ret = exec_ptr(
                ctypes.c_void_p(0), ctypes.c_uint64(0),
                executor, ctypes.c_void_p(stream),
            )
            if ret != 0:
                raise RuntimeError(f"Execute failed: {ret}")

        exec_int = int(executor.value) if executor.value else 0  # pylint: disable=using-constant-test
        return (ws_size.value, exec_int)
    finally:
        _fn_destroy_tensor(self_t)
        _fn_destroy_tensor(other_t)
        _fn_destroy_tensor(out_t)
        _ = (self_keep, other_keep, out_keep)
