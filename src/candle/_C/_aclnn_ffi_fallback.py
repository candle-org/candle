"""Pure-Python fallback for _aclnn_ffi.pyx.

Provides the same public API using ctypes, delegating to the existing
``candle._backends.npu.aclnn`` bindings.  This module is imported when
the Cython extension is not compiled.
"""

import ctypes

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_initialized = False

# Cache: op_name -> (getws_ptr, exec_ptr) as ints
_op_cache = {}

# Ops that must prefer libopapi.so (mirrors .pyx logic)
_PREFER_OPAPI = frozenset({
    "Add", "Sub", "Mul", "Div", "Adds", "Subs",
})

# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------

_bindings = None


def _get_bindings():
    global _bindings  # pylint: disable=global-statement
    if _bindings is None:
        from candle._backends.npu.aclnn import get_bindings  # pylint: disable=import-outside-toplevel
        _bindings = get_bindings()
    return _bindings


# ---------------------------------------------------------------------------
# init / is_initialized
# ---------------------------------------------------------------------------

def init(lib_paths):  # pylint: disable=unused-argument
    """No-op — aclnn.py handles its own initialisation."""


def is_initialized():
    """Always False — the Cython fast-path is not available."""
    return False


# ---------------------------------------------------------------------------
# Tensor creation / destruction
# ---------------------------------------------------------------------------

def create_tensor(shape_tuple, stride_tuple, dtype_code, data_ptr, fmt=2):
    """Create an ACL tensor descriptor.  Returns handle as int."""
    b = _get_bindings()
    ndim = len(shape_tuple)
    c_shape = (ctypes.c_int64 * ndim)(*shape_tuple)
    c_stride = (ctypes.c_int64 * ndim)(*stride_tuple)
    tensor = b.acl_create_tensor(
        c_shape, ndim, dtype_code,
        c_stride, 0, fmt,
        c_shape, ndim,  # storageDims == viewDims
        ctypes.c_void_p(data_ptr),
    )
    if tensor is None or tensor == 0:
        raise RuntimeError("aclCreateTensor returned null")
    return tensor


def destroy_tensor(handle):
    """Destroy an ACL tensor descriptor."""
    return _get_bindings().acl_destroy_tensor(ctypes.c_void_p(handle))


# ---------------------------------------------------------------------------
# Scalar creation / destruction
# ---------------------------------------------------------------------------

def create_scalar(scalar_bytes, dtype_code):
    """Create an ACL scalar from pre-encoded bytes.  Returns handle as int."""
    buf = ctypes.cast(ctypes.c_char_p(scalar_bytes), ctypes.c_void_p)
    scalar = _get_bindings().acl_create_scalar(buf, dtype_code)
    if scalar is None or scalar == 0:
        raise RuntimeError("aclCreateScalar returned null")
    return scalar


def destroy_scalar(handle):
    return _get_bindings().acl_destroy_scalar(ctypes.c_void_p(handle))


# ---------------------------------------------------------------------------
# IntArray creation / destruction
# ---------------------------------------------------------------------------

def create_int_array(values_tuple):
    """Create an ACL int array from a tuple of ints.  Returns handle as int."""
    n = len(values_tuple)
    if n == 0:
        return 0
    c_vals = (ctypes.c_int64 * n)(*values_tuple)
    arr = _get_bindings().acl_create_int_array(c_vals, n)
    if arr is None or arr == 0:
        raise RuntimeError("aclCreateIntArray returned null")
    return arr


def destroy_int_array(handle):
    return _get_bindings().acl_destroy_int_array(ctypes.c_void_p(handle))


# ---------------------------------------------------------------------------
# Executor destruction
# ---------------------------------------------------------------------------

def destroy_executor(handle):
    if handle == 0:
        return 0
    return _get_bindings().acl_destroy_executor(ctypes.c_void_p(handle))


# ---------------------------------------------------------------------------
# Op symbol resolution
# ---------------------------------------------------------------------------

def resolve_op(op_name):
    """Resolve GetWorkspaceSize and Execute function pointers for an op.

    Returns (getws_ptr, exec_ptr) as ints.  Cached after first call.
    """
    cached = _op_cache.get(op_name)
    if cached is not None:
        return cached

    b = _get_bindings()
    ws_name = f"aclnn{op_name}GetWorkspaceSize"
    exec_name = f"aclnn{op_name}"

    libs = b.libs

    # For arithmetic ops, prefer libopapi.so
    ws_func = None
    exec_func = None
    if op_name in _PREFER_OPAPI:
        for lib in libs:
            path = getattr(lib, "_name", "") or ""
            if path.endswith("libopapi.so"):
                if hasattr(lib, ws_name):
                    ws_func = ctypes.cast(
                        getattr(lib, ws_name), ctypes.c_void_p).value
                if hasattr(lib, exec_name):
                    exec_func = ctypes.cast(
                        getattr(lib, exec_name), ctypes.c_void_p).value
                break

    # Fall back to searching all handles
    if ws_func is None:
        for lib in libs:
            if hasattr(lib, ws_name):
                ws_func = ctypes.cast(
                    getattr(lib, ws_name), ctypes.c_void_p).value
                break
        else:
            raise RuntimeError(
                f"Symbol not found in any library: {ws_name}")

    if exec_func is None:
        for lib in libs:
            if hasattr(lib, exec_name):
                exec_func = ctypes.cast(
                    getattr(lib, exec_name), ctypes.c_void_p).value
                break
        else:
            raise RuntimeError(
                f"Symbol not found in any library: {exec_name}")

    result = (ws_func, exec_func)
    _op_cache[op_name] = result
    return result


def resolve_op_optional(op_name):
    """Like resolve_op but returns None if symbol not found."""
    try:
        return resolve_op(op_name)
    except RuntimeError:
        return None


# ---------------------------------------------------------------------------
# Generic execute
# ---------------------------------------------------------------------------

def execute(exec_ptr, workspace_ptr, workspace_size, executor, stream):
    """Call aclnn<Op>(workspace, wsSize, executor, stream).

    Returns the int32 return code.
    """
    fn_type = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p,
    )
    fn = fn_type(exec_ptr)
    return fn(
        ctypes.c_void_p(workspace_ptr),
        ctypes.c_uint64(workspace_size),
        ctypes.c_void_p(executor),
        ctypes.c_void_p(stream),
    )


# ---------------------------------------------------------------------------
# Binary op helpers
# ---------------------------------------------------------------------------

def binary_op_with_alpha(
        getws_ptr, exec_ptr,
        self_shape, self_stride,
        other_shape, other_stride,
        out_shape, out_stride,
        dtype_code, fmt,
        self_ptr, other_ptr, out_ptr,
        alpha_scalar,
        stream):
    """Binary op with alpha (add, sub): create tensors, getws, exec, destroy.

    Returns (ws_size: int, executor_ptr: int).
    If ws_size == 0, execute has already been called.
    """
    self_t = create_tensor(self_shape, self_stride, dtype_code, self_ptr, fmt)
    other_t = create_tensor(other_shape, other_stride, dtype_code,
                            other_ptr, fmt)
    out_t = create_tensor(out_shape, out_stride, dtype_code, out_ptr, fmt)

    try:
        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p(None)

        getws_type = ctypes.CFUNCTYPE(
            ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        )
        getws_fn = getws_type(getws_ptr)
        ret = getws_fn(
            ctypes.c_void_p(self_t),
            ctypes.c_void_p(other_t),
            ctypes.c_void_p(alpha_scalar),
            ctypes.c_void_p(out_t),
            ctypes.byref(ws_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")

        ws = ws_size.value
        exec_handle = executor.value or 0

        # Fast path: no workspace needed, execute immediately
        if ws == 0:
            ret = execute(exec_ptr, 0, 0, exec_handle, stream)
            if ret != 0:
                raise RuntimeError(f"Execute failed: {ret}")

        return (ws, exec_handle)
    finally:
        destroy_tensor(self_t)
        destroy_tensor(other_t)
        destroy_tensor(out_t)


def binary_op_no_alpha(
        getws_ptr, exec_ptr,
        self_shape, self_stride,
        other_shape, other_stride,
        out_shape, out_stride,
        dtype_code, fmt,
        self_ptr, other_ptr, out_ptr,
        stream):
    """Binary op without alpha (mul, div): create tensors, getws, exec, destroy.

    Returns (ws_size: int, executor_ptr: int).
    If ws_size == 0, execute has already been called.
    """
    self_t = create_tensor(self_shape, self_stride, dtype_code, self_ptr, fmt)
    other_t = create_tensor(other_shape, other_stride, dtype_code,
                            other_ptr, fmt)
    out_t = create_tensor(out_shape, out_stride, dtype_code, out_ptr, fmt)

    try:
        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p(None)

        getws_type = ctypes.CFUNCTYPE(
            ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        )
        getws_fn = getws_type(getws_ptr)
        ret = getws_fn(
            ctypes.c_void_p(self_t),
            ctypes.c_void_p(other_t),
            ctypes.c_void_p(out_t),
            ctypes.byref(ws_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")

        ws = ws_size.value
        exec_handle = executor.value or 0

        if ws == 0:
            ret = execute(exec_ptr, 0, 0, exec_handle, stream)
            if ret != 0:
                raise RuntimeError(f"Execute failed: {ret}")

        return (ws, exec_handle)
    finally:
        destroy_tensor(self_t)
        destroy_tensor(other_t)
        destroy_tensor(out_t)
