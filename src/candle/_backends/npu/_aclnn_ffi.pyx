# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot-path for ACLNN FFI calls.

Replaces ctypes-based tensor/scalar creation and op execution with direct
C function-pointer calls resolved via dlopen/dlsym at runtime.

The 243 op functions in aclnn.py stay as Python.  This module accelerates
the primitives they call: create_tensor, create_scalar, destroy_*, and
the generic execute pattern.
"""

from libc.stdint cimport int32_t, int64_t, uint64_t, uint8_t, uintptr_t

cdef extern from "dlfcn.h":
    void* dlopen(const char* filename, int flags) nogil
    void* dlsym(void* handle, const char* symbol) nogil
    char* dlerror() nogil
    int RTLD_LAZY
    int RTLD_GLOBAL

# ---------------------------------------------------------------------------
# Function pointer typedefs
# ---------------------------------------------------------------------------

ctypedef void* (*aclCreateTensor_t)(
    const int64_t*, uint64_t, int32_t,
    const int64_t*, int64_t, int32_t,
    const int64_t*, uint64_t, void*) noexcept nogil

ctypedef int32_t (*aclDestroyTensor_t)(void*) noexcept nogil
ctypedef void* (*aclCreateScalar_t)(void*, int32_t) noexcept nogil
ctypedef int32_t (*aclDestroyScalar_t)(void*) noexcept nogil
ctypedef void* (*aclCreateIntArray_t)(const int64_t*, uint64_t) noexcept nogil
ctypedef int32_t (*aclDestroyIntArray_t)(void*) noexcept nogil
ctypedef int32_t (*aclDestroyExecutor_t)(void*) noexcept nogil

# Generic execute: aclnn<Op>(workspace, wsSize, executor, stream) -> int32
ctypedef int32_t (*aclnnExec_t)(void*, uint64_t, void*, void*) noexcept nogil

# ---------------------------------------------------------------------------
# Module-level cached function pointers
# ---------------------------------------------------------------------------

cdef aclCreateTensor_t    _fn_create_tensor    = NULL
cdef aclDestroyTensor_t   _fn_destroy_tensor   = NULL
cdef aclCreateScalar_t    _fn_create_scalar    = NULL
cdef aclDestroyScalar_t   _fn_destroy_scalar   = NULL
cdef aclCreateIntArray_t  _fn_create_int_array  = NULL
cdef aclDestroyIntArray_t _fn_destroy_int_array = NULL
cdef aclDestroyExecutor_t _fn_destroy_executor  = NULL

cdef bint _initialized = 0

DEF MAX_NDIM = 16

# Stored dlopen handles (as uintptr_t) for op symbol resolution
_lib_handles = []

# Handle for libopapi.so — preferred for arithmetic ops (add/sub/mul/div)
_opapi_handle = 0  # uintptr_t, 0 means not found

# Ops that must prefer libopapi.so (matches _bind_symbol logic in aclnn.py)
_PREFER_OPAPI = frozenset({
    "Add", "Sub", "Mul", "Div", "Adds", "Subs",
})

# Cache: op_name -> (getws_ptr, exec_ptr) as uintptr_t
_op_cache = {}

# ---------------------------------------------------------------------------
# dlsym helpers
# ---------------------------------------------------------------------------

cdef void* _find_symbol(list handles, const char* name) except NULL:
    """Search all opened handles for *name*, return first match."""
    cdef void* h
    cdef void* sym
    dlerror()  # clear prior error
    for h_int in handles:
        h = <void*><uintptr_t>h_int
        sym = dlsym(h, name)
        if sym != NULL:
            return sym
    raise RuntimeError(
        f"Symbol not found in any library: {name.decode('utf-8')}")

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init(list lib_paths):
    """Resolve core CANN function pointers from the given .so paths.

    Called once from aclnn.py after library discovery.
    """
    global _fn_create_tensor, _fn_destroy_tensor
    global _fn_create_scalar, _fn_destroy_scalar
    global _fn_create_int_array, _fn_destroy_int_array
    global _fn_destroy_executor, _initialized

    if _initialized:
        return

    cdef void* handle
    cdef list handles = []

    for path in lib_paths:
        bpath = path.encode("utf-8") if isinstance(path, str) else path
        handle = dlopen(<const char*>bpath, RTLD_LAZY | RTLD_GLOBAL)
        if handle == NULL:
            err = dlerror()
            msg = err.decode("utf-8") if err != NULL else "unknown error"
            raise RuntimeError(f"dlopen failed for {path}: {msg}")
        handles.append(<uintptr_t>handle)

    _fn_create_tensor = <aclCreateTensor_t>_find_symbol(
        handles, b"aclCreateTensor")
    _fn_destroy_tensor = <aclDestroyTensor_t>_find_symbol(
        handles, b"aclDestroyTensor")
    _fn_create_scalar = <aclCreateScalar_t>_find_symbol(
        handles, b"aclCreateScalar")
    _fn_destroy_scalar = <aclDestroyScalar_t>_find_symbol(
        handles, b"aclDestroyScalar")
    _fn_create_int_array = <aclCreateIntArray_t>_find_symbol(
        handles, b"aclCreateIntArray")
    _fn_destroy_int_array = <aclDestroyIntArray_t>_find_symbol(
        handles, b"aclDestroyIntArray")
    _fn_destroy_executor = <aclDestroyExecutor_t>_find_symbol(
        handles, b"aclDestroyAclOpExecutor")

    # Store handles for later op resolution
    _lib_handles.clear()
    _lib_handles.extend(handles)

    # Identify the libopapi.so handle for arithmetic op preference
    global _opapi_handle
    for i, path in enumerate(lib_paths):
        p = path if isinstance(path, str) else path.decode("utf-8")
        if p.endswith("libopapi.so"):
            _opapi_handle = handles[i]
            break

    _initialized = 1


def is_initialized():
    return _initialized != 0

# ---------------------------------------------------------------------------
# Tensor creation / destruction
# ---------------------------------------------------------------------------

cdef inline void* _fast_create_tensor(
    const int64_t* shape, const int64_t* stride, uint64_t ndim,
    int32_t dtype_code, int32_t fmt, void* data_ptr) nogil:
    return _fn_create_tensor(
        shape, ndim, dtype_code,
        stride, 0, fmt,
        shape, ndim,  # storageDims == viewDims
        data_ptr)


cdef inline int32_t _fast_destroy_tensor(void* tensor) nogil:
    return _fn_destroy_tensor(tensor)


def create_tensor(shape_tuple, stride_tuple, int32_t dtype_code,
                  uintptr_t data_ptr, int32_t fmt=2):
    """Create an ACL tensor descriptor.  Returns handle as int."""
    cdef int ndim = len(shape_tuple)
    if ndim > MAX_NDIM:
        raise ValueError(f"ndim {ndim} exceeds MAX_NDIM {MAX_NDIM}")
    cdef int64_t[MAX_NDIM] shape_buf
    cdef int64_t[MAX_NDIM] stride_buf
    cdef int i
    for i in range(ndim):
        shape_buf[i] = shape_tuple[i]
        stride_buf[i] = stride_tuple[i]
    cdef void* tensor = _fast_create_tensor(
        shape_buf, stride_buf, <uint64_t>ndim,
        dtype_code, fmt, <void*>data_ptr)
    if tensor == NULL:
        raise RuntimeError("aclCreateTensor returned null")
    return <uintptr_t>tensor


def destroy_tensor(uintptr_t handle):
    """Destroy an ACL tensor descriptor."""
    cdef int32_t ret
    with nogil:
        ret = _fast_destroy_tensor(<void*>handle)
    return ret

# ---------------------------------------------------------------------------
# Scalar creation / destruction
# ---------------------------------------------------------------------------

def create_scalar(bytes scalar_bytes, int32_t dtype_code):
    """Create an ACL scalar from pre-encoded bytes.  Returns handle as int."""
    cdef const uint8_t* buf = <const uint8_t*>scalar_bytes
    cdef void* scalar
    with nogil:
        scalar = _fn_create_scalar(<void*>buf, dtype_code)
    if scalar == NULL:
        raise RuntimeError("aclCreateScalar returned null")
    return <uintptr_t>scalar


def destroy_scalar(uintptr_t handle):
    cdef int32_t ret
    with nogil:
        ret = _fn_destroy_scalar(<void*>handle)
    return ret

# ---------------------------------------------------------------------------
# IntArray creation / destruction
# ---------------------------------------------------------------------------

def create_int_array(values_tuple):
    """Create an ACL int array from a tuple of ints.  Returns handle as int."""
    cdef int n = len(values_tuple)
    if n == 0:
        return 0
    if n > MAX_NDIM:
        raise ValueError(f"int array length {n} exceeds MAX_NDIM {MAX_NDIM}")
    cdef int64_t[MAX_NDIM] buf
    cdef int i
    for i in range(n):
        buf[i] = values_tuple[i]
    cdef void* arr
    with nogil:
        arr = _fn_create_int_array(buf, <uint64_t>n)
    if arr == NULL:
        raise RuntimeError("aclCreateIntArray returned null")
    return <uintptr_t>arr


def destroy_int_array(uintptr_t handle):
    cdef int32_t ret
    with nogil:
        ret = _fn_destroy_int_array(<void*>handle)
    return ret

# ---------------------------------------------------------------------------
# Executor destruction
# ---------------------------------------------------------------------------

def destroy_executor(uintptr_t handle):
    if handle == 0:
        return 0
    cdef int32_t ret
    with nogil:
        ret = _fn_destroy_executor(<void*>handle)
    return ret

# ---------------------------------------------------------------------------
# Op symbol resolution
# ---------------------------------------------------------------------------

def resolve_op(str op_name):
    """Resolve GetWorkspaceSize and Execute function pointers for an op.

    Returns (getws_ptr, exec_ptr) as ints.  Cached after first call.
    For arithmetic ops (Add/Sub/Mul/Div), prefers libopapi.so to match
    the ctypes _bind_symbol preference in aclnn.py.
    """
    cached = _op_cache.get(op_name)
    if cached is not None:
        return cached

    ws_name = f"aclnn{op_name}GetWorkspaceSize".encode("utf-8")
    exec_name = f"aclnn{op_name}".encode("utf-8")

    cdef void* ws_sym = NULL
    cdef void* exec_sym = NULL
    cdef void* h

    # For arithmetic ops, try libopapi.so first
    if op_name in _PREFER_OPAPI and _opapi_handle != 0:
        h = <void*><uintptr_t>_opapi_handle
        dlerror()
        ws_sym = dlsym(h, ws_name)
        exec_sym = dlsym(h, exec_name)

    # Fall back to searching all handles
    if ws_sym == NULL:
        ws_sym = _find_symbol(_lib_handles, ws_name)
    if exec_sym == NULL:
        exec_sym = _find_symbol(_lib_handles, exec_name)

    result = (<uintptr_t>ws_sym, <uintptr_t>exec_sym)
    _op_cache[op_name] = result
    return result


def resolve_op_optional(str op_name):
    """Like resolve_op but returns None if symbol not found."""
    try:
        return resolve_op(op_name)
    except RuntimeError:
        return None

# ---------------------------------------------------------------------------
# Generic execute (the aclnn<Op> call that all ops share)
# ---------------------------------------------------------------------------

def execute(uintptr_t exec_ptr, uintptr_t workspace_ptr,
            uint64_t workspace_size, uintptr_t executor,
            uintptr_t stream):
    """Call aclnn<Op>(workspace, wsSize, executor, stream).

    This is the second half of every op — the signature is identical for
    all 243 ops.  Returns the int32 return code.
    """
    cdef aclnnExec_t fn = <aclnnExec_t>exec_ptr
    cdef int32_t ret
    with nogil:
        ret = fn(<void*>workspace_ptr, workspace_size,
                 <void*>executor, <void*>stream)
    return ret

# ---------------------------------------------------------------------------
# Binary op helpers — full create+getws+(exec)+destroy in minimal Python calls
#
# These handle the common pattern: create 3 tensors, call GetWorkspaceSize,
# optionally execute (if ws_size == 0), destroy tensors.
#
# If ws_size > 0, the caller must:
#   1. Allocate workspace via acl.rt.malloc(ws_size)
#   2. Call execute(exec_ptr, workspace_ptr, ws_size, executor, stream)
#   3. Free workspace via runtime.defer_raw_free(workspace)
#
# The executor is always returned for the caller to defer-destroy.
# ---------------------------------------------------------------------------

def binary_op_with_alpha(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        other_shape, other_stride,
        out_shape, out_stride,
        int32_t dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
        uintptr_t alpha_scalar,
        uintptr_t stream):
    """Binary op with alpha (add, sub): create tensors, getws, exec, destroy.

    Returns (ws_size: int, executor_ptr: int).
    If ws_size == 0, execute has already been called.
    If ws_size > 0, caller must allocate workspace and call execute().
    """
    cdef int self_ndim = len(self_shape)
    cdef int other_ndim = len(other_shape)
    cdef int out_ndim = len(out_shape)

    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] o_shape, o_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i

    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(other_ndim):
        o_shape[i] = other_shape[i]
        o_stride[i] = other_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]

    cdef void* self_t = NULL
    cdef void* other_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret

    with nogil:
        self_t = _fast_create_tensor(
            s_shape, s_stride, <uint64_t>self_ndim,
            dtype_code, fmt, <void*>self_ptr)
        other_t = _fast_create_tensor(
            o_shape, o_stride, <uint64_t>other_ndim,
            dtype_code, fmt, <void*>other_ptr)
        out_t = _fast_create_tensor(
            r_shape, r_stride, <uint64_t>out_ndim,
            dtype_code, fmt, <void*>out_ptr)

    if self_t == NULL or other_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if other_t != NULL: _fast_destroy_tensor(other_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")

    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*,
                                uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, other_t, <void*>alpha_scalar, out_t,
                &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")

        # Fast path: no workspace needed, execute immediately
        if ws_size == 0:
            with nogil:
                ret = (<aclnnExec_t>exec_ptr)(
                    NULL, 0, executor, <void*>stream)
            if ret != 0:
                raise RuntimeError(f"Execute failed: {ret}")

        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(other_t)
            _fast_destroy_tensor(out_t)


def binary_op_no_alpha(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        other_shape, other_stride,
        out_shape, out_stride,
        int32_t dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    """Binary op without alpha (mul, div): create tensors, getws, exec, destroy.

    Returns (ws_size: int, executor_ptr: int).
    If ws_size == 0, execute has already been called.
    If ws_size > 0, caller must allocate workspace and call execute().
    """
    cdef int self_ndim = len(self_shape)
    cdef int other_ndim = len(other_shape)
    cdef int out_ndim = len(out_shape)

    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] o_shape, o_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i

    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(other_ndim):
        o_shape[i] = other_shape[i]
        o_stride[i] = other_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]

    cdef void* self_t = NULL
    cdef void* other_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret

    with nogil:
        self_t = _fast_create_tensor(
            s_shape, s_stride, <uint64_t>self_ndim,
            dtype_code, fmt, <void*>self_ptr)
        other_t = _fast_create_tensor(
            o_shape, o_stride, <uint64_t>other_ndim,
            dtype_code, fmt, <void*>other_ptr)
        out_t = _fast_create_tensor(
            r_shape, r_stride, <uint64_t>out_ndim,
            dtype_code, fmt, <void*>out_ptr)

    if self_t == NULL or other_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if other_t != NULL: _fast_destroy_tensor(other_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")

    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*,
                                uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, other_t, out_t,
                &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")

        if ws_size == 0:
            with nogil:
                ret = (<aclnnExec_t>exec_ptr)(
                    NULL, 0, executor, <void*>stream)
            if ret != 0:
                raise RuntimeError(f"Execute failed: {ret}")

        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(other_t)
            _fast_destroy_tensor(out_t)
