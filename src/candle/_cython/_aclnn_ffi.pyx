# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot-path for ACLNN FFI calls.

Replaces ctypes-based tensor/scalar creation and op execution with direct
C function-pointer calls resolved via dlopen/dlsym at runtime.

The 243 op functions in aclnn.py stay as Python.  This module accelerates
the primitives they call: create_tensor, create_scalar, destroy_*, and
the generic execute pattern.
"""

from libc.stdint cimport int8_t, int32_t, int64_t, uint64_t, uint8_t, uintptr_t
from libc.stdlib cimport malloc, free

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
ctypedef void* (*aclCreateBoolArray_t)(const uint8_t*, uint64_t) noexcept nogil
ctypedef int32_t (*aclDestroyBoolArray_t)(void*) noexcept nogil
ctypedef int32_t (*aclDestroyExecutor_t)(void*) noexcept nogil
ctypedef void* (*aclCreateTensorList_t)(void**, uint64_t) noexcept nogil
ctypedef int32_t (*aclDestroyTensorList_t)(void*) noexcept nogil

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
cdef aclCreateBoolArray_t  _fn_create_bool_array  = NULL
cdef aclDestroyBoolArray_t _fn_destroy_bool_array = NULL
cdef aclDestroyExecutor_t _fn_destroy_executor  = NULL
cdef aclCreateTensorList_t  _fn_create_tensor_list = NULL
cdef aclDestroyTensorList_t _fn_destroy_tensor_list = NULL

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

cdef dict _executor_cleanup = {}

# ---------------------------------------------------------------------------
# Tensor descriptor cache — reuse aclTensor handles for input tensors
# ---------------------------------------------------------------------------

cdef class TensorDescCache:
    """LRU cache mapping (data_ptr, shape, stride, dtype_code, fmt) -> aclTensor handle.

    Only input tensors should be cached.  Output tensors get a new device ptr
    each op and must NOT be placed in this cache.

    Thread safety: relies on the Python GIL (get_or_create is called with GIL).
    """

    cdef dict _cache      # key -> handle (uintptr_t as Python int)
    cdef list _order      # insertion-order keys for LRU eviction
    cdef int _max_size

    def __cinit__(self, int max_size=64):
        self._cache = {}
        self._order = []
        self._max_size = max_size

    def get_or_create(self, int64_t data_ptr, shape, stride,
                      int32_t dtype_code, int32_t fmt):
        """Return a cached aclTensor handle, creating one if necessary.

        *shape* and *stride* are Python sequences (tuple or torch.Size).
        Returns the handle as an integer (uintptr_t).
        """
        cdef object key = (data_ptr, tuple(shape), tuple(stride), dtype_code, fmt)
        cdef object existing = self._cache.get(key)
        if existing is not None:
            return existing

        # Cache miss: create a new descriptor
        cdef int ndim = len(shape)
        if ndim > MAX_NDIM:
            raise ValueError(f"ndim {ndim} exceeds MAX_NDIM {MAX_NDIM}")

        cdef int64_t[MAX_NDIM] shape_buf
        cdef int64_t[MAX_NDIM] stride_buf
        cdef int i
        for i in range(ndim):
            shape_buf[i] = shape[i]
            stride_buf[i] = stride[i]

        cdef void* handle
        with nogil:
            handle = _fast_create_tensor(
                shape_buf, stride_buf, <uint64_t>ndim,
                dtype_code, fmt, <void*>data_ptr)
        if handle == NULL:
            raise RuntimeError("aclCreateTensor returned null in TensorDescCache")

        cdef object handle_int = <uintptr_t>handle
        cdef object old_key
        cdef object old_handle

        # LRU eviction: if at capacity, remove the oldest entry
        cdef uintptr_t evict_h
        if len(self._cache) >= self._max_size:
            old_key = self._order[0]
            del self._order[0]
            old_handle = self._cache.pop(old_key, None)
            if old_handle is not None:
                evict_h = <uintptr_t>old_handle
                with nogil:
                    _fast_destroy_tensor(<void*>evict_h)

        self._cache[key] = handle_int
        self._order.append(key)
        return handle_int

    def invalidate_range(self, int64_t base_ptr, int64_t size):
        """Remove all entries whose data_ptr falls within [base_ptr, base_ptr+size)."""
        cdef object key
        cdef int64_t dp
        cdef list to_remove = []
        for key in self._cache:
            dp = <int64_t>key[0]
            if base_ptr <= dp < base_ptr + size:
                to_remove.append(key)
        cdef uintptr_t remove_h
        for key in to_remove:
            handle = self._cache.pop(key, None)
            self._order.remove(key)
            if handle is not None:
                remove_h = <uintptr_t>handle
                with nogil:
                    _fast_destroy_tensor(<void*>remove_h)

    def clear(self):
        """Destroy all cached descriptor handles."""
        cdef object handle
        cdef uintptr_t clear_h
        for handle in self._cache.values():
            clear_h = <uintptr_t>handle
            with nogil:
                _fast_destroy_tensor(<void*>clear_h)
        self._cache.clear()
        self._order.clear()

    def size(self):
        """Return the number of cached entries."""
        return len(self._cache)


# Module-level singleton
cdef TensorDescCache _tensor_desc_cache = TensorDescCache()


def get_tensor_desc_cache():
    """Return the module-level TensorDescCache singleton."""
    return _tensor_desc_cache

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
    global _fn_create_bool_array, _fn_destroy_bool_array
    global _fn_destroy_executor, _fn_create_tensor_list, _fn_destroy_tensor_list, _initialized

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
    _fn_create_bool_array = <aclCreateBoolArray_t>_find_symbol(
        handles, b"aclCreateBoolArray")
    _fn_destroy_bool_array = <aclDestroyBoolArray_t>_find_symbol(
        handles, b"aclDestroyBoolArray")
    _fn_destroy_executor = <aclDestroyExecutor_t>_find_symbol(
        handles, b"aclDestroyAclOpExecutor")
    _fn_create_tensor_list = <aclCreateTensorList_t>_find_symbol(
        handles, b"aclCreateTensorList")
    _fn_destroy_tensor_list = <aclDestroyTensorList_t>_find_symbol(
        handles, b"aclDestroyTensorList")

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

cdef inline void* _fast_create_tensor_ex(
    const int64_t* shape, const int64_t* stride, uint64_t ndim,
    int32_t dtype_code, int32_t fmt,
    const int64_t* storage_dims, uint64_t storage_ndim,
    int64_t storage_offset, void* data_ptr) nogil:
    return _fn_create_tensor(
        shape, ndim, dtype_code,
        stride, storage_offset, fmt,
        storage_dims, storage_ndim,
        data_ptr)


cdef inline void* _fast_create_tensor(
    const int64_t* shape, const int64_t* stride, uint64_t ndim,
    int32_t dtype_code, int32_t fmt, void* data_ptr) nogil:
    return _fast_create_tensor_ex(
        shape, stride, ndim, dtype_code, fmt,
        shape, ndim,
        0,
        data_ptr)


cdef inline int32_t _fast_destroy_tensor(void* tensor) nogil:
    return _fn_destroy_tensor(tensor)


cdef inline void* _fast_create_bool_array(const uint8_t* values, uint64_t size) nogil:
    return _fn_create_bool_array(values, size)


cdef inline int32_t _fast_destroy_bool_array(void* array) nogil:
    return _fn_destroy_bool_array(array)


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
    cdef int i
    cdef int64_t* buf = NULL
    cdef void* arr
    if n == 0:
        return 0
    buf = <int64_t*>malloc(n * sizeof(int64_t))
    if buf == NULL:
        raise MemoryError("malloc failed for int array buffer")
    try:
        for i in range(n):
            buf[i] = values_tuple[i]
        with nogil:
            arr = _fn_create_int_array(buf, <uint64_t>n)
        if arr == NULL:
            raise RuntimeError("aclCreateIntArray returned null")
        return <uintptr_t>arr
    finally:
        free(buf)


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
    _release_executor_cleanup(handle)
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

cdef void _release_executor_cleanup(uintptr_t executor):
    cdef object cleanup
    cdef object item
    cdef object kind
    cdef uintptr_t handle
    if executor == 0:
        return
    cleanup = _executor_cleanup.pop(int(executor), None)
    if cleanup is None:
        return
    for item in cleanup:
        kind = item[0]
        handle = <uintptr_t>item[1]
        if handle == 0:
            continue
        if _fn_destroy_tensor_list != NULL and kind == 'l':
            with nogil:
                _fn_destroy_tensor_list(<void*>handle)
        elif _fn_destroy_tensor != NULL and kind == 't':
            with nogil:
                _fn_destroy_tensor(<void*>handle)
        elif _fn_destroy_int_array != NULL and kind == 'i':
            with nogil:
                _fn_destroy_int_array(<void*>handle)
        elif _fn_destroy_bool_array != NULL and kind == 'b':
            with nogil:
                _fn_destroy_bool_array(<void*>handle)
        elif _fn_destroy_scalar != NULL and kind == 's':
            with nogil:
                _fn_destroy_scalar(<void*>handle)


cdef void _register_executor_cleanup(uintptr_t executor, object cleanup):
    if executor == 0 or cleanup is None:
        return
    _executor_cleanup[int(executor)] = cleanup


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

    # Input tensors: use descriptor cache (skips aclCreateTensor on cache hit)
    self_t = <void*><uintptr_t>_tensor_desc_cache.get_or_create(
        <int64_t>self_ptr,
        tuple(self_shape[:self_ndim]), tuple(self_stride[:self_ndim]),
        dtype_code, fmt)
    other_t = <void*><uintptr_t>_tensor_desc_cache.get_or_create(
        <int64_t>other_ptr,
        tuple(other_shape[:other_ndim]), tuple(other_stride[:other_ndim]),
        dtype_code, fmt)
    # Output tensor: always create fresh (new device ptr each op)
    with nogil:
        out_t = _fast_create_tensor(
            r_shape, r_stride, <uint64_t>out_ndim,
            dtype_code, fmt, <void*>out_ptr)

    if self_t == NULL or other_t == NULL or out_t == NULL:
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
        # Only out_t goes into executor cleanup — self_t and other_t are owned by cache
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        out_t = NULL

        # Fast path: no workspace needed, execute immediately
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(
                        NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise

        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if out_t != NULL:
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

    # Input tensors: use descriptor cache (skips aclCreateTensor on cache hit)
    self_t = <void*><uintptr_t>_tensor_desc_cache.get_or_create(
        <int64_t>self_ptr,
        tuple(self_shape[:self_ndim]), tuple(self_stride[:self_ndim]),
        dtype_code, fmt)
    other_t = <void*><uintptr_t>_tensor_desc_cache.get_or_create(
        <int64_t>other_ptr,
        tuple(other_shape[:other_ndim]), tuple(other_stride[:other_ndim]),
        dtype_code, fmt)
    # Output tensor: always create fresh (new device ptr each op)
    with nogil:
        out_t = _fast_create_tensor(
            r_shape, r_stride, <uint64_t>out_ndim,
            dtype_code, fmt, <void*>out_ptr)

    if self_t == NULL or other_t == NULL or out_t == NULL:
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
        # Only out_t goes into executor cleanup — self_t and other_t are owned by cache
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        out_t = NULL

        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(
                        NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise

        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def tensor_scalar_op_with_alpha(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int32_t dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t scalar_handle, uintptr_t alpha_scalar,
        uintptr_t stream):
    """Tensor-scalar op with alpha (Adds): create tensors, getws, exec, destroy."""
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)

    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i

    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]

    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret

    with nogil:
        self_t = _fast_create_tensor(
            s_shape, s_stride, <uint64_t>self_ndim,
            dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(
            r_shape, r_stride, <uint64_t>out_ndim,
            dtype_code, fmt, <void*>out_ptr)

    if self_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")

    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*,
                                uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, <void*>scalar_handle, <void*>alpha_scalar, out_t,
                &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL

        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(
                        NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise

        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def tensor_scalar_op_no_alpha(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int32_t dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t scalar_handle,
        uintptr_t stream):
    """Tensor-scalar op without alpha (Subs): create tensors, getws, exec, destroy."""
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)

    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i

    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]

    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret

    with nogil:
        self_t = _fast_create_tensor(
            s_shape, s_stride, <uint64_t>self_ndim,
            dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(
            r_shape, r_stride, <uint64_t>out_ndim,
            dtype_code, fmt, <void*>out_ptr)

    if self_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")

    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*,
                                uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, <void*>scalar_handle, out_t,
                &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL

        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(
                        NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise

        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def unary_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    """Unary op helper: create input/output tensors, getws, exec, destroy."""
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)

    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i

    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]

    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret

    with nogil:
        self_t = _fast_create_tensor(
            s_shape, s_stride, <uint64_t>self_ndim,
            self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(
            r_shape, r_stride, <uint64_t>out_ndim,
            out_dtype_code, fmt, <void*>out_ptr)

    if self_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")

    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL

        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise

        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def unary_out_dtype_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def reduce_sum_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        dims_tuple, bint keepdim,
        int32_t dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    """ReduceSum helper with C-side tensor/int-array descriptors."""
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int dims_ndim = len(dims_tuple)

    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t[MAX_NDIM] dims_buf
    cdef int i

    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    for i in range(dims_ndim):
        dims_buf[i] = dims_tuple[i]

    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* dims_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret

    with nogil:
        self_t = _fast_create_tensor(
            s_shape, s_stride, <uint64_t>self_ndim,
            dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(
            r_shape, r_stride, <uint64_t>out_ndim,
            dtype_code, fmt, <void*>out_ptr)
        if dims_ndim > 0:
            dims_handle = _fn_create_int_array(dims_buf, <uint64_t>dims_ndim)

    if self_t == NULL or out_t == NULL or (dims_ndim > 0 and dims_handle == NULL):
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        if dims_handle != NULL: _fn_destroy_int_array(dims_handle)
        raise RuntimeError("ACLNN reduction descriptor creation failed")

    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, bint, int32_t, void*,
                                uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dims_handle, keepdim, dtype_code, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")

        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>dims_handle)] if dims_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        dims_handle = NULL
        self_t = NULL
        out_t = NULL

        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise

        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dims_handle != NULL:
                _fn_destroy_int_array(dims_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def reduce_dims_dtype_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        dims_tuple, bint keepdim,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int dims_ndim = len(dims_tuple)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t[MAX_NDIM] dims_buf
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    for i in range(dims_ndim):
        dims_buf[i] = dims_tuple[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* dims_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if dims_ndim > 0:
            dims_handle = _fn_create_int_array(dims_buf, <uint64_t>dims_ndim)
    if self_t == NULL or out_t == NULL or (dims_ndim > 0 and dims_handle == NULL):
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        if dims_handle != NULL: _fn_destroy_int_array(dims_handle)
        raise RuntimeError("ACLNN reduction descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, bint, int32_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dims_handle, keepdim, out_dtype_code, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>dims_handle)] if dims_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        dims_handle = NULL
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dims_handle != NULL:
                _fn_destroy_int_array(dims_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def arg_reduce_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int64_t dim, bint keepdim,
        int32_t self_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    """Arg-reduce helper for ArgMax/ArgMin."""
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)

    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i

    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]

    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret

    with nogil:
        self_t = _fast_create_tensor(
            s_shape, s_stride, <uint64_t>self_ndim,
            self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(
            r_shape, r_stride, <uint64_t>out_ndim,
            9, fmt, <void*>out_ptr)

    if self_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")

    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, bint, void*,
                                uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dim, keepdim, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL

        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise

        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def cast_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int32_t src_dtype_code, int32_t dst_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, src_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, dst_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int32_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dst_dtype_code, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def argsort_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int64_t dim, bint descending, int32_t self_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, 9, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, bint, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dim, descending, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def dual_output_with_indices_op(
        str variant, uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        values_shape, values_stride,
        indices_shape, indices_stride,
        int64_t dim, bint flag_a, bint flag_b, int64_t k,
        int32_t self_dtype_code, int32_t values_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t values_ptr, uintptr_t indices_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int values_ndim = len(values_shape)
    cdef int indices_ndim = len(indices_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] v_shape, v_stride
    cdef int64_t[MAX_NDIM] i_shape, i_stride
    cdef int idx
    for idx in range(self_ndim):
        s_shape[idx] = self_shape[idx]
        s_stride[idx] = self_stride[idx]
    for idx in range(values_ndim):
        v_shape[idx] = values_shape[idx]
        v_stride[idx] = values_stride[idx]
    for idx in range(indices_ndim):
        i_shape[idx] = indices_shape[idx]
        i_stride[idx] = indices_stride[idx]
    cdef void* self_t = NULL
    cdef void* values_t = NULL
    cdef void* indices_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        values_t = _fast_create_tensor(v_shape, v_stride, <uint64_t>values_ndim, values_dtype_code, fmt, <void*>values_ptr)
        indices_t = _fast_create_tensor(i_shape, i_stride, <uint64_t>indices_ndim, 9, fmt, <void*>indices_ptr)
    if self_t == NULL or values_t == NULL or indices_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if values_t != NULL: _fast_destroy_tensor(values_t)
        if indices_t != NULL: _fast_destroy_tensor(indices_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        if variant == "dim_reduce":
            with nogil:
                ret = (<int32_t (*)(void*, int64_t, bint, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                    self_t, dim, flag_a, values_t, indices_t, &ws_size, &executor)
        elif variant == "sort":
            with nogil:
                ret = (<int32_t (*)(void*, bint, int64_t, bint, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                    self_t, flag_a, dim, flag_b, values_t, indices_t, &ws_size, &executor)
        elif variant == "topk":
            with nogil:
                ret = (<int32_t (*)(void*, int64_t, int64_t, bint, bint, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                    self_t, k, dim, flag_a, flag_b, values_t, indices_t, &ws_size, &executor)
        elif variant == "cummax":
            with nogil:
                ret = (<int32_t (*)(void*, int64_t, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                    self_t, dim, values_t, indices_t, &ws_size, &executor)
        elif variant == "cummin":
            with nogil:
                ret = (<int32_t (*)(void*, int64_t, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                    self_t, dim, values_t, indices_t, &ws_size, &executor)
        else:
            raise RuntimeError(f"unsupported dual output variant: {variant}")
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>values_t)] if values_t != NULL else [])
            + ([('t', <uintptr_t>indices_t)] if indices_t != NULL else []),
        )
        self_t = NULL
        values_t = NULL
        indices_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if values_t != NULL:
                _fast_destroy_tensor(values_t)
            if indices_t != NULL:
                _fast_destroy_tensor(indices_t)


def reduce_all_dtype_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int32_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, out_dtype_code, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)



def axis_keepdim_dtype_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int64_t dim, bint keepdim,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, bint, int32_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dim, keepdim, out_dtype_code, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def axis_dtype_unary_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int64_t dim, int32_t out_dtype_code, int32_t self_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, int32_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dim, out_dtype_code, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)



def tensor_scalar_bool_out_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int32_t self_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t scalar_handle,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t[MAX_NDIM] out_storage_shape, out_storage_stride
    cdef int i
    cdef int32_t bool_dtype_code = 12
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
        out_storage_shape[i] = out_shape[i]
        out_storage_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(
            s_shape, s_stride, <uint64_t>self_ndim,
            self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor_ex(
            r_shape, r_stride, <uint64_t>out_ndim,
            bool_dtype_code, fmt,
            out_storage_shape, <uint64_t>out_ndim,
            0,
            <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, <void*>scalar_handle, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def tensor_scalar_dtype_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int32_t out_dtype_code, int32_t self_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t scalar_handle,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, int32_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, <void*>scalar_handle, out_dtype_code, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)



def axis_unary_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int64_t dim, int32_t dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL: _fast_destroy_tensor(self_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dim, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def layer_norm_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        input_shape, input_stride,
        out_shape, out_stride,
        stats_shape, stats_stride,
        weight_shape, weight_stride,
        bias_shape, bias_stride,
        normalized_shape, double eps,
        int32_t dtype_code, int32_t fmt,
        uintptr_t input_ptr, uintptr_t weight_ptr, uintptr_t bias_ptr,
        uintptr_t out_ptr, uintptr_t mean_ptr, uintptr_t rstd_ptr,
        uintptr_t stream):
    cdef int input_ndim = len(input_shape)
    cdef int out_ndim = len(out_shape)
    cdef int stats_ndim = len(stats_shape)
    cdef int weight_ndim = len(weight_shape)
    cdef int bias_ndim = len(bias_shape)
    cdef int norm_ndim = len(normalized_shape)
    cdef int64_t[MAX_NDIM] input_shape_buf, input_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int64_t[MAX_NDIM] stats_shape_buf, stats_stride_buf
    cdef int64_t[MAX_NDIM] weight_shape_buf, weight_stride_buf
    cdef int64_t[MAX_NDIM] bias_shape_buf, bias_stride_buf
    cdef int64_t[MAX_NDIM] norm_shape_buf
    cdef int i
    for i in range(input_ndim):
        input_shape_buf[i] = input_shape[i]
        input_stride_buf[i] = input_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    for i in range(stats_ndim):
        stats_shape_buf[i] = stats_shape[i]
        stats_stride_buf[i] = stats_stride[i]
    for i in range(weight_ndim):
        weight_shape_buf[i] = weight_shape[i]
        weight_stride_buf[i] = weight_stride[i]
    for i in range(bias_ndim):
        bias_shape_buf[i] = bias_shape[i]
        bias_stride_buf[i] = bias_stride[i]
    for i in range(norm_ndim):
        norm_shape_buf[i] = normalized_shape[i]

    cdef void* input_t = NULL
    cdef void* out_t = NULL
    cdef void* mean_t = NULL
    cdef void* rstd_t = NULL
    cdef void* weight_t = NULL
    cdef void* bias_t = NULL
    cdef void* norm_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret

    with nogil:
        input_t = _fast_create_tensor(input_shape_buf, input_stride_buf, <uint64_t>input_ndim, dtype_code, fmt, <void*>input_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, dtype_code, fmt, <void*>out_ptr)
        if mean_ptr != 0:
            mean_t = _fast_create_tensor(stats_shape_buf, stats_stride_buf, <uint64_t>stats_ndim, 0, fmt, <void*>mean_ptr)
        if rstd_ptr != 0:
            rstd_t = _fast_create_tensor(stats_shape_buf, stats_stride_buf, <uint64_t>stats_ndim, 0, fmt, <void*>rstd_ptr)
        if weight_ptr != 0:
            weight_t = _fast_create_tensor(weight_shape_buf, weight_stride_buf, <uint64_t>weight_ndim, dtype_code, fmt, <void*>weight_ptr)
        if bias_ptr != 0:
            bias_t = _fast_create_tensor(bias_shape_buf, bias_stride_buf, <uint64_t>bias_ndim, dtype_code, fmt, <void*>bias_ptr)
        if norm_ndim > 0:
            norm_handle = _fn_create_int_array(norm_shape_buf, <uint64_t>norm_ndim)

    if input_t == NULL or out_t == NULL or (mean_ptr != 0 and mean_t == NULL) or (rstd_ptr != 0 and rstd_t == NULL) or (weight_ptr != 0 and weight_t == NULL) or (bias_ptr != 0 and bias_t == NULL) or (norm_ndim > 0 and norm_handle == NULL):
        if input_t != NULL: _fast_destroy_tensor(input_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        if mean_t != NULL: _fast_destroy_tensor(mean_t)
        if rstd_t != NULL: _fast_destroy_tensor(rstd_t)
        if weight_t != NULL: _fast_destroy_tensor(weight_t)
        if bias_t != NULL: _fast_destroy_tensor(bias_t)
        if norm_handle != NULL: _fn_destroy_int_array(norm_handle)
        raise RuntimeError("ACLNN layer_norm descriptor creation failed")

    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, double, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                input_t,
                norm_handle,
                weight_t,
                bias_t,
                eps,
                out_t,
                mean_t,
                rstd_t,
                &ws_size,
                &executor,
            )
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if norm_handle != NULL:
                _fn_destroy_int_array(norm_handle)
            _fast_destroy_tensor(input_t)
            _fast_destroy_tensor(out_t)
            if mean_t != NULL:
                _fast_destroy_tensor(mean_t)
            if rstd_t != NULL:
                _fast_destroy_tensor(rstd_t)
            if weight_t != NULL:
                _fast_destroy_tensor(weight_t)
            if bias_t != NULL:
                _fast_destroy_tensor(bias_t)



def tensor_two_scalars_dim_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int64_t dim,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t scalar_a, uintptr_t scalar_b,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, int64_t, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, <void*>scalar_a, dim, <void*>scalar_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)



def tensor_int_array_two_bools_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        dims_tuple, bint flag_a, bint flag_b,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int dims_ndim = len(dims_tuple)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t[MAX_NDIM] dims_buf
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    for i in range(dims_ndim):
        dims_buf[i] = dims_tuple[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* dims_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if dims_ndim > 0:
            dims_handle = _fn_create_int_array(dims_buf, <uint64_t>dims_ndim)
    if self_t == NULL or out_t == NULL or (dims_ndim > 0 and dims_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if dims_handle != NULL:
            _fn_destroy_int_array(dims_handle)
        raise RuntimeError("ACLNN descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, bint, bint, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dims_handle, flag_a, flag_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>dims_handle)] if dims_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        dims_handle = NULL
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dims_handle != NULL:
                _fn_destroy_int_array(dims_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)



def tensor_int_array_bool_two_doubles_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        dims_tuple, bint flag,
        double scalar_a, double scalar_b,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int dims_ndim = len(dims_tuple)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t[MAX_NDIM] dims_buf
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    for i in range(dims_ndim):
        dims_buf[i] = dims_tuple[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* dims_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if dims_ndim > 0:
            dims_handle = _fn_create_int_array(dims_buf, <uint64_t>dims_ndim)
    if self_t == NULL or out_t == NULL or (dims_ndim > 0 and dims_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if dims_handle != NULL:
            _fn_destroy_int_array(dims_handle)
        raise RuntimeError("ACLNN descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, bint, double, double, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dims_handle, flag, scalar_a, scalar_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>dims_handle)] if dims_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        dims_handle = NULL
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dims_handle != NULL:
                _fn_destroy_int_array(dims_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def tensor_int_array_bool_double_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        dims_tuple, bint flag,
        double scalar_value,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int dims_ndim = len(dims_tuple)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t[MAX_NDIM] dims_buf
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    for i in range(dims_ndim):
        dims_buf[i] = dims_tuple[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* dims_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if dims_ndim > 0:
            dims_handle = _fn_create_int_array(dims_buf, <uint64_t>dims_ndim)
    if self_t == NULL or out_t == NULL or (dims_ndim > 0 and dims_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if dims_handle != NULL:
            _fn_destroy_int_array(dims_handle)
        raise RuntimeError("ACLNN descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, bint, double, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dims_handle, flag, scalar_value, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>dims_handle)] if dims_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        dims_handle = NULL
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dims_handle != NULL:
                _fn_destroy_int_array(dims_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def tensor_scalar_int_array_bool_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        dims_tuple, bint keepdim,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t scalar_handle,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int dims_ndim = len(dims_tuple)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t[MAX_NDIM] dims_buf
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    for i in range(dims_ndim):
        dims_buf[i] = dims_tuple[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* dims_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if dims_ndim > 0:
            dims_handle = _fn_create_int_array(dims_buf, <uint64_t>dims_ndim)
    if self_t == NULL or out_t == NULL or (dims_ndim > 0 and dims_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if dims_handle != NULL:
            _fn_destroy_int_array(dims_handle)
        raise RuntimeError("ACLNN descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, bint, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, <void*>scalar_handle, dims_handle, keepdim, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>dims_handle)] if dims_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        dims_handle = NULL
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dims_handle != NULL:
                _fn_destroy_int_array(dims_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def tensor_three_scalars_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t scalar_a, uintptr_t scalar_b, uintptr_t scalar_c,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, <void*>scalar_a, <void*>scalar_b, <void*>scalar_c, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def inplace_unary_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        shape, stride,
        int32_t dtype_code, int32_t fmt,
        uintptr_t ptr,
        uintptr_t stream):
    cdef int ndim = len(shape)
    cdef int64_t[MAX_NDIM] shape_buf, stride_buf
    cdef int i
    for i in range(ndim):
        shape_buf[i] = shape[i]
        stride_buf[i] = stride[i]
    cdef void* tensor = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        tensor = _fast_create_tensor(shape_buf, stride_buf, <uint64_t>ndim, dtype_code, fmt, <void*>ptr)
    if tensor == NULL:
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                tensor, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            [('t', <uintptr_t>tensor)] if tensor != NULL else [],
        )
        tensor = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if tensor != NULL:
                _fast_destroy_tensor(tensor)


def inplace_normal_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        shape, stride,
        int32_t dtype_code, int32_t fmt,
        uintptr_t ptr,
        float mean,
        float std,
        int64_t seed,
        int64_t offset,
        uintptr_t stream):
    cdef int ndim = len(shape)
    cdef int64_t[MAX_NDIM] shape_buf, stride_buf
    cdef int i
    for i in range(ndim):
        shape_buf[i] = shape[i]
        stride_buf[i] = stride[i]
    cdef void* tensor = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        tensor = _fast_create_tensor(shape_buf, stride_buf, <uint64_t>ndim, dtype_code, fmt, <void*>ptr)
    if tensor == NULL:
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, float, float, int64_t, int64_t, uint64_t*, void**) noexcept nogil>getws_ptr)(
                tensor, mean, std, seed, offset, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(tensor)


def inplace_uniform_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        shape, stride,
        int32_t dtype_code, int32_t fmt,
        uintptr_t ptr,
        double low,
        double high,
        uint64_t seed,
        uint64_t offset,
        uintptr_t stream):
    cdef int ndim = len(shape)
    cdef int64_t[MAX_NDIM] shape_buf, stride_buf
    cdef int i
    for i in range(ndim):
        shape_buf[i] = shape[i]
        stride_buf[i] = stride[i]
    cdef void* tensor = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        tensor = _fast_create_tensor(shape_buf, stride_buf, <uint64_t>ndim, dtype_code, fmt, <void*>ptr)
    if tensor == NULL:
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, double, double, uint64_t, uint64_t, uint64_t*, void**) noexcept nogil>getws_ptr)(
                tensor, low, high, seed, offset, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(tensor)


def inplace_fill_scalar_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        shape, stride,
        int32_t dtype_code, int32_t fmt,
        uintptr_t ptr,
        uintptr_t scalar_handle,
        uintptr_t stream):
    cdef int ndim = len(shape)
    cdef int64_t[MAX_NDIM] shape_buf, stride_buf
    cdef int i
    for i in range(ndim):
        shape_buf[i] = shape[i]
        stride_buf[i] = stride[i]
    cdef void* tensor = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        tensor = _fast_create_tensor(shape_buf, stride_buf, <uint64_t>ndim, dtype_code, fmt, <void*>ptr)
    if tensor == NULL:
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                tensor, <void*>scalar_handle, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            [('t', <uintptr_t>tensor)] if tensor != NULL else [],
        )
        tensor = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if tensor != NULL:
                _fast_destroy_tensor(tensor)


def inplace_copy_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        dst_shape, dst_stride,
        src_shape, src_stride,
        int32_t dst_dtype_code, int32_t src_dtype_code, int32_t fmt,
        uintptr_t dst_ptr, uintptr_t src_ptr,
        uintptr_t stream):
    cdef int dst_ndim = len(dst_shape)
    cdef int src_ndim = len(src_shape)
    cdef int64_t[MAX_NDIM] d_shape, d_stride
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int i
    for i in range(dst_ndim):
        d_shape[i] = dst_shape[i]
        d_stride[i] = dst_stride[i]
    for i in range(src_ndim):
        s_shape[i] = src_shape[i]
        s_stride[i] = src_stride[i]
    cdef void* dst_t = NULL
    cdef void* src_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        dst_t = _fast_create_tensor(d_shape, d_stride, <uint64_t>dst_ndim, dst_dtype_code, fmt, <void*>dst_ptr)
        src_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>src_ndim, src_dtype_code, fmt, <void*>src_ptr)
    if dst_t == NULL or src_t == NULL:
        if dst_t != NULL:
            _fast_destroy_tensor(dst_t)
        if src_t != NULL:
            _fast_destroy_tensor(src_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                dst_t, src_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>dst_t)] if dst_t != NULL else [])
            + ([('t', <uintptr_t>src_t)] if src_t != NULL else []),
        )
        dst_t = NULL
        src_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dst_t != NULL:
                _fast_destroy_tensor(dst_t)
            if src_t != NULL:
                _fast_destroy_tensor(src_t)


def inplace_masked_fill_scalar_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        mask_shape, mask_stride,
        int32_t self_dtype_code, int32_t mask_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t mask_ptr,
        uintptr_t scalar_handle,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int mask_ndim = len(mask_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] m_shape, m_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(mask_ndim):
        m_shape[i] = mask_shape[i]
        m_stride[i] = mask_stride[i]
    cdef void* self_t = NULL
    cdef void* mask_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        mask_t = _fast_create_tensor(m_shape, m_stride, <uint64_t>mask_ndim, mask_dtype_code, fmt, <void*>mask_ptr)
    if self_t == NULL or mask_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if mask_t != NULL:
            _fast_destroy_tensor(mask_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, mask_t, <void*>scalar_handle, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>mask_t)] if mask_t != NULL else []),
        )
        self_t = NULL
        mask_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if mask_t != NULL:
                _fast_destroy_tensor(mask_t)


def inplace_index_fill_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        index_shape, index_stride,
        int64_t dim,
        int32_t self_dtype_code, int32_t index_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t index_ptr,
        uintptr_t scalar_handle,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int index_ndim = len(index_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] i_shape, i_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(index_ndim):
        i_shape[i] = index_shape[i]
        i_stride[i] = index_stride[i]
    cdef void* self_t = NULL
    cdef void* index_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        index_t = _fast_create_tensor(i_shape, i_stride, <uint64_t>index_ndim, index_dtype_code, fmt, <void*>index_ptr)
    if self_t == NULL or index_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if index_t != NULL:
            _fast_destroy_tensor(index_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dim, index_t, <void*>scalar_handle, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>index_t)] if index_t != NULL else []),
        )
        self_t = NULL
        index_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if index_t != NULL:
                _fast_destroy_tensor(index_t)


def inplace_index_copy_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        index_shape, index_stride,
        source_shape, source_stride,
        int64_t dim,
        int32_t self_dtype_code, int32_t index_dtype_code, int32_t source_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t index_ptr, uintptr_t source_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int index_ndim = len(index_shape)
    cdef int source_ndim = len(source_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] i_shape, i_stride
    cdef int64_t[MAX_NDIM] src_shape_buf, src_stride_buf
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(index_ndim):
        i_shape[i] = index_shape[i]
        i_stride[i] = index_stride[i]
    for i in range(source_ndim):
        src_shape_buf[i] = source_shape[i]
        src_stride_buf[i] = source_stride[i]
    cdef void* self_t = NULL
    cdef void* index_t = NULL
    cdef void* source_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        index_t = _fast_create_tensor(i_shape, i_stride, <uint64_t>index_ndim, index_dtype_code, fmt, <void*>index_ptr)
        source_t = _fast_create_tensor(src_shape_buf, src_stride_buf, <uint64_t>source_ndim, source_dtype_code, fmt, <void*>source_ptr)
    if self_t == NULL or index_t == NULL or source_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if index_t != NULL:
            _fast_destroy_tensor(index_t)
        if source_t != NULL:
            _fast_destroy_tensor(source_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dim, index_t, source_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>index_t)] if index_t != NULL else [])
            + ([('t', <uintptr_t>source_t)] if source_t != NULL else []),
        )
        self_t = NULL
        index_t = NULL
        source_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if index_t != NULL:
                _fast_destroy_tensor(index_t)
            if source_t != NULL:
                _fast_destroy_tensor(source_t)


def scatter_add_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        index_shape, index_stride,
        src_shape, src_stride,
        out_shape, out_stride,
        int64_t dim,
        int32_t self_dtype_code, int32_t index_dtype_code,
        int32_t src_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t index_ptr, uintptr_t src_ptr,
        uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int index_ndim = len(index_shape)
    cdef int src_ndim = len(src_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] i_shape, i_stride
    cdef int64_t[MAX_NDIM] src_shape_buf, src_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(index_ndim):
        i_shape[i] = index_shape[i]
        i_stride[i] = index_stride[i]
    for i in range(src_ndim):
        src_shape_buf[i] = src_shape[i]
        src_stride_buf[i] = src_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* index_t = NULL
    cdef void* src_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        index_t = _fast_create_tensor(i_shape, i_stride, <uint64_t>index_ndim, index_dtype_code, fmt, <void*>index_ptr)
        src_t = _fast_create_tensor(src_shape_buf, src_stride_buf, <uint64_t>src_ndim, src_dtype_code, fmt, <void*>src_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or index_t == NULL or src_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if index_t != NULL:
            _fast_destroy_tensor(index_t)
        if src_t != NULL:
            _fast_destroy_tensor(src_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dim, index_t, src_t, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>index_t)] if index_t != NULL else [])
            + ([('t', <uintptr_t>src_t)] if src_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        index_t = NULL
        src_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if index_t != NULL:
                _fast_destroy_tensor(index_t)
            if src_t != NULL:
                _fast_destroy_tensor(src_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)



def inplace_masked_scatter_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        mask_shape, mask_stride,
        source_shape, source_stride,
        int32_t self_dtype_code, int32_t mask_dtype_code, int32_t source_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t mask_ptr, uintptr_t source_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int mask_ndim = len(mask_shape)
    cdef int source_ndim = len(source_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] m_shape, m_stride
    cdef int64_t[MAX_NDIM] src_shape_buf, src_stride_buf
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(mask_ndim):
        m_shape[i] = mask_shape[i]
        m_stride[i] = mask_stride[i]
    for i in range(source_ndim):
        src_shape_buf[i] = source_shape[i]
        src_stride_buf[i] = source_stride[i]
    cdef void* self_t = NULL
    cdef void* mask_t = NULL
    cdef void* source_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        mask_t = _fast_create_tensor(m_shape, m_stride, <uint64_t>mask_ndim, mask_dtype_code, fmt, <void*>mask_ptr)
        source_t = _fast_create_tensor(src_shape_buf, src_stride_buf, <uint64_t>source_ndim, source_dtype_code, fmt, <void*>source_ptr)
    if self_t == NULL or mask_t == NULL or source_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if mask_t != NULL:
            _fast_destroy_tensor(mask_t)
        if source_t != NULL:
            _fast_destroy_tensor(source_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, mask_t, source_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(mask_t)
            _fast_destroy_tensor(source_t)



def index_add_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        index_shape, index_stride,
        source_shape, source_stride,
        out_shape, out_stride,
        int64_t dim,
        int32_t self_dtype_code, int32_t index_dtype_code,
        int32_t source_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t index_ptr, uintptr_t source_ptr,
        uintptr_t alpha_handle, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int index_ndim = len(index_shape)
    cdef int source_ndim = len(source_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] i_shape, i_stride
    cdef int64_t[MAX_NDIM] src_shape_buf, src_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(index_ndim):
        i_shape[i] = index_shape[i]
        i_stride[i] = index_stride[i]
    for i in range(source_ndim):
        src_shape_buf[i] = source_shape[i]
        src_stride_buf[i] = source_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* index_t = NULL
    cdef void* source_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        index_t = _fast_create_tensor(i_shape, i_stride, <uint64_t>index_ndim, index_dtype_code, fmt, <void*>index_ptr)
        source_t = _fast_create_tensor(src_shape_buf, src_stride_buf, <uint64_t>source_ndim, source_dtype_code, fmt, <void*>source_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or index_t == NULL or source_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if index_t != NULL:
            _fast_destroy_tensor(index_t)
        if source_t != NULL:
            _fast_destroy_tensor(source_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dim, index_t, source_t, <void*>alpha_handle, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>index_t)] if index_t != NULL else [])
            + ([('t', <uintptr_t>source_t)] if source_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        index_t = NULL
        source_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if index_t != NULL:
                _fast_destroy_tensor(index_t)
            if source_t != NULL:
                _fast_destroy_tensor(source_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)



def index_put_impl_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride, int32_t self_dtype_code,
        index_entries,
        values_shape, values_stride, int32_t values_dtype_code,
        bint accumulate, bint unsafe,
        uintptr_t self_ptr, uintptr_t values_ptr,
        int32_t fmt,
        uintptr_t stream,
        int64_t self_storage_offset=0,
        self_storage_dims=None,
        uintptr_t self_storage_ptr=0,
        int64_t values_storage_offset=0,
        values_storage_dims=None,
        uintptr_t values_storage_ptr=0):
    cdef int self_ndim = len(self_shape)
    cdef int values_ndim = len(values_shape)
    cdef int n = len(index_entries)
    cdef int self_storage_ndim = self_ndim if self_storage_dims is None else len(self_storage_dims)
    cdef int values_storage_ndim = values_ndim if values_storage_dims is None else len(values_storage_dims)
    cdef int64_t[MAX_NDIM] self_shape_buf, self_stride_buf, self_storage_dims_buf
    cdef int64_t[MAX_NDIM] values_shape_buf, values_stride_buf, values_storage_dims_buf
    cdef void* self_t = NULL
    cdef void* values_t = NULL
    cdef void* tensor_list = NULL
    cdef void* tensor_array_buf[64]
    cdef void* created_tensors[64]
    cdef uintptr_t ptr_buf[64]
    cdef uintptr_t storage_ptr_buf[64]
    cdef int64_t storage_offset_buf[64]
    cdef int32_t dtype_buf[64]
    cdef int ndim_buf[64]
    cdef int storage_ndim_buf[64]
    cdef int64_t shape_bufs[64][MAX_NDIM]
    cdef int64_t stride_bufs[64][MAX_NDIM]
    cdef int64_t storage_shape_bufs[64][MAX_NDIM]
    cdef int i, j, ndim
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    if n > 64:
        raise RuntimeError("index_put_impl_op supports at most 64 indices")
    if self_storage_ndim > MAX_NDIM or values_storage_ndim > MAX_NDIM:
        raise RuntimeError("storage ndim exceeds MAX_NDIM")
    for i in range(self_ndim):
        self_shape_buf[i] = self_shape[i]
        self_stride_buf[i] = self_stride[i]
    for i in range(self_storage_ndim):
        self_storage_dims_buf[i] = self_shape[i] if self_storage_dims is None else self_storage_dims[i]
    for i in range(values_ndim):
        values_shape_buf[i] = values_shape[i]
        values_stride_buf[i] = values_stride[i]
    for i in range(values_storage_ndim):
        values_storage_dims_buf[i] = values_shape[i] if values_storage_dims is None else values_storage_dims[i]
    if self_storage_ptr == 0:
        self_storage_ptr = self_ptr
    if values_storage_ptr == 0:
        values_storage_ptr = values_ptr
    for i in range(n):
        created_tensors[i] = NULL
        if index_entries[i] is None:
            tensor_array_buf[i] = NULL
            storage_ptr_buf[i] = 0
            storage_offset_buf[i] = 0
            storage_ndim_buf[i] = 0
            continue
        ptr_buf[i] = <uintptr_t>index_entries[i][0]
        ndim = len(index_entries[i][1])
        if ndim > MAX_NDIM:
            raise RuntimeError("index ndim exceeds MAX_NDIM")
        ndim_buf[i] = ndim
        dtype_buf[i] = _dtype_to_acl_runtime(index_entries[i][3])
        storage_offset_buf[i] = 0 if len(index_entries[i]) < 5 or index_entries[i][4] is None else <int64_t>index_entries[i][4]
        if len(index_entries[i]) >= 7 and index_entries[i][6] is not None:
            storage_ptr_buf[i] = <uintptr_t>index_entries[i][6]
        else:
            storage_ptr_buf[i] = ptr_buf[i]
        if len(index_entries[i]) >= 6 and index_entries[i][5] is not None:
            storage_ndim_buf[i] = len(index_entries[i][5])
            if storage_ndim_buf[i] > MAX_NDIM:
                raise RuntimeError("index storage ndim exceeds MAX_NDIM")
        else:
            storage_ndim_buf[i] = ndim
        for j in range(ndim):
            shape_bufs[i][j] = index_entries[i][1][j]
            stride_bufs[i][j] = index_entries[i][2][j]
        for j in range(storage_ndim_buf[i]):
            if len(index_entries[i]) >= 6 and index_entries[i][5] is not None:
                storage_shape_bufs[i][j] = index_entries[i][5][j]
            else:
                storage_shape_bufs[i][j] = index_entries[i][1][j]
    with nogil:
        self_t = _fast_create_tensor_ex(
            self_shape_buf, self_stride_buf, <uint64_t>self_ndim,
            self_dtype_code, fmt,
            self_storage_dims_buf, <uint64_t>self_storage_ndim,
            self_storage_offset, <void*>self_storage_ptr)
        values_t = _fast_create_tensor_ex(
            values_shape_buf, values_stride_buf, <uint64_t>values_ndim,
            values_dtype_code, fmt,
            values_storage_dims_buf, <uint64_t>values_storage_ndim,
            values_storage_offset, <void*>values_storage_ptr)
    if self_t == NULL or values_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if values_t != NULL:
            _fast_destroy_tensor(values_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        for i in range(n):
            if index_entries[i] is None:
                continue
            with nogil:
                created_tensors[i] = _fast_create_tensor_ex(
                    shape_bufs[i], stride_bufs[i], <uint64_t>ndim_buf[i], dtype_buf[i], fmt,
                    storage_shape_bufs[i], <uint64_t>storage_ndim_buf[i],
                    storage_offset_buf[i], <void*>storage_ptr_buf[i])
            if created_tensors[i] == NULL:
                raise RuntimeError("aclCreateTensor returned null")
            tensor_array_buf[i] = created_tensors[i]
        with nogil:
            tensor_list = _fn_create_tensor_list(tensor_array_buf, <uint64_t>n)
        if tensor_list == NULL:
            raise RuntimeError("aclCreateTensorList failed")
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, bint, bint, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, tensor_list, values_t, accumulate, unsafe, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if tensor_list != NULL:
                _fn_destroy_tensor_list(tensor_list)
            for i in range(n):
                if created_tensors[i] != NULL:
                    _fast_destroy_tensor(created_tensors[i])
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if values_t != NULL:
                _fast_destroy_tensor(values_t)


def index_with_optional_tensor_list_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride, int32_t self_dtype_code,
        index_entries,
        out_shape, out_stride, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream,
        int64_t self_storage_offset=0,
        self_storage_dims=None,
        uintptr_t self_storage_ptr=0):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int n = len(index_entries)
    cdef int self_storage_ndim = self_ndim if self_storage_dims is None else len(self_storage_dims)
    cdef int64_t[MAX_NDIM] self_shape_buf, self_stride_buf, self_storage_dims_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* tensor_list = NULL
    cdef void* tensor_array_buf[64]
    cdef void* created_tensors[64]
    cdef uintptr_t ptr_buf[64]
    cdef uintptr_t storage_ptr_buf[64]
    cdef int64_t storage_offset_buf[64]
    cdef int32_t dtype_buf[64]
    cdef int ndim_buf[64]
    cdef int storage_ndim_buf[64]
    cdef int64_t shape_bufs[64][MAX_NDIM]
    cdef int64_t stride_bufs[64][MAX_NDIM]
    cdef int64_t storage_shape_bufs[64][MAX_NDIM]
    cdef int i, j, ndim
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    if n > 64:
        raise RuntimeError("index_with_optional_tensor_list_op supports at most 64 entries")
    if self_storage_ndim > MAX_NDIM:
        raise RuntimeError("storage ndim exceeds MAX_NDIM")
    for i in range(self_ndim):
        self_shape_buf[i] = self_shape[i]
        self_stride_buf[i] = self_stride[i]
    for i in range(self_storage_ndim):
        self_storage_dims_buf[i] = self_shape[i] if self_storage_dims is None else self_storage_dims[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    if self_storage_ptr == 0:
        self_storage_ptr = self_ptr
    for i in range(n):
        created_tensors[i] = NULL
        if index_entries[i] is None:
            tensor_array_buf[i] = NULL
            storage_ptr_buf[i] = 0
            storage_offset_buf[i] = 0
            storage_ndim_buf[i] = 0
            continue
        ptr_buf[i] = <uintptr_t>index_entries[i][0]
        ndim = len(index_entries[i][1])
        if ndim > MAX_NDIM:
            raise RuntimeError("index ndim exceeds MAX_NDIM")
        ndim_buf[i] = ndim
        dtype_buf[i] = _dtype_to_acl_runtime(index_entries[i][3])
        storage_offset_buf[i] = 0 if len(index_entries[i]) < 5 or index_entries[i][4] is None else <int64_t>index_entries[i][4]
        if len(index_entries[i]) >= 7 and index_entries[i][6] is not None:
            storage_ptr_buf[i] = <uintptr_t>index_entries[i][6]
        else:
            storage_ptr_buf[i] = ptr_buf[i]
        if len(index_entries[i]) >= 6 and index_entries[i][5] is not None:
            storage_ndim_buf[i] = len(index_entries[i][5])
            if storage_ndim_buf[i] > MAX_NDIM:
                raise RuntimeError("index storage ndim exceeds MAX_NDIM")
        else:
            storage_ndim_buf[i] = ndim
        for j in range(ndim):
            shape_bufs[i][j] = index_entries[i][1][j]
            stride_bufs[i][j] = index_entries[i][2][j]
        for j in range(storage_ndim_buf[i]):
            if len(index_entries[i]) >= 6 and index_entries[i][5] is not None:
                storage_shape_bufs[i][j] = index_entries[i][5][j]
            else:
                storage_shape_bufs[i][j] = index_entries[i][1][j]
    with nogil:
        self_t = _fast_create_tensor_ex(
            self_shape_buf, self_stride_buf, <uint64_t>self_ndim,
            self_dtype_code, fmt,
            self_storage_dims_buf, <uint64_t>self_storage_ndim,
            self_storage_offset, <void*>self_storage_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        for i in range(n):
            if index_entries[i] is None:
                continue
            with nogil:
                created_tensors[i] = _fast_create_tensor_ex(
                    shape_bufs[i], stride_bufs[i], <uint64_t>ndim_buf[i], dtype_buf[i], fmt,
                    storage_shape_bufs[i], <uint64_t>storage_ndim_buf[i],
                    storage_offset_buf[i], <void*>storage_ptr_buf[i])
            if created_tensors[i] == NULL:
                raise RuntimeError("aclCreateTensor returned null")
            tensor_array_buf[i] = created_tensors[i]
        with nogil:
            tensor_list = _fn_create_tensor_list(tensor_array_buf, <uint64_t>n)
        if tensor_list == NULL:
            raise RuntimeError("aclCreateTensorList failed")
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, tensor_list, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if tensor_list != NULL:
                _fn_destroy_tensor_list(tensor_list)
            for i in range(n):
                if created_tensors[i] != NULL:
                    _fast_destroy_tensor(created_tensors[i])
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def tensor_list_axis_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        tensor_ptrs, shapes, strides, dtypes,
        int64_t dim,
        out_shape, out_stride,
        int32_t out_dtype_code, int32_t fmt,
        uintptr_t out_ptr,
        uintptr_t stream):
    cdef int n = len(tensor_ptrs)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef void* out_t = NULL
    cdef void* tensor_list = NULL
    cdef void* tensor_array_buf[64]
    cdef void* created_tensors[64]
    cdef uintptr_t ptr_buf[64]
    cdef int32_t dtype_buf[64]
    cdef int ndim_buf[64]
    cdef int64_t shape_bufs[64][MAX_NDIM]
    cdef int64_t stride_bufs[64][MAX_NDIM]
    cdef int i, j, ndim
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    if n > 64:
        raise RuntimeError("tensor_list_axis_op supports at most 64 tensors")
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    for i in range(n):
        created_tensors[i] = NULL
        ptr_buf[i] = <uintptr_t>tensor_ptrs[i]
        dtype_buf[i] = _dtype_to_acl_runtime(dtypes[i])
        ndim = len(shapes[i])
        ndim_buf[i] = ndim
        for j in range(ndim):
            shape_bufs[i][j] = shapes[i][j]
            stride_bufs[i][j] = strides[i][j]
    with nogil:
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if out_t == NULL:
        raise RuntimeError("aclCreateTensor returned null")
    try:
        for i in range(n):
            with nogil:
                created_tensors[i] = _fast_create_tensor(
                    shape_bufs[i], stride_bufs[i], <uint64_t>ndim_buf[i], dtype_buf[i], fmt, <void*>ptr_buf[i])
            if created_tensors[i] == NULL:
                raise RuntimeError("aclCreateTensor returned null")
            tensor_array_buf[i] = created_tensors[i]
        with nogil:
            tensor_list = _fn_create_tensor_list(tensor_array_buf, <uint64_t>n)
        if tensor_list == NULL:
            raise RuntimeError("aclCreateTensorList failed")
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                tensor_list, dim, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            [('l', <uintptr_t>tensor_list)]
            + [('t', <uintptr_t>created_tensors[i]) for i in range(n) if created_tensors[i] != NULL]
            + [('t', <uintptr_t>out_t)],
        )
        tensor_list = NULL
        for i in range(n):
            created_tensors[i] = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                # cat/stack fast-path must keep tensor-list cleanup deferred until a
                # later runtime synchronize; releasing it immediately corrupts ACLNN
                # state for subsequent ops on 910A.
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if tensor_list != NULL:
                _fn_destroy_tensor_list(tensor_list)
            for i in range(n):
                if created_tensors[i] != NULL:
                    _fast_destroy_tensor(created_tensors[i])
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


cdef inline int32_t _dtype_to_acl_runtime(object dtype_name):
    cdef object normalized = dtype_name
    if not isinstance(normalized, str):
        normalized = getattr(normalized, "name", None)
        if normalized is None:
            normalized = str(dtype_name)
    if normalized in ("float16", "torch.float16", "candle.float16"):
        return 1
    if normalized in ("float32", "torch.float32", "candle.float32"):
        return 0
    if normalized in ("float64", "torch.float64", "candle.float64"):
        return 11
    if normalized in ("int8", "torch.int8", "candle.int8"):
        return 2
    if normalized in ("uint8", "torch.uint8", "candle.uint8"):
        return 4
    if normalized in ("int16", "torch.int16", "candle.int16"):
        return 6
    if normalized in ("int32", "torch.int32", "candle.int32"):
        return 3
    if normalized in ("int64", "torch.int64", "candle.int64"):
        return 9
    if normalized in ("bool", "torch.bool", "candle.bool"):
        return 12
    raise RuntimeError(f"unsupported dtype for tensor list helper: {dtype_name}")


def tensor_list_string_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        tensor_ptrs, shapes, strides, dtypes,
        equation,
        out_shape, out_stride,
        int32_t out_dtype_code, int32_t fmt,
        uintptr_t out_ptr,
        uintptr_t stream):
    cdef int n = len(tensor_ptrs)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef void* out_t = NULL
    cdef void* tensor_list = NULL
    cdef void* tensor_array_buf[64]
    cdef void* created_tensors[64]
    cdef uintptr_t ptr_buf[64]
    cdef int32_t dtype_buf[64]
    cdef int ndim_buf[64]
    cdef int64_t shape_bufs[64][MAX_NDIM]
    cdef int64_t stride_bufs[64][MAX_NDIM]
    cdef int i, j, ndim
    cdef bytes eq_bytes = equation.encode("utf-8") + b"\x00"
    cdef const char* eq_ptr = eq_bytes
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    if n > 64:
        raise RuntimeError("tensor_list_string_op supports at most 64 tensors")
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    for i in range(n):
        created_tensors[i] = NULL
        ptr_buf[i] = <uintptr_t>tensor_ptrs[i]
        dtype_buf[i] = _dtype_to_acl_runtime(dtypes[i])
        ndim = len(shapes[i])
        ndim_buf[i] = ndim
        for j in range(ndim):
            shape_bufs[i][j] = shapes[i][j]
            stride_bufs[i][j] = strides[i][j]
    with nogil:
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if out_t == NULL:
        raise RuntimeError("aclCreateTensor returned null")
    try:
        for i in range(n):
            with nogil:
                created_tensors[i] = _fast_create_tensor(
                    shape_bufs[i], stride_bufs[i], <uint64_t>ndim_buf[i], dtype_buf[i], fmt, <void*>ptr_buf[i])
            if created_tensors[i] == NULL:
                raise RuntimeError("aclCreateTensor returned null")
            tensor_array_buf[i] = created_tensors[i]
        with nogil:
            tensor_list = _fn_create_tensor_list(tensor_array_buf, <uint64_t>n)
        if tensor_list == NULL:
            raise RuntimeError("aclCreateTensorList failed")
        with nogil:
            ret = (<int32_t (*)(void*, const char*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                tensor_list, eq_ptr, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if tensor_list != NULL:
                _fn_destroy_tensor_list(tensor_list)
            for i in range(n):
                if created_tensors[i] != NULL:
                    _fast_destroy_tensor(created_tensors[i])
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def two_tensor_two_bools_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        first_shape, first_stride,
        second_shape, second_stride,
        out_shape, out_stride,
        bint flag_a, bint flag_b,
        int32_t first_dtype_code, int32_t second_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t first_ptr, uintptr_t second_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int first_ndim = len(first_shape)
    cdef int second_ndim = len(second_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] first_shape_buf, first_stride_buf
    cdef int64_t[MAX_NDIM] second_shape_buf, second_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(first_ndim):
        first_shape_buf[i] = first_shape[i]
        first_stride_buf[i] = first_stride[i]
    for i in range(second_ndim):
        second_shape_buf[i] = second_shape[i]
        second_stride_buf[i] = second_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* first_t = NULL
    cdef void* second_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        first_t = _fast_create_tensor(first_shape_buf, first_stride_buf, <uint64_t>first_ndim, first_dtype_code, fmt, <void*>first_ptr)
        second_t = _fast_create_tensor(second_shape_buf, second_stride_buf, <uint64_t>second_ndim, second_dtype_code, fmt, <void*>second_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if first_t == NULL or second_t == NULL or out_t == NULL:
        if first_t != NULL:
            _fast_destroy_tensor(first_t)
        if second_t != NULL:
            _fast_destroy_tensor(second_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, bint, bint, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                first_t, second_t, flag_a, flag_b, NULL, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>first_t)] if first_t != NULL else [])
            + ([('t', <uintptr_t>second_t)] if second_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        first_t = NULL
        second_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if first_t != NULL:
                _fast_destroy_tensor(first_t)
            if second_t != NULL:
                _fast_destroy_tensor(second_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)



def unary_two_bools_two_outputs_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        inverse_shape, inverse_stride,
        bint flag_a, bint flag_b,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t inverse_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr, uintptr_t inverse_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int inverse_ndim = len(inverse_shape)
    cdef int64_t[MAX_NDIM] self_shape_buf, self_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int64_t[MAX_NDIM] inverse_shape_buf, inverse_stride_buf
    cdef int i
    for i in range(self_ndim):
        self_shape_buf[i] = self_shape[i]
        self_stride_buf[i] = self_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    for i in range(inverse_ndim):
        inverse_shape_buf[i] = inverse_shape[i]
        inverse_stride_buf[i] = inverse_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* inverse_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(self_shape_buf, self_stride_buf, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        inverse_t = _fast_create_tensor(inverse_shape_buf, inverse_stride_buf, <uint64_t>inverse_ndim, inverse_dtype_code, fmt, <void*>inverse_ptr)
    if self_t == NULL or out_t == NULL or inverse_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if inverse_t != NULL:
            _fast_destroy_tensor(inverse_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, bint, bint, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, flag_a, flag_b, out_t, inverse_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else [])
            + ([('t', <uintptr_t>inverse_t)] if inverse_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        inverse_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
            if inverse_t != NULL:
                _fast_destroy_tensor(inverse_t)
def output_tensor_three_scalars_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        out_shape, out_stride,
        int32_t out_dtype_code, int32_t fmt,
        uintptr_t out_ptr,
        uintptr_t scalar_a, uintptr_t scalar_b, uintptr_t scalar_c,
        uintptr_t stream):
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if out_t == NULL:
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                <void*>scalar_a, <void*>scalar_b, <void*>scalar_c, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            [('t', <uintptr_t>out_t)] if out_t != NULL else [],
        )
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
def output_tensor_two_ints_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        out_shape, out_stride,
        int64_t value_a, int64_t value_b,
        int32_t out_dtype_code, int32_t fmt,
        uintptr_t out_ptr,
        uintptr_t stream):
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if out_t == NULL:
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(int64_t, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                value_a, value_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            [('t', <uintptr_t>out_t)] if out_t != NULL else [],
        )
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
def output_tensor_three_ints_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        out_shape, out_stride,
        int64_t value_a, int64_t value_b, int64_t value_c,
        int32_t out_dtype_code, int32_t fmt,
        uintptr_t out_ptr,
        uintptr_t stream):
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if out_t == NULL:
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(int64_t, int64_t, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                value_a, value_b, value_c, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            [('t', <uintptr_t>out_t)] if out_t != NULL else [],
        )
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
def output_tensor_int_array_double_two_ints_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        out_shape, out_stride,
        dims_tuple, double scalar_value, int64_t value_a, int64_t value_b,
        int32_t out_dtype_code, int32_t fmt,
        uintptr_t out_ptr,
        uintptr_t stream):
    cdef int out_ndim = len(out_shape)
    cdef int dims_ndim = len(dims_tuple)
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int64_t[MAX_NDIM] dims_buf
    cdef int i
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    for i in range(dims_ndim):
        dims_buf[i] = dims_tuple[i]
    cdef void* out_t = NULL
    cdef void* dims_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if dims_ndim > 0:
            dims_handle = _fn_create_int_array(dims_buf, <uint64_t>dims_ndim)
    if out_t == NULL or (dims_ndim > 0 and dims_handle == NULL):
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if dims_handle != NULL:
            _fn_destroy_int_array(dims_handle)
        raise RuntimeError("ACLNN descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, double, int64_t, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                dims_handle, scalar_value, value_a, value_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dims_handle != NULL:
                _fn_destroy_int_array(dims_handle)
            _fast_destroy_tensor(out_t)


def tensor_int_array_bool_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        dims_tuple, bint flag,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int dims_ndim = len(dims_tuple)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t[MAX_NDIM] dims_buf
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    for i in range(dims_ndim):
        dims_buf[i] = dims_tuple[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* dims_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if dims_ndim > 0:
            dims_handle = _fn_create_int_array(dims_buf, <uint64_t>dims_ndim)
    if self_t == NULL or out_t == NULL or (dims_ndim > 0 and dims_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if dims_handle != NULL:
            _fn_destroy_int_array(dims_handle)
        raise RuntimeError("ACLNN descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, bint, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dims_handle, flag, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dims_handle != NULL:
                _fn_destroy_int_array(dims_handle)
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(out_t)


def tensor_two_ints_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int64_t value_a, int64_t value_b,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, value_a, value_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)



def tensor_three_ints_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int64_t value_a, int64_t value_b, int64_t value_c,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, int64_t, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, value_a, value_b, value_c, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(out_t)


def tensor_int_array_scalar_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        pad_values,
        out_shape, out_stride,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t scalar_handle,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int pad_ndim = len(pad_values)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t* pad_buf = NULL
    cdef int i
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* pad_arr = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    if pad_ndim > 0:
        pad_buf = <int64_t*>malloc(pad_ndim * sizeof(int64_t))
        if pad_buf == NULL:
            raise MemoryError("malloc failed for pad buffer")
        for i in range(pad_ndim):
            pad_buf[i] = pad_values[i]
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if pad_ndim > 0:
            pad_arr = _fn_create_int_array(pad_buf, <uint64_t>pad_ndim)
    if self_t == NULL or out_t == NULL or (pad_ndim > 0 and pad_arr == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if pad_arr != NULL:
            _fn_destroy_int_array(pad_arr)
        if pad_buf != NULL:
            free(pad_buf)
        if pad_ndim > 0 and pad_arr == NULL:
            raise RuntimeError("aclCreateIntArray returned null")
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, pad_arr, <void*>scalar_handle, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>pad_arr)] if pad_arr != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        pad_arr = NULL
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if pad_arr != NULL:
                _fn_destroy_int_array(pad_arr)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
        if pad_buf != NULL:
            free(pad_buf)


def tensor_int_array_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        array_values,
        out_shape, out_stride,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int array_ndim = len(array_values)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t* array_buf = NULL
    cdef int i
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* array_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    if array_ndim > 0:
        array_buf = <int64_t*>malloc(array_ndim * sizeof(int64_t))
        if array_buf == NULL:
            raise MemoryError("malloc failed for int array buffer")
        for i in range(array_ndim):
            array_buf[i] = array_values[i]
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if array_ndim > 0:
            array_handle = _fn_create_int_array(array_buf, <uint64_t>array_ndim)
    if self_t == NULL or out_t == NULL or (array_ndim > 0 and array_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if array_handle != NULL:
            _fn_destroy_int_array(array_handle)
        if array_buf != NULL:
            free(array_buf)
        if array_ndim > 0 and array_handle == NULL:
            raise RuntimeError("aclCreateIntArray returned null")
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, array_handle, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>array_handle)] if array_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        array_handle = NULL
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if array_handle != NULL:
                _fn_destroy_int_array(array_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
        if array_buf != NULL:
            free(array_buf)




def batch_norm_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        input_shape, input_stride,
        weight_shape, weight_stride,
        bias_shape, bias_stride,
        running_mean_shape, running_mean_stride,
        running_var_shape, running_var_stride,
        out_shape, out_stride,
        save_mean_shape, save_mean_stride,
        save_invstd_shape, save_invstd_stride,
        bint training, double momentum, double eps,
        int32_t tensor_dtype_code, int32_t stats_dtype_code,
        int32_t io_fmt, int32_t param_fmt, int32_t stats_fmt,
        uintptr_t input_ptr, uintptr_t weight_ptr, uintptr_t bias_ptr,
        uintptr_t running_mean_ptr, uintptr_t running_var_ptr,
        uintptr_t out_ptr, uintptr_t save_mean_ptr, uintptr_t save_invstd_ptr,
        uintptr_t stream):
    cdef int input_ndim = len(input_shape)
    cdef int weight_ndim = 0 if weight_shape is None else len(weight_shape)
    cdef int bias_ndim = 0 if bias_shape is None else len(bias_shape)
    cdef int running_mean_ndim = 0 if running_mean_shape is None else len(running_mean_shape)
    cdef int running_var_ndim = 0 if running_var_shape is None else len(running_var_shape)
    cdef int out_ndim = len(out_shape)
    cdef int save_mean_ndim = len(save_mean_shape)
    cdef int save_invstd_ndim = len(save_invstd_shape)
    cdef bint has_weight = weight_ptr != 0 and weight_shape is not None
    cdef bint has_bias = bias_ptr != 0 and bias_shape is not None
    cdef bint has_running_mean = running_mean_ptr != 0 and running_mean_shape is not None
    cdef bint has_running_var = running_var_ptr != 0 and running_var_shape is not None
    cdef int64_t[MAX_NDIM] input_shape_buf, input_stride_buf
    cdef int64_t[MAX_NDIM] weight_shape_buf, weight_stride_buf
    cdef int64_t[MAX_NDIM] bias_shape_buf, bias_stride_buf
    cdef int64_t[MAX_NDIM] running_mean_shape_buf, running_mean_stride_buf
    cdef int64_t[MAX_NDIM] running_var_shape_buf, running_var_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int64_t[MAX_NDIM] save_mean_shape_buf, save_mean_stride_buf
    cdef int64_t[MAX_NDIM] save_invstd_shape_buf, save_invstd_stride_buf
    cdef int i
    for i in range(input_ndim):
        input_shape_buf[i] = input_shape[i]
        input_stride_buf[i] = input_stride[i]
    if weight_shape is not None:
        for i in range(weight_ndim):
            weight_shape_buf[i] = weight_shape[i]
            weight_stride_buf[i] = weight_stride[i]
    if bias_shape is not None:
        for i in range(bias_ndim):
            bias_shape_buf[i] = bias_shape[i]
            bias_stride_buf[i] = bias_stride[i]
    if running_mean_shape is not None:
        for i in range(running_mean_ndim):
            running_mean_shape_buf[i] = running_mean_shape[i]
            running_mean_stride_buf[i] = running_mean_stride[i]
    if running_var_shape is not None:
        for i in range(running_var_ndim):
            running_var_shape_buf[i] = running_var_shape[i]
            running_var_stride_buf[i] = running_var_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    for i in range(save_mean_ndim):
        save_mean_shape_buf[i] = save_mean_shape[i]
        save_mean_stride_buf[i] = save_mean_stride[i]
    for i in range(save_invstd_ndim):
        save_invstd_shape_buf[i] = save_invstd_shape[i]
        save_invstd_stride_buf[i] = save_invstd_stride[i]
    cdef void* input_t = NULL
    cdef void* weight_t = NULL
    cdef void* bias_t = NULL
    cdef void* running_mean_t = NULL
    cdef void* running_var_t = NULL
    cdef void* out_t = NULL
    cdef void* save_mean_t = NULL
    cdef void* save_invstd_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        input_t = _fast_create_tensor(input_shape_buf, input_stride_buf, <uint64_t>input_ndim, tensor_dtype_code, io_fmt, <void*>input_ptr)
        if has_weight:
            weight_t = _fast_create_tensor(weight_shape_buf, weight_stride_buf, <uint64_t>weight_ndim, tensor_dtype_code, param_fmt, <void*>weight_ptr)
        if has_bias:
            bias_t = _fast_create_tensor(bias_shape_buf, bias_stride_buf, <uint64_t>bias_ndim, tensor_dtype_code, param_fmt, <void*>bias_ptr)
        if has_running_mean:
            running_mean_t = _fast_create_tensor(running_mean_shape_buf, running_mean_stride_buf, <uint64_t>running_mean_ndim, tensor_dtype_code, param_fmt, <void*>running_mean_ptr)
        if has_running_var:
            running_var_t = _fast_create_tensor(running_var_shape_buf, running_var_stride_buf, <uint64_t>running_var_ndim, tensor_dtype_code, param_fmt, <void*>running_var_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, tensor_dtype_code, io_fmt, <void*>out_ptr)
        save_mean_t = _fast_create_tensor(save_mean_shape_buf, save_mean_stride_buf, <uint64_t>save_mean_ndim, stats_dtype_code, stats_fmt, <void*>save_mean_ptr)
        save_invstd_t = _fast_create_tensor(save_invstd_shape_buf, save_invstd_stride_buf, <uint64_t>save_invstd_ndim, stats_dtype_code, stats_fmt, <void*>save_invstd_ptr)
    if input_t == NULL or out_t == NULL or save_mean_t == NULL or save_invstd_t == NULL or (has_weight and weight_t == NULL) or (has_bias and bias_t == NULL) or (has_running_mean and running_mean_t == NULL) or (has_running_var and running_var_t == NULL):
        if input_t != NULL:
            _fast_destroy_tensor(input_t)
        if weight_t != NULL:
            _fast_destroy_tensor(weight_t)
        if bias_t != NULL:
            _fast_destroy_tensor(bias_t)
        if running_mean_t != NULL:
            _fast_destroy_tensor(running_mean_t)
        if running_var_t != NULL:
            _fast_destroy_tensor(running_var_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if save_mean_t != NULL:
            _fast_destroy_tensor(save_mean_t)
        if save_invstd_t != NULL:
            _fast_destroy_tensor(save_invstd_t)
        raise RuntimeError("ACLNN batch_norm descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, bint, double, double, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                input_t,
                weight_t,
                bias_t,
                running_mean_t,
                running_var_t,
                training,
                momentum,
                eps,
                out_t,
                save_mean_t,
                save_invstd_t,
                &ws_size,
                &executor,
            )
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(input_t)
            if weight_t != NULL:
                _fast_destroy_tensor(weight_t)
            if bias_t != NULL:
                _fast_destroy_tensor(bias_t)
            if running_mean_t != NULL:
                _fast_destroy_tensor(running_mean_t)
            if running_var_t != NULL:
                _fast_destroy_tensor(running_var_t)
            _fast_destroy_tensor(out_t)
            _fast_destroy_tensor(save_mean_t)
            _fast_destroy_tensor(save_invstd_t)


def group_norm_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        input_shape, input_stride,
        weight_shape, weight_stride,
        bias_shape, bias_stride,
        out_shape, out_stride,
        mean_shape, mean_stride,
        rstd_shape, rstd_stride,
        int64_t batch_size, int64_t channels, int64_t spatial_size, int64_t group_count, double eps,
        int32_t tensor_dtype_code, int32_t stats_dtype_code,
        int32_t io_fmt, int32_t param_fmt, int32_t stats_fmt,
        uintptr_t input_ptr, uintptr_t weight_ptr, uintptr_t bias_ptr,
        uintptr_t out_ptr, uintptr_t mean_ptr, uintptr_t rstd_ptr,
        uintptr_t stream):
    cdef int input_ndim = len(input_shape)
    cdef int weight_ndim = 0 if weight_shape is None else len(weight_shape)
    cdef int bias_ndim = 0 if bias_shape is None else len(bias_shape)
    cdef int out_ndim = len(out_shape)
    cdef int mean_ndim = len(mean_shape)
    cdef int rstd_ndim = len(rstd_shape)
    cdef bint has_weight = weight_ptr != 0 and weight_shape is not None
    cdef bint has_bias = bias_ptr != 0 and bias_shape is not None
    cdef int64_t[MAX_NDIM] input_shape_buf, input_stride_buf
    cdef int64_t[MAX_NDIM] weight_shape_buf, weight_stride_buf
    cdef int64_t[MAX_NDIM] bias_shape_buf, bias_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int64_t[MAX_NDIM] mean_shape_buf, mean_stride_buf
    cdef int64_t[MAX_NDIM] rstd_shape_buf, rstd_stride_buf
    cdef int i
    for i in range(input_ndim):
        input_shape_buf[i] = input_shape[i]
        input_stride_buf[i] = input_stride[i]
    if weight_shape is not None:
        for i in range(weight_ndim):
            weight_shape_buf[i] = weight_shape[i]
            weight_stride_buf[i] = weight_stride[i]
    if bias_shape is not None:
        for i in range(bias_ndim):
            bias_shape_buf[i] = bias_shape[i]
            bias_stride_buf[i] = bias_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    for i in range(mean_ndim):
        mean_shape_buf[i] = mean_shape[i]
        mean_stride_buf[i] = mean_stride[i]
    for i in range(rstd_ndim):
        rstd_shape_buf[i] = rstd_shape[i]
        rstd_stride_buf[i] = rstd_stride[i]
    cdef void* input_t = NULL
    cdef void* weight_t = NULL
    cdef void* bias_t = NULL
    cdef void* out_t = NULL
    cdef void* mean_t = NULL
    cdef void* rstd_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        input_t = _fast_create_tensor(input_shape_buf, input_stride_buf, <uint64_t>input_ndim, tensor_dtype_code, io_fmt, <void*>input_ptr)
        if has_weight:
            weight_t = _fast_create_tensor(weight_shape_buf, weight_stride_buf, <uint64_t>weight_ndim, tensor_dtype_code, param_fmt, <void*>weight_ptr)
        if has_bias:
            bias_t = _fast_create_tensor(bias_shape_buf, bias_stride_buf, <uint64_t>bias_ndim, tensor_dtype_code, param_fmt, <void*>bias_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, tensor_dtype_code, io_fmt, <void*>out_ptr)
        mean_t = _fast_create_tensor(mean_shape_buf, mean_stride_buf, <uint64_t>mean_ndim, stats_dtype_code, stats_fmt, <void*>mean_ptr)
        rstd_t = _fast_create_tensor(rstd_shape_buf, rstd_stride_buf, <uint64_t>rstd_ndim, stats_dtype_code, stats_fmt, <void*>rstd_ptr)
    if input_t == NULL or out_t == NULL or mean_t == NULL or rstd_t == NULL or (has_weight and weight_t == NULL) or (has_bias and bias_t == NULL):
        if input_t != NULL:
            _fast_destroy_tensor(input_t)
        if weight_t != NULL:
            _fast_destroy_tensor(weight_t)
        if bias_t != NULL:
            _fast_destroy_tensor(bias_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if mean_t != NULL:
            _fast_destroy_tensor(mean_t)
        if rstd_t != NULL:
            _fast_destroy_tensor(rstd_t)
        raise RuntimeError("ACLNN group_norm descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, int64_t, int64_t, int64_t, int64_t, double, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                input_t,
                weight_t,
                bias_t,
                batch_size,
                channels,
                spatial_size,
                group_count,
                eps,
                out_t,
                mean_t,
                rstd_t,
                &ws_size,
                &executor,
            )
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(input_t)
            if weight_t != NULL:
                _fast_destroy_tensor(weight_t)
            if bias_t != NULL:
                _fast_destroy_tensor(bias_t)
            _fast_destroy_tensor(out_t)
            _fast_destroy_tensor(mean_t)
            _fast_destroy_tensor(rstd_t)


def convolution_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        input_shape, input_stride,
        weight_shape, weight_stride,
        bias_shape, bias_stride,
        out_shape, out_stride,
        stride_values, padding_values, dilation_values, output_padding_values,
        bint transposed, int64_t groups, int8_t cube_math_type,
        int32_t tensor_dtype_code, int32_t fmt,
        uintptr_t input_ptr, uintptr_t weight_ptr, uintptr_t bias_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int input_ndim = len(input_shape)
    cdef int weight_ndim = len(weight_shape)
    cdef int bias_ndim = 0 if bias_shape is None else len(bias_shape)
    cdef int out_ndim = len(out_shape)
    cdef int stride_ndim = len(stride_values)
    cdef int padding_ndim = len(padding_values)
    cdef int dilation_ndim = len(dilation_values)
    cdef int output_padding_ndim = len(output_padding_values)
    cdef bint has_bias = bias_ptr != 0 and bias_shape is not None
    cdef int64_t[MAX_NDIM] input_shape_buf, input_stride_buf
    cdef int64_t[MAX_NDIM] weight_shape_buf, weight_stride_buf
    cdef int64_t[MAX_NDIM] bias_shape_buf, bias_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int64_t[MAX_NDIM] stride_buf
    cdef int64_t[MAX_NDIM] padding_buf
    cdef int64_t[MAX_NDIM] dilation_buf
    cdef int64_t[MAX_NDIM] output_padding_buf
    cdef int i
    for i in range(input_ndim):
        input_shape_buf[i] = input_shape[i]
        input_stride_buf[i] = input_stride[i]
    for i in range(weight_ndim):
        weight_shape_buf[i] = weight_shape[i]
        weight_stride_buf[i] = weight_stride[i]
    if bias_shape is not None:
        for i in range(bias_ndim):
            bias_shape_buf[i] = bias_shape[i]
            bias_stride_buf[i] = bias_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    for i in range(stride_ndim):
        stride_buf[i] = stride_values[i]
    for i in range(padding_ndim):
        padding_buf[i] = padding_values[i]
    for i in range(dilation_ndim):
        dilation_buf[i] = dilation_values[i]
    for i in range(output_padding_ndim):
        output_padding_buf[i] = output_padding_values[i]
    cdef void* input_t = NULL
    cdef void* weight_t = NULL
    cdef void* bias_t = NULL
    cdef void* out_t = NULL
    cdef void* stride_handle = NULL
    cdef void* padding_handle = NULL
    cdef void* dilation_handle = NULL
    cdef void* output_padding_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        input_t = _fast_create_tensor(input_shape_buf, input_stride_buf, <uint64_t>input_ndim, tensor_dtype_code, fmt, <void*>input_ptr)
        weight_t = _fast_create_tensor(weight_shape_buf, weight_stride_buf, <uint64_t>weight_ndim, tensor_dtype_code, fmt, <void*>weight_ptr)
        if has_bias:
            bias_t = _fast_create_tensor(bias_shape_buf, bias_stride_buf, <uint64_t>bias_ndim, tensor_dtype_code, fmt, <void*>bias_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, tensor_dtype_code, fmt, <void*>out_ptr)
        if stride_ndim > 0:
            stride_handle = _fn_create_int_array(stride_buf, <uint64_t>stride_ndim)
        if padding_ndim > 0:
            padding_handle = _fn_create_int_array(padding_buf, <uint64_t>padding_ndim)
        if dilation_ndim > 0:
            dilation_handle = _fn_create_int_array(dilation_buf, <uint64_t>dilation_ndim)
        if output_padding_ndim > 0:
            output_padding_handle = _fn_create_int_array(output_padding_buf, <uint64_t>output_padding_ndim)
    if input_t == NULL or weight_t == NULL or out_t == NULL or (has_bias and bias_t == NULL) or (stride_ndim > 0 and stride_handle == NULL) or (padding_ndim > 0 and padding_handle == NULL) or (dilation_ndim > 0 and dilation_handle == NULL) or (output_padding_ndim > 0 and output_padding_handle == NULL):
        if input_t != NULL:
            _fast_destroy_tensor(input_t)
        if weight_t != NULL:
            _fast_destroy_tensor(weight_t)
        if bias_t != NULL:
            _fast_destroy_tensor(bias_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if stride_handle != NULL:
            _fn_destroy_int_array(stride_handle)
        if padding_handle != NULL:
            _fn_destroy_int_array(padding_handle)
        if dilation_handle != NULL:
            _fn_destroy_int_array(dilation_handle)
        if output_padding_handle != NULL:
            _fn_destroy_int_array(output_padding_handle)
        raise RuntimeError("ACLNN convolution descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, void*, bint, void*, int64_t, void*, int8_t, uint64_t*, void**) noexcept nogil>getws_ptr)(
                input_t,
                weight_t,
                bias_t,
                stride_handle,
                padding_handle,
                dilation_handle,
                transposed,
                output_padding_handle,
                groups,
                out_t,
                cube_math_type,
                &ws_size,
                &executor,
            )
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if stride_handle != NULL:
                _fn_destroy_int_array(stride_handle)
            if padding_handle != NULL:
                _fn_destroy_int_array(padding_handle)
            if dilation_handle != NULL:
                _fn_destroy_int_array(dilation_handle)
            if output_padding_handle != NULL:
                _fn_destroy_int_array(output_padding_handle)
            _fast_destroy_tensor(input_t)
            _fast_destroy_tensor(weight_t)
            if bias_t != NULL:
                _fast_destroy_tensor(bias_t)
            _fast_destroy_tensor(out_t)

def tensor_two_int_arrays_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        first_values, second_values,
        out_shape, out_stride,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int first_ndim = len(first_values)
    cdef int second_ndim = len(second_values)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t* first_buf = NULL
    cdef int64_t* second_buf = NULL
    cdef int i
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* first_handle = NULL
    cdef void* second_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    if first_ndim > 0:
        first_buf = <int64_t*>malloc(first_ndim * sizeof(int64_t))
        if first_buf == NULL:
            raise MemoryError("malloc failed for first int array buffer")
        for i in range(first_ndim):
            first_buf[i] = first_values[i]
    if second_ndim > 0:
        second_buf = <int64_t*>malloc(second_ndim * sizeof(int64_t))
        if second_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            raise MemoryError("malloc failed for second int array buffer")
        for i in range(second_ndim):
            second_buf[i] = second_values[i]
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if first_ndim > 0:
            first_handle = _fn_create_int_array(first_buf, <uint64_t>first_ndim)
        if second_ndim > 0:
            second_handle = _fn_create_int_array(second_buf, <uint64_t>second_ndim)
    if self_t == NULL or out_t == NULL or (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if first_handle != NULL:
            _fn_destroy_int_array(first_handle)
        if second_handle != NULL:
            _fn_destroy_int_array(second_handle)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL):
            raise RuntimeError("aclCreateIntArray returned null")
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, first_handle, second_handle, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>first_handle)] if first_handle != NULL else [])
            + ([('i', <uintptr_t>second_handle)] if second_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        first_handle = NULL
        second_handle = NULL
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if first_handle != NULL:
                _fn_destroy_int_array(first_handle)
            if second_handle != NULL:
                _fn_destroy_int_array(second_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
def tensor_four_int_arrays_two_ints_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        first_values, second_values, third_values, fourth_values,
        out_shape, out_stride,
        int64_t value_a, int64_t value_b,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int first_ndim = len(first_values)
    cdef int second_ndim = len(second_values)
    cdef int third_ndim = len(third_values)
    cdef int fourth_ndim = len(fourth_values)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t* first_buf = NULL
    cdef int64_t* second_buf = NULL
    cdef int64_t* third_buf = NULL
    cdef int64_t* fourth_buf = NULL
    cdef int i
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* first_handle = NULL
    cdef void* second_handle = NULL
    cdef void* third_handle = NULL
    cdef void* fourth_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    if first_ndim > 0:
        first_buf = <int64_t*>malloc(first_ndim * sizeof(int64_t))
        if first_buf == NULL:
            raise MemoryError("malloc failed for first int array buffer")
        for i in range(first_ndim):
            first_buf[i] = first_values[i]
    if second_ndim > 0:
        second_buf = <int64_t*>malloc(second_ndim * sizeof(int64_t))
        if second_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            raise MemoryError("malloc failed for second int array buffer")
        for i in range(second_ndim):
            second_buf[i] = second_values[i]
    if third_ndim > 0:
        third_buf = <int64_t*>malloc(third_ndim * sizeof(int64_t))
        if third_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            if second_buf != NULL:
                free(second_buf)
            raise MemoryError("malloc failed for third int array buffer")
        for i in range(third_ndim):
            third_buf[i] = third_values[i]
    if fourth_ndim > 0:
        fourth_buf = <int64_t*>malloc(fourth_ndim * sizeof(int64_t))
        if fourth_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            if second_buf != NULL:
                free(second_buf)
            if third_buf != NULL:
                free(third_buf)
            raise MemoryError("malloc failed for fourth int array buffer")
        for i in range(fourth_ndim):
            fourth_buf[i] = fourth_values[i]
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if first_ndim > 0:
            first_handle = _fn_create_int_array(first_buf, <uint64_t>first_ndim)
        if second_ndim > 0:
            second_handle = _fn_create_int_array(second_buf, <uint64_t>second_ndim)
        if third_ndim > 0:
            third_handle = _fn_create_int_array(third_buf, <uint64_t>third_ndim)
        if fourth_ndim > 0:
            fourth_handle = _fn_create_int_array(fourth_buf, <uint64_t>fourth_ndim)
    if self_t == NULL or out_t == NULL or (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL) or (third_ndim > 0 and third_handle == NULL) or (fourth_ndim > 0 and fourth_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if first_handle != NULL:
            _fn_destroy_int_array(first_handle)
        if second_handle != NULL:
            _fn_destroy_int_array(second_handle)
        if third_handle != NULL:
            _fn_destroy_int_array(third_handle)
        if fourth_handle != NULL:
            _fn_destroy_int_array(fourth_handle)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if third_buf != NULL:
            free(third_buf)
        if fourth_buf != NULL:
            free(fourth_buf)
        if (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL) or (third_ndim > 0 and third_handle == NULL) or (fourth_ndim > 0 and fourth_handle == NULL):
            raise RuntimeError("aclCreateIntArray returned null")
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, int64_t, void*, void*, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, first_handle, second_handle, value_a, third_handle, fourth_handle, value_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>first_handle)] if first_handle != NULL else [])
            + ([('i', <uintptr_t>second_handle)] if second_handle != NULL else [])
            + ([('i', <uintptr_t>third_handle)] if third_handle != NULL else [])
            + ([('i', <uintptr_t>fourth_handle)] if fourth_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        first_handle = NULL
        second_handle = NULL
        third_handle = NULL
        fourth_handle = NULL
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if first_handle != NULL:
                _fn_destroy_int_array(first_handle)
            if second_handle != NULL:
                _fn_destroy_int_array(second_handle)
            if third_handle != NULL:
                _fn_destroy_int_array(third_handle)
            if fourth_handle != NULL:
                _fn_destroy_int_array(fourth_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if third_buf != NULL:
            free(third_buf)
        if fourth_buf != NULL:
            free(fourth_buf)
def tensor_three_int_arrays_two_bools_int64_int8_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        first_values, second_values, third_values,
        out_shape, out_stride,
        bint flag_a, bint flag_b, int64_t value_a, int8_t value_b,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int first_ndim = len(first_values)
    cdef int second_ndim = len(second_values)
    cdef int third_ndim = len(third_values)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t* first_buf = NULL
    cdef int64_t* second_buf = NULL
    cdef int64_t* third_buf = NULL
    cdef int i
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* first_handle = NULL
    cdef void* second_handle = NULL
    cdef void* third_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    if first_ndim > 0:
        first_buf = <int64_t*>malloc(first_ndim * sizeof(int64_t))
        if first_buf == NULL:
            raise MemoryError("malloc failed for first int array buffer")
        for i in range(first_ndim):
            first_buf[i] = first_values[i]
    if second_ndim > 0:
        second_buf = <int64_t*>malloc(second_ndim * sizeof(int64_t))
        if second_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            raise MemoryError("malloc failed for second int array buffer")
        for i in range(second_ndim):
            second_buf[i] = second_values[i]
    if third_ndim > 0:
        third_buf = <int64_t*>malloc(third_ndim * sizeof(int64_t))
        if third_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            if second_buf != NULL:
                free(second_buf)
            raise MemoryError("malloc failed for third int array buffer")
        for i in range(third_ndim):
            third_buf[i] = third_values[i]
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if first_ndim > 0:
            first_handle = _fn_create_int_array(first_buf, <uint64_t>first_ndim)
        if second_ndim > 0:
            second_handle = _fn_create_int_array(second_buf, <uint64_t>second_ndim)
        if third_ndim > 0:
            third_handle = _fn_create_int_array(third_buf, <uint64_t>third_ndim)
    if self_t == NULL or out_t == NULL or (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL) or (third_ndim > 0 and third_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if first_handle != NULL:
            _fn_destroy_int_array(first_handle)
        if second_handle != NULL:
            _fn_destroy_int_array(second_handle)
        if third_handle != NULL:
            _fn_destroy_int_array(third_handle)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if third_buf != NULL:
            free(third_buf)
        if (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL) or (third_ndim > 0 and third_handle == NULL):
            raise RuntimeError("aclCreateIntArray returned null")
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, bint, bint, int64_t, int8_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, first_handle, second_handle, third_handle, flag_a, flag_b, value_a, value_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>first_handle)] if first_handle != NULL else [])
            + ([('i', <uintptr_t>second_handle)] if second_handle != NULL else [])
            + ([('i', <uintptr_t>third_handle)] if third_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        first_handle = NULL
        second_handle = NULL
        third_handle = NULL
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if first_handle != NULL:
                _fn_destroy_int_array(first_handle)
            if second_handle != NULL:
                _fn_destroy_int_array(second_handle)
            if third_handle != NULL:
                _fn_destroy_int_array(third_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if third_buf != NULL:
            free(third_buf)
def tensor_four_int_arrays_bool_two_outputs_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        first_values, second_values, third_values, fourth_values,
        out_a_shape, out_a_stride,
        out_b_shape, out_b_stride,
        bint flag,
        int32_t self_dtype_code, int32_t out_a_dtype_code, int32_t out_b_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_a_ptr, uintptr_t out_b_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_a_ndim = len(out_a_shape)
    cdef int out_b_ndim = len(out_b_shape)
    cdef int first_ndim = len(first_values)
    cdef int second_ndim = len(second_values)
    cdef int third_ndim = len(third_values)
    cdef int fourth_ndim = len(fourth_values)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] out_a_shape_buf, out_a_stride_buf
    cdef int64_t[MAX_NDIM] out_b_shape_buf, out_b_stride_buf
    cdef int64_t* first_buf = NULL
    cdef int64_t* second_buf = NULL
    cdef int64_t* third_buf = NULL
    cdef int64_t* fourth_buf = NULL
    cdef int i
    cdef void* self_t = NULL
    cdef void* out_a_t = NULL
    cdef void* out_b_t = NULL
    cdef void* first_handle = NULL
    cdef void* second_handle = NULL
    cdef void* third_handle = NULL
    cdef void* fourth_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_a_ndim):
        out_a_shape_buf[i] = out_a_shape[i]
        out_a_stride_buf[i] = out_a_stride[i]
    for i in range(out_b_ndim):
        out_b_shape_buf[i] = out_b_shape[i]
        out_b_stride_buf[i] = out_b_stride[i]
    if first_ndim > 0:
        first_buf = <int64_t*>malloc(first_ndim * sizeof(int64_t))
        if first_buf == NULL:
            raise MemoryError("malloc failed for first int array buffer")
        for i in range(first_ndim):
            first_buf[i] = first_values[i]
    if second_ndim > 0:
        second_buf = <int64_t*>malloc(second_ndim * sizeof(int64_t))
        if second_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            raise MemoryError("malloc failed for second int array buffer")
        for i in range(second_ndim):
            second_buf[i] = second_values[i]
    if third_ndim > 0:
        third_buf = <int64_t*>malloc(third_ndim * sizeof(int64_t))
        if third_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            if second_buf != NULL:
                free(second_buf)
            raise MemoryError("malloc failed for third int array buffer")
        for i in range(third_ndim):
            third_buf[i] = third_values[i]
    if fourth_ndim > 0:
        fourth_buf = <int64_t*>malloc(fourth_ndim * sizeof(int64_t))
        if fourth_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            if second_buf != NULL:
                free(second_buf)
            if third_buf != NULL:
                free(third_buf)
            raise MemoryError("malloc failed for fourth int array buffer")
        for i in range(fourth_ndim):
            fourth_buf[i] = fourth_values[i]
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_a_t = _fast_create_tensor(out_a_shape_buf, out_a_stride_buf, <uint64_t>out_a_ndim, out_a_dtype_code, fmt, <void*>out_a_ptr)
        out_b_t = _fast_create_tensor(out_b_shape_buf, out_b_stride_buf, <uint64_t>out_b_ndim, out_b_dtype_code, fmt, <void*>out_b_ptr)
        if first_ndim > 0:
            first_handle = _fn_create_int_array(first_buf, <uint64_t>first_ndim)
        if second_ndim > 0:
            second_handle = _fn_create_int_array(second_buf, <uint64_t>second_ndim)
        if third_ndim > 0:
            third_handle = _fn_create_int_array(third_buf, <uint64_t>third_ndim)
        if fourth_ndim > 0:
            fourth_handle = _fn_create_int_array(fourth_buf, <uint64_t>fourth_ndim)
    if self_t == NULL or out_a_t == NULL or out_b_t == NULL or (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL) or (third_ndim > 0 and third_handle == NULL) or (fourth_ndim > 0 and fourth_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_a_t != NULL:
            _fast_destroy_tensor(out_a_t)
        if out_b_t != NULL:
            _fast_destroy_tensor(out_b_t)
        if first_handle != NULL:
            _fn_destroy_int_array(first_handle)
        if second_handle != NULL:
            _fn_destroy_int_array(second_handle)
        if third_handle != NULL:
            _fn_destroy_int_array(third_handle)
        if fourth_handle != NULL:
            _fn_destroy_int_array(fourth_handle)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if third_buf != NULL:
            free(third_buf)
        if fourth_buf != NULL:
            free(fourth_buf)
        if (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL) or (third_ndim > 0 and third_handle == NULL) or (fourth_ndim > 0 and fourth_handle == NULL):
            raise RuntimeError("aclCreateIntArray returned null")
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, bint, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, first_handle, second_handle, third_handle, fourth_handle, flag, out_a_t, out_b_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>first_handle)] if first_handle != NULL else [])
            + ([('i', <uintptr_t>second_handle)] if second_handle != NULL else [])
            + ([('i', <uintptr_t>third_handle)] if third_handle != NULL else [])
            + ([('i', <uintptr_t>fourth_handle)] if fourth_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_a_t)] if out_a_t != NULL else [])
            + ([('t', <uintptr_t>out_b_t)] if out_b_t != NULL else []),
        )
        first_handle = NULL
        second_handle = NULL
        third_handle = NULL
        fourth_handle = NULL
        self_t = NULL
        out_a_t = NULL
        out_b_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if first_handle != NULL:
                _fn_destroy_int_array(first_handle)
            if second_handle != NULL:
                _fn_destroy_int_array(second_handle)
            if third_handle != NULL:
                _fn_destroy_int_array(third_handle)
            if fourth_handle != NULL:
                _fn_destroy_int_array(fourth_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_a_t != NULL:
                _fast_destroy_tensor(out_a_t)
            if out_b_t != NULL:
                _fast_destroy_tensor(out_b_t)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if third_buf != NULL:
            free(third_buf)
        if fourth_buf != NULL:
            free(fourth_buf)
def optional_tensor_int_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        optional_shape, optional_stride,
        out_shape, out_stride,
        int64_t scalar_value,
        int32_t self_dtype_code, int32_t optional_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t optional_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int optional_ndim = 0 if optional_shape is None else len(optional_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] o_shape, o_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    if optional_shape is not None:
        for i in range(optional_ndim):
            o_shape[i] = optional_shape[i]
            o_stride[i] = optional_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* optional_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if optional_shape is not None and optional_ptr != 0:
            optional_t = _fast_create_tensor(o_shape, o_stride, <uint64_t>optional_ndim, optional_dtype_code, fmt, <void*>optional_ptr)
    if self_t == NULL or out_t == NULL or (optional_shape is not None and optional_ptr != 0 and optional_t == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if optional_t != NULL:
            _fast_destroy_tensor(optional_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, optional_t, scalar_value, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>optional_t)] if optional_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        optional_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if optional_t != NULL:
                _fast_destroy_tensor(optional_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def ternary_two_inputs_with_dims_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        index_shape, index_stride,
        src_shape, src_stride,
        out_shape, out_stride,
        int64_t dim, int64_t reduce,
        int32_t self_dtype_code, int32_t index_dtype_code, int32_t src_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t index_ptr, uintptr_t src_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int index_ndim = len(index_shape)
    cdef int src_ndim = len(src_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] i_shape, i_stride
    cdef int64_t[MAX_NDIM] src_shape_buf, src_stride_buf
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(index_ndim):
        i_shape[i] = index_shape[i]
        i_stride[i] = index_stride[i]
    for i in range(src_ndim):
        src_shape_buf[i] = src_shape[i]
        src_stride_buf[i] = src_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* index_t = NULL
    cdef void* src_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        index_t = _fast_create_tensor(i_shape, i_stride, <uint64_t>index_ndim, index_dtype_code, fmt, <void*>index_ptr)
        src_t = _fast_create_tensor(src_shape_buf, src_stride_buf, <uint64_t>src_ndim, src_dtype_code, fmt, <void*>src_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or index_t == NULL or src_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if index_t != NULL:
            _fast_destroy_tensor(index_t)
        if src_t != NULL:
            _fast_destroy_tensor(src_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, void*, void*, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dim, index_t, src_t, reduce, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>index_t)] if index_t != NULL else [])
            + ([('t', <uintptr_t>src_t)] if src_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        index_t = NULL
        src_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if index_t != NULL:
                _fast_destroy_tensor(index_t)
            if src_t != NULL:
                _fast_destroy_tensor(src_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def slice_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int64_t dim, int64_t start, int64_t end, int64_t step,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, int64_t, int64_t, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dim, start, end, step, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def three_tensor_one_int_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        q_shape, q_stride,
        r_shape, r_stride,
        int64_t value,
        int32_t self_dtype_code, int32_t q_dtype_code, int32_t r_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t q_ptr, uintptr_t r_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int q_ndim = len(q_shape)
    cdef int r_ndim = len(r_shape)
    cdef int64_t[MAX_NDIM] self_shape_buf, self_stride_buf
    cdef int64_t[MAX_NDIM] q_shape_buf, q_stride_buf
    cdef int64_t[MAX_NDIM] r_shape_buf, r_stride_buf
    cdef int i
    for i in range(self_ndim):
        self_shape_buf[i] = self_shape[i]
        self_stride_buf[i] = self_stride[i]
    for i in range(q_ndim):
        q_shape_buf[i] = q_shape[i]
        q_stride_buf[i] = q_stride[i]
    for i in range(r_ndim):
        r_shape_buf[i] = r_shape[i]
        r_stride_buf[i] = r_stride[i]
    cdef void* self_t = NULL
    cdef void* q_t = NULL
    cdef void* r_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(self_shape_buf, self_stride_buf, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        q_t = _fast_create_tensor(q_shape_buf, q_stride_buf, <uint64_t>q_ndim, q_dtype_code, fmt, <void*>q_ptr)
        r_t = _fast_create_tensor(r_shape_buf, r_stride_buf, <uint64_t>r_ndim, r_dtype_code, fmt, <void*>r_ptr)
    if self_t == NULL or q_t == NULL or r_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if q_t != NULL:
            _fast_destroy_tensor(q_t)
        if r_t != NULL:
            _fast_destroy_tensor(r_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, value, q_t, r_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(q_t)
            _fast_destroy_tensor(r_t)



def four_tensor_two_ints_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        b_shape, b_stride,
        c_shape, c_stride,
        out_shape, out_stride,
        int32_t value_a, int64_t value_b,
        int32_t self_dtype_code, int32_t b_dtype_code, int32_t c_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t b_ptr, uintptr_t c_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int b_ndim = len(b_shape)
    cdef int c_ndim = len(c_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] self_shape_buf, self_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] c_shape_buf, c_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(self_ndim):
        self_shape_buf[i] = self_shape[i]
        self_stride_buf[i] = self_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(c_ndim):
        c_shape_buf[i] = c_shape[i]
        c_stride_buf[i] = c_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* b_t = NULL
    cdef void* c_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(self_shape_buf, self_stride_buf, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        c_t = _fast_create_tensor(c_shape_buf, c_stride_buf, <uint64_t>c_ndim, c_dtype_code, fmt, <void*>c_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or b_t == NULL or c_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if b_t != NULL:
            _fast_destroy_tensor(b_t)
        if c_t != NULL:
            _fast_destroy_tensor(c_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int32_t, void*, void*, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, value_a, b_t, c_t, value_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(b_t)
            _fast_destroy_tensor(c_t)
            _fast_destroy_tensor(out_t)



def three_tensor_two_ints_bool_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        values_shape, values_stride,
        indices_shape, indices_stride,
        int64_t value_a, int64_t value_b, bint flag,
        int32_t self_dtype_code, int32_t values_dtype_code, int32_t indices_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t values_ptr, uintptr_t indices_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int values_ndim = len(values_shape)
    cdef int indices_ndim = len(indices_shape)
    cdef int64_t[MAX_NDIM] self_shape_buf, self_stride_buf
    cdef int64_t[MAX_NDIM] values_shape_buf, values_stride_buf
    cdef int64_t[MAX_NDIM] indices_shape_buf, indices_stride_buf
    cdef int i
    for i in range(self_ndim):
        self_shape_buf[i] = self_shape[i]
        self_stride_buf[i] = self_stride[i]
    for i in range(values_ndim):
        values_shape_buf[i] = values_shape[i]
        values_stride_buf[i] = values_stride[i]
    for i in range(indices_ndim):
        indices_shape_buf[i] = indices_shape[i]
        indices_stride_buf[i] = indices_stride[i]
    cdef void* self_t = NULL
    cdef void* values_t = NULL
    cdef void* indices_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(self_shape_buf, self_stride_buf, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        values_t = _fast_create_tensor(values_shape_buf, values_stride_buf, <uint64_t>values_ndim, values_dtype_code, fmt, <void*>values_ptr)
        indices_t = _fast_create_tensor(indices_shape_buf, indices_stride_buf, <uint64_t>indices_ndim, indices_dtype_code, fmt, <void*>indices_ptr)
    if self_t == NULL or values_t == NULL or indices_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if values_t != NULL:
            _fast_destroy_tensor(values_t)
        if indices_t != NULL:
            _fast_destroy_tensor(indices_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, int64_t, bint, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, value_a, value_b, flag, values_t, indices_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>values_t)] if values_t != NULL else [])
            + ([('t', <uintptr_t>indices_t)] if indices_t != NULL else []),
        )
        self_t = NULL
        values_t = NULL
        indices_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if values_t != NULL:
                _fast_destroy_tensor(values_t)
            if indices_t != NULL:
                _fast_destroy_tensor(indices_t)



def four_tensor_two_scalars_one_int8_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        c_shape, c_stride,
        out_shape, out_stride,
        int8_t cube_math_type,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t c_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t c_ptr, uintptr_t out_ptr,
        uintptr_t scalar_a, uintptr_t scalar_b,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int c_ndim = len(c_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] c_shape_buf, c_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(c_ndim):
        c_shape_buf[i] = c_shape[i]
        c_stride_buf[i] = c_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* c_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        c_t = _fast_create_tensor(c_shape_buf, c_stride_buf, <uint64_t>c_ndim, c_dtype_code, fmt, <void*>c_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if a_t == NULL or b_t == NULL or c_t == NULL or out_t == NULL:
        if a_t != NULL:
            _fast_destroy_tensor(a_t)
        if b_t != NULL:
            _fast_destroy_tensor(b_t)
        if c_t != NULL:
            _fast_destroy_tensor(c_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, void*, int8_t, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, b_t, c_t, <void*>scalar_a, <void*>scalar_b, out_t, cube_math_type, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(a_t)
            _fast_destroy_tensor(b_t)
            _fast_destroy_tensor(c_t)
            _fast_destroy_tensor(out_t)



def tensor_int_array_two_outputs_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_a_shape, out_a_stride,
        out_b_shape, out_b_stride,
        dims_tuple,
        int32_t self_dtype_code, int32_t out_a_dtype_code, int32_t out_b_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_a_ptr, uintptr_t out_b_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_a_ndim = len(out_a_shape)
    cdef int out_b_ndim = len(out_b_shape)
    cdef int dims_ndim = len(dims_tuple)
    cdef int64_t[MAX_NDIM] self_shape_buf, self_stride_buf
    cdef int64_t[MAX_NDIM] out_a_shape_buf, out_a_stride_buf
    cdef int64_t[MAX_NDIM] out_b_shape_buf, out_b_stride_buf
    cdef int64_t[MAX_NDIM] dims_buf
    cdef int i
    for i in range(self_ndim):
        self_shape_buf[i] = self_shape[i]
        self_stride_buf[i] = self_stride[i]
    for i in range(out_a_ndim):
        out_a_shape_buf[i] = out_a_shape[i]
        out_a_stride_buf[i] = out_a_stride[i]
    for i in range(out_b_ndim):
        out_b_shape_buf[i] = out_b_shape[i]
        out_b_stride_buf[i] = out_b_stride[i]
    for i in range(dims_ndim):
        dims_buf[i] = dims_tuple[i]
    cdef void* self_t = NULL
    cdef void* out_a_t = NULL
    cdef void* out_b_t = NULL
    cdef void* dims_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(self_shape_buf, self_stride_buf, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_a_t = _fast_create_tensor(out_a_shape_buf, out_a_stride_buf, <uint64_t>out_a_ndim, out_a_dtype_code, fmt, <void*>out_a_ptr)
        out_b_t = _fast_create_tensor(out_b_shape_buf, out_b_stride_buf, <uint64_t>out_b_ndim, out_b_dtype_code, fmt, <void*>out_b_ptr)
        if dims_ndim > 0:
            dims_handle = _fn_create_int_array(dims_buf, <uint64_t>dims_ndim)
    if self_t == NULL or out_a_t == NULL or out_b_t == NULL or (dims_ndim > 0 and dims_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_a_t != NULL:
            _fast_destroy_tensor(out_a_t)
        if out_b_t != NULL:
            _fast_destroy_tensor(out_b_t)
        if dims_handle != NULL:
            _fn_destroy_int_array(dims_handle)
        raise RuntimeError("ACLNN descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dims_handle, out_a_t, out_b_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>dims_handle)] if dims_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_a_t)] if out_a_t != NULL else [])
            + ([('t', <uintptr_t>out_b_t)] if out_b_t != NULL else []),
        )
        dims_handle = NULL
        self_t = NULL
        out_a_t = NULL
        out_b_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dims_handle != NULL:
                _fn_destroy_int_array(dims_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_a_t != NULL:
                _fast_destroy_tensor(out_a_t)
            if out_b_t != NULL:
                _fast_destroy_tensor(out_b_t)
def tensor_int_array_bool_two_outputs_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_a_shape, out_a_stride,
        out_b_shape, out_b_stride,
        dims_tuple, bint keepdim,
        int32_t self_dtype_code, int32_t out_a_dtype_code, int32_t out_b_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_a_ptr, uintptr_t out_b_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_a_ndim = len(out_a_shape)
    cdef int out_b_ndim = len(out_b_shape)
    cdef int dims_ndim = len(dims_tuple)
    cdef int64_t[MAX_NDIM] self_shape_buf, self_stride_buf
    cdef int64_t[MAX_NDIM] out_a_shape_buf, out_a_stride_buf
    cdef int64_t[MAX_NDIM] out_b_shape_buf, out_b_stride_buf
    cdef int64_t[MAX_NDIM] dims_buf
    cdef int i
    for i in range(self_ndim):
        self_shape_buf[i] = self_shape[i]
        self_stride_buf[i] = self_stride[i]
    for i in range(out_a_ndim):
        out_a_shape_buf[i] = out_a_shape[i]
        out_a_stride_buf[i] = out_a_stride[i]
    for i in range(out_b_ndim):
        out_b_shape_buf[i] = out_b_shape[i]
        out_b_stride_buf[i] = out_b_stride[i]
    for i in range(dims_ndim):
        dims_buf[i] = dims_tuple[i]
    cdef void* self_t = NULL
    cdef void* out_a_t = NULL
    cdef void* out_b_t = NULL
    cdef void* dims_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(self_shape_buf, self_stride_buf, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_a_t = _fast_create_tensor(out_a_shape_buf, out_a_stride_buf, <uint64_t>out_a_ndim, out_a_dtype_code, fmt, <void*>out_a_ptr)
        out_b_t = _fast_create_tensor(out_b_shape_buf, out_b_stride_buf, <uint64_t>out_b_ndim, out_b_dtype_code, fmt, <void*>out_b_ptr)
        if dims_ndim > 0:
            dims_handle = _fn_create_int_array(dims_buf, <uint64_t>dims_ndim)
    if self_t == NULL or out_a_t == NULL or out_b_t == NULL or (dims_ndim > 0 and dims_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_a_t != NULL:
            _fast_destroy_tensor(out_a_t)
        if out_b_t != NULL:
            _fast_destroy_tensor(out_b_t)
        if dims_handle != NULL:
            _fn_destroy_int_array(dims_handle)
        raise RuntimeError("ACLNN descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, bint, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dims_handle, keepdim, out_a_t, out_b_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>dims_handle)] if dims_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_a_t)] if out_a_t != NULL else [])
            + ([('t', <uintptr_t>out_b_t)] if out_b_t != NULL else []),
        )
        dims_handle = NULL
        self_t = NULL
        out_a_t = NULL
        out_b_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dims_handle != NULL:
                _fn_destroy_int_array(dims_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_a_t != NULL:
                _fast_destroy_tensor(out_a_t)
            if out_b_t != NULL:
                _fast_destroy_tensor(out_b_t)
def four_tensor_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        c_shape, c_stride,
        out_shape, out_stride,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t c_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t c_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int c_ndim = len(c_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] c_shape_buf, c_stride_buf
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(c_ndim):
        c_shape_buf[i] = c_shape[i]
        c_stride_buf[i] = c_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* c_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        c_t = _fast_create_tensor(c_shape_buf, c_stride_buf, <uint64_t>c_ndim, c_dtype_code, fmt, <void*>c_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if a_t == NULL or b_t == NULL or c_t == NULL or out_t == NULL:
        if a_t != NULL:
            _fast_destroy_tensor(a_t)
        if b_t != NULL:
            _fast_destroy_tensor(b_t)
        if c_t != NULL:
            _fast_destroy_tensor(c_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, b_t, c_t, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>a_t)] if a_t != NULL else [])
            + ([('t', <uintptr_t>b_t)] if b_t != NULL else [])
            + ([('t', <uintptr_t>c_t)] if c_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        a_t = NULL
        b_t = NULL
        c_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if a_t != NULL:
                _fast_destroy_tensor(a_t)
            if b_t != NULL:
                _fast_destroy_tensor(b_t)
            if c_t != NULL:
                _fast_destroy_tensor(c_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
def four_tensor_three_int_arrays_two_bools_int64_int8_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        out_shape, out_stride,
        first_values, second_values, third_values,
        bint flag_a, bint flag_b, int64_t value_a, int8_t value_b,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int out_ndim = len(out_shape)
    cdef int first_ndim = len(first_values)
    cdef int second_ndim = len(second_values)
    cdef int third_ndim = len(third_values)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int64_t* first_buf = NULL
    cdef int64_t* second_buf = NULL
    cdef int64_t* third_buf = NULL
    cdef int i
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* out_t = NULL
    cdef void* first_handle = NULL
    cdef void* second_handle = NULL
    cdef void* third_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    if first_ndim > 0:
        first_buf = <int64_t*>malloc(first_ndim * sizeof(int64_t))
        if first_buf == NULL:
            raise MemoryError("malloc failed for first int array buffer")
        for i in range(first_ndim):
            first_buf[i] = first_values[i]
    if second_ndim > 0:
        second_buf = <int64_t*>malloc(second_ndim * sizeof(int64_t))
        if second_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            raise MemoryError("malloc failed for second int array buffer")
        for i in range(second_ndim):
            second_buf[i] = second_values[i]
    if third_ndim > 0:
        third_buf = <int64_t*>malloc(third_ndim * sizeof(int64_t))
        if third_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            if second_buf != NULL:
                free(second_buf)
            raise MemoryError("malloc failed for third int array buffer")
        for i in range(third_ndim):
            third_buf[i] = third_values[i]
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if first_ndim > 0:
            first_handle = _fn_create_int_array(first_buf, <uint64_t>first_ndim)
        if second_ndim > 0:
            second_handle = _fn_create_int_array(second_buf, <uint64_t>second_ndim)
        if third_ndim > 0:
            third_handle = _fn_create_int_array(third_buf, <uint64_t>third_ndim)
    if a_t == NULL or b_t == NULL or out_t == NULL or (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL) or (third_ndim > 0 and third_handle == NULL):
        if a_t != NULL:
            _fast_destroy_tensor(a_t)
        if b_t != NULL:
            _fast_destroy_tensor(b_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if first_handle != NULL:
            _fn_destroy_int_array(first_handle)
        if second_handle != NULL:
            _fn_destroy_int_array(second_handle)
        if third_handle != NULL:
            _fn_destroy_int_array(third_handle)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if third_buf != NULL:
            free(third_buf)
        if (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL) or (third_ndim > 0 and third_handle == NULL):
            raise RuntimeError("aclCreateIntArray returned null")
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, bint, bint, int64_t, int8_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, b_t, first_handle, second_handle, third_handle, flag_a, flag_b, value_a, value_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>first_handle)] if first_handle != NULL else [])
            + ([('i', <uintptr_t>second_handle)] if second_handle != NULL else [])
            + ([('i', <uintptr_t>third_handle)] if third_handle != NULL else [])
            + ([('t', <uintptr_t>a_t)] if a_t != NULL else [])
            + ([('t', <uintptr_t>b_t)] if b_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        first_handle = NULL
        second_handle = NULL
        third_handle = NULL
        a_t = NULL
        b_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if first_handle != NULL:
                _fn_destroy_int_array(first_handle)
            if second_handle != NULL:
                _fn_destroy_int_array(second_handle)
            if third_handle != NULL:
                _fn_destroy_int_array(third_handle)
            if a_t != NULL:
                _fast_destroy_tensor(a_t)
            if b_t != NULL:
                _fast_destroy_tensor(b_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if third_buf != NULL:
            free(third_buf)
def four_tensor_four_int_arrays_bool_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        c_shape, c_stride,
        out_shape, out_stride,
        first_values, second_values, third_values, fourth_values,
        bint flag,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t c_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t c_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int c_ndim = len(c_shape)
    cdef int out_ndim = len(out_shape)
    cdef int first_ndim = len(first_values)
    cdef int second_ndim = len(second_values)
    cdef int third_ndim = len(third_values)
    cdef int fourth_ndim = len(fourth_values)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] c_shape_buf, c_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int64_t* first_buf = NULL
    cdef int64_t* second_buf = NULL
    cdef int64_t* third_buf = NULL
    cdef int64_t* fourth_buf = NULL
    cdef int i
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* c_t = NULL
    cdef void* out_t = NULL
    cdef void* first_handle = NULL
    cdef void* second_handle = NULL
    cdef void* third_handle = NULL
    cdef void* fourth_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(c_ndim):
        c_shape_buf[i] = c_shape[i]
        c_stride_buf[i] = c_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    if first_ndim > 0:
        first_buf = <int64_t*>malloc(first_ndim * sizeof(int64_t))
        if first_buf == NULL:
            raise MemoryError("malloc failed for first int array buffer")
        for i in range(first_ndim):
            first_buf[i] = first_values[i]
    if second_ndim > 0:
        second_buf = <int64_t*>malloc(second_ndim * sizeof(int64_t))
        if second_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            raise MemoryError("malloc failed for second int array buffer")
        for i in range(second_ndim):
            second_buf[i] = second_values[i]
    if third_ndim > 0:
        third_buf = <int64_t*>malloc(third_ndim * sizeof(int64_t))
        if third_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            if second_buf != NULL:
                free(second_buf)
            raise MemoryError("malloc failed for third int array buffer")
        for i in range(third_ndim):
            third_buf[i] = third_values[i]
    if fourth_ndim > 0:
        fourth_buf = <int64_t*>malloc(fourth_ndim * sizeof(int64_t))
        if fourth_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            if second_buf != NULL:
                free(second_buf)
            if third_buf != NULL:
                free(third_buf)
            raise MemoryError("malloc failed for fourth int array buffer")
        for i in range(fourth_ndim):
            fourth_buf[i] = fourth_values[i]
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        c_t = _fast_create_tensor(c_shape_buf, c_stride_buf, <uint64_t>c_ndim, c_dtype_code, fmt, <void*>c_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if first_ndim > 0:
            first_handle = _fn_create_int_array(first_buf, <uint64_t>first_ndim)
        if second_ndim > 0:
            second_handle = _fn_create_int_array(second_buf, <uint64_t>second_ndim)
        if third_ndim > 0:
            third_handle = _fn_create_int_array(third_buf, <uint64_t>third_ndim)
        if fourth_ndim > 0:
            fourth_handle = _fn_create_int_array(fourth_buf, <uint64_t>fourth_ndim)
    if a_t == NULL or b_t == NULL or c_t == NULL or out_t == NULL or (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL) or (third_ndim > 0 and third_handle == NULL) or (fourth_ndim > 0 and fourth_handle == NULL):
        if a_t != NULL:
            _fast_destroy_tensor(a_t)
        if b_t != NULL:
            _fast_destroy_tensor(b_t)
        if c_t != NULL:
            _fast_destroy_tensor(c_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if first_handle != NULL:
            _fn_destroy_int_array(first_handle)
        if second_handle != NULL:
            _fn_destroy_int_array(second_handle)
        if third_handle != NULL:
            _fn_destroy_int_array(third_handle)
        if fourth_handle != NULL:
            _fn_destroy_int_array(fourth_handle)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if third_buf != NULL:
            free(third_buf)
        if fourth_buf != NULL:
            free(fourth_buf)
        if (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL) or (third_ndim > 0 and third_handle == NULL) or (fourth_ndim > 0 and fourth_handle == NULL):
            raise RuntimeError("aclCreateIntArray returned null")
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, void*, void*, bint, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, b_t, c_t, first_handle, second_handle, third_handle, fourth_handle, flag, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>first_handle)] if first_handle != NULL else [])
            + ([('i', <uintptr_t>second_handle)] if second_handle != NULL else [])
            + ([('i', <uintptr_t>third_handle)] if third_handle != NULL else [])
            + ([('i', <uintptr_t>fourth_handle)] if fourth_handle != NULL else [])
            + ([('t', <uintptr_t>a_t)] if a_t != NULL else [])
            + ([('t', <uintptr_t>b_t)] if b_t != NULL else [])
            + ([('t', <uintptr_t>c_t)] if c_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        first_handle = NULL
        second_handle = NULL
        third_handle = NULL
        fourth_handle = NULL
        a_t = NULL
        b_t = NULL
        c_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if first_handle != NULL:
                _fn_destroy_int_array(first_handle)
            if second_handle != NULL:
                _fn_destroy_int_array(second_handle)
            if third_handle != NULL:
                _fn_destroy_int_array(third_handle)
            if fourth_handle != NULL:
                _fn_destroy_int_array(fourth_handle)
            if a_t != NULL:
                _fast_destroy_tensor(a_t)
            if b_t != NULL:
                _fast_destroy_tensor(b_t)
            if c_t != NULL:
                _fast_destroy_tensor(c_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if third_buf != NULL:
            free(third_buf)
        if fourth_buf != NULL:
            free(fourth_buf)
def tensor_int_array_three_ints_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        dims_tuple,
        out_shape, out_stride,
        int64_t value_a, int64_t value_b, int64_t value_c,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int dims_ndim = len(dims_tuple)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] self_shape_buf, self_stride_buf
    cdef int64_t[MAX_NDIM] dims_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(self_ndim):
        self_shape_buf[i] = self_shape[i]
        self_stride_buf[i] = self_stride[i]
    for i in range(dims_ndim):
        dims_buf[i] = dims_tuple[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* dims_handle = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(self_shape_buf, self_stride_buf, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if dims_ndim > 0:
            dims_handle = _fn_create_int_array(dims_buf, <uint64_t>dims_ndim)
    if self_t == NULL or out_t == NULL or (dims_ndim > 0 and dims_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if dims_handle != NULL:
            _fn_destroy_int_array(dims_handle)
        raise RuntimeError("ACLNN descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, int64_t, int64_t, int64_t, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dims_handle, value_a, value_b, value_c, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('i', <uintptr_t>dims_handle)] if dims_handle != NULL else [])
            + ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        dims_handle = NULL
        self_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dims_handle != NULL:
                _fn_destroy_int_array(dims_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def two_tensor_ints_bool_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        first_shape, first_stride,
        second_shape, second_stride,
        out_shape, out_stride,
        int64_t value_a, bint flag,
        int32_t first_dtype_code, int32_t second_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t first_ptr, uintptr_t second_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int first_ndim = len(first_shape)
    cdef int second_ndim = len(second_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] first_shape_buf, first_stride_buf
    cdef int64_t[MAX_NDIM] second_shape_buf, second_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(first_ndim):
        first_shape_buf[i] = first_shape[i]
        first_stride_buf[i] = first_stride[i]
    for i in range(second_ndim):
        second_shape_buf[i] = second_shape[i]
        second_stride_buf[i] = second_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* first_t = NULL
    cdef void* second_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        first_t = _fast_create_tensor(first_shape_buf, first_stride_buf, <uint64_t>first_ndim, first_dtype_code, fmt, <void*>first_ptr)
        second_t = _fast_create_tensor(second_shape_buf, second_stride_buf, <uint64_t>second_ndim, second_dtype_code, fmt, <void*>second_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if first_t == NULL or second_t == NULL or out_t == NULL:
        if first_t != NULL:
            _fast_destroy_tensor(first_t)
        if second_t != NULL:
            _fast_destroy_tensor(second_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, int64_t, bint, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                first_t, second_t, value_a, flag, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>first_t)] if first_t != NULL else [])
            + ([('t', <uintptr_t>second_t)] if second_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        first_t = NULL
        second_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if first_t != NULL:
                _fast_destroy_tensor(first_t)
            if second_t != NULL:
                _fast_destroy_tensor(second_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def tensor_two_int_arrays_bool_two_doubles_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        first_values, second_values,
        out_shape, out_stride,
        bint flag, double value_a, double value_b,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int first_ndim = len(first_values)
    cdef int second_ndim = len(second_values)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t* first_buf = NULL
    cdef int64_t* second_buf = NULL
    cdef int i
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* first_handle = NULL
    cdef void* second_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    if first_ndim > 0:
        first_buf = <int64_t*>malloc(first_ndim * sizeof(int64_t))
        if first_buf == NULL:
            raise MemoryError("malloc failed for first int array buffer")
        for i in range(first_ndim):
            first_buf[i] = first_values[i]
    if second_ndim > 0:
        second_buf = <int64_t*>malloc(second_ndim * sizeof(int64_t))
        if second_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            raise MemoryError("malloc failed for second int array buffer")
        for i in range(second_ndim):
            second_buf[i] = second_values[i]
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if first_ndim > 0:
            first_handle = _fn_create_int_array(first_buf, <uint64_t>first_ndim)
        if second_ndim > 0:
            second_handle = _fn_create_int_array(second_buf, <uint64_t>second_ndim)
    if self_t == NULL or out_t == NULL or (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if first_handle != NULL:
            _fn_destroy_int_array(first_handle)
        if second_handle != NULL:
            _fn_destroy_int_array(second_handle)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL):
            raise RuntimeError("aclCreateIntArray returned null")
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, bint, double, double, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, first_handle, second_handle, flag, value_a, value_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if first_handle != NULL:
                _fn_destroy_int_array(first_handle)
            if second_handle != NULL:
                _fn_destroy_int_array(second_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)


def tensor_two_int_arrays_bool_double_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        first_values, second_values,
        out_shape, out_stride,
        bint flag, double value,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int first_ndim = len(first_values)
    cdef int second_ndim = len(second_values)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int64_t* first_buf = NULL
    cdef int64_t* second_buf = NULL
    cdef int i
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef void* first_handle = NULL
    cdef void* second_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    if first_ndim > 0:
        first_buf = <int64_t*>malloc(first_ndim * sizeof(int64_t))
        if first_buf == NULL:
            raise MemoryError("malloc failed for first int array buffer")
        for i in range(first_ndim):
            first_buf[i] = first_values[i]
    if second_ndim > 0:
        second_buf = <int64_t*>malloc(second_ndim * sizeof(int64_t))
        if second_buf == NULL:
            if first_buf != NULL:
                free(first_buf)
            raise MemoryError("malloc failed for second int array buffer")
        for i in range(second_ndim):
            second_buf[i] = second_values[i]
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
        if first_ndim > 0:
            first_handle = _fn_create_int_array(first_buf, <uint64_t>first_ndim)
        if second_ndim > 0:
            second_handle = _fn_create_int_array(second_buf, <uint64_t>second_ndim)
    if self_t == NULL or out_t == NULL or (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL):
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        if first_handle != NULL:
            _fn_destroy_int_array(first_handle)
        if second_handle != NULL:
            _fn_destroy_int_array(second_handle)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)
        if (first_ndim > 0 and first_handle == NULL) or (second_ndim > 0 and second_handle == NULL):
            raise RuntimeError("aclCreateIntArray returned null")
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, bint, double, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, first_handle, second_handle, flag, value, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if first_handle != NULL:
                _fn_destroy_int_array(first_handle)
            if second_handle != NULL:
                _fn_destroy_int_array(second_handle)
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
        if first_buf != NULL:
            free(first_buf)
        if second_buf != NULL:
            free(second_buf)


def two_tensor_two_scalars_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        out_shape, out_stride,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t out_ptr,
        uintptr_t scalar_a, uintptr_t scalar_b,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if a_t == NULL or b_t == NULL or out_t == NULL:
        if a_t != NULL:
            _fast_destroy_tensor(a_t)
        if b_t != NULL:
            _fast_destroy_tensor(b_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, b_t, <void*>scalar_a, <void*>scalar_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>a_t)] if a_t != NULL else [])
            + ([('t', <uintptr_t>b_t)] if b_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        a_t = NULL
        b_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if a_t != NULL:
                _fast_destroy_tensor(a_t)
            if b_t != NULL:
                _fast_destroy_tensor(b_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
def two_tensor_scalar_bool_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        out_shape, out_stride,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t out_ptr,
        uintptr_t scalar_handle, bint flag,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if a_t == NULL or b_t == NULL or out_t == NULL:
        if a_t != NULL:
            _fast_destroy_tensor(a_t)
        if b_t != NULL:
            _fast_destroy_tensor(b_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, bint, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, b_t, <void*>scalar_handle, flag, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>a_t)] if a_t != NULL else [])
            + ([('t', <uintptr_t>b_t)] if b_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        a_t = NULL
        b_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if a_t != NULL:
                _fast_destroy_tensor(a_t)
            if b_t != NULL:
                _fast_destroy_tensor(b_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
def two_tensor_three_scalars_bool_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        out_shape, out_stride,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t out_ptr,
        uintptr_t scalar_a, uintptr_t scalar_b, uintptr_t scalar_c, bint flag,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if a_t == NULL or b_t == NULL or out_t == NULL:
        if a_t != NULL:
            _fast_destroy_tensor(a_t)
        if b_t != NULL:
            _fast_destroy_tensor(b_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, bint, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, <void*>scalar_a, <void*>scalar_b, <void*>scalar_c, flag, b_t, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>a_t)] if a_t != NULL else [])
            + ([('t', <uintptr_t>b_t)] if b_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        a_t = NULL
        b_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if a_t != NULL:
                _fast_destroy_tensor(a_t)
            if b_t != NULL:
                _fast_destroy_tensor(b_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
def layer_norm_backward_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        grad_shape, grad_stride,
        input_shape, input_stride,
        grad_input_shape, grad_input_stride,
        stats_shape, stats_stride,
        weight_shape, weight_stride,
        bias_shape, bias_stride,
        normalized_shape,
        output_mask,
        int32_t tensor_dtype_code, int32_t stats_dtype_code, int32_t fmt,
        uintptr_t grad_ptr, uintptr_t input_ptr,
        uintptr_t mean_ptr, uintptr_t rstd_ptr,
        uintptr_t weight_ptr, uintptr_t bias_ptr,
        uintptr_t grad_input_ptr, uintptr_t grad_weight_ptr, uintptr_t grad_bias_ptr,
        uintptr_t stream):
    cdef int grad_ndim = len(grad_shape)
    cdef int input_ndim = len(input_shape)
    cdef int gi_ndim = len(grad_input_shape)
    cdef int stats_ndim = len(stats_shape)
    cdef int weight_ndim = 0 if weight_shape is None else len(weight_shape)
    cdef int bias_ndim = 0 if bias_shape is None else len(bias_shape)
    cdef int norm_ndim = len(normalized_shape)
    cdef int mask_ndim = len(output_mask)
    cdef bint has_weight = weight_ptr != 0 and weight_shape is not None
    cdef bint has_bias = bias_ptr != 0 and bias_shape is not None
    cdef bint has_grad_weight = grad_weight_ptr != 0 and weight_shape is not None
    cdef bint has_grad_bias = grad_bias_ptr != 0 and bias_shape is not None
    cdef int64_t[MAX_NDIM] grad_shape_buf, grad_stride_buf
    cdef int64_t[MAX_NDIM] input_shape_buf, input_stride_buf
    cdef int64_t[MAX_NDIM] gi_shape_buf, gi_stride_buf
    cdef int64_t[MAX_NDIM] stats_shape_buf, stats_stride_buf
    cdef int64_t[MAX_NDIM] weight_shape_buf, weight_stride_buf
    cdef int64_t[MAX_NDIM] bias_shape_buf, bias_stride_buf
    cdef int64_t[MAX_NDIM] norm_shape_buf
    cdef uint8_t[MAX_NDIM] mask_buf
    cdef int i
    for i in range(grad_ndim):
        grad_shape_buf[i] = grad_shape[i]
        grad_stride_buf[i] = grad_stride[i]
    for i in range(input_ndim):
        input_shape_buf[i] = input_shape[i]
        input_stride_buf[i] = input_stride[i]
    for i in range(gi_ndim):
        gi_shape_buf[i] = grad_input_shape[i]
        gi_stride_buf[i] = grad_input_stride[i]
    for i in range(stats_ndim):
        stats_shape_buf[i] = stats_shape[i]
        stats_stride_buf[i] = stats_stride[i]
    if weight_shape is not None:
        for i in range(weight_ndim):
            weight_shape_buf[i] = weight_shape[i]
            weight_stride_buf[i] = weight_stride[i]
    if bias_shape is not None:
        for i in range(bias_ndim):
            bias_shape_buf[i] = bias_shape[i]
            bias_stride_buf[i] = bias_stride[i]
    for i in range(norm_ndim):
        norm_shape_buf[i] = normalized_shape[i]
    for i in range(mask_ndim):
        mask_buf[i] = 1 if output_mask[i] else 0
    cdef void* grad_t = NULL
    cdef void* input_t = NULL
    cdef void* mean_t = NULL
    cdef void* rstd_t = NULL
    cdef void* weight_t = NULL
    cdef void* bias_t = NULL
    cdef void* grad_input_t = NULL
    cdef void* grad_weight_t = NULL
    cdef void* grad_bias_t = NULL
    cdef void* norm_handle = NULL
    cdef void* mask_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        grad_t = _fast_create_tensor(grad_shape_buf, grad_stride_buf, <uint64_t>grad_ndim, tensor_dtype_code, fmt, <void*>grad_ptr)
        input_t = _fast_create_tensor(input_shape_buf, input_stride_buf, <uint64_t>input_ndim, tensor_dtype_code, fmt, <void*>input_ptr)
        mean_t = _fast_create_tensor(stats_shape_buf, stats_stride_buf, <uint64_t>stats_ndim, stats_dtype_code, fmt, <void*>mean_ptr)
        rstd_t = _fast_create_tensor(stats_shape_buf, stats_stride_buf, <uint64_t>stats_ndim, stats_dtype_code, fmt, <void*>rstd_ptr)
        grad_input_t = _fast_create_tensor(gi_shape_buf, gi_stride_buf, <uint64_t>gi_ndim, tensor_dtype_code, fmt, <void*>grad_input_ptr)
        if has_weight:
            weight_t = _fast_create_tensor(weight_shape_buf, weight_stride_buf, <uint64_t>weight_ndim, tensor_dtype_code, fmt, <void*>weight_ptr)
        if has_bias:
            bias_t = _fast_create_tensor(bias_shape_buf, bias_stride_buf, <uint64_t>bias_ndim, tensor_dtype_code, fmt, <void*>bias_ptr)
        if has_grad_weight:
            grad_weight_t = _fast_create_tensor(weight_shape_buf, weight_stride_buf, <uint64_t>weight_ndim, tensor_dtype_code, fmt, <void*>grad_weight_ptr)
        if has_grad_bias:
            grad_bias_t = _fast_create_tensor(bias_shape_buf, bias_stride_buf, <uint64_t>bias_ndim, tensor_dtype_code, fmt, <void*>grad_bias_ptr)
        if norm_ndim > 0:
            norm_handle = _fn_create_int_array(norm_shape_buf, <uint64_t>norm_ndim)
        if mask_ndim > 0:
            mask_handle = _fast_create_bool_array(mask_buf, <uint64_t>mask_ndim)
    if grad_t == NULL or input_t == NULL or mean_t == NULL or rstd_t == NULL or grad_input_t == NULL or (has_weight and weight_t == NULL) or (has_bias and bias_t == NULL) or (has_grad_weight and grad_weight_t == NULL) or (has_grad_bias and grad_bias_t == NULL) or (norm_ndim > 0 and norm_handle == NULL) or (mask_ndim > 0 and mask_handle == NULL):
        if grad_t != NULL: _fast_destroy_tensor(grad_t)
        if input_t != NULL: _fast_destroy_tensor(input_t)
        if mean_t != NULL: _fast_destroy_tensor(mean_t)
        if rstd_t != NULL: _fast_destroy_tensor(rstd_t)
        if weight_t != NULL: _fast_destroy_tensor(weight_t)
        if bias_t != NULL: _fast_destroy_tensor(bias_t)
        if grad_input_t != NULL: _fast_destroy_tensor(grad_input_t)
        if grad_weight_t != NULL: _fast_destroy_tensor(grad_weight_t)
        if grad_bias_t != NULL: _fast_destroy_tensor(grad_bias_t)
        if norm_handle != NULL: _fn_destroy_int_array(norm_handle)
        if mask_handle != NULL: _fast_destroy_bool_array(mask_handle)
        raise RuntimeError("ACLNN layer_norm_backward descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                grad_t,
                input_t,
                norm_handle,
                mean_t,
                rstd_t,
                weight_t,
                bias_t,
                mask_handle,
                grad_input_t,
                grad_weight_t,
                grad_bias_t,
                &ws_size,
                &executor,
            )
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if norm_handle != NULL: _fn_destroy_int_array(norm_handle)
            if mask_handle != NULL: _fast_destroy_bool_array(mask_handle)
            if grad_t != NULL: _fast_destroy_tensor(grad_t)
            if input_t != NULL: _fast_destroy_tensor(input_t)
            if mean_t != NULL: _fast_destroy_tensor(mean_t)
            if rstd_t != NULL: _fast_destroy_tensor(rstd_t)
            if weight_t != NULL: _fast_destroy_tensor(weight_t)
            if bias_t != NULL: _fast_destroy_tensor(bias_t)
            if grad_input_t != NULL: _fast_destroy_tensor(grad_input_t)
            if grad_weight_t != NULL: _fast_destroy_tensor(grad_weight_t)
            if grad_bias_t != NULL: _fast_destroy_tensor(grad_bias_t)

def rms_norm_grad_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        dy_shape, dy_stride,
        x_shape, x_stride,
        rstd_shape, rstd_stride,
        gamma_shape, gamma_stride,
        dx_shape, dx_stride,
        dgamma_shape, dgamma_stride,
        int32_t tensor_dtype_code, int32_t fmt,
        uintptr_t dy_ptr, uintptr_t x_ptr, uintptr_t rstd_ptr, uintptr_t gamma_ptr,
        uintptr_t dx_ptr, uintptr_t dgamma_ptr,
        uintptr_t stream):
    cdef int dy_ndim = len(dy_shape)
    cdef int x_ndim = len(x_shape)
    cdef int rstd_ndim = len(rstd_shape)
    cdef int gamma_ndim = len(gamma_shape)
    cdef int dx_ndim = len(dx_shape)
    cdef int dgamma_ndim = len(dgamma_shape)
    cdef int64_t[MAX_NDIM] dy_shape_buf, dy_stride_buf
    cdef int64_t[MAX_NDIM] x_shape_buf, x_stride_buf
    cdef int64_t[MAX_NDIM] rstd_shape_buf, rstd_stride_buf
    cdef int64_t[MAX_NDIM] gamma_shape_buf, gamma_stride_buf
    cdef int64_t[MAX_NDIM] dx_shape_buf, dx_stride_buf
    cdef int64_t[MAX_NDIM] dgamma_shape_buf, dgamma_stride_buf
    cdef int i
    for i in range(dy_ndim):
        dy_shape_buf[i] = dy_shape[i]
        dy_stride_buf[i] = dy_stride[i]
    for i in range(x_ndim):
        x_shape_buf[i] = x_shape[i]
        x_stride_buf[i] = x_stride[i]
    for i in range(rstd_ndim):
        rstd_shape_buf[i] = rstd_shape[i]
        rstd_stride_buf[i] = rstd_stride[i]
    for i in range(gamma_ndim):
        gamma_shape_buf[i] = gamma_shape[i]
        gamma_stride_buf[i] = gamma_stride[i]
    for i in range(dx_ndim):
        dx_shape_buf[i] = dx_shape[i]
        dx_stride_buf[i] = dx_stride[i]
    for i in range(dgamma_ndim):
        dgamma_shape_buf[i] = dgamma_shape[i]
        dgamma_stride_buf[i] = dgamma_stride[i]
    cdef void* dy_t = NULL
    cdef void* x_t = NULL
    cdef void* rstd_t = NULL
    cdef void* gamma_t = NULL
    cdef void* dx_t = NULL
    cdef void* dgamma_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        dy_t = _fast_create_tensor(dy_shape_buf, dy_stride_buf, <uint64_t>dy_ndim, tensor_dtype_code, fmt, <void*>dy_ptr)
        x_t = _fast_create_tensor(x_shape_buf, x_stride_buf, <uint64_t>x_ndim, tensor_dtype_code, fmt, <void*>x_ptr)
        rstd_t = _fast_create_tensor(rstd_shape_buf, rstd_stride_buf, <uint64_t>rstd_ndim, tensor_dtype_code, fmt, <void*>rstd_ptr)
        gamma_t = _fast_create_tensor(gamma_shape_buf, gamma_stride_buf, <uint64_t>gamma_ndim, tensor_dtype_code, fmt, <void*>gamma_ptr)
        dx_t = _fast_create_tensor(dx_shape_buf, dx_stride_buf, <uint64_t>dx_ndim, tensor_dtype_code, fmt, <void*>dx_ptr)
        dgamma_t = _fast_create_tensor(dgamma_shape_buf, dgamma_stride_buf, <uint64_t>dgamma_ndim, tensor_dtype_code, fmt, <void*>dgamma_ptr)
    if dy_t == NULL or x_t == NULL or rstd_t == NULL or gamma_t == NULL or dx_t == NULL or dgamma_t == NULL:
        if dy_t != NULL: _fast_destroy_tensor(dy_t)
        if x_t != NULL: _fast_destroy_tensor(x_t)
        if rstd_t != NULL: _fast_destroy_tensor(rstd_t)
        if gamma_t != NULL: _fast_destroy_tensor(gamma_t)
        if dx_t != NULL: _fast_destroy_tensor(dx_t)
        if dgamma_t != NULL: _fast_destroy_tensor(dgamma_t)
        raise RuntimeError("ACLNN rms_norm_grad descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                dy_t, x_t, rstd_t, gamma_t, dx_t, dgamma_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if dy_t != NULL: _fast_destroy_tensor(dy_t)
            if x_t != NULL: _fast_destroy_tensor(x_t)
            if rstd_t != NULL: _fast_destroy_tensor(rstd_t)
            if gamma_t != NULL: _fast_destroy_tensor(gamma_t)
            if dx_t != NULL: _fast_destroy_tensor(dx_t)
            if dgamma_t != NULL: _fast_destroy_tensor(dgamma_t)


def batch_norm_backward_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        grad_shape, grad_stride,
        input_shape, input_stride,
        weight_shape, weight_stride,
        rm_shape, rm_stride,
        rv_shape, rv_stride,
        sm_shape, sm_stride,
        si_shape, si_stride,
        gi_shape, gi_stride,
        gw_shape, gw_stride,
        gb_shape, gb_stride,
        output_mask,
        bint training, double eps,
        int32_t tensor_dtype_code, int32_t stats_dtype_code,
        int32_t io_fmt, int32_t param_fmt, int32_t stats_fmt,
        uintptr_t grad_ptr, uintptr_t input_ptr,
        uintptr_t weight_ptr, uintptr_t rm_ptr, uintptr_t rv_ptr,
        uintptr_t sm_ptr, uintptr_t si_ptr,
        uintptr_t gi_ptr, uintptr_t gw_ptr, uintptr_t gb_ptr,
        uintptr_t stream):
    cdef int grad_ndim = len(grad_shape)
    cdef int input_ndim = len(input_shape)
    cdef int weight_ndim = 0 if weight_shape is None else len(weight_shape)
    cdef int rm_ndim = 0 if rm_shape is None else len(rm_shape)
    cdef int rv_ndim = 0 if rv_shape is None else len(rv_shape)
    cdef int sm_ndim = len(sm_shape)
    cdef int si_ndim = len(si_shape)
    cdef int gi_ndim = len(gi_shape)
    cdef int gw_ndim = 0 if gw_shape is None else len(gw_shape)
    cdef int gb_ndim = 0 if gb_shape is None else len(gb_shape)
    cdef int mask_ndim = len(output_mask)
    cdef bint has_weight = weight_ptr != 0 and weight_shape is not None
    cdef bint has_rm = rm_ptr != 0 and rm_shape is not None
    cdef bint has_rv = rv_ptr != 0 and rv_shape is not None
    cdef bint has_gw = gw_ptr != 0 and gw_shape is not None
    cdef bint has_gb = gb_ptr != 0 and gb_shape is not None
    cdef int64_t[MAX_NDIM] grad_shape_buf, grad_stride_buf
    cdef int64_t[MAX_NDIM] input_shape_buf, input_stride_buf
    cdef int64_t[MAX_NDIM] weight_shape_buf, weight_stride_buf
    cdef int64_t[MAX_NDIM] rm_shape_buf, rm_stride_buf
    cdef int64_t[MAX_NDIM] rv_shape_buf, rv_stride_buf
    cdef int64_t[MAX_NDIM] sm_shape_buf, sm_stride_buf
    cdef int64_t[MAX_NDIM] si_shape_buf, si_stride_buf
    cdef int64_t[MAX_NDIM] gi_shape_buf, gi_stride_buf
    cdef int64_t[MAX_NDIM] gw_shape_buf, gw_stride_buf
    cdef int64_t[MAX_NDIM] gb_shape_buf, gb_stride_buf
    cdef uint8_t[MAX_NDIM] mask_buf
    cdef int i
    for i in range(grad_ndim):
        grad_shape_buf[i] = grad_shape[i]
        grad_stride_buf[i] = grad_stride[i]
    for i in range(input_ndim):
        input_shape_buf[i] = input_shape[i]
        input_stride_buf[i] = input_stride[i]
    if weight_shape is not None:
        for i in range(weight_ndim):
            weight_shape_buf[i] = weight_shape[i]
            weight_stride_buf[i] = weight_stride[i]
    if rm_shape is not None:
        for i in range(rm_ndim):
            rm_shape_buf[i] = rm_shape[i]
            rm_stride_buf[i] = rm_stride[i]
    if rv_shape is not None:
        for i in range(rv_ndim):
            rv_shape_buf[i] = rv_shape[i]
            rv_stride_buf[i] = rv_stride[i]
    for i in range(sm_ndim):
        sm_shape_buf[i] = sm_shape[i]
        sm_stride_buf[i] = sm_stride[i]
    for i in range(si_ndim):
        si_shape_buf[i] = si_shape[i]
        si_stride_buf[i] = si_stride[i]
    for i in range(gi_ndim):
        gi_shape_buf[i] = gi_shape[i]
        gi_stride_buf[i] = gi_stride[i]
    if gw_shape is not None:
        for i in range(gw_ndim):
            gw_shape_buf[i] = gw_shape[i]
            gw_stride_buf[i] = gw_stride[i]
    if gb_shape is not None:
        for i in range(gb_ndim):
            gb_shape_buf[i] = gb_shape[i]
            gb_stride_buf[i] = gb_stride[i]
    for i in range(mask_ndim):
        mask_buf[i] = 1 if output_mask[i] else 0
    cdef void* grad_t = NULL
    cdef void* input_t = NULL
    cdef void* weight_t = NULL
    cdef void* rm_t = NULL
    cdef void* rv_t = NULL
    cdef void* sm_t = NULL
    cdef void* si_t = NULL
    cdef void* gi_t = NULL
    cdef void* gw_t = NULL
    cdef void* gb_t = NULL
    cdef void* mask_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        grad_t = _fast_create_tensor(grad_shape_buf, grad_stride_buf, <uint64_t>grad_ndim, tensor_dtype_code, io_fmt, <void*>grad_ptr)
        input_t = _fast_create_tensor(input_shape_buf, input_stride_buf, <uint64_t>input_ndim, tensor_dtype_code, io_fmt, <void*>input_ptr)
        if has_weight:
            weight_t = _fast_create_tensor(weight_shape_buf, weight_stride_buf, <uint64_t>weight_ndim, tensor_dtype_code, param_fmt, <void*>weight_ptr)
        if has_rm:
            rm_t = _fast_create_tensor(rm_shape_buf, rm_stride_buf, <uint64_t>rm_ndim, tensor_dtype_code, param_fmt, <void*>rm_ptr)
        if has_rv:
            rv_t = _fast_create_tensor(rv_shape_buf, rv_stride_buf, <uint64_t>rv_ndim, tensor_dtype_code, param_fmt, <void*>rv_ptr)
        sm_t = _fast_create_tensor(sm_shape_buf, sm_stride_buf, <uint64_t>sm_ndim, stats_dtype_code, stats_fmt, <void*>sm_ptr)
        si_t = _fast_create_tensor(si_shape_buf, si_stride_buf, <uint64_t>si_ndim, stats_dtype_code, stats_fmt, <void*>si_ptr)
        gi_t = _fast_create_tensor(gi_shape_buf, gi_stride_buf, <uint64_t>gi_ndim, tensor_dtype_code, io_fmt, <void*>gi_ptr)
        if has_gw:
            gw_t = _fast_create_tensor(gw_shape_buf, gw_stride_buf, <uint64_t>gw_ndim, tensor_dtype_code, param_fmt, <void*>gw_ptr)
        if has_gb:
            gb_t = _fast_create_tensor(gb_shape_buf, gb_stride_buf, <uint64_t>gb_ndim, tensor_dtype_code, param_fmt, <void*>gb_ptr)
        if mask_ndim > 0:
            mask_handle = _fast_create_bool_array(mask_buf, <uint64_t>mask_ndim)
    if grad_t == NULL or input_t == NULL or sm_t == NULL or si_t == NULL or gi_t == NULL or (has_weight and weight_t == NULL) or (has_rm and rm_t == NULL) or (has_rv and rv_t == NULL) or (has_gw and gw_t == NULL) or (has_gb and gb_t == NULL) or (mask_ndim > 0 and mask_handle == NULL):
        if grad_t != NULL: _fast_destroy_tensor(grad_t)
        if input_t != NULL: _fast_destroy_tensor(input_t)
        if weight_t != NULL: _fast_destroy_tensor(weight_t)
        if rm_t != NULL: _fast_destroy_tensor(rm_t)
        if rv_t != NULL: _fast_destroy_tensor(rv_t)
        if sm_t != NULL: _fast_destroy_tensor(sm_t)
        if si_t != NULL: _fast_destroy_tensor(si_t)
        if gi_t != NULL: _fast_destroy_tensor(gi_t)
        if gw_t != NULL: _fast_destroy_tensor(gw_t)
        if gb_t != NULL: _fast_destroy_tensor(gb_t)
        if mask_handle != NULL: _fast_destroy_bool_array(mask_handle)
        raise RuntimeError("ACLNN batch_norm_backward descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, void*, void*, bint, double, void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                grad_t, input_t, weight_t, rm_t, rv_t, sm_t, si_t, training, eps, mask_handle, gi_t, gw_t, gb_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if mask_handle != NULL: _fast_destroy_bool_array(mask_handle)
            if grad_t != NULL: _fast_destroy_tensor(grad_t)
            if input_t != NULL: _fast_destroy_tensor(input_t)
            if weight_t != NULL: _fast_destroy_tensor(weight_t)
            if rm_t != NULL: _fast_destroy_tensor(rm_t)
            if rv_t != NULL: _fast_destroy_tensor(rv_t)
            if sm_t != NULL: _fast_destroy_tensor(sm_t)
            if si_t != NULL: _fast_destroy_tensor(si_t)
            if gi_t != NULL: _fast_destroy_tensor(gi_t)
            if gw_t != NULL: _fast_destroy_tensor(gw_t)
            if gb_t != NULL: _fast_destroy_tensor(gb_t)


def group_norm_backward_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        grad_shape, grad_stride,
        input_shape, input_stride,
        mean_shape, mean_stride,
        rstd_shape, rstd_stride,
        gamma_shape, gamma_stride,
        gi_shape, gi_stride,
        gg_shape, gg_stride,
        gb_shape, gb_stride,
        output_mask,
        int64_t N, int64_t C, int64_t HxW, int64_t group,
        int32_t tensor_dtype_code, int32_t stats_dtype_code,
        int32_t io_fmt, int32_t param_fmt, int32_t stats_fmt,
        uintptr_t grad_ptr, uintptr_t input_ptr,
        uintptr_t mean_ptr, uintptr_t rstd_ptr, uintptr_t gamma_ptr,
        uintptr_t gi_ptr, uintptr_t gg_ptr, uintptr_t gb_ptr,
        uintptr_t stream):
    cdef int grad_ndim = len(grad_shape)
    cdef int input_ndim = len(input_shape)
    cdef int mean_ndim = len(mean_shape)
    cdef int rstd_ndim = len(rstd_shape)
    cdef int gamma_ndim = 0 if gamma_shape is None else len(gamma_shape)
    cdef int gi_ndim = len(gi_shape)
    cdef int gg_ndim = 0 if gg_shape is None else len(gg_shape)
    cdef int gb_ndim = 0 if gb_shape is None else len(gb_shape)
    cdef int mask_ndim = len(output_mask)
    cdef bint has_gamma = gamma_ptr != 0 and gamma_shape is not None
    cdef bint has_gg = gg_ptr != 0 and gg_shape is not None
    cdef bint has_gb = gb_ptr != 0 and gb_shape is not None
    cdef int64_t[MAX_NDIM] grad_shape_buf, grad_stride_buf
    cdef int64_t[MAX_NDIM] input_shape_buf, input_stride_buf
    cdef int64_t[MAX_NDIM] mean_shape_buf, mean_stride_buf
    cdef int64_t[MAX_NDIM] rstd_shape_buf, rstd_stride_buf
    cdef int64_t[MAX_NDIM] gamma_shape_buf, gamma_stride_buf
    cdef int64_t[MAX_NDIM] gi_shape_buf, gi_stride_buf
    cdef int64_t[MAX_NDIM] gg_shape_buf, gg_stride_buf
    cdef int64_t[MAX_NDIM] gb_shape_buf, gb_stride_buf
    cdef uint8_t[MAX_NDIM] mask_buf
    cdef int i
    for i in range(grad_ndim):
        grad_shape_buf[i] = grad_shape[i]
        grad_stride_buf[i] = grad_stride[i]
    for i in range(input_ndim):
        input_shape_buf[i] = input_shape[i]
        input_stride_buf[i] = input_stride[i]
    for i in range(mean_ndim):
        mean_shape_buf[i] = mean_shape[i]
        mean_stride_buf[i] = mean_stride[i]
    for i in range(rstd_ndim):
        rstd_shape_buf[i] = rstd_shape[i]
        rstd_stride_buf[i] = rstd_stride[i]
    if gamma_shape is not None:
        for i in range(gamma_ndim):
            gamma_shape_buf[i] = gamma_shape[i]
            gamma_stride_buf[i] = gamma_stride[i]
    for i in range(gi_ndim):
        gi_shape_buf[i] = gi_shape[i]
        gi_stride_buf[i] = gi_stride[i]
    if gg_shape is not None:
        for i in range(gg_ndim):
            gg_shape_buf[i] = gg_shape[i]
            gg_stride_buf[i] = gg_stride[i]
    if gb_shape is not None:
        for i in range(gb_ndim):
            gb_shape_buf[i] = gb_shape[i]
            gb_stride_buf[i] = gb_stride[i]
    for i in range(mask_ndim):
        mask_buf[i] = 1 if output_mask[i] else 0
    cdef void* grad_t = NULL
    cdef void* input_t = NULL
    cdef void* mean_t = NULL
    cdef void* rstd_t = NULL
    cdef void* gamma_t = NULL
    cdef void* gi_t = NULL
    cdef void* gg_t = NULL
    cdef void* gb_t = NULL
    cdef void* mask_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        grad_t = _fast_create_tensor(grad_shape_buf, grad_stride_buf, <uint64_t>grad_ndim, tensor_dtype_code, io_fmt, <void*>grad_ptr)
        input_t = _fast_create_tensor(input_shape_buf, input_stride_buf, <uint64_t>input_ndim, tensor_dtype_code, io_fmt, <void*>input_ptr)
        mean_t = _fast_create_tensor(mean_shape_buf, mean_stride_buf, <uint64_t>mean_ndim, stats_dtype_code, stats_fmt, <void*>mean_ptr)
        rstd_t = _fast_create_tensor(rstd_shape_buf, rstd_stride_buf, <uint64_t>rstd_ndim, stats_dtype_code, stats_fmt, <void*>rstd_ptr)
        if has_gamma:
            gamma_t = _fast_create_tensor(gamma_shape_buf, gamma_stride_buf, <uint64_t>gamma_ndim, tensor_dtype_code, param_fmt, <void*>gamma_ptr)
        gi_t = _fast_create_tensor(gi_shape_buf, gi_stride_buf, <uint64_t>gi_ndim, tensor_dtype_code, io_fmt, <void*>gi_ptr)
        if has_gg:
            gg_t = _fast_create_tensor(gg_shape_buf, gg_stride_buf, <uint64_t>gg_ndim, tensor_dtype_code, param_fmt, <void*>gg_ptr)
        if has_gb:
            gb_t = _fast_create_tensor(gb_shape_buf, gb_stride_buf, <uint64_t>gb_ndim, tensor_dtype_code, param_fmt, <void*>gb_ptr)
        if mask_ndim > 0:
            mask_handle = _fast_create_bool_array(mask_buf, <uint64_t>mask_ndim)
    if grad_t == NULL or input_t == NULL or mean_t == NULL or rstd_t == NULL or gi_t == NULL or (has_gamma and gamma_t == NULL) or (has_gg and gg_t == NULL) or (has_gb and gb_t == NULL) or (mask_ndim > 0 and mask_handle == NULL):
        if grad_t != NULL: _fast_destroy_tensor(grad_t)
        if input_t != NULL: _fast_destroy_tensor(input_t)
        if mean_t != NULL: _fast_destroy_tensor(mean_t)
        if rstd_t != NULL: _fast_destroy_tensor(rstd_t)
        if gamma_t != NULL: _fast_destroy_tensor(gamma_t)
        if gi_t != NULL: _fast_destroy_tensor(gi_t)
        if gg_t != NULL: _fast_destroy_tensor(gg_t)
        if gb_t != NULL: _fast_destroy_tensor(gb_t)
        if mask_handle != NULL: _fast_destroy_bool_array(mask_handle)
        raise RuntimeError("ACLNN group_norm_backward descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, int64_t, int64_t, int64_t, int64_t, void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                grad_t, input_t, mean_t, rstd_t, gamma_t, N, C, HxW, group, mask_handle, gi_t, gg_t, gb_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if mask_handle != NULL: _fast_destroy_bool_array(mask_handle)
            if grad_t != NULL: _fast_destroy_tensor(grad_t)
            if input_t != NULL: _fast_destroy_tensor(input_t)
            if mean_t != NULL: _fast_destroy_tensor(mean_t)
            if rstd_t != NULL: _fast_destroy_tensor(rstd_t)
            if gamma_t != NULL: _fast_destroy_tensor(gamma_t)
            if gi_t != NULL: _fast_destroy_tensor(gi_t)
            if gg_t != NULL: _fast_destroy_tensor(gg_t)
            if gb_t != NULL: _fast_destroy_tensor(gb_t)


def convolution_backward_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        grad_shape, grad_stride,
        input_shape, input_stride,
        weight_shape, weight_stride,
        gi_shape, gi_stride,
        gw_shape, gw_stride,
        gb_shape, gb_stride,
        bias_sizes, stride_values, padding_values, dilation_values, output_padding_values,
        output_mask,
        bint transposed, int64_t groups, int8_t cube_math_type,
        int32_t tensor_dtype_code, int32_t fmt,
        uintptr_t grad_ptr, uintptr_t input_ptr, uintptr_t weight_ptr,
        uintptr_t gi_ptr, uintptr_t gw_ptr, uintptr_t gb_ptr,
        uintptr_t stream):
    cdef int grad_ndim = len(grad_shape)
    cdef int input_ndim = len(input_shape)
    cdef int weight_ndim = len(weight_shape)
    cdef int gi_ndim = 0 if gi_shape is None else len(gi_shape)
    cdef int gw_ndim = 0 if gw_shape is None else len(gw_shape)
    cdef int gb_ndim = 0 if gb_shape is None else len(gb_shape)
    cdef int bias_sizes_ndim = 0 if bias_sizes is None else len(bias_sizes)
    cdef int stride_ndim = len(stride_values)
    cdef int padding_ndim = len(padding_values)
    cdef int dilation_ndim = len(dilation_values)
    cdef int output_padding_ndim = len(output_padding_values)
    cdef int mask_ndim = len(output_mask)
    cdef bint has_gi = gi_ptr != 0 and gi_shape is not None
    cdef bint has_gw = gw_ptr != 0 and gw_shape is not None
    cdef bint has_gb = gb_ptr != 0 and gb_shape is not None
    cdef int64_t[MAX_NDIM] grad_shape_buf, grad_stride_buf
    cdef int64_t[MAX_NDIM] input_shape_buf, input_stride_buf
    cdef int64_t[MAX_NDIM] weight_shape_buf, weight_stride_buf
    cdef int64_t[MAX_NDIM] gi_shape_buf, gi_stride_buf
    cdef int64_t[MAX_NDIM] gw_shape_buf, gw_stride_buf
    cdef int64_t[MAX_NDIM] gb_shape_buf, gb_stride_buf
    cdef int64_t[MAX_NDIM] bias_sizes_buf
    cdef int64_t[MAX_NDIM] stride_buf
    cdef int64_t[MAX_NDIM] padding_buf
    cdef int64_t[MAX_NDIM] dilation_buf
    cdef int64_t[MAX_NDIM] output_padding_buf
    cdef uint8_t[MAX_NDIM] mask_buf
    cdef int i
    for i in range(grad_ndim):
        grad_shape_buf[i] = grad_shape[i]
        grad_stride_buf[i] = grad_stride[i]
    for i in range(input_ndim):
        input_shape_buf[i] = input_shape[i]
        input_stride_buf[i] = input_stride[i]
    for i in range(weight_ndim):
        weight_shape_buf[i] = weight_shape[i]
        weight_stride_buf[i] = weight_stride[i]
    if gi_shape is not None:
        for i in range(gi_ndim):
            gi_shape_buf[i] = gi_shape[i]
            gi_stride_buf[i] = gi_stride[i]
    if gw_shape is not None:
        for i in range(gw_ndim):
            gw_shape_buf[i] = gw_shape[i]
            gw_stride_buf[i] = gw_stride[i]
    if gb_shape is not None:
        for i in range(gb_ndim):
            gb_shape_buf[i] = gb_shape[i]
            gb_stride_buf[i] = gb_stride[i]
    if bias_sizes is not None:
        for i in range(bias_sizes_ndim):
            bias_sizes_buf[i] = bias_sizes[i]
    for i in range(stride_ndim):
        stride_buf[i] = stride_values[i]
    for i in range(padding_ndim):
        padding_buf[i] = padding_values[i]
    for i in range(dilation_ndim):
        dilation_buf[i] = dilation_values[i]
    for i in range(output_padding_ndim):
        output_padding_buf[i] = output_padding_values[i]
    for i in range(mask_ndim):
        mask_buf[i] = 1 if output_mask[i] else 0
    cdef void* grad_t = NULL
    cdef void* input_t = NULL
    cdef void* weight_t = NULL
    cdef void* gi_t = NULL
    cdef void* gw_t = NULL
    cdef void* gb_t = NULL
    cdef void* bias_sizes_handle = NULL
    cdef void* stride_handle = NULL
    cdef void* padding_handle = NULL
    cdef void* dilation_handle = NULL
    cdef void* output_padding_handle = NULL
    cdef void* mask_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        grad_t = _fast_create_tensor(grad_shape_buf, grad_stride_buf, <uint64_t>grad_ndim, tensor_dtype_code, fmt, <void*>grad_ptr)
        input_t = _fast_create_tensor(input_shape_buf, input_stride_buf, <uint64_t>input_ndim, tensor_dtype_code, fmt, <void*>input_ptr)
        weight_t = _fast_create_tensor(weight_shape_buf, weight_stride_buf, <uint64_t>weight_ndim, tensor_dtype_code, fmt, <void*>weight_ptr)
        if has_gi:
            gi_t = _fast_create_tensor(gi_shape_buf, gi_stride_buf, <uint64_t>gi_ndim, tensor_dtype_code, fmt, <void*>gi_ptr)
        if has_gw:
            gw_t = _fast_create_tensor(gw_shape_buf, gw_stride_buf, <uint64_t>gw_ndim, tensor_dtype_code, fmt, <void*>gw_ptr)
        if has_gb:
            gb_t = _fast_create_tensor(gb_shape_buf, gb_stride_buf, <uint64_t>gb_ndim, tensor_dtype_code, fmt, <void*>gb_ptr)
        if bias_sizes_ndim > 0:
            bias_sizes_handle = _fn_create_int_array(bias_sizes_buf, <uint64_t>bias_sizes_ndim)
        if stride_ndim > 0:
            stride_handle = _fn_create_int_array(stride_buf, <uint64_t>stride_ndim)
        if padding_ndim > 0:
            padding_handle = _fn_create_int_array(padding_buf, <uint64_t>padding_ndim)
        if dilation_ndim > 0:
            dilation_handle = _fn_create_int_array(dilation_buf, <uint64_t>dilation_ndim)
        if output_padding_ndim > 0:
            output_padding_handle = _fn_create_int_array(output_padding_buf, <uint64_t>output_padding_ndim)
        if mask_ndim > 0:
            mask_handle = _fast_create_bool_array(mask_buf, <uint64_t>mask_ndim)
    if grad_t == NULL or input_t == NULL or weight_t == NULL or (has_gi and gi_t == NULL) or (has_gw and gw_t == NULL) or (has_gb and gb_t == NULL) or (bias_sizes_ndim > 0 and bias_sizes_handle == NULL) or (stride_ndim > 0 and stride_handle == NULL) or (padding_ndim > 0 and padding_handle == NULL) or (dilation_ndim > 0 and dilation_handle == NULL) or (output_padding_ndim > 0 and output_padding_handle == NULL) or (mask_ndim > 0 and mask_handle == NULL):
        if grad_t != NULL: _fast_destroy_tensor(grad_t)
        if input_t != NULL: _fast_destroy_tensor(input_t)
        if weight_t != NULL: _fast_destroy_tensor(weight_t)
        if gi_t != NULL: _fast_destroy_tensor(gi_t)
        if gw_t != NULL: _fast_destroy_tensor(gw_t)
        if gb_t != NULL: _fast_destroy_tensor(gb_t)
        if bias_sizes_handle != NULL: _fn_destroy_int_array(bias_sizes_handle)
        if stride_handle != NULL: _fn_destroy_int_array(stride_handle)
        if padding_handle != NULL: _fn_destroy_int_array(padding_handle)
        if dilation_handle != NULL: _fn_destroy_int_array(dilation_handle)
        if output_padding_handle != NULL: _fn_destroy_int_array(output_padding_handle)
        if mask_handle != NULL: _fast_destroy_bool_array(mask_handle)
        raise RuntimeError("ACLNN convolution_backward descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, void*, void*, bint, void*, int64_t, void*, int8_t, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                grad_t,
                input_t,
                weight_t,
                bias_sizes_handle,
                stride_handle,
                padding_handle,
                dilation_handle,
                transposed,
                output_padding_handle,
                groups,
                mask_handle,
                cube_math_type,
                gi_t,
                gw_t,
                gb_t,
                &ws_size,
                &executor,
            )
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if bias_sizes_handle != NULL: _fn_destroy_int_array(bias_sizes_handle)
            if stride_handle != NULL: _fn_destroy_int_array(stride_handle)
            if padding_handle != NULL: _fn_destroy_int_array(padding_handle)
            if dilation_handle != NULL: _fn_destroy_int_array(dilation_handle)
            if output_padding_handle != NULL: _fn_destroy_int_array(output_padding_handle)
            if mask_handle != NULL: _fast_destroy_bool_array(mask_handle)
            if grad_t != NULL: _fast_destroy_tensor(grad_t)
            if input_t != NULL: _fast_destroy_tensor(input_t)
            if weight_t != NULL: _fast_destroy_tensor(weight_t)
            if gi_t != NULL: _fast_destroy_tensor(gi_t)
            if gw_t != NULL: _fast_destroy_tensor(gw_t)
            if gb_t != NULL: _fast_destroy_tensor(gb_t)

def grid_sampler2d_backward_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        grad_shape, grad_stride,
        input_shape, input_stride,
        grid_shape, grid_stride,
        ig_shape, ig_stride,
        gg_shape, gg_stride,
        output_mask,
        int64_t interpolation_mode, int64_t padding_mode, bint align_corners,
        int32_t tensor_dtype_code,
        uintptr_t grad_ptr, uintptr_t input_ptr, uintptr_t grid_ptr,
        uintptr_t ig_ptr, uintptr_t gg_ptr,
        uintptr_t stream):
    cdef int grad_ndim = len(grad_shape)
    cdef int input_ndim = len(input_shape)
    cdef int grid_ndim = len(grid_shape)
    cdef int ig_ndim = len(ig_shape)
    cdef int gg_ndim = len(gg_shape)
    cdef int mask_ndim = len(output_mask)
    cdef int64_t[MAX_NDIM] grad_shape_buf, grad_stride_buf
    cdef int64_t[MAX_NDIM] input_shape_buf, input_stride_buf
    cdef int64_t[MAX_NDIM] grid_shape_buf, grid_stride_buf
    cdef int64_t[MAX_NDIM] ig_shape_buf, ig_stride_buf
    cdef int64_t[MAX_NDIM] gg_shape_buf, gg_stride_buf
    cdef uint8_t[MAX_NDIM] mask_buf
    cdef int i
    for i in range(grad_ndim):
        grad_shape_buf[i] = grad_shape[i]
        grad_stride_buf[i] = grad_stride[i]
    for i in range(input_ndim):
        input_shape_buf[i] = input_shape[i]
        input_stride_buf[i] = input_stride[i]
    for i in range(grid_ndim):
        grid_shape_buf[i] = grid_shape[i]
        grid_stride_buf[i] = grid_stride[i]
    for i in range(ig_ndim):
        ig_shape_buf[i] = ig_shape[i]
        ig_stride_buf[i] = ig_stride[i]
    for i in range(gg_ndim):
        gg_shape_buf[i] = gg_shape[i]
        gg_stride_buf[i] = gg_stride[i]
    for i in range(mask_ndim):
        mask_buf[i] = 1 if output_mask[i] else 0
    cdef void* grad_t = NULL
    cdef void* input_t = NULL
    cdef void* grid_t = NULL
    cdef void* ig_t = NULL
    cdef void* gg_t = NULL
    cdef void* mask_handle = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        grad_t = _fast_create_tensor(grad_shape_buf, grad_stride_buf, <uint64_t>grad_ndim, tensor_dtype_code, 0, <void*>grad_ptr)
        input_t = _fast_create_tensor(input_shape_buf, input_stride_buf, <uint64_t>input_ndim, tensor_dtype_code, 0, <void*>input_ptr)
        grid_t = _fast_create_tensor(grid_shape_buf, grid_stride_buf, <uint64_t>grid_ndim, tensor_dtype_code, 2, <void*>grid_ptr)
        ig_t = _fast_create_tensor(ig_shape_buf, ig_stride_buf, <uint64_t>ig_ndim, tensor_dtype_code, 0, <void*>ig_ptr)
        gg_t = _fast_create_tensor(gg_shape_buf, gg_stride_buf, <uint64_t>gg_ndim, tensor_dtype_code, 2, <void*>gg_ptr)
        if mask_ndim > 0:
            mask_handle = _fast_create_bool_array(mask_buf, <uint64_t>mask_ndim)
    if grad_t == NULL or input_t == NULL or grid_t == NULL or ig_t == NULL or gg_t == NULL or (mask_ndim > 0 and mask_handle == NULL):
        if grad_t != NULL: _fast_destroy_tensor(grad_t)
        if input_t != NULL: _fast_destroy_tensor(input_t)
        if grid_t != NULL: _fast_destroy_tensor(grid_t)
        if ig_t != NULL: _fast_destroy_tensor(ig_t)
        if gg_t != NULL: _fast_destroy_tensor(gg_t)
        if mask_handle != NULL: _fast_destroy_bool_array(mask_handle)
        raise RuntimeError("ACLNN grid_sampler2d_backward descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, int64_t, int64_t, bint, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                grad_t, input_t, grid_t, interpolation_mode, padding_mode, align_corners, mask_handle, ig_t, gg_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if mask_handle != NULL: _fast_destroy_bool_array(mask_handle)
            if grad_t != NULL: _fast_destroy_tensor(grad_t)
            if input_t != NULL: _fast_destroy_tensor(input_t)
            if grid_t != NULL: _fast_destroy_tensor(grid_t)
            if ig_t != NULL: _fast_destroy_tensor(ig_t)
            if gg_t != NULL: _fast_destroy_tensor(gg_t)

def three_tensor_two_outputs_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        c_shape, c_stride,
        out_a_shape, out_a_stride,
        out_b_shape, out_b_stride,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t c_dtype_code,
        int32_t out_a_dtype_code, int32_t out_b_dtype_code, int32_t fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t c_ptr,
        uintptr_t out_a_ptr, uintptr_t out_b_ptr,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int c_ndim = len(c_shape)
    cdef int out_a_ndim = len(out_a_shape)
    cdef int out_b_ndim = len(out_b_shape)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] c_shape_buf, c_stride_buf
    cdef int64_t[MAX_NDIM] out_a_shape_buf, out_a_stride_buf
    cdef int64_t[MAX_NDIM] out_b_shape_buf, out_b_stride_buf
    cdef int i
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(c_ndim):
        c_shape_buf[i] = c_shape[i]
        c_stride_buf[i] = c_stride[i]
    for i in range(out_a_ndim):
        out_a_shape_buf[i] = out_a_shape[i]
        out_a_stride_buf[i] = out_a_stride[i]
    for i in range(out_b_ndim):
        out_b_shape_buf[i] = out_b_shape[i]
        out_b_stride_buf[i] = out_b_stride[i]
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* c_t = NULL
    cdef void* out_a_t = NULL
    cdef void* out_b_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        c_t = _fast_create_tensor(c_shape_buf, c_stride_buf, <uint64_t>c_ndim, c_dtype_code, fmt, <void*>c_ptr)
        out_a_t = _fast_create_tensor(out_a_shape_buf, out_a_stride_buf, <uint64_t>out_a_ndim, out_a_dtype_code, fmt, <void*>out_a_ptr)
        out_b_t = _fast_create_tensor(out_b_shape_buf, out_b_stride_buf, <uint64_t>out_b_ndim, out_b_dtype_code, fmt, <void*>out_b_ptr)
    if a_t == NULL or b_t == NULL or c_t == NULL or out_a_t == NULL or out_b_t == NULL:
        if a_t != NULL:
            _fast_destroy_tensor(a_t)
        if b_t != NULL:
            _fast_destroy_tensor(b_t)
        if c_t != NULL:
            _fast_destroy_tensor(c_t)
        if out_a_t != NULL:
            _fast_destroy_tensor(out_a_t)
        if out_b_t != NULL:
            _fast_destroy_tensor(out_b_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, b_t, c_t, out_a_t, out_b_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>a_t)] if a_t != NULL else [])
            + ([('t', <uintptr_t>b_t)] if b_t != NULL else [])
            + ([('t', <uintptr_t>c_t)] if c_t != NULL else [])
            + ([('t', <uintptr_t>out_a_t)] if out_a_t != NULL else [])
            + ([('t', <uintptr_t>out_b_t)] if out_b_t != NULL else []),
        )
        a_t = NULL
        b_t = NULL
        c_t = NULL
        out_a_t = NULL
        out_b_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if a_t != NULL:
                _fast_destroy_tensor(a_t)
            if b_t != NULL:
                _fast_destroy_tensor(b_t)
            if c_t != NULL:
                _fast_destroy_tensor(c_t)
            if out_a_t != NULL:
                _fast_destroy_tensor(out_a_t)
            if out_b_t != NULL:
                _fast_destroy_tensor(out_b_t)
def two_tensor_scalar_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        out_shape, out_stride,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t out_ptr,
        uintptr_t scalar_handle,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if a_t == NULL or b_t == NULL or out_t == NULL:
        if a_t != NULL:
            _fast_destroy_tensor(a_t)
        if b_t != NULL:
            _fast_destroy_tensor(b_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, b_t, <void*>scalar_handle, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>a_t)] if a_t != NULL else [])
            + ([('t', <uintptr_t>b_t)] if b_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        a_t = NULL
        b_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if a_t != NULL:
                _fast_destroy_tensor(a_t)
            if b_t != NULL:
                _fast_destroy_tensor(b_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
def two_tensor_one_double_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        out_shape, out_stride,
        double scalar_value,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if a_t == NULL or b_t == NULL or out_t == NULL:
        if a_t != NULL:
            _fast_destroy_tensor(a_t)
        if b_t != NULL:
            _fast_destroy_tensor(b_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, double, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, b_t, scalar_value, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>a_t)] if a_t != NULL else [])
            + ([('t', <uintptr_t>b_t)] if b_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        a_t = NULL
        b_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if a_t != NULL:
                _fast_destroy_tensor(a_t)
            if b_t != NULL:
                _fast_destroy_tensor(b_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
def three_tensor_scalar_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        c_shape, c_stride,
        out_shape, out_stride,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t c_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t c_ptr, uintptr_t out_ptr,
        uintptr_t scalar_handle,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int c_ndim = len(c_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] c_shape_buf, c_stride_buf
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(c_ndim):
        c_shape_buf[i] = c_shape[i]
        c_stride_buf[i] = c_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* c_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        c_t = _fast_create_tensor(c_shape_buf, c_stride_buf, <uint64_t>c_ndim, c_dtype_code, fmt, <void*>c_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if a_t == NULL or b_t == NULL or c_t == NULL or out_t == NULL:
        if a_t != NULL:
            _fast_destroy_tensor(a_t)
        if b_t != NULL:
            _fast_destroy_tensor(b_t)
        if c_t != NULL:
            _fast_destroy_tensor(c_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, b_t, c_t, <void*>scalar_handle, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>a_t)] if a_t != NULL else [])
            + ([('t', <uintptr_t>b_t)] if b_t != NULL else [])
            + ([('t', <uintptr_t>c_t)] if c_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        a_t = NULL
        b_t = NULL
        c_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if a_t != NULL:
                _fast_destroy_tensor(a_t)
            if b_t != NULL:
                _fast_destroy_tensor(b_t)
            if c_t != NULL:
                _fast_destroy_tensor(c_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)
def six_tensor_string_double_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        c_shape, c_stride,
        out_shape, out_stride,
        stats_a_shape, stats_a_stride,
        stats_b_shape, stats_b_stride,
        bytes string_value, double scalar_value,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t c_dtype_code,
        int32_t out_dtype_code, int32_t stats_a_dtype_code, int32_t stats_b_dtype_code,
        int32_t a_fmt, int32_t bc_fmt, int32_t out_fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t c_ptr, uintptr_t out_ptr, uintptr_t stats_a_ptr, uintptr_t stats_b_ptr,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int c_ndim = len(c_shape)
    cdef int out_ndim = len(out_shape)
    cdef int stats_a_ndim = len(stats_a_shape)
    cdef int stats_b_ndim = len(stats_b_shape)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] c_shape_buf, c_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int64_t[MAX_NDIM] stats_a_shape_buf, stats_a_stride_buf
    cdef int64_t[MAX_NDIM] stats_b_shape_buf, stats_b_stride_buf
    cdef int i
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(c_ndim):
        c_shape_buf[i] = c_shape[i]
        c_stride_buf[i] = c_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    for i in range(stats_a_ndim):
        stats_a_shape_buf[i] = stats_a_shape[i]
        stats_a_stride_buf[i] = stats_a_stride[i]
    for i in range(stats_b_ndim):
        stats_b_shape_buf[i] = stats_b_shape[i]
        stats_b_stride_buf[i] = stats_b_stride[i]
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* c_t = NULL
    cdef void* out_t = NULL
    cdef void* stats_a_t = NULL
    cdef void* stats_b_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    cdef const char* string_ptr = string_value
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, a_fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, bc_fmt, <void*>b_ptr)
        c_t = _fast_create_tensor(c_shape_buf, c_stride_buf, <uint64_t>c_ndim, c_dtype_code, bc_fmt, <void*>c_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, out_fmt, <void*>out_ptr)
        stats_a_t = _fast_create_tensor(stats_a_shape_buf, stats_a_stride_buf, <uint64_t>stats_a_ndim, stats_a_dtype_code, bc_fmt, <void*>stats_a_ptr)
        stats_b_t = _fast_create_tensor(stats_b_shape_buf, stats_b_stride_buf, <uint64_t>stats_b_ndim, stats_b_dtype_code, bc_fmt, <void*>stats_b_ptr)
    if a_t == NULL or b_t == NULL or c_t == NULL or out_t == NULL or stats_a_t == NULL or stats_b_t == NULL:
        if a_t != NULL: _fast_destroy_tensor(a_t)
        if b_t != NULL: _fast_destroy_tensor(b_t)
        if c_t != NULL: _fast_destroy_tensor(c_t)
        if out_t != NULL: _fast_destroy_tensor(out_t)
        if stats_a_t != NULL: _fast_destroy_tensor(stats_a_t)
        if stats_b_t != NULL: _fast_destroy_tensor(stats_b_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, const char*, double, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, b_t, c_t, string_ptr, scalar_value, out_t, stats_a_t, stats_b_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(a_t)
            _fast_destroy_tensor(b_t)
            _fast_destroy_tensor(c_t)
            _fast_destroy_tensor(out_t)
            _fast_destroy_tensor(stats_a_t)
            _fast_destroy_tensor(stats_b_t)


def six_tensor_five_floats_two_bools_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        a_shape, a_stride,
        b_shape, b_stride,
        c_shape, c_stride,
        d_shape, d_stride,
        e_shape, e_stride,
        f_shape, f_stride,
        double fa, double fb, double fc, double fd, double fe,
        bint flag_a, bint flag_b,
        int32_t a_dtype_code, int32_t b_dtype_code, int32_t c_dtype_code,
        int32_t d_dtype_code, int32_t e_dtype_code, int32_t f_dtype_code,
        int32_t fmt,
        uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t c_ptr, uintptr_t d_ptr, uintptr_t e_ptr, uintptr_t f_ptr,
        uintptr_t stream):
    cdef int a_ndim = len(a_shape)
    cdef int b_ndim = len(b_shape)
    cdef int c_ndim = len(c_shape)
    cdef int d_ndim = len(d_shape)
    cdef int e_ndim = len(e_shape)
    cdef int f_ndim = len(f_shape)
    cdef int64_t[MAX_NDIM] a_shape_buf, a_stride_buf
    cdef int64_t[MAX_NDIM] b_shape_buf, b_stride_buf
    cdef int64_t[MAX_NDIM] c_shape_buf, c_stride_buf
    cdef int64_t[MAX_NDIM] d_shape_buf, d_stride_buf
    cdef int64_t[MAX_NDIM] e_shape_buf, e_stride_buf
    cdef int64_t[MAX_NDIM] f_shape_buf, f_stride_buf
    cdef int i
    for i in range(a_ndim):
        a_shape_buf[i] = a_shape[i]
        a_stride_buf[i] = a_stride[i]
    for i in range(b_ndim):
        b_shape_buf[i] = b_shape[i]
        b_stride_buf[i] = b_stride[i]
    for i in range(c_ndim):
        c_shape_buf[i] = c_shape[i]
        c_stride_buf[i] = c_stride[i]
    for i in range(d_ndim):
        d_shape_buf[i] = d_shape[i]
        d_stride_buf[i] = d_stride[i]
    for i in range(e_ndim):
        e_shape_buf[i] = e_shape[i]
        e_stride_buf[i] = e_stride[i]
    for i in range(f_ndim):
        f_shape_buf[i] = f_shape[i]
        f_stride_buf[i] = f_stride[i]
    cdef void* a_t = NULL
    cdef void* b_t = NULL
    cdef void* c_t = NULL
    cdef void* d_t = NULL
    cdef void* e_t = NULL
    cdef void* f_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        a_t = _fast_create_tensor(a_shape_buf, a_stride_buf, <uint64_t>a_ndim, a_dtype_code, fmt, <void*>a_ptr)
        b_t = _fast_create_tensor(b_shape_buf, b_stride_buf, <uint64_t>b_ndim, b_dtype_code, fmt, <void*>b_ptr)
        c_t = _fast_create_tensor(c_shape_buf, c_stride_buf, <uint64_t>c_ndim, c_dtype_code, fmt, <void*>c_ptr)
        d_t = NULL if d_ptr == 0 else _fast_create_tensor(d_shape_buf, d_stride_buf, <uint64_t>d_ndim, d_dtype_code, fmt, <void*>d_ptr)
        e_t = _fast_create_tensor(e_shape_buf, e_stride_buf, <uint64_t>e_ndim, e_dtype_code, fmt, <void*>e_ptr)
        f_t = _fast_create_tensor(f_shape_buf, f_stride_buf, <uint64_t>f_ndim, f_dtype_code, fmt, <void*>f_ptr)
    if a_t == NULL or b_t == NULL or c_t == NULL or e_t == NULL or f_t == NULL or (d_ptr != 0 and d_t == NULL):
        if a_t != NULL: _fast_destroy_tensor(a_t)
        if b_t != NULL: _fast_destroy_tensor(b_t)
        if c_t != NULL: _fast_destroy_tensor(c_t)
        if d_t != NULL: _fast_destroy_tensor(d_t)
        if e_t != NULL: _fast_destroy_tensor(e_t)
        if f_t != NULL: _fast_destroy_tensor(f_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, void*, void*, float, float, float, float, float, bint, bint, uint64_t*, void**) noexcept nogil>getws_ptr)(
                a_t, b_t, c_t, d_t, e_t, f_t, <float>fa, <float>fb, <float>fc, <float>fd, <float>fe, flag_a, flag_b, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(a_t)
            _fast_destroy_tensor(b_t)
            _fast_destroy_tensor(c_t)
            if d_t != NULL:
                _fast_destroy_tensor(d_t)
            _fast_destroy_tensor(e_t)
            _fast_destroy_tensor(f_t)


def where_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        cond_shape, cond_stride,
        self_shape, self_stride,
        other_shape, other_stride,
        out_shape, out_stride,
        int32_t cond_dtype_code, int32_t self_dtype_code, int32_t other_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t cond_ptr, uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int cond_ndim = len(cond_shape)
    cdef int self_ndim = len(self_shape)
    cdef int other_ndim = len(other_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] c_shape, c_stride
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] o_shape, o_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(cond_ndim):
        c_shape[i] = cond_shape[i]
        c_stride[i] = cond_stride[i]
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(other_ndim):
        o_shape[i] = other_shape[i]
        o_stride[i] = other_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* cond_t = NULL
    cdef void* self_t = NULL
    cdef void* other_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        cond_t = _fast_create_tensor(c_shape, c_stride, <uint64_t>cond_ndim, cond_dtype_code, fmt, <void*>cond_ptr)
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        other_t = _fast_create_tensor(o_shape, o_stride, <uint64_t>other_ndim, other_dtype_code, fmt, <void*>other_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if cond_t == NULL or self_t == NULL or other_t == NULL or out_t == NULL:
        if cond_t != NULL:
            _fast_destroy_tensor(cond_t)
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if other_t != NULL:
            _fast_destroy_tensor(other_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                cond_t, self_t, other_t, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(cond_t)
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(other_t)
            _fast_destroy_tensor(out_t)


def clamp_optional_scalars_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int32_t dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t min_scalar, uintptr_t max_scalar,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, <void*>min_scalar, <void*>max_scalar, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(out_t)


def clamp_tensor_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        min_shape, min_stride,
        max_shape, max_stride,
        out_shape, out_stride,
        int32_t dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t min_ptr, uintptr_t max_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int min_ndim = len(min_shape)
    cdef int max_ndim = len(max_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] min_shape_buf, min_stride_buf
    cdef int64_t[MAX_NDIM] max_shape_buf, max_stride_buf
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(min_ndim):
        min_shape_buf[i] = min_shape[i]
        min_stride_buf[i] = min_stride[i]
    for i in range(max_ndim):
        max_shape_buf[i] = max_shape[i]
        max_stride_buf[i] = max_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* min_t = NULL
    cdef void* max_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, dtype_code, fmt, <void*>self_ptr)
        min_t = _fast_create_tensor(min_shape_buf, min_stride_buf, <uint64_t>min_ndim, dtype_code, fmt, <void*>min_ptr)
        max_t = _fast_create_tensor(max_shape_buf, max_stride_buf, <uint64_t>max_ndim, dtype_code, fmt, <void*>max_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or min_t == NULL or max_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if min_t != NULL:
            _fast_destroy_tensor(min_t)
        if max_t != NULL:
            _fast_destroy_tensor(max_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, min_t, max_t, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(min_t)
            _fast_destroy_tensor(max_t)
            _fast_destroy_tensor(out_t)


def tensor_two_scalars_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t scalar_a, uintptr_t scalar_b,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, <void*>scalar_a, <void*>scalar_b, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(out_t)


def binary_two_inputs_with_dim_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        other_shape, other_stride,
        out_shape, out_stride,
        int64_t dim,
        int32_t self_dtype_code, int32_t other_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
        uintptr_t stream):
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
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        other_t = _fast_create_tensor(o_shape, o_stride, <uint64_t>other_ndim, other_dtype_code, fmt, <void*>other_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or other_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if other_t != NULL:
            _fast_destroy_tensor(other_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, int64_t, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, dim, other_t, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>other_t)] if other_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        other_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if other_t != NULL:
                _fast_destroy_tensor(other_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def binary_two_inputs_with_int8_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        other_shape, other_stride,
        out_shape, out_stride,
        int8_t extra_flag,
        int32_t self_dtype_code, int32_t other_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
        uintptr_t stream):
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
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        other_t = _fast_create_tensor(o_shape, o_stride, <uint64_t>other_ndim, other_dtype_code, fmt, <void*>other_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or other_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if other_t != NULL:
            _fast_destroy_tensor(other_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, int8_t, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, other_t, out_t, <int8_t>extra_flag, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(other_t)
            _fast_destroy_tensor(out_t)


def two_tensor_two_ints_bool_mixed_fmt_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        first_shape, first_stride,
        second_shape, second_stride,
        out_shape, out_stride,
        int64_t value_a, int64_t value_b, bint flag,
        int32_t first_dtype_code, int32_t second_dtype_code, int32_t out_dtype_code,
        int32_t first_fmt, int32_t second_fmt, int32_t out_fmt,
        uintptr_t first_ptr, uintptr_t second_ptr, uintptr_t out_ptr,
        uintptr_t stream):
    cdef int first_ndim = len(first_shape)
    cdef int second_ndim = len(second_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] first_shape_buf, first_stride_buf
    cdef int64_t[MAX_NDIM] second_shape_buf, second_stride_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int i
    for i in range(first_ndim):
        first_shape_buf[i] = first_shape[i]
        first_stride_buf[i] = first_stride[i]
    for i in range(second_ndim):
        second_shape_buf[i] = second_shape[i]
        second_stride_buf[i] = second_stride[i]
    for i in range(out_ndim):
        out_shape_buf[i] = out_shape[i]
        out_stride_buf[i] = out_stride[i]
    cdef void* first_t = NULL
    cdef void* second_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        first_t = _fast_create_tensor(first_shape_buf, first_stride_buf, <uint64_t>first_ndim, first_dtype_code, first_fmt, <void*>first_ptr)
        second_t = _fast_create_tensor(second_shape_buf, second_stride_buf, <uint64_t>second_ndim, second_dtype_code, second_fmt, <void*>second_ptr)
        out_t = _fast_create_tensor(out_shape_buf, out_stride_buf, <uint64_t>out_ndim, out_dtype_code, out_fmt, <void*>out_ptr)
    if first_t == NULL or second_t == NULL or out_t == NULL:
        if first_t != NULL:
            _fast_destroy_tensor(first_t)
        if second_t != NULL:
            _fast_destroy_tensor(second_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, int64_t, int64_t, bint, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                first_t, second_t, value_a, value_b, flag, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(first_t)
            _fast_destroy_tensor(second_t)
            _fast_destroy_tensor(out_t)


def binary_two_inputs_three_attrs_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        other_shape, other_stride,
        out_shape, out_stride,
        double attr0, double attr1, bint attr2,
        int32_t self_dtype_code, int32_t other_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
        uintptr_t stream):
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
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        other_t = _fast_create_tensor(o_shape, o_stride, <uint64_t>other_ndim, other_dtype_code, fmt, <void*>other_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or other_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if other_t != NULL:
            _fast_destroy_tensor(other_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, double, double, bint, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, other_t, attr0, attr1, attr2, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(other_t)
            _fast_destroy_tensor(out_t)


def binary_two_inputs_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        other_shape, other_stride,
        out_shape, out_stride,
        int32_t self_dtype_code, int32_t other_dtype_code, int32_t out_dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
        uintptr_t stream):
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
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, self_dtype_code, fmt, <void*>self_ptr)
        other_t = _fast_create_tensor(o_shape, o_stride, <uint64_t>other_ndim, other_dtype_code, fmt, <void*>other_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, out_dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or other_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if other_t != NULL:
            _fast_destroy_tensor(other_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, other_t, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        _register_executor_cleanup(
            <uintptr_t>executor,
            ([('t', <uintptr_t>self_t)] if self_t != NULL else [])
            + ([('t', <uintptr_t>other_t)] if other_t != NULL else [])
            + ([('t', <uintptr_t>out_t)] if out_t != NULL else []),
        )
        self_t = NULL
        other_t = NULL
        out_t = NULL
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            if self_t != NULL:
                _fast_destroy_tensor(self_t)
            if other_t != NULL:
                _fast_destroy_tensor(other_t)
            if out_t != NULL:
                _fast_destroy_tensor(out_t)


def leaky_relu_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        self_shape, self_stride,
        out_shape, out_stride,
        int32_t dtype_code, int32_t fmt,
        uintptr_t self_ptr, uintptr_t out_ptr,
        uintptr_t slope_scalar,
        uintptr_t stream):
    cdef int self_ndim = len(self_shape)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] s_shape, s_stride
    cdef int64_t[MAX_NDIM] r_shape, r_stride
    cdef int i
    for i in range(self_ndim):
        s_shape[i] = self_shape[i]
        s_stride[i] = self_stride[i]
    for i in range(out_ndim):
        r_shape[i] = out_shape[i]
        r_stride[i] = out_stride[i]
    cdef void* self_t = NULL
    cdef void* out_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        self_t = _fast_create_tensor(s_shape, s_stride, <uint64_t>self_ndim, dtype_code, fmt, <void*>self_ptr)
        out_t = _fast_create_tensor(r_shape, r_stride, <uint64_t>out_ndim, dtype_code, fmt, <void*>out_ptr)
    if self_t == NULL or out_t == NULL:
        if self_t != NULL:
            _fast_destroy_tensor(self_t)
        if out_t != NULL:
            _fast_destroy_tensor(out_t)
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                self_t, <void*>slope_scalar, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(self_t)
            _fast_destroy_tensor(out_t)


def rms_norm_op(
        uintptr_t getws_ptr, uintptr_t exec_ptr,
        x_shape, x_stride,
        gamma_shape, gamma_stride,
        y_shape, y_stride,
        rstd_shape, rstd_stride,
        double eps,
        int32_t dtype_code, int32_t fmt,
        uintptr_t x_ptr, uintptr_t gamma_ptr, uintptr_t y_ptr, uintptr_t rstd_ptr,
        uintptr_t stream):
    cdef int x_ndim = len(x_shape)
    cdef int gamma_ndim = len(gamma_shape)
    cdef int y_ndim = len(y_shape)
    cdef int rstd_ndim = len(rstd_shape)
    cdef int64_t[MAX_NDIM] x_shape_buf, x_stride_buf
    cdef int64_t[MAX_NDIM] gamma_shape_buf, gamma_stride_buf
    cdef int64_t[MAX_NDIM] y_shape_buf, y_stride_buf
    cdef int64_t[MAX_NDIM] rstd_shape_buf, rstd_stride_buf
    cdef int i
    for i in range(x_ndim):
        x_shape_buf[i] = x_shape[i]
        x_stride_buf[i] = x_stride[i]
    for i in range(gamma_ndim):
        gamma_shape_buf[i] = gamma_shape[i]
        gamma_stride_buf[i] = gamma_stride[i]
    for i in range(y_ndim):
        y_shape_buf[i] = y_shape[i]
        y_stride_buf[i] = y_stride[i]
    for i in range(rstd_ndim):
        rstd_shape_buf[i] = rstd_shape[i]
        rstd_stride_buf[i] = rstd_stride[i]
    cdef void* x_t = NULL
    cdef void* gamma_t = NULL
    cdef void* y_t = NULL
    cdef void* rstd_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef int32_t ret
    with nogil:
        x_t = _fast_create_tensor(x_shape_buf, x_stride_buf, <uint64_t>x_ndim, dtype_code, fmt, <void*>x_ptr)
        gamma_t = _fast_create_tensor(gamma_shape_buf, gamma_stride_buf, <uint64_t>gamma_ndim, dtype_code, fmt, <void*>gamma_ptr)
        y_t = _fast_create_tensor(y_shape_buf, y_stride_buf, <uint64_t>y_ndim, dtype_code, fmt, <void*>y_ptr)
        rstd_t = _fast_create_tensor(rstd_shape_buf, rstd_stride_buf, <uint64_t>rstd_ndim, 0, fmt, <void*>rstd_ptr)
    if x_t == NULL or gamma_t == NULL or y_t == NULL or rstd_t == NULL:
        if x_t != NULL:
            _fast_destroy_tensor(x_t)
        if gamma_t != NULL:
            _fast_destroy_tensor(gamma_t)
        if y_t != NULL:
            _fast_destroy_tensor(y_t)
        if rstd_t != NULL:
            _fast_destroy_tensor(rstd_t)
        raise RuntimeError("ACLNN rms_norm descriptor creation failed")
    try:
        with nogil:
            ret = (<int32_t (*)(void*, void*, double, void*, void*, uint64_t*, void**) noexcept nogil>getws_ptr)(
                x_t, gamma_t, eps, y_t, rstd_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: {ret}")
        if ws_size == 0:
            try:
                with nogil:
                    ret = (<aclnnExec_t>exec_ptr)(NULL, 0, executor, <void*>stream)
                if ret != 0:
                    raise RuntimeError(f"Execute failed: {ret}")
                _release_executor_cleanup(<uintptr_t>executor)
                executor = NULL
            except Exception:
                destroy_executor(<uintptr_t>executor)
                executor = NULL
                raise
        return (ws_size, <uintptr_t>executor)
    finally:
        with nogil:
            _fast_destroy_tensor(x_t)
            _fast_destroy_tensor(gamma_t)
            _fast_destroy_tensor(y_t)
            _fast_destroy_tensor(rstd_t)
