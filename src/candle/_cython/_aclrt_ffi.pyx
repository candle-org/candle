# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython FFI for ACL runtime: stream/event management + aclmdlRI graph capture.

Loads libascendcl.so via dlopen/dlsym at runtime.  Provides Python-visible
wrappers for stream, event, and aclmdlRI APIs.  The aclgraph state machine
in _aclgraph.pyx cimports the cdef functions declared in _aclrt_ffi.pxd.
"""

from libc.stdint cimport int32_t, uint32_t, uint64_t, uintptr_t
from libc.string cimport memset

cdef extern from "dlfcn.h":
    void* dlopen(const char* filename, int flags) nogil
    void* dlsym(void* handle, const char* symbol) nogil
    char* dlerror() nogil
    int RTLD_LAZY
    int RTLD_GLOBAL

# ---------------------------------------------------------------------------
# Enum constants (exported to Python)
# ---------------------------------------------------------------------------

ACL_MODEL_RI_CAPTURE_MODE_GLOBAL = 0
ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL = 1
ACL_MODEL_RI_CAPTURE_MODE_RELAXED = 2

ACL_MODEL_RI_CAPTURE_STATUS_NONE = 0
ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE = 1
ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED = 2

ACL_EVENT_STATUS_COMPLETE = 0
ACL_EVENT_STATUS_NOT_READY = 1

# ---------------------------------------------------------------------------
# Function pointer typedefs — stream/event
# ---------------------------------------------------------------------------

ctypedef int32_t (*fn_aclrtCreateStream_t)(void**) noexcept nogil
ctypedef int32_t (*fn_aclrtCreateStreamWithConfig_t)(void**, uint32_t, uint32_t) noexcept nogil
ctypedef int32_t (*fn_aclrtDestroyStream_t)(void*) noexcept nogil
ctypedef int32_t (*fn_aclrtSynchronizeStream_t)(void*) noexcept nogil
ctypedef int32_t (*fn_aclrtStreamWaitEvent_t)(void*, void*) noexcept nogil

ctypedef int32_t (*fn_aclrtCreateEvent_t)(void**) noexcept nogil
ctypedef int32_t (*fn_aclrtCreateEventWithFlag_t)(void**, uint32_t) noexcept nogil
ctypedef int32_t (*fn_aclrtDestroyEvent_t)(void*) noexcept nogil
ctypedef int32_t (*fn_aclrtRecordEvent_t)(void*, void*) noexcept nogil
ctypedef int32_t (*fn_aclrtQueryEvent_t)(void*, int32_t*) noexcept nogil
ctypedef int32_t (*fn_aclrtSynchronizeEvent_t)(void*) noexcept nogil
ctypedef int32_t (*fn_aclrtEventElapsedTime_t)(float*, void*, void*) noexcept nogil
ctypedef int32_t (*fn_aclrtSynchronizeDevice_t)() noexcept nogil

# ---------------------------------------------------------------------------
# Function pointer typedefs — aclmdlRI capture/replay
# ---------------------------------------------------------------------------

ctypedef int32_t (*fn_aclmdlRICaptureBegin_t)(void*, int32_t) noexcept nogil
ctypedef int32_t (*fn_aclmdlRICaptureEnd_t)(void*, void**) noexcept nogil
ctypedef int32_t (*fn_aclmdlRICaptureGetInfo_t)(void*, int32_t*, void**) noexcept nogil
ctypedef int32_t (*fn_aclmdlRICaptureThreadExchangeMode_t)(int32_t*) noexcept nogil
ctypedef int32_t (*fn_aclmdlRIExecuteAsync_t)(void*, void*) noexcept nogil
ctypedef int32_t (*fn_aclmdlRIExecute_t)(void*, int32_t) noexcept nogil
ctypedef int32_t (*fn_aclmdlRIDestroy_t)(void*) noexcept nogil
ctypedef int32_t (*fn_aclmdlRISetName_t)(void*, const char*) noexcept nogil
ctypedef int32_t (*fn_aclmdlRIGetName_t)(void*, uint32_t, char*) noexcept nogil
ctypedef int32_t (*fn_aclmdlRIDebugJsonPrint_t)(void*, const char*, uint32_t) noexcept nogil
ctypedef int32_t (*fn_aclmdlRIAbort_t)(void*) noexcept nogil

# Task group (bound now, unused — future-proofing)
ctypedef int32_t (*fn_aclmdlRICaptureTaskGrpBegin_t)(void*) noexcept nogil
ctypedef int32_t (*fn_aclmdlRICaptureTaskGrpEnd_t)(void*, void**) noexcept nogil
ctypedef int32_t (*fn_aclmdlRICaptureTaskUpdateBegin_t)(void*, void*) noexcept nogil
ctypedef int32_t (*fn_aclmdlRICaptureTaskUpdateEnd_t)(void*) noexcept nogil

# ---------------------------------------------------------------------------
# Cached function pointers
# ---------------------------------------------------------------------------

# Stream/event
cdef fn_aclrtCreateStream_t             _fn_create_stream = NULL
cdef fn_aclrtCreateStreamWithConfig_t   _fn_create_stream_cfg = NULL
cdef fn_aclrtDestroyStream_t            _fn_destroy_stream = NULL
cdef fn_aclrtSynchronizeStream_t        _fn_sync_stream = NULL
cdef fn_aclrtStreamWaitEvent_t          _fn_stream_wait_event = NULL
cdef fn_aclrtCreateEvent_t              _fn_create_event = NULL
cdef fn_aclrtCreateEventWithFlag_t      _fn_create_event_flag = NULL
cdef fn_aclrtCreateEventWithFlag_t      _fn_create_event_ex_flag = NULL  # same sig
cdef fn_aclrtDestroyEvent_t             _fn_destroy_event = NULL
cdef fn_aclrtRecordEvent_t              _fn_record_event = NULL
cdef fn_aclrtQueryEvent_t               _fn_query_event = NULL
cdef fn_aclrtSynchronizeEvent_t         _fn_sync_event = NULL
cdef fn_aclrtEventElapsedTime_t         _fn_event_elapsed = NULL
cdef fn_aclrtSynchronizeDevice_t        _fn_sync_device = NULL

# aclmdlRI
cdef fn_aclmdlRICaptureBegin_t          _fn_capture_begin = NULL
cdef fn_aclmdlRICaptureEnd_t            _fn_capture_end = NULL
cdef fn_aclmdlRICaptureGetInfo_t        _fn_capture_get_info = NULL
cdef fn_aclmdlRICaptureThreadExchangeMode_t _fn_capture_exchange_mode = NULL
cdef fn_aclmdlRIExecuteAsync_t          _fn_ri_exec_async = NULL
cdef fn_aclmdlRIExecute_t               _fn_ri_exec = NULL
cdef fn_aclmdlRIDestroy_t               _fn_ri_destroy = NULL
cdef fn_aclmdlRISetName_t               _fn_ri_set_name = NULL
cdef fn_aclmdlRIGetName_t               _fn_ri_get_name = NULL
cdef fn_aclmdlRIDebugJsonPrint_t        _fn_ri_debug_json = NULL
cdef fn_aclmdlRIAbort_t                 _fn_ri_abort = NULL

# Task group
cdef fn_aclmdlRICaptureTaskGrpBegin_t   _fn_task_grp_begin = NULL
cdef fn_aclmdlRICaptureTaskGrpEnd_t     _fn_task_grp_end = NULL
cdef fn_aclmdlRICaptureTaskUpdateBegin_t _fn_task_update_begin = NULL
cdef fn_aclmdlRICaptureTaskUpdateEnd_t  _fn_task_update_end = NULL

cdef void* _acl_handle = NULL
cdef bint _initialized = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cdef inline void _check_error(int32_t ret, const char* op_name) except *:
    if ret != 0:
        raise RuntimeError(
            f"{op_name.decode('utf-8')} failed with error code {ret}")

cdef void* _sym(void* handle, const char* name) except NULL:
    """Resolve a required symbol, raise on failure."""
    dlerror()
    cdef void* sym = dlsym(handle, name)
    if sym == NULL:
        err = dlerror()
        msg = err.decode("utf-8") if err != NULL else "unknown"
        raise RuntimeError(f"dlsym({name.decode('utf-8')}) failed: {msg}")
    return sym

cdef void* _sym_optional(void* handle, const char* name) nogil:
    """Resolve an optional symbol, return NULL on failure."""
    dlerror()
    return dlsym(handle, name)

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init(str lib_path=None):
    """Load libascendcl.so and resolve all function pointers.

    Called once at NPU backend startup.
    """
    global _acl_handle, _initialized
    global _fn_create_stream, _fn_create_stream_cfg, _fn_destroy_stream
    global _fn_sync_stream, _fn_stream_wait_event
    global _fn_create_event, _fn_create_event_flag, _fn_create_event_ex_flag
    global _fn_destroy_event, _fn_record_event, _fn_query_event
    global _fn_sync_event, _fn_event_elapsed, _fn_sync_device
    global _fn_capture_begin, _fn_capture_end, _fn_capture_get_info
    global _fn_capture_exchange_mode
    global _fn_ri_exec_async, _fn_ri_exec, _fn_ri_destroy
    global _fn_ri_set_name, _fn_ri_get_name, _fn_ri_debug_json, _fn_ri_abort
    global _fn_task_grp_begin, _fn_task_grp_end
    global _fn_task_update_begin, _fn_task_update_end

    if _initialized:
        return

    cdef bytes bpath
    if lib_path is not None:
        bpath = lib_path.encode("utf-8")
    else:
        bpath = b"libascendcl.so"

    _acl_handle = dlopen(<const char*>bpath, RTLD_LAZY | RTLD_GLOBAL)
    if _acl_handle == NULL:
        err = dlerror()
        msg = err.decode("utf-8") if err != NULL else "unknown"
        raise RuntimeError(f"dlopen({bpath.decode()}) failed: {msg}")

    cdef void* h = _acl_handle

    # -- Stream
    _fn_create_stream = <fn_aclrtCreateStream_t>_sym(h, b"aclrtCreateStream")
    _fn_create_stream_cfg = <fn_aclrtCreateStreamWithConfig_t>_sym_optional(
        h, b"aclrtCreateStreamWithConfig")
    _fn_destroy_stream = <fn_aclrtDestroyStream_t>_sym(h, b"aclrtDestroyStream")
    _fn_sync_stream = <fn_aclrtSynchronizeStream_t>_sym(h, b"aclrtSynchronizeStream")
    _fn_stream_wait_event = <fn_aclrtStreamWaitEvent_t>_sym(h, b"aclrtStreamWaitEvent")

    # -- Event
    _fn_create_event = <fn_aclrtCreateEvent_t>_sym(h, b"aclrtCreateEvent")
    _fn_create_event_flag = <fn_aclrtCreateEventWithFlag_t>_sym_optional(
        h, b"aclrtCreateEventWithFlag")
    _fn_create_event_ex_flag = <fn_aclrtCreateEventWithFlag_t>_sym_optional(
        h, b"aclrtCreateEventExWithFlag")
    _fn_destroy_event = <fn_aclrtDestroyEvent_t>_sym(h, b"aclrtDestroyEvent")
    _fn_record_event = <fn_aclrtRecordEvent_t>_sym(h, b"aclrtRecordEvent")
    _fn_query_event = <fn_aclrtQueryEvent_t>_sym(h, b"aclrtQueryEvent")
    _fn_sync_event = <fn_aclrtSynchronizeEvent_t>_sym(h, b"aclrtSynchronizeEvent")
    _fn_event_elapsed = <fn_aclrtEventElapsedTime_t>_sym(h, b"aclrtEventElapsedTime")
    _fn_sync_device = <fn_aclrtSynchronizeDevice_t>_sym(h, b"aclrtSynchronizeDevice")

    # -- aclmdlRI capture/replay
    _fn_capture_begin = <fn_aclmdlRICaptureBegin_t>_sym(h, b"aclmdlRICaptureBegin")
    _fn_capture_end = <fn_aclmdlRICaptureEnd_t>_sym(h, b"aclmdlRICaptureEnd")
    _fn_capture_get_info = <fn_aclmdlRICaptureGetInfo_t>_sym(
        h, b"aclmdlRICaptureGetInfo")
    _fn_capture_exchange_mode = <fn_aclmdlRICaptureThreadExchangeMode_t>_sym(
        h, b"aclmdlRICaptureThreadExchangeMode")
    _fn_ri_exec_async = <fn_aclmdlRIExecuteAsync_t>_sym(h, b"aclmdlRIExecuteAsync")
    _fn_ri_exec = <fn_aclmdlRIExecute_t>_sym(h, b"aclmdlRIExecute")
    _fn_ri_destroy = <fn_aclmdlRIDestroy_t>_sym(h, b"aclmdlRIDestroy")
    _fn_ri_set_name = <fn_aclmdlRISetName_t>_sym(h, b"aclmdlRISetName")
    _fn_ri_get_name = <fn_aclmdlRIGetName_t>_sym(h, b"aclmdlRIGetName")
    _fn_ri_debug_json = <fn_aclmdlRIDebugJsonPrint_t>_sym(
        h, b"aclmdlRIDebugJsonPrint")
    _fn_ri_abort = <fn_aclmdlRIAbort_t>_sym(h, b"aclmdlRIAbort")

    # -- Task group (optional — may not exist in older CANN)
    _fn_task_grp_begin = <fn_aclmdlRICaptureTaskGrpBegin_t>_sym_optional(
        h, b"aclmdlRICaptureTaskGrpBegin")
    _fn_task_grp_end = <fn_aclmdlRICaptureTaskGrpEnd_t>_sym_optional(
        h, b"aclmdlRICaptureTaskGrpEnd")
    _fn_task_update_begin = <fn_aclmdlRICaptureTaskUpdateBegin_t>_sym_optional(
        h, b"aclmdlRICaptureTaskUpdateBegin")
    _fn_task_update_end = <fn_aclmdlRICaptureTaskUpdateEnd_t>_sym_optional(
        h, b"aclmdlRICaptureTaskUpdateEnd")

    _initialized = 1


cdef inline void _ensure_loaded() except *:
    if not _initialized:
        init()


def is_initialized():
    return _initialized != 0

# ===================================================================
# Python-visible stream/event wrappers
# ===================================================================

def create_stream(uint32_t priority=0):
    """Create an ACL stream. Returns handle as Python int."""
    _ensure_loaded()
    cdef void* stream = NULL
    cdef int32_t ret
    if _fn_create_stream_cfg != NULL:
        with nogil:
            ret = _fn_create_stream_cfg(&stream, priority, 0)
        if ret == 0:
            return <uintptr_t>stream
    # fallback to basic create
    with nogil:
        ret = _fn_create_stream(&stream)
    _check_error(ret, b"aclrtCreateStream")
    return <uintptr_t>stream

def destroy_stream(uintptr_t handle):
    _ensure_loaded()
    cdef int32_t ret
    with nogil:
        ret = _fn_destroy_stream(<void*>handle)
    _check_error(ret, b"aclrtDestroyStream")

cpdef synchronize_stream(uintptr_t handle):
    _ensure_loaded()
    cdef int32_t ret
    with nogil:
        ret = _fn_sync_stream(<void*>handle)
    _check_error(ret, b"aclrtSynchronizeStream")

def create_event(uint32_t flag=0):
    """Create an ACL event. Returns handle as Python int."""
    _ensure_loaded()
    cdef void* event = NULL
    cdef int32_t ret
    if flag != 0 and _fn_create_event_ex_flag != NULL:
        with nogil:
            ret = _fn_create_event_ex_flag(&event, flag)
        if ret == 0:
            return <uintptr_t>event
    if flag != 0 and _fn_create_event_flag != NULL:
        with nogil:
            ret = _fn_create_event_flag(&event, flag)
        if ret == 0:
            return <uintptr_t>event
    with nogil:
        ret = _fn_create_event(&event)
    _check_error(ret, b"aclrtCreateEvent")
    return <uintptr_t>event

def destroy_event(uintptr_t handle):
    _ensure_loaded()
    cdef int32_t ret
    with nogil:
        ret = _fn_destroy_event(<void*>handle)
    _check_error(ret, b"aclrtDestroyEvent")

def record_event(uintptr_t event, uintptr_t stream):
    _ensure_loaded()
    cdef int32_t ret
    with nogil:
        ret = _fn_record_event(<void*>event, <void*>stream)
    _check_error(ret, b"aclrtRecordEvent")

def query_event(uintptr_t event):
    """Returns True if event is complete, False otherwise."""
    _ensure_loaded()
    cdef int32_t status = 1  # NOT_READY
    cdef int32_t ret
    with nogil:
        ret = _fn_query_event(<void*>event, &status)
    # ret != 0 typically means NOT_READY rather than hard error
    if ret != 0:
        return False
    return status == 0  # ACL_EVENT_STATUS_COMPLETE

def synchronize_event(uintptr_t event):
    _ensure_loaded()
    cdef int32_t ret
    with nogil:
        ret = _fn_sync_event(<void*>event)
    _check_error(ret, b"aclrtSynchronizeEvent")

def event_elapsed_time(uintptr_t start, uintptr_t end):
    """Returns elapsed time in milliseconds (float)."""
    _ensure_loaded()
    cdef float ms = 0.0
    cdef int32_t ret
    with nogil:
        ret = _fn_event_elapsed(&ms, <void*>start, <void*>end)
    _check_error(ret, b"aclrtEventElapsedTime")
    return float(ms)

def stream_wait_event(uintptr_t stream, uintptr_t event):
    _ensure_loaded()
    cdef int32_t ret
    with nogil:
        ret = _fn_stream_wait_event(<void*>stream, <void*>event)
    _check_error(ret, b"aclrtStreamWaitEvent")

def synchronize_device():
    _ensure_loaded()
    cdef int32_t ret
    with nogil:
        ret = _fn_sync_device()
    _check_error(ret, b"aclrtSynchronizeDevice")

# ===================================================================
# Python-visible aclmdlRI wrappers
# ===================================================================

cpdef capture_begin(uintptr_t stream, int mode):
    """Start recording ops on the stream."""
    _ensure_loaded()
    cdef int32_t ret
    with nogil:
        ret = _fn_capture_begin(<void*>stream, <int32_t>mode)
    _check_error(ret, b"aclmdlRICaptureBegin")

cpdef uintptr_t capture_end(uintptr_t stream):
    """Stop recording and return the aclmdlRI handle."""
    _ensure_loaded()
    cdef void* model_ri = NULL
    cdef int32_t ret
    with nogil:
        ret = _fn_capture_end(<void*>stream, &model_ri)
    _check_error(ret, b"aclmdlRICaptureEnd")
    return <uintptr_t>model_ri

cpdef tuple capture_get_info(uintptr_t stream):
    """Query stream capture status. Returns (status_int, model_ri_handle)."""
    _ensure_loaded()
    cdef int32_t status = 0
    cdef void* model_ri = NULL
    cdef int32_t ret
    with nogil:
        ret = _fn_capture_get_info(<void*>stream, &status, &model_ri)
    _check_error(ret, b"aclmdlRICaptureGetInfo")
    return (<int>status, <uintptr_t>model_ri)

cpdef int capture_thread_exchange_mode(int mode):
    """Exchange capture mode for current thread. Returns old mode."""
    _ensure_loaded()
    cdef int32_t m = <int32_t>mode
    cdef int32_t ret
    with nogil:
        ret = _fn_capture_exchange_mode(&m)
    _check_error(ret, b"aclmdlRICaptureThreadExchangeMode")
    return <int>m

cpdef ri_execute_async(uintptr_t model_ri, uintptr_t stream):
    """Replay captured graph asynchronously on the given stream."""
    _ensure_loaded()
    cdef int32_t ret
    with nogil:
        ret = _fn_ri_exec_async(<void*>model_ri, <void*>stream)
    _check_error(ret, b"aclmdlRIExecuteAsync")

cpdef ri_execute(uintptr_t model_ri, int timeout=-1):
    """Execute captured graph synchronously with timeout (ms)."""
    _ensure_loaded()
    cdef int32_t ret
    with nogil:
        ret = _fn_ri_exec(<void*>model_ri, <int32_t>timeout)
    _check_error(ret, b"aclmdlRIExecute")

cpdef ri_destroy(uintptr_t model_ri):
    """Destroy a captured aclmdlRI handle."""
    _ensure_loaded()
    cdef int32_t ret
    with nogil:
        ret = _fn_ri_destroy(<void*>model_ri)
    _check_error(ret, b"aclmdlRIDestroy")

cpdef ri_set_name(uintptr_t model_ri, str name):
    _ensure_loaded()
    cdef bytes bname = name.encode("utf-8")
    cdef const char* cname = <const char*>bname
    cdef int32_t ret
    with nogil:
        ret = _fn_ri_set_name(<void*>model_ri, cname)
    _check_error(ret, b"aclmdlRISetName")

cpdef str ri_get_name(uintptr_t model_ri):
    _ensure_loaded()
    cdef char buf[256]
    memset(buf, 0, 256)
    cdef int32_t ret
    with nogil:
        ret = _fn_ri_get_name(<void*>model_ri, 255, buf)
    _check_error(ret, b"aclmdlRIGetName")
    return buf.decode("utf-8")

cpdef ri_debug_json_print(uintptr_t model_ri, str path, uint32_t flags=0):
    _ensure_loaded()
    cdef bytes bpath = path.encode("utf-8")
    cdef const char* cpath = <const char*>bpath
    cdef int32_t ret
    with nogil:
        ret = _fn_ri_debug_json(<void*>model_ri, cpath, flags)
    _check_error(ret, b"aclmdlRIDebugJsonPrint")

cpdef ri_abort(uintptr_t model_ri):
    """Abort an in-progress capture."""
    _ensure_loaded()
    cdef int32_t ret
    with nogil:
        ret = _fn_ri_abort(<void*>model_ri)
    _check_error(ret, b"aclmdlRIAbort")