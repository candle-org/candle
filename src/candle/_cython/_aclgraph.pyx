# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython aclgraph state machine over aclmdlRI handles."""

from libc.stdint cimport uint32_t, uintptr_t
from candle._cython cimport _aclrt_ffi

cdef int STATE_IDLE = 0
cdef int STATE_CAPTURING = 1
cdef int STATE_CAPTURED = 2

cdef inline const char* _state_name(int state):
    if state == STATE_IDLE:
        return b"IDLE"
    if state == STATE_CAPTURING:
        return b"CAPTURING"
    if state == STATE_CAPTURED:
        return b"CAPTURED"
    return b"UNKNOWN"


cdef class _NPUGraphImpl:
    cdef void* _model_ri
    cdef void* _capture_stream
    cdef int _state
    cdef bytes _name

    def __cinit__(self):
        self._model_ri = NULL
        self._capture_stream = NULL
        self._state = STATE_IDLE
        self._name = b""

    def __dealloc__(self):
        if self._state == STATE_CAPTURING and self._capture_stream != NULL:
            # Abort in-progress capture: end it to get the partial handle,
            # then destroy that handle immediately.
            try:
                handle = _aclrt_ffi.capture_end(<uintptr_t>self._capture_stream)
                if handle:
                    _aclrt_ffi.ri_destroy(handle)
            except Exception:
                pass
        elif self._state == STATE_CAPTURED and self._model_ri != NULL:
            try:
                _aclrt_ffi.ri_destroy(<uintptr_t>self._model_ri)
            except Exception:
                pass
        self._model_ri = NULL
        self._capture_stream = NULL
        self._state = STATE_IDLE

    cdef inline void _require_state(self, int expected, str opname):
        if self._state != expected:
            raise RuntimeError(
                f"{opname} requires state {_state_name(expected).decode()}, "
                f"got {_state_name(self._state).decode()}")

    cdef inline void _clear_handle(self):
        self._model_ri = NULL
        self._capture_stream = NULL

    @property
    def capture_stream(self):
        return <uintptr_t>self._capture_stream

    cpdef capture_begin(self, uintptr_t stream, int mode):
        self._require_state(STATE_IDLE, "capture_begin")
        _aclrt_ffi.capture_begin(stream, mode)
        self._capture_stream = <void*>stream
        self._state = STATE_CAPTURING

    cpdef capture_end(self):
        self._require_state(STATE_CAPTURING, "capture_end")
        cdef uintptr_t handle = _aclrt_ffi.capture_end(<uintptr_t>self._capture_stream)
        self._model_ri = <void*>handle
        self._state = STATE_CAPTURED

    cpdef replay_async(self, uintptr_t stream=0):
        self._require_state(STATE_CAPTURED, "replay_async")
        if stream == 0:
            stream = <uintptr_t>self._capture_stream
        _aclrt_ffi.ri_execute_async(<uintptr_t>self._model_ri, stream)

    cpdef reset(self):
        if self._state == STATE_CAPTURING:
            raise RuntimeError("reset() is not allowed during CAPTURING; use abort()")
        if self._state == STATE_CAPTURED and self._model_ri != NULL:
            _aclrt_ffi.ri_destroy(<uintptr_t>self._model_ri)
        self._clear_handle()
        self._state = STATE_IDLE

    cpdef abort(self):
        self._require_state(STATE_CAPTURING, "abort")
        # aclmdlRIAbort() aborts an existing materialized modelRI. During
        # CAPTURING there is no modelRI handle yet, so aborting capture means
        # ending the capture on the stream to materialize the partial graph,
        # then immediately destroying that handle.
        if self._capture_stream != NULL:
            try:
                handle = _aclrt_ffi.capture_end(<uintptr_t>self._capture_stream)
                if handle:
                    _aclrt_ffi.ri_destroy(handle)
            except Exception:
                pass
        self._clear_handle()
        self._state = STATE_IDLE

    cpdef debug_dump(self, str path, uint32_t flags=0):
        self._require_state(STATE_CAPTURED, "debug_dump")
        _aclrt_ffi.ri_debug_json_print(<uintptr_t>self._model_ri, path, flags)
