# ACLGraph Support Design Spec

**Date:** 2026-03-20
**Status:** Draft

## Goal

Implement `candle.npu.NPUGraph` and `candle.npu.graph()` mirroring PyTorch's `torch.cuda.CUDAGraph` / `torch.cuda.graph()` API, backed by CANN's `aclmdlRI` stream capture mechanism. Simultaneously migrate stream/event management from ctypes to Cython.

## Background

### aclmdlRI API (CANN 8.5, acl_rt.h)

The `aclmdlRI` (Model Runtime Instance) API in `libascendcl.so` provides CUDA-Graph-style stream capture for Ascend NPU. It is a pure C ABI (`extern "C"`), directly callable via dlopen/dlsym.

**Core flow:**
```
aclmdlRICaptureBegin(stream, mode)   // start recording
  aclnnAdd(..., stream)               // ops are recorded, not executed
  aclnnMatmul(..., stream)
aclmdlRICaptureEnd(stream, &modelRI) // stop recording, get handle
aclmdlRIExecuteAsync(modelRI, stream) // replay captured ops
aclmdlRIDestroy(modelRI)             // free resources
```

**Key difference from GE:** aclmdlRI captures existing aclnn kernel calls on a stream — no op mapping table needed, no C++ ABI issues.

### PyTorch CUDA Graph API (reference)

| API | Purpose |
|-----|---------|
| `torch.cuda.CUDAGraph` | Low-level class: capture_begin/end, replay, reset |
| `torch.cuda.graph(g, pool, stream, capture_error_mode)` | Context manager wrapping capture lifecycle |
| `torch.cuda.is_current_stream_capturing()` | Query capture status |
| `torch.cuda.graph_pool_handle()` | Memory pool sharing (not applicable to aclmdlRI) |
| `torch.cuda.make_graphed_callables()` | High-level auto-graph for fwd+bwd (future work) |

## Architecture

Three-layer design:

```
┌─────────────────────────────────────────────────┐
│  Python API Layer (candle/npu/graphs.py)        │
│  NPUGraph, graph(), is_current_stream_capturing │
├─────────────────────────────────────────────────┤
│  Cython Core Layer (_cython/_aclgraph.pyx)      │
│  cdef class _NPUGraphImpl (state machine)       │
├─────────────────────────────────────────────────┤
│  Cython FFI Layer (_cython/_aclrt_ffi.pyx)      │
│  dlopen/dlsym bindings: aclmdlRI + stream/event │
└─────────────────────────────────────────────────┘
```

## Layer 1: Cython FFI (`_aclrt_ffi.pyx`)

Binds C functions from `libascendcl.so` via dlopen/dlsym. Follows the same pattern as `_aclnn_ffi.pyx`: cached function pointers, `nogil` calls, `_ensure_loaded()` guard.

### Enum constants

Defined in `_aclrt_ffi.pyx` (and exported via `.pxd`):

```
ACL_MODEL_RI_CAPTURE_MODE_GLOBAL = 0
ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL = 1
ACL_MODEL_RI_CAPTURE_MODE_RELAXED = 2

ACL_MODEL_RI_CAPTURE_STATUS_NONE = 0
ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE = 1
ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED = 2
```

### aclmdlRI bindings

| C function | Cython cdef |
|-----------|-------------|
| `aclmdlRICaptureBegin(stream, mode)` | `c_capture_begin` |
| `aclmdlRICaptureEnd(stream, &modelRI)` | `c_capture_end` |
| `aclmdlRICaptureGetInfo(stream, &status, &modelRI)` | `c_capture_get_info` |
| `aclmdlRIExecuteAsync(modelRI, stream)` | `c_ri_execute_async` |
| `aclmdlRIExecute(modelRI, timeout)` | `c_ri_execute` |
| `aclmdlRIDestroy(modelRI)` | `c_ri_destroy` |
| `aclmdlRICaptureThreadExchangeMode(&mode)` | `c_capture_thread_exchange_mode` |
| `aclmdlRISetName(modelRI, name)` | `c_ri_set_name` |
| `aclmdlRIGetName(modelRI, maxLen, name)` | `c_ri_get_name` |
| `aclmdlRIDebugJsonPrint(modelRI, path, flags)` | `c_ri_debug_json_print` |
| `aclmdlRIAbort(modelRI)` | `c_ri_abort` |

### Out-parameter handling

Functions with `void**` out-parameters (`capture_end`, `capture_get_info`) use stack-local `void*` and return the result:

```cython
cdef void* c_capture_end(void* stream):
    cdef void* model_ri = NULL
    cdef int ret
    with nogil:
        ret = _fn_capture_end(stream, &model_ri)
    _check_error(ret)
    return model_ri

cdef (int, void*) c_capture_get_info(void* stream):
    cdef int status = 0
    cdef void* model_ri = NULL
    cdef int ret
    with nogil:
        ret = _fn_capture_get_info(stream, &status, &model_ri)
    _check_error(ret)
    return status, model_ri
```

### Stream/event bindings (migrated from runtime.py ctypes)

| C function | Cython cdef |
|-----------|-------------|
| `aclrtCreateStream(&stream)` | `c_create_stream` |
| `aclrtCreateStreamWithConfig(&stream, priority, flag)` | `c_create_stream_with_config` |
| `aclrtDestroyStream(stream)` | `c_destroy_stream` |
| `aclrtSynchronizeStream(stream)` | `c_synchronize_stream` |
| `aclrtStreamWaitEvent(stream, event)` | `c_stream_wait_event` |
| `aclrtCreateEvent(&event)` | `c_create_event` |
| `aclrtCreateEventWithFlag(&event, flag)` | `c_create_event_with_flag` |
| `aclrtDestroyEvent(event)` | `c_destroy_event` |
| `aclrtRecordEvent(event, stream)` | `c_record_event` |
| `aclrtSynchronizeEvent(event)` | `c_synchronize_event` |
| `aclrtEventElapsedTime(&time, start, end)` | `c_event_elapsed_time` |

### Shared .pxd declaration

`_aclrt_ffi.pxd` declares all `cdef` functions so `_aclgraph.pyx` can `cimport` them directly without Python call overhead.

### No fallback

Cython extension is a hard dependency. If compilation fails, NPU backend is unavailable.

## Layer 2: Cython Core (`_aclgraph.pyx`)

### State machine

```
IDLE ──capture_begin──→ CAPTURING ──capture_end──→ CAPTURED ──reset──→ IDLE
                            │                         │
                          abort                   replay (N times)
                            │                     destroy via __dealloc__
                            ↓
                          IDLE
```

### cdef class _NPUGraphImpl

```
Fields:
  _model_ri: void*        # aclmdlRI handle, NULL when not captured
  _capture_stream: void*  # stream used during capture (stored for replay default)
  _state: int             # IDLE=0, CAPTURING=1, CAPTURED=2
  _name: bytes            # optional debug name

Methods:
  cpdef capture_begin(self, uintptr_t stream, int mode)
  cpdef capture_end(self)
  cpdef replay_async(self, uintptr_t stream)  # defaults to capture stream
  cpdef reset(self)
  cpdef abort(self)
  cpdef debug_dump(self, str path, uint32_t flags=0)
  readonly property capture_stream -> uintptr_t
  __dealloc__: auto-destroy if CAPTURED
```

### Error handling

- State violations raise `RuntimeError` with descriptive message
- Non-zero aclError from C calls raises `RuntimeError` with `aclGetRecentErrMsg()` detail
- `capture_begin` failure rolls state back to IDLE
- `capture_end` transitions state to CAPTURED only after successful C call (not before)
- `abort` during CAPTURING calls `aclmdlRIAbort` and resets to IDLE
- `__dealloc__`: if `_state == CAPTURED` and `_model_ri != NULL`, auto-calls `c_ri_destroy` to prevent resource leaks

## Layer 3: Python API (`candle/npu/graphs.py`)

### NPUGraph class

```python
class NPUGraph:
    def __init__(self):
        self._impl = _NPUGraphImpl()

    def capture_begin(self, pool=None, capture_error_mode="global"):
        stream = npu.current_stream()
        mode = _ERROR_MODE_MAP[capture_error_mode]
        self._impl.capture_begin(stream._handle, mode)

    def capture_end(self):
        self._impl.capture_end()

    def replay(self):
        self._impl.replay_async(self._impl.capture_stream)

    def reset(self):
        self._impl.reset()

    def pool(self):
        return None  # reserved for future

    def debug_dump(self, path):
        self._impl.debug_dump(path)
```

`capture_error_mode` values: `"global"` (default), `"thread_local"`, `"relaxed"` — mapped to `aclmdlRICaptureMode` enum values.

### graph context manager

```python
class graph:
    def __init__(self, npu_graph, pool=None, stream=None,
                 capture_error_mode="global"):
        self._graph = npu_graph
        self._stream = stream
        self._pool = pool
        self._capture_error_mode = capture_error_mode
        self._stream_ctx = None

    def __enter__(self):
        npu.synchronize()
        if self._stream is not None:
            self._stream_ctx = npu.stream(self._stream)
            self._stream_ctx.__enter__()
        self._graph.capture_begin(
            pool=self._pool,
            capture_error_mode=self._capture_error_mode,
        )
        return self._graph

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._graph._impl.abort()
        else:
            self._graph.capture_end()
        if self._stream_ctx is not None:
            self._stream_ctx.__exit__(exc_type, exc_val, exc_tb)
        npu.current_stream().synchronize()
        return False
```

### is_current_stream_capturing

```python
def is_current_stream_capturing():
    stream = npu.current_stream()
    status, _ = _aclrt_ffi.capture_get_info(stream._handle)
    return status == _aclrt_ffi.ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE
```

### Module exports

`candle/npu/` may need to be created if it doesn't exist yet. It should use lazy imports to avoid loading NPU code on non-NPU machines.

```python
# candle/npu/__init__.py
from .graphs import NPUGraph, graph, is_current_stream_capturing
```

## Stream migration

### Scope

Only stream/event APIs move from ctypes to `_aclrt_ffi.pyx`. Memory management (malloc, memcpy, device management) stays in `runtime.py` ctypes for now.

### Approach

Bypass `_Runtime` class for stream/event operations. `streams.py` imports directly from `_aclrt_ffi` Cython module. The `_Runtime` methods for stream/event become thin wrappers that delegate to `_aclrt_ffi` (for backward compat with any internal callers).

### streams.py changes

```python
# Before: runtime.create_stream(priority) via ctypes
# After:  _aclrt_ffi.create_stream(priority) via Cython

from candle._cython._aclrt_ffi import (
    create_stream, create_stream_with_config, destroy_stream,
    synchronize_stream, stream_wait_event,
    create_event, create_event_with_flag, destroy_event,
    record_event, synchronize_event, event_elapsed_time,
)
```

### Stream handle bridging

The `Stream` class stores the raw handle as a Python `int` (from `uintptr_t`). The `_handle` property returns this int directly. Cython methods accept `uintptr_t` and cast to `void*` internally. No ctypes pointer conversion needed.

## File structure

```
src/candle/
├── _cython/
│   ├── _aclrt_ffi.pyx          # NEW: ACL runtime FFI
│   ├── _aclrt_ffi.pxd          # NEW: cdef declarations for cimport
│   └── _aclgraph.pyx           # NEW: cdef class _NPUGraphImpl
├── _backends/npu/
│   └── streams.py              # MODIFIED: ctypes → _aclrt_ffi
└── npu/
    ├── __init__.py              # MODIFIED: export graph APIs
    └── graphs.py                # NEW: Python API layer

tests/npu/
    └── test_aclgraph.py         # NEW: graph capture/replay tests
```

### setup.py additions

```python
# In linux_only_extensions:
Extension("candle._cython._aclrt_ffi",
          ["src/candle/_cython/_aclrt_ffi.pyx"],
          libraries=["dl"]),
Extension("candle._cython._aclgraph",
          ["src/candle/_cython/_aclgraph.pyx"]),
```

## Typical usage

```python
import candle as torch

static_input = torch.randn(8, 64, device="npu")

# warmup
for _ in range(3):
    out = model(static_input)
torch.npu.synchronize()

# capture
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    static_output = model(static_input)

# replay loop
for batch in dataloader:
    static_input.copy_(batch)
    g.replay()
    result = static_output.clone()
```

## Test plan

File: `tests/npu/test_aclgraph.py` (auto-skipped when NPU unavailable)

| # | Test | Validates |
|---|------|-----------|
| 1 | Basic capture/replay | Single add op captured and replayed correctly |
| 2 | Static tensor pattern | copy_() input → replay → output changes |
| 3 | Context manager happy path | `with npu.graph(g)` capture + replay |
| 4 | Context manager exception | Exception in with block → abort, state → IDLE |
| 5 | Multiple replays | Same graph replayed N times, consistent results |
| 6 | Reset and recapture | capture → replay → reset → capture new ops |
| 7 | Illegal state transitions | replay from IDLE → RuntimeError |
| 8 | is_current_stream_capturing | True inside capture block, False outside |
| 9 | Multi-op capture | matmul + add + relu sequence matches eager |
| 10 | debug_dump | Calls debug_dump without crash |
| 11 | Double capture_begin | capture_begin while CAPTURING → RuntimeError |

Each replay result is validated against eager execution via `allclose`.

## Out of scope (future work)

- Memory pool sharing (`graph_pool_handle`)
- `make_graphed_callables` (requires autograd custom function integration)
- TaskGrp dynamic update (`aclmdlRICaptureTaskGrpBegin/End`, `TaskUpdateBegin/End`) — note: FFI layer should bind these symbols now (unused) to avoid a second migration later
- Integration with Pipeline deferred execution
- `torch.compile` backend using aclgraph
