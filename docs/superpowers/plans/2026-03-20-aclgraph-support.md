# ACLGraph Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ascend aclgraph support to candle via `torch.npu.NPUGraph`, `torch.npu.graph`, and `torch.npu.is_current_stream_capturing`, backed by CANN `aclmdlRI`, while migrating NPU stream/event runtime calls from ctypes to Cython.

**Architecture:** Implement three layers: a low-level Cython FFI module (`_aclrt_ffi.pyx`) that dlopens `libascendcl.so` and binds aclmdlRI plus stream/event APIs; a Cython state-machine wrapper (`_aclgraph.pyx`) that owns `aclmdlRI` handles and enforces lifecycle rules; and a Python API layer wired into the existing public `candle.npu` module (`src/candle/npu.py`). Stream and event methods are migrated incrementally by delegating `_Runtime` and `streams.py` to the new Cython FFI without touching unrelated memory-management paths.

**Tech Stack:** Python 3.11, Cython, CANN 8.5 `libascendcl.so`, candle NPU backend, pytest

**Spec:** `docs/superpowers/specs/2026-03-20-aclgraph-support-design.md`

---

## File Structure

```
src/candle/
├── _cython/
│   ├── _aclrt_ffi.pyx          # NEW: ACL runtime + aclmdlRI dlopen/dlsym FFI
│   ├── _aclrt_ffi.pxd          # NEW: cdef declarations for _aclgraph.pyx cimport
│   └── _aclgraph.pyx           # NEW: cdef class _NPUGraphImpl state machine
├── _backends/npu/
│   ├── runtime.py              # MODIFY: delegate stream/event methods to _aclrt_ffi
│   └── streams.py              # MODIFY: use _handle property and Cython-backed runtime
├── npu.py                      # MODIFY: export NPUGraph APIs and graph context manager
└── _backends/npu/aclnn.py      # OPTIONAL SMALL MODIFY: if tests need helper injection, keep minimal

tests/npu/
└── test_aclgraph.py            # NEW: aclgraph behavior tests

setup.py                        # MODIFY: add _aclrt_ffi and _aclgraph extensions
```

Notes for implementers:
- Public API lives in `src/candle/npu.py`, not `src/candle/npu/__init__.py`.
- Existing Cython FFI pattern to copy is `src/candle/_cython/_aclnn_ffi.pyx`.
- Existing stream/event behavior and test style live in `src/candle/_backends/npu/streams.py` and `tests/npu/test_npu_streams.py`.
- Do not add a fallback path. Cython is a hard dependency for this feature.
- Do not touch allocator/malloc/memcpy/device-management paths in this plan.

---

## Chunk 1: Build the low-level ACL runtime FFI

### Task 1: Add setup.py extension entries for aclgraph modules

**Files:**
- Modify: `setup.py:96-112`

- [ ] **Step 1: Write the failing test by trying to import the future extension names from a small Python smoke snippet in the plan comments**

```python
# smoke expectation after implementation
import importlib
importlib.import_module("candle._cython._aclrt_ffi")
importlib.import_module("candle._cython._aclgraph")
```

- [ ] **Step 2: Add the two new Linux-only Cython extensions**

```python
Extension(
    "candle._cython._aclrt_ffi",
    ["src/candle/_cython/_aclrt_ffi.pyx"],
    libraries=["dl"],
),
Extension(
    "candle._cython._aclgraph",
    ["src/candle/_cython/_aclgraph.pyx"],
    libraries=["dl"],
),
```

Keep them in `linux_only_extensions` next to `_aclnn_ffi`.

- [ ] **Step 3: Run a targeted build to verify setup.py sees the new extensions**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python setup.py build_ext --inplace
```

Expected: build fails because the new `.pyx` files do not exist yet.

- [ ] **Step 4: Commit setup-only scaffolding**

```bash
git add setup.py
git commit -m "build(cython): register aclgraph extension modules"
```

---

### Task 2: Create `_aclrt_ffi.pyx` skeleton with shared dlopen pattern

**Files:**
- Create: `src/candle/_cython/_aclrt_ffi.pyx`
- Modify: `setup.py`

- [ ] **Step 1: Write the failing test by building extensions again**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python setup.py build_ext --inplace
```

Expected: fail with missing `_aclrt_ffi.pyx`.

- [ ] **Step 2: Create `_aclrt_ffi.pyx` module header and basic dlopen/dlsym scaffolding**

```cython
# cython: language_level=3, boundscheck=False, wraparound=False
from libc.stdint cimport int32_t, uint32_t, uint64_t, uintptr_t

cdef extern from "dlfcn.h":
    void* dlopen(const char* filename, int flags) nogil
    void* dlsym(void* handle, const char* symbol) nogil
    char* dlerror() nogil
    int RTLD_LAZY
    int RTLD_GLOBAL

ctypedef int32_t aclError_t
ctypedef void* aclrtStream_t
ctypedef void* aclrtEvent_t
ctypedef void* aclmdlRI_t

cdef void* _acl_handle = NULL
cdef bint _initialized = 0

def is_initialized():
    return _initialized != 0
```

Also add `_load_acl()` and `_find_symbol()` matching `_aclnn_ffi.pyx` style, but targeting only `libascendcl.so`.

- [ ] **Step 3: Add `_check_error(ret, op_name)` helper**

Make it raise `RuntimeError(f"{op_name} failed: {ret}")` first. Do not over-design error-message lookup in the first task.

- [ ] **Step 4: Run the build**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python setup.py build_ext --inplace
```

Expected: `_aclrt_ffi` now compiles or fails only because `_aclgraph.pyx` is still missing.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_cython/_aclrt_ffi.pyx setup.py
git commit -m "feat(cython): add acl runtime ffi skeleton"
```

---

### Task 3: Bind stream and event functions in `_aclrt_ffi.pyx`

**Files:**
- Modify: `src/candle/_cython/_aclrt_ffi.pyx`
- Test: `tests/npu/test_npu_streams.py`

- [ ] **Step 1: Add function pointer typedefs and cached symbols for stream/event APIs**

Add typedefs for:
- `aclrtCreateStream(aclrtStream*)`
- `aclrtCreateStreamWithConfig(aclrtStream*, uint32_t, uint32_t)`
- `aclrtDestroyStream(aclrtStream)`
- `aclrtSynchronizeStream(aclrtStream)`
- `aclrtCreateEvent(aclrtEvent*)`
- `aclrtCreateEventWithFlag(aclrtEvent*, uint32_t)`
- `aclrtCreateEventExWithFlag(aclrtEvent*, uint32_t)`
- `aclrtDestroyEvent(aclrtEvent)`
- `aclrtRecordEvent(aclrtEvent, aclrtStream)`
- `aclrtQueryEvent(aclrtEvent, aclrtEventStatus*)`
- `aclrtSynchronizeEvent(aclrtEvent)`
- `aclrtEventElapsedTime(float*, aclrtEvent, aclrtEvent)`
- `aclrtStreamWaitEvent(aclrtStream, aclrtEvent)`
- `aclrtSynchronizeDevice(void)`

- [ ] **Step 2: Resolve all stream/event symbols in `_ensure_loaded()`**

Follow `_aclnn_ffi.pyx` pattern exactly: cache symbols once, raise if a required symbol is missing, but allow `create_event_ex_with_flag` / `create_event_with_flag` style optional fallback by resolving them lazily in priority order.

- [ ] **Step 3: Add Python-visible wrappers returning Python ints/floats/bools**

Implement these functions:
- `create_stream(priority=0) -> int`
- `destroy_stream(handle) -> None`
- `synchronize_stream(handle) -> None`
- `create_event(flag=0) -> int`
- `destroy_event(handle) -> None`
- `record_event(event, stream) -> None`
- `query_event(event) -> bool`
- `synchronize_event(event) -> None`
- `event_elapsed_time(start, end) -> float`
- `stream_wait_event(stream, event) -> None`
- `synchronize_device() -> None`

Use `uintptr_t`/Python `int` as the public bridge type.

- [ ] **Step 4: Run the existing stream tests to ensure no behavior regressions once runtime is rewired later**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_npu_streams.py -q
```

Expected: still failing or partially failing because runtime/streams are not yet using `_aclrt_ffi`.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_cython/_aclrt_ffi.pyx
git commit -m "feat(cython): bind npu stream and event runtime apis"
```

---

### Task 4: Bind aclmdlRI capture/replay APIs in `_aclrt_ffi.pyx`

**Files:**
- Modify: `src/candle/_cython/_aclrt_ffi.pyx`

- [ ] **Step 1: Add enum constants and function pointer typedefs for aclmdlRI**

Define module-level constants:
```python
ACL_MODEL_RI_CAPTURE_MODE_GLOBAL = 0
ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL = 1
ACL_MODEL_RI_CAPTURE_MODE_RELAXED = 2
ACL_MODEL_RI_CAPTURE_STATUS_NONE = 0
ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE = 1
ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED = 2
```

Add typedefs for:
- `aclmdlRICaptureBegin`
- `aclmdlRICaptureGetInfo`
- `aclmdlRICaptureEnd`
- `aclmdlRICaptureThreadExchangeMode`
- `aclmdlRIExecuteAsync`
- `aclmdlRIExecute`
- `aclmdlRIDestroy`
- `aclmdlRISetName`
- `aclmdlRIGetName`
- `aclmdlRIDebugJsonPrint`
- `aclmdlRIAbort`
- task-group APIs mentioned in the spec (bind now, unused)

- [ ] **Step 2: Resolve symbols in `_ensure_loaded()`**

Keep task-group functions optional only if header/library variance requires it; otherwise resolve them eagerly.

- [ ] **Step 3: Add out-parameter wrappers exactly as designed**

Implement:
- `capture_begin(stream, mode) -> None`
- `capture_get_info(stream) -> (status, model_ri_handle)`
- `capture_end(stream) -> model_ri_handle`
- `capture_thread_exchange_mode(mode) -> old_mode`
- `ri_execute_async(model_ri, stream) -> None`
- `ri_execute(model_ri, timeout) -> None`
- `ri_destroy(model_ri) -> None`
- `ri_set_name(model_ri, name) -> None`
- `ri_get_name(model_ri) -> str`
- `ri_debug_json_print(model_ri, path, flags=0) -> None`
- `ri_abort(model_ri) -> None`

Use Python `str.encode()` for string inputs and decode returned buffers for `ri_get_name`.

- [ ] **Step 4: Run build_ext again**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python setup.py build_ext --inplace
```

Expected: `_aclrt_ffi` builds; `_aclgraph` still missing.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_cython/_aclrt_ffi.pyx
git commit -m "feat(cython): bind aclmdlri capture and replay apis"
```

---

## Chunk 2: Build the Cython graph state machine

### Task 5: Create `_aclrt_ffi.pxd` for direct cimport

**Files:**
- Create: `src/candle/_cython/_aclrt_ffi.pxd`
- Modify: `src/candle/_cython/_aclrt_ffi.pyx`

- [ ] **Step 1: Write the failing build**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python setup.py build_ext --inplace
```

Expected: future `_aclgraph.pyx` cannot cimport `_aclrt_ffi` until declarations exist.

- [ ] **Step 2: Create `.pxd` with cdef declarations for the functions `_aclgraph.pyx` needs**

Expose only what the graph core needs:
- enum constants
- `cdef`/`cpdef` declarations for `capture_begin`, `capture_get_info`, `capture_end`, `ri_execute_async`, `ri_execute`, `ri_destroy`, `ri_debug_json_print`, `ri_abort`, and stream synchronization helpers if needed

Use `cpdef` in `.pyx` if you want both Python import and Cython cimport with one implementation.

- [ ] **Step 3: Build to verify `_aclrt_ffi` still compiles with the new `.pxd`**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python setup.py build_ext --inplace
```

Expected: success for `_aclrt_ffi`.

- [ ] **Step 4: Commit**

```bash
git add src/candle/_cython/_aclrt_ffi.pxd src/candle/_cython/_aclrt_ffi.pyx
git commit -m "feat(cython): add acl runtime ffi declarations"
```

---

### Task 6: Create `_aclgraph.pyx` skeleton with state machine fields

**Files:**
- Create: `src/candle/_cython/_aclgraph.pyx`
- Modify: `setup.py`

- [ ] **Step 1: Write the failing build**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python setup.py build_ext --inplace
```

Expected: fail because `_aclgraph.pyx` does not exist.

- [ ] **Step 2: Create `_aclgraph.pyx` with the state constants and cdef class**

```cython
# cython: language_level=3, boundscheck=False, wraparound=False
from libc.stdint cimport uint32_t, uintptr_t
from candle._cython cimport _aclrt_ffi

cdef int STATE_IDLE = 0
cdef int STATE_CAPTURING = 1
cdef int STATE_CAPTURED = 2

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
```

- [ ] **Step 3: Add a readonly `capture_stream` property and tiny helpers**

Implement:
- `property capture_stream`
- `_require_state(expected, opname)` helper
- `_clear_handle()` helper

- [ ] **Step 4: Build**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python setup.py build_ext --inplace
```

Expected: `_aclgraph` now compiles, even with methods still stubbed.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_cython/_aclgraph.pyx
git commit -m "feat(cython): add aclgraph state machine skeleton"
```

---

### Task 7: Implement capture lifecycle in `_NPUGraphImpl`

**Files:**
- Modify: `src/candle/_cython/_aclgraph.pyx`
- Test: `tests/npu/test_aclgraph.py`

- [ ] **Step 1: Write the failing tests for state transitions**

Create `tests/npu/test_aclgraph.py` with stub-runtime style tests first. Start with these tests:
- `test_npu_graph_replay_from_idle_raises()`
- `test_npu_graph_double_capture_begin_raises()`
- `test_npu_graph_capture_end_without_begin_raises()`

Use monkeypatch/stub style matching `tests/npu/test_npu_streams.py`.

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_aclgraph.py -q
```

Expected: fail because public API does not exist yet.

- [ ] **Step 3: Implement `capture_begin`, `capture_end`, `abort`, `reset`, and `__dealloc__` in `_NPUGraphImpl`**

Required behavior:
- `capture_begin(stream, mode)` allowed only in IDLE; stores `_capture_stream`; transitions to CAPTURING only after success.
- `capture_end()` allowed only in CAPTURING; calls `_aclrt_ffi.capture_end`; stores handle; transitions to CAPTURED only after success.
- `abort()` allowed in CAPTURING; calls `ri_abort` if appropriate; clears stream/handle; returns to IDLE.
- `reset()` destroys handle if present and returns to IDLE.
- `__dealloc__()` destroys a live captured handle safely.

Do not add extra abstractions.

- [ ] **Step 4: Re-run tests**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_aclgraph.py -q
```

Expected: state-transition tests still fail until public API exists, but low-level behavior can be exercised indirectly once imported.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_cython/_aclgraph.pyx tests/npu/test_aclgraph.py
git commit -m "feat(npu): implement aclgraph capture lifecycle"
```

---

### Task 8: Implement replay and debug methods in `_NPUGraphImpl`

**Files:**
- Modify: `src/candle/_cython/_aclgraph.pyx`
- Test: `tests/npu/test_aclgraph.py`

- [ ] **Step 1: Add failing tests for replay path**

Add tests:
- `test_npu_graph_replay_calls_execute_async_on_capture_stream()`
- `test_npu_graph_reset_clears_handle_and_state()`
- `test_npu_graph_debug_dump_calls_runtime()`

Use monkeypatched `_aclrt_ffi` functions or public API stubs.

- [ ] **Step 2: Run tests to verify failure**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_aclgraph.py -q
```

Expected: fail until replay/debug methods are implemented.

- [ ] **Step 3: Implement `replay_async`, optional `replay`, and `debug_dump`**

Behavior:
- `replay_async(stream=0)` allowed only in CAPTURED.
- If `stream == 0`, use `_capture_stream`.
- `debug_dump(path, flags=0)` allowed only with a valid handle.
- Keep sync replay at Python layer if simpler; no need to add duplicate sync wrapper in Cython beyond `ri_execute` unless it simplifies tests.

- [ ] **Step 4: Run tests again**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_aclgraph.py -q
```

Expected: low-level graph tests pass once public API wiring is done in the next chunk.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_cython/_aclgraph.pyx tests/npu/test_aclgraph.py
git commit -m "feat(npu): add aclgraph replay and debug methods"
```

---

## Chunk 3: Wire the public API and migrate runtime/streams

### Task 9: Migrate `_Runtime` stream/event methods to `_aclrt_ffi`

**Files:**
- Modify: `src/candle/_backends/npu/runtime.py`
- Test: `tests/npu/test_npu_streams.py`

- [ ] **Step 1: Write failing/updated expectations in stream tests if necessary**

Existing tests should largely stay as-is. Only adjust stubs if they patch `get_runtime` in a way that conflicts with direct delegation.

- [ ] **Step 2: Import `_aclrt_ffi` in runtime.py and delegate the stream/event methods**

Update these methods only:
- `create_stream`
- `destroy_stream`
- `synchronize_stream`
- `create_event`
- `destroy_event`
- `record_event`
- `synchronize_event`
- `query_event`
- `event_elapsed_time`
- `stream_wait_event`
- `synchronize_device`

Keep the `_Runtime` interface stable. Do not touch init/context/malloc/memcpy.

- [ ] **Step 3: Run the existing stream test file**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_npu_streams.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/candle/_backends/npu/runtime.py tests/npu/test_npu_streams.py
git commit -m "refactor(npu): route stream and event runtime calls through cython ffi"
```

---

### Task 10: Update `streams.py` to expose `_handle` and keep behavior unchanged

**Files:**
- Modify: `src/candle/_backends/npu/streams.py`
- Test: `tests/npu/test_npu_streams.py`

- [ ] **Step 1: Add failing tests for `_handle` bridging and current-stream behavior**

Add or extend tests:
- `test_npu_stream_exposes_handle_alias()`
- `test_npu_event_exposes_event_handle_unchanged()`

- [ ] **Step 2: Modify `Stream` and `Event` with the smallest API changes**

For `Stream`:
- keep `.stream` property unchanged
- add `._handle` property returning `self._stream`

For `Event`:
- keep `.event` property unchanged
- no extra abstraction unless needed by tests

Do not rename existing fields.

- [ ] **Step 3: Run stream tests again**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_npu_streams.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/candle/_backends/npu/streams.py tests/npu/test_npu_streams.py
git commit -m "feat(npu): expose stream handles for aclgraph integration"
```

---

### Task 11: Add public Python graph APIs to `src/candle/npu.py`

**Files:**
- Modify: `src/candle/npu.py`
- Test: `tests/npu/test_aclgraph.py`

- [ ] **Step 1: Write the failing public API tests**

Add tests:
- `test_npu_graph_api_symbols_exist()`
- `test_npu_graph_context_manager_captures_and_replays()`
- `test_npu_is_current_stream_capturing_reflects_capture_state()`
- `test_npu_graph_context_abort_on_exception()`

Use monkeypatch on `candle._cython._aclrt_ffi` / `_aclgraph` to avoid real hardware dependency where possible.

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_aclgraph.py -q
```

Expected: fail because `torch.npu.NPUGraph` and friends do not exist.

- [ ] **Step 3: Implement the public API in `src/candle/npu.py`**

Add:
- `class NPUGraph`
- `class graph`
- `def is_current_stream_capturing()`

Implementation details:
- import `_NPUGraphImpl` from `candle._cython._aclgraph`
- import enum/status helpers from `candle._cython._aclrt_ffi`
- `NPUGraph.capture_begin(pool=None, capture_error_mode="global")` ignores `pool`
- `NPUGraph.replay()` uses stored capture stream
- `graph.__enter__()` does `torch.npu.synchronize()`, optional `with torch.npu.stream(...)`, then `capture_begin`
- `graph.__exit__()` aborts on exception, otherwise capture_end, exits stream ctx, then synchronizes current stream
- add names to `__all__`

Do not create a new `src/candle/npu/` package. The public API belongs in the existing `src/candle/npu.py`.

- [ ] **Step 4: Run the aclgraph tests**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_aclgraph.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/candle/npu.py tests/npu/test_aclgraph.py
git commit -m "feat(npu): add public aclgraph capture and replay api"
```

---

## Chunk 4: Expand coverage and verify real integration

### Task 12: Add integration-style graph tests covering multi-op capture and static-tensor replay

**Files:**
- Modify: `tests/npu/test_aclgraph.py`

- [ ] **Step 1: Add failing tests for the behavior promised in the spec**

Add tests:
- `test_npu_graph_multiple_replays()`
- `test_npu_graph_reset_then_recapture()`
- `test_npu_graph_multi_op_capture_matches_eager()`
- `test_npu_graph_static_tensor_copy_pattern()`

Follow the same style as `tests/npu/test_npu_streams.py`: use stubs/monkeypatch where possible; only use real NPU ops if the suite already relies on hardware-gated tests.

- [ ] **Step 2: Run the target file**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_aclgraph.py -q
```

Expected: fail only on the newly added behaviors.

- [ ] **Step 3: Make the smallest implementation adjustments needed**

Expected small fixes only, such as:
- ensuring `capture_begin` stores stream consistently
- ensuring `reset` clears `_capture_stream`
- ensuring `graph` context manager restores prior stream correctly
- ensuring replay path does not mutate state incorrectly

Do not broaden scope.

- [ ] **Step 4: Re-run aclgraph tests**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_aclgraph.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tests/npu/test_aclgraph.py src/candle/_cython/_aclgraph.pyx src/candle/npu.py
git commit -m "test(npu): cover aclgraph replay and multi-op behavior"
```

---

### Task 13: Run the full targeted verification set

**Files:**
- No code changes required unless failures surface

- [ ] **Step 1: Build all extensions in-place**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python setup.py build_ext --inplace
```

Expected: success.

- [ ] **Step 2: Run stream tests**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_npu_streams.py -q
```

Expected: pass.

- [ ] **Step 3: Run aclgraph tests**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_aclgraph.py -q
```

Expected: pass.

- [ ] **Step 4: Run the combined NPU-focused suite**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_npu_streams.py tests/npu/test_aclgraph.py -q
```

Expected: pass.

- [ ] **Step 5: Commit any final fixes only if verification required changes**

```bash
git add src/candle/_cython/_aclrt_ffi.pyx src/candle/_cython/_aclrt_ffi.pxd src/candle/_cython/_aclgraph.pyx src/candle/_backends/npu/runtime.py src/candle/_backends/npu/streams.py src/candle/npu.py tests/npu/test_aclgraph.py tests/npu/test_npu_streams.py setup.py
git commit -m "fix(npu): finalize aclgraph integration verification"
```

Skip this step if no files changed.

---

### Task 14: Run project-required lint/test gate before handoff

**Files:**
- No new files expected unless lint fixes are needed

- [ ] **Step 1: Run pylint on candle source**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && pylint src/candle/ --rcfile=.github/pylint.conf
```

Expected: 0 errors.

- [ ] **Step 2: If pylint reports issues in modified files, fix only those issues minimally**

Typical acceptable fixes:
- unused imports
- import-order issues
- missing encoding/bytes handling warnings

Do not refactor unrelated code.

- [ ] **Step 3: Re-run the two NPU test files after lint fixes**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/aclgraph-support && python -m pytest tests/npu/test_npu_streams.py tests/npu/test_aclgraph.py -q
```

Expected: pass.

- [ ] **Step 4: Commit the final polished state**

```bash
git add setup.py src/candle/_cython/_aclrt_ffi.pyx src/candle/_cython/_aclrt_ffi.pxd src/candle/_cython/_aclgraph.pyx src/candle/_backends/npu/runtime.py src/candle/_backends/npu/streams.py src/candle/npu.py tests/npu/test_aclgraph.py tests/npu/test_npu_streams.py
git commit -m "feat(npu): add aclgraph capture and replay support"
```

---

## Implementation Notes

- Mirror the coding style of `src/candle/_cython/_aclnn_ffi.pyx`: cached function pointers, dlerror clearing, `nogil` blocks, Python `int` bridge via `uintptr_t`.
- Keep stream/event runtime migration surgical. `_Runtime` stays as the façade; only its stream/event methods change implementation.
- Public API must be exported from `src/candle/npu.py` because that file already powers `torch.npu`.
- Tests should prefer monkeypatch/stub style for state-machine validation and only rely on hardware-gated behavior where necessary.
- Do not implement memory-pool sharing, `make_graphed_callables`, Pipeline integration, or `torch.compile` integration in this plan.

## Chunk Review Checklist

For each chunk above, run a focused plan/code review before moving on:
- Verify file paths match the real repo layout.
- Verify commands are runnable from the worktree.
- Verify every public behavior in the spec has at least one test.
- Verify no task broadens scope beyond aclgraph + stream/event migration.

Plan complete and saved to `docs/superpowers/plans/2026-03-20-aclgraph-support.md`. Ready to execute?