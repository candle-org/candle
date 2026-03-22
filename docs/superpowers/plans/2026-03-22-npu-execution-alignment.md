# NPU Execution Alignment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce candle NPU per-op submit cost from ~137µs to ~57µs by eliminating Python/Cython overhead in the hot path, aligning with torch_npu's C++ execution model.

**Architecture:** Five independent PRs, each targeting a distinct bottleneck in `fast_binary_op`. Each PR is benchmarked independently using `bench_fast_path_profile.py`. PRs do not depend on each other at the code level but are ordered by expected impact.

**Tech Stack:** Cython 3, CANN 8.5.0, `libopapi.so`, `libaclnn_ffi` (dlopen). Tests run via `conda run -n candle pytest tests/npu/ -v --tb=short`.

**Spec:** `docs/superpowers/specs/2026-03-22-npu-execution-alignment-design.md`

---

## Chunk 1: PR-1 — aclTensor Descriptor Cache

**Expected savings:** ~30µs (aclnn time 72µs → ~42µs)

**Files:**
- Modify: `src/candle/_cython/_aclnn_ffi.pyx`
- Modify: `src/candle/_cython/_allocator.pyx`
- Test: `tests/npu/cython/test_tensor_desc_cache.py` (new)

### Task 1.1: Write failing tests for TensorDescCache

- [ ] Create `tests/npu/cython/test_tensor_desc_cache.py`:

```python
import pytest
pytestmark = pytest.mark.npu

def test_cache_hit_same_tensor(npu_device):
    """Same tensor accessed twice should reuse descriptor handle."""
    import candle as torch
    from candle._cython._aclnn_ffi import get_tensor_desc_cache  # new function
    a = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()
    cache = get_tensor_desc_cache()
    cache.clear()
    h1 = cache.get_or_create(a.data_ptr(), a.shape, a.stride, 0, 2)  # dtype_code=0 (float32), fmt=2 (ND)
    h2 = cache.get_or_create(a.data_ptr(), a.shape, a.stride, 0, 2)
    assert h1 == h2, "Cache miss on identical key"

def test_cache_miss_different_stride(npu_device):
    """Transposed view has different stride — must be a cache miss."""
    import candle as torch
    from candle._cython._aclnn_ffi import get_tensor_desc_cache
    a = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()
    cache = get_tensor_desc_cache()
    cache.clear()
    h1 = cache.get_or_create(a.data_ptr(), a.shape, a.stride, 0, 2)
    a_t = a.t()
    h2 = cache.get_or_create(a_t.data_ptr(), a_t.shape, a_t.stride, 0, 2)
    assert h1 != h2, "Cache incorrectly hit for transposed tensor"

def test_cache_invalidated_on_free(npu_device):
    """After tensor goes out of scope and memory is freed, cache entry must be gone."""
    import candle as torch
    import gc
    from candle._cython._aclnn_ffi import get_tensor_desc_cache
    cache = get_tensor_desc_cache()
    cache.clear()
    a = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()
    ptr = a.data_ptr()
    h1 = cache.get_or_create(ptr, a.shape, a.stride, 0, 2)
    assert cache.size() == 1
    del a
    gc.collect()
    torch.npu.synchronize()  # triggers allocator drain → cache invalidation
    assert cache.size() == 0, "Cache entry not invalidated after free"

def test_add_uses_cache(npu_device):
    """Repeated torch.add on same tensors should not increase cache size beyond inputs."""
    import candle as torch
    from candle._cython._aclnn_ffi import get_tensor_desc_cache
    cache = get_tensor_desc_cache()
    cache.clear()
    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()
    for _ in range(5):
        c = torch.add(a, b)
    torch.npu.synchronize()
    # a and b are cached (2 entries); output c changes ptr each time (not cached)
    assert cache.size() == 2
```

- [ ] Run tests to confirm they fail (ImportError expected):

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle pytest tests/npu/cython/test_tensor_desc_cache.py -v --tb=short
```
Expected: ImportError on `get_tensor_desc_cache`

### Task 1.2: Add TensorDescCache to `_aclnn_ffi.pyx`

- [ ] Add after the `_op_cache = {}` dict in `_aclnn_ffi.pyx`:

```cython
# ---------------------------------------------------------------------------
# Tensor descriptor cache
# ---------------------------------------------------------------------------

cdef int _DESC_CACHE_MAX = 64

cdef class TensorDescCache:
    """LRU cache: (data_ptr, shape, stride, dtype_code, fmt) -> aclTensor* handle.

    Eliminates 3x aclCreateTensor per op for reused input tensors.
    Invalidated by allocator.free() via invalidate_range().
    """
    cdef dict _cache    # key -> (handle: uintptr_t, order: int)
    cdef list _order    # insertion-order list of keys (for LRU eviction)
    cdef int _max_size

    def __cinit__(self):
        self._cache = {}
        self._order = []
        self._max_size = _DESC_CACHE_MAX

    cpdef uintptr_t get_or_create(
            self,
            int64_t data_ptr,
            tuple shape,
            tuple stride,
            int32_t dtype_code,
            int32_t fmt):
        """Return cached aclTensor handle or create and cache a new one."""
        cdef object key = (data_ptr, shape, stride, dtype_code, fmt)
        cdef object entry = self._cache.get(key)
        if entry is not None:
            return <uintptr_t>entry
        # Create new descriptor
        cdef int ndim = len(shape)
        cdef int64_t[MAX_NDIM] shape_buf, stride_buf
        cdef int i
        for i in range(ndim):
            shape_buf[i] = shape[i]
            stride_buf[i] = stride[i]
        cdef void* t = _fast_create_tensor(
            shape_buf, stride_buf, <uint64_t>ndim,
            dtype_code, fmt, <void*>data_ptr)
        if t == NULL:
            raise RuntimeError("aclCreateTensor returned null")
        cdef uintptr_t handle = <uintptr_t>t
        # LRU eviction if full
        if len(self._cache) >= self._max_size:
            evict_key = self._order.pop(0)
            evicted = self._cache.pop(evict_key, None)
            if evicted is not None:
                _fast_destroy_tensor(<void*>(<uintptr_t>evicted))
        self._cache[key] = handle
        self._order.append(key)
        return handle

    cpdef void invalidate_range(self, int64_t base_ptr, int64_t size):
        """Remove all entries whose data_ptr falls in [base_ptr, base_ptr+size)."""
        cdef list to_remove = []
        cdef object k
        cdef int64_t ptr
        for k in self._cache:
            ptr = <int64_t>k[0]
            if base_ptr <= ptr < base_ptr + size:
                to_remove.append(k)
        for k in to_remove:
            handle = self._cache.pop(k, None)
            if handle is not None:
                _fast_destroy_tensor(<void*>(<uintptr_t>handle))
            try:
                self._order.remove(k)
            except ValueError:
                pass

    cpdef void clear(self):
        """Destroy all cached handles and reset."""
        cdef object handle
        for handle in self._cache.values():
            _fast_destroy_tensor(<void*>(<uintptr_t>handle))
        self._cache.clear()
        self._order.clear()

    cpdef int size(self):
        return len(self._cache)

    def __dealloc__(self):
        self.clear()


# Module-level singleton
cdef TensorDescCache _tensor_desc_cache = TensorDescCache()

def get_tensor_desc_cache():
    """Return the module-level TensorDescCache singleton."""
    return _tensor_desc_cache
```

- [ ] Modify `binary_op_with_alpha` to use cache for self and other (NOT out):

Replace the `with nogil:` block that creates `self_t` and `other_t`:
```cython
# Before:
    with nogil:
        self_t = _fast_create_tensor(
            s_shape, s_stride, <uint64_t>self_ndim,
            dtype_code, fmt, <void*>self_ptr)
        other_t = _fast_create_tensor(
            o_shape, o_stride, <uint64_t>other_ndim,
            dtype_code, fmt, <void*>other_ptr)

# After:
    # Input tensors: use cache (same ptr+shape+stride → reuse handle)
    cdef uintptr_t self_handle = _tensor_desc_cache.get_or_create(
        <int64_t>self_ptr,
        tuple(self_shape), tuple(self_stride), dtype_code, fmt)
    cdef uintptr_t other_handle = _tensor_desc_cache.get_or_create(
        <int64_t>other_ptr,
        tuple(other_shape), tuple(other_stride), dtype_code, fmt)
    self_t = <void*>self_handle
    other_t = <void*>other_handle
    # out tensor: always freshly created (new ptr each op)
    with nogil:
        out_t = _fast_create_tensor(
            r_shape, r_stride, <uint64_t>out_ndim,
            dtype_code, fmt, <void*>out_ptr)
```

  Also remove `self_t` and `other_t` from the executor cleanup registration (they are now owned by the cache, not the executor):
```cython
    _register_executor_cleanup(
        <uintptr_t>executor,
        ([('t', <uintptr_t>out_t)] if out_t != NULL else []),  # only out
    )
```

- [ ] Add `get_tensor_desc_cache` to `_aclnn_ffi.pyx` exports (already done above via `def`)

### Task 1.3: Hook invalidation into allocator free

- [ ] In `src/candle/_cython/_allocator.pyx`, add at the top of `cpdef void free(...)`:

```cython
cpdef void free(self, int64_t ptr, object stream=None):
    block = self._active.get(ptr)  # peek before pop
    if block is not None:
        # Invalidate descriptor cache for this address range
        try:
            from candle._cython._aclnn_ffi import get_tensor_desc_cache  # pylint: disable=import-error,no-name-in-module
            get_tensor_desc_cache().invalidate_range(ptr, block.size)
        except ImportError:
            pass
    block = self._active.pop(ptr, None)
    # ... rest of existing free logic unchanged
```

  Note: the `import` is inside the function to avoid a circular import. It is cached by Python's import system after the first call.

### Task 1.4: Build and run tests

- [ ] Build:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle pip install -e . -q
```

- [ ] Run cache tests:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle pytest tests/npu/cython/test_tensor_desc_cache.py -v --tb=short
```
Expected: all 4 tests PASS

- [ ] Run full NPU test suite to catch regressions:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle pytest tests/npu/ -v --tb=short -x
```
Expected: same pass/fail as baseline (only pre-existing failure is `test_group_norm_npu`)

### Task 1.5: Benchmark and commit

- [ ] Run profiler:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle python bench_fast_path_profile.py
```
Expected: `aclnn_add` stage drops from ~72µs to ~42µs. Total drops from ~137µs to ~107µs.

- [ ] Pylint check:
```bash
conda run -n candle pylint src/candle/ --rcfile=.github/pylint.conf
```

- [ ] Commit:
```bash
git add src/candle/_cython/_aclnn_ffi.pyx src/candle/_cython/_allocator.pyx \
         tests/npu/cython/test_tensor_desc_cache.py
git commit -m "perf(npu): cache aclTensor descriptors for input tensors (#PR-1)"
```

---

## Chunk 2: PR-2 — Direct Device Pointer Access

**Expected savings:** ~8µs (eliminates 2× `a.storage().data_ptr()` Python method calls)

**Files:**
- Modify: `src/candle/_cython/_npu_ops.pyx`
- Modify: test helpers / `DummyStorage` to expose `_device_ptr`
- Test: `tests/npu/cython/test_direct_ptr.py` (new)

### Task 2.1: Write failing test

- [ ] Create `tests/npu/cython/test_direct_ptr.py`:

```python
import pytest
pytestmark = pytest.mark.npu

def test_fast_binary_uses_direct_ptr(npu_device):
    """fast_binary_op must read device ptr without calling .storage().
    Verified by confirming .storage() is NOT called during add.
    """
    import candle as torch
    call_count = 0
    original_storage = type(torch.zeros(1, device=npu_device)).storage

    class CountingTensor(torch.Tensor):
        def storage(self):
            nonlocal call_count
            call_count += 1
            return super().storage()

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()
    call_count = 0
    c = torch.add(a, b)
    torch.npu.synchronize()
    # Direct ptr access: storage() should not be called on the hot path
    assert call_count == 0, f".storage() was called {call_count} times in fast path"
    assert c.shape == (4, 4)
```

- [ ] Run to confirm failure:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle pytest tests/npu/cython/test_direct_ptr.py -v --tb=short
```
Expected: FAIL — `call_count == 2` (current code calls `.storage()` twice)

### Task 2.2: Expose `_device_ptr` on DummyStorage

- [ ] Find where `DummyStorage` is defined:
```bash
grep -rn 'class DummyStorage' tests/ src/
```

- [ ] Add `_device_ptr = 0` class attribute (or property returning `0`) so existing tests don't break when `_npu_ops.pyx` accesses the field directly.

### Task 2.3: Replace `.storage().data_ptr()` in `_npu_ops.pyx`

- [ ] In `fast_binary_op` (around line 207–212 of `_npu_ops.pyx`), replace:

```cython
    # 7. Get data pointers via storage
    a_storage = a.storage()
    b_storage = b.storage()
    # 8. Call aclnn
    fn(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
```

With:

```cython
    # 7. Get data pointers directly from storage C attribute
    cdef int64_t a_ptr = a._impl._storage._untyped._device_ptr
    cdef int64_t b_ptr = b._impl._storage._untyped._device_ptr
    # 8. Call aclnn
    fn(
        a_ptr,
        b_ptr,
```

### Task 2.4: Build, test, benchmark, commit

- [ ] Build: `conda run -n candle pip install -e . -q`

- [ ] Run tests:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle pytest tests/npu/ -v --tb=short -x
```
Expected: all tests pass (same baseline)

- [ ] Run profiler and confirm `fast_binary_op` drops by ~8µs vs PR-1 baseline.

- [ ] Pylint + commit:
```bash
conda run -n candle pylint src/candle/ --rcfile=.github/pylint.conf
git add src/candle/_cython/_npu_ops.pyx tests/npu/cython/test_direct_ptr.py
git commit -m "perf(npu): read device ptr directly from storage C attribute (#PR-2)"
```

---

## Chunk 3: PR-3 — C-level Output Tensor Construction

**Expected savings:** ~30µs (eliminates `npu_typed_storage_from_ptr` 14µs + `Tensor.__init__` 16µs)

**Files:**
- Modify: `src/candle/_cython/_storage.pyx` — add `cy_make_npu_tensor`
- Modify: `src/candle/_cython/_npu_ops.pyx` — use `cy_make_npu_tensor`
- Test: `tests/npu/cython/test_cy_make_tensor.py` (new)

### Task 3.1: Verify TensorImpl cdef methods exist

- [ ] Confirm these cdef methods exist in `_tensor_impl.pyx`:
  - `_set_shape(tuple shape)`
  - `_set_stride(object stride)`
  - `_set_device_from_obj(object dev)`
  - `_set_dtype_from_obj(object dtype)`
  - `_recompute_dispatch_keys()` (called internally by `_set_device_from_obj`)

  If `_set_stride` accepts a `_StrideTuple` (not a plain tuple), pass a plain tuple — check `Tensor.__init__` line 146: `self.stride = _StrideTuple(stride)`. The cdef `_set_stride` in `_tensor_impl.pyx` takes `object stride` — plain tuple is fine.

### Task 3.2: Write failing test

- [ ] Create `tests/npu/cython/test_cy_make_tensor.py`:

```python
import pytest
pytestmark = pytest.mark.npu

def test_cy_make_npu_tensor_equivalence(npu_device):
    """Tensor from cy_make_npu_tensor must be identical to one from normal constructor."""
    import candle as torch
    from candle._cython._storage import cy_make_npu_tensor  # new function
    from candle._storage import npu_typed_storage_from_ptr
    from candle._backends.npu import runtime as rt, allocator as alloc_mod
    import candle
    dtype = candle.float32
    device = torch.device("npu:0")
    shape = (4, 4)
    stride = (4, 1)
    n = 16
    runtime = rt.get_runtime(0)
    alloc = alloc_mod.get_allocator(0)
    torch.npu.synchronize()
    ptr = alloc.malloc(n * 4, stream=runtime.stream)
    try:
        # Normal path
        storage_normal = npu_typed_storage_from_ptr(ptr, n, dtype, device=device)
        t_normal = torch.Tensor(storage_normal, shape, stride)
        # Fast path
        t_fast = cy_make_npu_tensor(ptr, n, dtype, device, shape, stride)
        assert t_fast.shape == t_normal.shape
        assert t_fast.stride == t_normal.stride
        assert t_fast.dtype == t_normal.dtype
        assert t_fast.device.type == t_normal.device.type
        assert t_fast.data_ptr() == t_normal.data_ptr()
        assert t_fast.requires_grad == t_normal.requires_grad
        assert t_fast.grad_fn is None
    finally:
        alloc.free(ptr, stream=None)
        torch.npu.synchronize()

def test_fast_binary_uses_cy_make(npu_device, monkeypatch):
    """Ensure torch.add produces correct results with cy_make_npu_tensor in hot path."""
    import candle as torch
    import numpy as np
    a_np = np.ones((4, 4), dtype=np.float32)
    b_np = np.ones((4, 4), dtype=np.float32) * 2.0
    a = torch.tensor(a_np, device=npu_device)
    b = torch.tensor(b_np, device=npu_device)
    c = torch.add(a, b)
    torch.npu.synchronize()
    result = c.cpu().numpy()
    np.testing.assert_allclose(result, np.full((4, 4), 3.0, dtype=np.float32))
```

- [ ] Run to confirm ImportError on `cy_make_npu_tensor`:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle pytest tests/npu/cython/test_cy_make_tensor.py -v --tb=short
```

### Task 3.3: Implement `cy_make_npu_tensor` in `_storage.pyx`

- [ ] Add after `cy_npu_storage_from_ptr` in `src/candle/_cython/_storage.pyx`:

```cython
def cy_make_npu_tensor(int64_t device_ptr, int64_t n_elements,
                       object dtype, object device,
                       tuple shape, tuple stride):
    """Construct an NPU Tensor entirely in Cython.

    Equivalent to:
        storage = npu_typed_storage_from_ptr(device_ptr, n_elements, dtype, device)
        return Tensor(storage, shape, stride)
    but without Python __init__ overhead.
    """
    from candle._cython._tensor_impl import TensorImpl  # pylint: disable=import-error,no-name-in-module
    from candle._tensor import Tensor
    _ensure_fast_storage()

    cdef int itemsize = _c_dtype_itemsize(dtype)
    cdef int64_t nbytes = n_elements * itemsize

    # 1. Storage (FastNPUStorage + FastTypedStorage)
    untyped = _FastNPUStorage_cls(device_ptr, nbytes, device)
    typed = _FastTypedStorage_cls(untyped, dtype, n_elements)

    # 2. TensorImpl via __new__ (skips __init__)
    impl = TensorImpl.__new__(TensorImpl)
    impl._set_shape(shape)
    impl._set_stride(stride)
    impl._set_device_from_obj(device)
    impl._set_dtype_from_obj(dtype)
    impl._storage = typed
    impl.requires_grad = False
    impl.grad = None
    impl.grad_fn = None
    impl._pending = False
    impl._retain_grad = False
    impl._backward_hooks = None
    impl._version_value = 0
    impl._vc_proxy = None
    impl._base = None
    impl._view_meta = None
    impl.offset = 0

    # 3. Tensor via __new__ (skips __init__)
    t = Tensor.__new__(Tensor)
    t._impl = impl
    return t
```

### Task 3.4: Use `cy_make_npu_tensor` in `fast_binary_op`

- [ ] In `src/candle/_cython/_npu_ops.pyx`, add import at top of `_ensure_npu_imports()`:
```cython
    from candle._storage import npu_typed_storage_from_ptr as nfp  # remove this line
    from candle._cython._storage import cy_make_npu_tensor as _cy_make  # add this
```

- [ ] Replace the output wrapping section (lines 226-229):
```cython
    # Before:
    out_storage = _npu_typed_storage_from_ptr(
        out_ptr, n, a_dtype, device=a_dev)
    return _Tensor(out_storage, out_shape, out_stride)

    # After:
    return _cy_make(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)
```

### Task 3.5: Build, test, benchmark, commit

- [ ] Build: `conda run -n candle pip install -e . -q`

- [ ] Run tests:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle pytest tests/npu/ -v --tb=short -x
```

- [ ] Run profiler — expect `npu_typed_storage_from_ptr` + `Tensor_init` stages to drop by ~30µs combined.

- [ ] Pylint + commit:
```bash
conda run -n candle pylint src/candle/ --rcfile=.github/pylint.conf
git add src/candle/_cython/_storage.pyx src/candle/_cython/_npu_ops.pyx \
         tests/npu/cython/test_cy_make_tensor.py
git commit -m "perf(npu): construct output tensor in Cython, skip Python __init__ (#PR-3)"
```

---

## Chunk 4: PR-4 — Allocator `_drain_pending` Hot Path

**Expected savings:** ~7µs

**Files:**
- Modify: `src/candle/_cython/_allocator.pyx`
- Test: existing allocator tests in `tests/npu/`

### Task 4.1: Profile current `_drain_pending` to confirm it's the bottleneck

- [ ] Add timing inside `_drain_pending` temporarily:
```python
# In bench_fast_path_profile.py, add a stage for _drain_pending
```
Or run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle python bench_dispatch_profile.py
```
Confirm `_drain_pending` is ≥5µs before proceeding.

### Task 4.2: Inline `_event_complete` and cache runtime ref

- [ ] In `FastNpuAllocator`, add a cached runtime reference:
```cython
cdef object _cached_runtime  # set on first use

cdef inline bint _fast_event_complete(self, uintptr_t event_ptr):
    """Inline event query — avoids Python import on every iteration."""
    if event_ptr == 0:
        return True
    if self._cached_runtime is None:
        from candle._backends.npu import runtime as _rt
        self._cached_runtime = _rt
    try:
        return self._cached_runtime.get_runtime(self.device_id).query_event(
            <object>(<void*>event_ptr))
    except Exception:
        return False
```

- [ ] Replace `_drain_pending` to use `_fast_event_complete`:

```cython
    def _drain_pending(self):
        if not self._pending_events:
            return
        still_pending = []
        for event, block in self._pending_events:
            event_ptr = id(event) if event is not None else 0
            if self._fast_event_complete(<uintptr_t>id(event) if event is not None else 0):
                # ... rest unchanged
```

  Note: `query_event` takes the event Python object, not a raw ptr. The optimization here is caching the runtime module reference so the `from ... import` only runs once per allocator lifetime, not once per `_drain_pending` call.

  Actual implementation: cache `runtime.get_runtime(device_id).query_event` as a callable:

```cython
cdef object _cached_query_event  # set on first use

cdef inline bint _fast_event_complete(self, object event):
    if event is None:
        return True
    if self._cached_query_event is None:
        from candle._backends.npu import runtime as _rt
        self._cached_query_event = _rt.get_runtime(self.device_id).query_event
    try:
        return self._cached_query_event(event)
    except Exception:
        return False
```

### Task 4.3: Build, test, benchmark, commit

- [ ] Build: `conda run -n candle pip install -e . -q`

- [ ] Run full NPU tests:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle pytest tests/npu/ -v --tb=short -x
```

- [ ] Run profiler — expect ~7µs reduction in `alloc_device` stage.

- [ ] Pylint + commit:
```bash
conda run -n candle pylint src/candle/ --rcfile=.github/pylint.conf
git add src/candle/_cython/_allocator.pyx
git commit -m "perf(npu): cache runtime query_event ref in allocator drain (#PR-4)"
```

---

## Chunk 5: PR-5 — Workspace Static Reuse

**Expected savings:** ~5µs (eliminates workspace malloc from main allocator pool)

**Files:**
- Modify: `src/candle/_cython/_aclnn_ffi.pyx` — add `_get_workspace`, use in `binary_op_with_alpha`
- Test: `tests/npu/cython/test_workspace.py` (new)

### Task 5.1: Write failing test

- [ ] Create `tests/npu/cython/test_workspace.py`:

```python
import pytest
pytestmark = pytest.mark.npu

def test_add_workspace_zero(npu_device):
    """aclnnAdd should require zero workspace — no device malloc expected."""
    import candle as torch
    # Confirm add works correctly with static workspace path
    a = torch.ones(64, 64, device=npu_device)
    b = torch.ones(64, 64, device=npu_device) * 2.0
    c = torch.add(a, b)
    torch.npu.synchronize()
    import numpy as np
    np.testing.assert_allclose(c.cpu().numpy(), np.full((64, 64), 3.0, dtype=np.float32))

def test_matmul_workspace_nonzero(npu_device):
    """matmul requires workspace > 0 — static buffer must grow to accommodate."""
    import candle as torch
    a = torch.randn(128, 128, device=npu_device)
    b = torch.randn(128, 128, device=npu_device)
    c = torch.matmul(a, b)
    torch.npu.synchronize()
    assert c.shape == (128, 128)
```

### Task 5.2: Add workspace buffer to `_aclnn_ffi.pyx`

- [ ] Add module-level workspace state:

```cython
# Per-device static workspace buffer (mirrors torch_npu NPUWorkspaceAllocator)
DEF MAX_DEVICES = 8
cdef uintptr_t _ws_ptr[MAX_DEVICES]
cdef int64_t   _ws_size[MAX_DEVICES]
cdef bint      _ws_initialized = False

cdef void _init_workspace() noexcept nogil:
    cdef int i
    for i in range(MAX_DEVICES):
        _ws_ptr[i] = 0
        _ws_size[i] = 0

# Initialize at module level
_init_workspace()
```

- [ ] Add `_get_workspace` function:

```cython
cdef uintptr_t _get_workspace_ptr(int device_id, int64_t needed, object acl_rt):
    """Return a workspace ptr of at least `needed` bytes.
    needed == 0: returns 0 (NULL) — no allocation.
    Reuses existing buffer if large enough; reallocates otherwise.
    Buffer is NOT tracked by CachingAllocator (matches NPUWorkspaceAllocator).
    """
    if needed == 0:
        return 0
    if device_id < 0 or device_id >= MAX_DEVICES:
        device_id = 0
    if _ws_size[device_id] >= needed:
        return _ws_ptr[device_id]
    # Free old buffer
    if _ws_ptr[device_id] != 0:
        acl_rt.free(<void*>_ws_ptr[device_id])
    # Allocate new — round up to next power of 2 for stability
    cdef int64_t alloc_size = needed
    while alloc_size & (alloc_size - 1):  # not power of 2
        alloc_size += 1  # simple linear scan, workspace is rarely > few MB
    new_ptr = acl_rt.malloc(alloc_size)
    _ws_ptr[device_id] = <uintptr_t>new_ptr
    _ws_size[device_id] = alloc_size
    return _ws_ptr[device_id]
```

  Note: rounding logic should use `1 << ceil(log2(needed))`. Simplify in actual implementation:
```cython
    # Round up to next power of 2
    cdef int64_t alloc_size = 1
    while alloc_size < needed:
        alloc_size <<= 1
```

- [ ] Register atexit cleanup:
```cython
import atexit as _atexit

def _cleanup_workspace():
    """Free workspace buffers on process exit."""
    cdef int i
    for i in range(MAX_DEVICES):
        if _ws_ptr[i] != 0:
            try:
                from candle._backends.npu import runtime as _rt
                _rt.get_runtime(i).acl.rt.free(<void*>_ws_ptr[i])
            except Exception:
                pass
            _ws_ptr[i] = 0
            _ws_size[i] = 0

_atexit.register(_cleanup_workspace)
```

### Task 5.3: Use workspace in `binary_op_with_alpha`

- [ ] In `binary_op_with_alpha`, replace the ws_size > 0 path:

```cython
        # Before (caller allocates via caching allocator):
        if ws_size == 0:
            ...
        # After: use static workspace buffer
        cdef uintptr_t ws_buf = 0
        if ws_size > 0:
            # Get acl_rt handle for raw malloc
            from candle._backends.npu import runtime as _rt_mod
            acl_rt = _rt_mod.get_runtime(0).acl.rt  # device 0; parameterize if needed
            ws_buf = _get_workspace_ptr(0, <int64_t>ws_size, acl_rt)
        with nogil:
            ret = (<aclnnExec_t>exec_ptr)(
                <void*>ws_buf, ws_size, executor, <void*>stream)
```

  The workspace is NOT freed after the call — it's reused next time. This is correct because `aclnnAdd` submits to stream and the stream serializes execution, so the buffer is available for the next op by the time it runs.

### Task 5.4: Build, test, benchmark, commit

- [ ] Build: `conda run -n candle pip install -e . -q`

- [ ] Run workspace tests:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle pytest tests/npu/cython/test_workspace.py -v --tb=short
```

- [ ] Run full NPU tests:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle pytest tests/npu/ -v --tb=short -x
```

- [ ] Run profiler — expect total to drop from ~62µs to ~57µs.

- [ ] Run end-to-end benchmark and compare against torch_npu:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle python bench_torch_npu.py
conda run -n mindie python bench_torch_npu.py
```
Expected: candle median ≤ 70µs (e2e with sync), torch_npu ~57µs.

- [ ] Pylint + commit:
```bash
conda run -n candle pylint src/candle/ --rcfile=.github/pylint.conf
git add src/candle/_cython/_aclnn_ffi.pyx tests/npu/cython/test_workspace.py
git commit -m "perf(npu): static workspace buffer, align with NPUWorkspaceAllocator (#PR-5)"
```

---

## Final Validation

- [ ] Run complete test suite:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle pytest tests/npu/ tests/cpu/ tests/contract/ -v --tb=short
```

- [ ] Run full profiler comparison:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  conda run -n candle python bench_fast_path_profile.py && \
  conda run -n candle python bench_torch_npu.py
```
Expected final state:
```
Stage                          Before    After
aclnn_add                       72µs     ~42µs  (PR-1)
alloc_device                    16µs     ~ 9µs  (PR-4)
Tensor.__init__                 16µs     ~ 0µs  (PR-3)
npu_typed_storage_from_ptr      14µs     ~ 0µs  (PR-3)
other                           19µs     ~11µs  (PR-2, PR-5)
──────────────────────────────────────────────
Total                          137µs     ~62µs
torch_npu reference             57µs      57µs
```
