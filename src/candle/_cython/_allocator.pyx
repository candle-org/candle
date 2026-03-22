# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython fast-path for NPU memory allocator.

Replaces dict-based stats tracking with flat C int64 array and
Block with a cdef class for minimal Python overhead.
"""

from libc.stdint cimport int64_t

DEF SMALL_POOL_THRESHOLD = 1048576  # 1 << 20
DEF ROUNDING_BYTES = 512
DEF MAX_SPLIT_SIZE = 1048576  # 1 << 20

# ---------------------------------------------------------------------------
# Stat indices — flat array replaces dict[str] -> int
# ---------------------------------------------------------------------------
# Layout: each metric group has 4 slots (current, peak, allocated, freed)
# Groups: allocated, allocated_bytes, active, active_bytes, segment,
#         reserved_bytes, inactive_split, inactive_split_bytes,
#         requested_bytes, oversize_allocations, oversize_segments
# Each group has 3 pools: all, small_pool, large_pool (except oversize: just one)
# Total: 9 groups * 3 pools * 4 slots + 2 groups * 1 pool * 4 slots + 5 scalars

DEF STAT_COUNT = 122  # 11 groups * 3 pools * 4 slots + 2 oversize * 4 + 6 scalars

# Pool indices
DEF POOL_ALL = 0
DEF POOL_SMALL = 1
DEF POOL_LARGE = 2

# Metric group base offsets (each group = 3 pools * 4 slots = 12)
DEF GRP_ALLOCATED = 0        # 0..11
DEF GRP_ALLOCATED_BYTES = 12 # 12..23
DEF GRP_ACTIVE = 24          # 24..35
DEF GRP_ACTIVE_BYTES = 36    # 36..47
DEF GRP_SEGMENT = 48         # 48..59
DEF GRP_RESERVED_BYTES = 60  # 60..71
DEF GRP_INACTIVE_SPLIT = 72  # 72..83
DEF GRP_INACTIVE_SPLIT_BYTES = 84  # 84..95

# Slot offsets within a pool
DEF SLOT_CURRENT = 0
DEF SLOT_PEAK = 1
DEF SLOT_ALLOCATED = 2
DEF SLOT_FREED = 3

# Requested bytes (same layout as above groups)
# But we need to fit it — let's use a simpler approach
# Actually let's just map the requested_bytes into the same pattern
# We have 9 groups * 12 = 108 but _ALL_STAT_KEYS has 96 entries
# Let me just use a dict mapping for the stat names and flat array

# Simpler approach: use the _ALL_STAT_KEYS list order directly
# Index 0 = "allocated.all.current", index 1 = "allocated.all.peak", etc.

# Scalar stats at the end
DEF IDX_NUM_ALLOC_RETRIES = 116
DEF IDX_NUM_OOMS = 117
DEF IDX_NUM_SYNC_ALL_STREAMS = 118
DEF IDX_NUM_DEVICE_ALLOC = 119
DEF IDX_NUM_DEVICE_FREE = 120

# Build name->index mapping at module level
_STAT_NAMES = [
    "allocated.all.current", "allocated.all.peak",
    "allocated.all.allocated", "allocated.all.freed",
    "allocated.small_pool.current", "allocated.small_pool.peak",
    "allocated.small_pool.allocated", "allocated.small_pool.freed",
    "allocated.large_pool.current", "allocated.large_pool.peak",
    "allocated.large_pool.allocated", "allocated.large_pool.freed",
    "allocated_bytes.all.current", "allocated_bytes.all.peak",
    "allocated_bytes.all.allocated", "allocated_bytes.all.freed",
    "allocated_bytes.small_pool.current", "allocated_bytes.small_pool.peak",
    "allocated_bytes.small_pool.allocated", "allocated_bytes.small_pool.freed",
    "allocated_bytes.large_pool.current", "allocated_bytes.large_pool.peak",
    "allocated_bytes.large_pool.allocated", "allocated_bytes.large_pool.freed",
    "active.all.current", "active.all.peak",
    "active.all.allocated", "active.all.freed",
    "active.small_pool.current", "active.small_pool.peak",
    "active.small_pool.allocated", "active.small_pool.freed",
    "active.large_pool.current", "active.large_pool.peak",
    "active.large_pool.allocated", "active.large_pool.freed",
    "active_bytes.all.current", "active_bytes.all.peak",
    "active_bytes.all.allocated", "active_bytes.all.freed",
    "active_bytes.small_pool.current", "active_bytes.small_pool.peak",
    "active_bytes.small_pool.allocated", "active_bytes.small_pool.freed",
    "active_bytes.large_pool.current", "active_bytes.large_pool.peak",
    "active_bytes.large_pool.allocated", "active_bytes.large_pool.freed",
    "segment.all.current", "segment.all.peak",
    "segment.all.allocated", "segment.all.freed",
    "segment.small_pool.current", "segment.small_pool.peak",
    "segment.small_pool.allocated", "segment.small_pool.freed",
    "segment.large_pool.current", "segment.large_pool.peak",
    "segment.large_pool.allocated", "segment.large_pool.freed",
    "reserved_bytes.all.current", "reserved_bytes.all.peak",
    "reserved_bytes.all.allocated", "reserved_bytes.all.freed",
    "reserved_bytes.small_pool.current", "reserved_bytes.small_pool.peak",
    "reserved_bytes.small_pool.allocated", "reserved_bytes.small_pool.freed",
    "reserved_bytes.large_pool.current", "reserved_bytes.large_pool.peak",
    "reserved_bytes.large_pool.allocated", "reserved_bytes.large_pool.freed",
    "inactive_split.all.current", "inactive_split.all.peak",
    "inactive_split.all.allocated", "inactive_split.all.freed",
    "inactive_split.small_pool.current", "inactive_split.small_pool.peak",
    "inactive_split.small_pool.allocated", "inactive_split.small_pool.freed",
    "inactive_split.large_pool.current", "inactive_split.large_pool.peak",
    "inactive_split.large_pool.allocated", "inactive_split.large_pool.freed",
    "inactive_split_bytes.all.current", "inactive_split_bytes.all.peak",
    "inactive_split_bytes.all.allocated", "inactive_split_bytes.all.freed",
    "inactive_split_bytes.small_pool.current", "inactive_split_bytes.small_pool.peak",
    "inactive_split_bytes.small_pool.allocated", "inactive_split_bytes.small_pool.freed",
    "inactive_split_bytes.large_pool.current", "inactive_split_bytes.large_pool.peak",
    "inactive_split_bytes.large_pool.allocated", "inactive_split_bytes.large_pool.freed",
    "oversize_allocations.current", "oversize_allocations.peak",
    "oversize_allocations.allocated", "oversize_allocations.freed",
    "oversize_segments.current", "oversize_segments.peak",
    "oversize_segments.allocated", "oversize_segments.freed",
    "requested_bytes.all.current", "requested_bytes.all.peak",
    "requested_bytes.all.allocated", "requested_bytes.all.freed",
    "requested_bytes.small_pool.current", "requested_bytes.small_pool.peak",
    "requested_bytes.small_pool.allocated", "requested_bytes.small_pool.freed",
    "requested_bytes.large_pool.current", "requested_bytes.large_pool.peak",
    "requested_bytes.large_pool.allocated", "requested_bytes.large_pool.freed",
    "num_alloc_retries", "num_ooms", "num_sync_all_streams",
    "num_device_alloc", "num_device_free", "max_split_size",
]

# Build reverse mapping
_NAME_TO_IDX = {name: i for i, name in enumerate(_STAT_NAMES)}

# Pre-compute bump indices for each (prefix, pool) combination
# _bump(prefix, pool, current, allocated, freed) updates:
#   prefix.pool.current, prefix.pool.peak, prefix.pool.allocated, prefix.pool.freed
#   prefix.all.current, prefix.all.peak, prefix.all.allocated, prefix.all.freed (if pool != "all")

cdef dict _BUMP_INDICES = {}

def _precompute_bump_indices():
    for prefix in ["allocated", "allocated_bytes", "active", "active_bytes",
                    "segment", "reserved_bytes", "inactive_split",
                    "inactive_split_bytes", "requested_bytes"]:
        for pool in ["small_pool", "large_pool", "all"]:
            targets = []
            if pool != "all":
                # Pool-specific indices
                base = _NAME_TO_IDX.get(f"{prefix}.{pool}.current")
                if base is not None:
                    targets.append((base, base + 1, base + 2, base + 3))
                # All-pool indices
                base_all = _NAME_TO_IDX.get(f"{prefix}.all.current")
                if base_all is not None:
                    targets.append((base_all, base_all + 1, base_all + 2, base_all + 3))
            else:
                base = _NAME_TO_IDX.get(f"{prefix}.all.current")
                if base is not None:
                    targets.append((base, base + 1, base + 2, base + 3))
            _BUMP_INDICES[(prefix, pool)] = targets
    # Handle oversize_ prefixes (no pool subdivision)
    for prefix in ["oversize_allocations", "oversize_segments"]:
        base = _NAME_TO_IDX.get(f"{prefix}.current")
        if base is not None:
            _BUMP_INDICES[(prefix, "all")] = [(base, base + 1, base + 2, base + 3)]

_precompute_bump_indices()


# ---------------------------------------------------------------------------
# FastBlock — cdef class replacing Python Block
# ---------------------------------------------------------------------------

cdef class FastBlock:
    cdef public int64_t ptr
    cdef public int64_t size
    cdef public int64_t requested
    cdef public str pool
    cdef public object stream
    cdef public object event
    cdef public set stream_uses
    cdef public int event_count

    def __init__(self, int64_t ptr, int64_t size, int64_t requested,
                 str pool, object stream):
        self.ptr = ptr
        self.size = size
        self.requested = requested
        self.pool = pool
        self.stream = stream
        self.event = None
        self.stream_uses = set()
        self.event_count = 0


# ---------------------------------------------------------------------------
# FastNpuAllocator
# ---------------------------------------------------------------------------

cdef inline int64_t _round_size(int64_t size) nogil:
    return ((size + ROUNDING_BYTES - 1) // ROUNDING_BYTES) * ROUNDING_BYTES


cdef class FastNpuAllocator:
    cdef public int device_id
    cdef int64_t _stats_arr[STAT_COUNT + 6]  # +6 for max_split_size and padding
    cdef public dict _active
    cdef public dict _cached
    cdef public list _pending_events
    cdef public list _event_pool
    cdef public bint _event_pool_ready
    cdef public object max_split_size
    cdef public object gc_threshold
    cdef dict __dict__  # allow monkey-patching (needed by tests)

    def __init__(self, int device_id):
        from candle._backends.npu.allocator import _load_alloc_conf, _parse_max_split_size_mb
        conf = _load_alloc_conf()
        self.max_split_size = _parse_max_split_size_mb(conf.get("max_split_size_mb"))
        self.gc_threshold = conf.get("garbage_collection_threshold")
        self.device_id = device_id
        self._active = {}
        self._cached = {"small_pool": [], "large_pool": []}
        self._pending_events = []
        self._event_pool = []
        self._event_pool_ready = False
        cdef int i
        for i in range(STAT_COUNT + 6):
            self._stats_arr[i] = 0

    cdef inline void _bump_fast(self, str prefix, str pool,
                                 int64_t current, int64_t allocated,
                                 int64_t freed):
        """Fast stat bump using pre-computed indices."""
        cdef list targets = _BUMP_INDICES.get((prefix, pool), [])
        cdef tuple t
        cdef int ci, pi, ai, fi
        for t in targets:
            ci = t[0]; pi = t[1]; ai = t[2]; fi = t[3]
            if current:
                self._stats_arr[ci] += current
                if self._stats_arr[ci] > self._stats_arr[pi]:
                    self._stats_arr[pi] = self._stats_arr[ci]
            if allocated:
                self._stats_arr[ai] += allocated
            if freed:
                self._stats_arr[fi] += freed

    def _bump(self, prefix, pool, current=0, allocated=0, freed=0):
        """Public bump — delegates to fast C path."""
        self._bump_fast(prefix, pool, current, allocated, freed)

    cdef void _track_alloc(self, int64_t requested, int64_t allocated, str pool):
        self._bump_fast("allocated_bytes", pool, allocated, allocated, 0)
        self._bump_fast("requested_bytes", pool, requested, requested, 0)
        self._bump_fast("active_bytes", pool, allocated, allocated, 0)
        self._bump_fast("allocated", pool, 1, 1, 0)
        self._bump_fast("active", pool, 1, 1, 0)
        self._bump_fast("segment", pool, 1, 1, 0)
        self._bump_fast("reserved_bytes", pool, allocated, allocated, 0)
        self._stats_arr[IDX_NUM_DEVICE_ALLOC] += 1

    cdef void _track_reuse(self, int64_t requested, int64_t allocated, str pool):
        self._bump_fast("allocated_bytes", pool, allocated, allocated, 0)
        self._bump_fast("requested_bytes", pool, requested, requested, 0)
        self._bump_fast("allocated", pool, 1, 1, 0)
        self._bump_fast("active_bytes", pool, allocated, allocated, 0)
        self._bump_fast("active", pool, 1, 1, 0)

    cdef void _track_free(self, FastBlock block):
        self._bump_fast("allocated_bytes", block.pool, -block.size, 0, block.size)
        self._bump_fast("requested_bytes", block.pool, -block.requested, 0, block.requested)
        self._bump_fast("allocated", block.pool, -1, 0, 1)
        self._bump_fast("active_bytes", block.pool, -block.size, 0, block.size)
        self._bump_fast("active", block.pool, -1, 0, 1)

    cdef str _pool_for_size(self, int64_t size):
        if size < SMALL_POOL_THRESHOLD:
            return "small_pool"
        return "large_pool"

    cdef FastBlock _find_cached(self, int64_t size, str pool):
        cdef list blocks = self._cached[pool]
        cdef int idx
        cdef FastBlock block
        for idx in range(len(blocks)):
            block = <FastBlock>blocks[idx]
            if block.size >= size:
                blocks.pop(idx)
                return block
        return None

    cdef tuple _split_block(self, FastBlock block, int64_t size):
        if block.size - size <= 0:
            return block, None
        if self.max_split_size is not None and block.size > self.max_split_size:
            return block, None
        if block.size > MAX_SPLIT_SIZE:
            return block, None
        cdef int64_t remaining = block.size - size
        block.size = size
        remainder = FastBlock(block.ptr + size, remaining, 0, block.pool, None)
        return block, remainder

    def _drain_pending(self):
        still_pending = []
        for event, block in self._pending_events:
            if self._event_complete(event):
                if event is not None:
                    self._event_pool.append(event)
                block.event_count -= 1
                if block.event_count == 0:
                    self._cached[block.pool].append(block)
                    self._bump_fast("inactive_split_bytes", block.pool,
                                    block.size, block.size, 0)
                    self._bump_fast("inactive_split", block.pool, 1, 1, 0)
            else:
                still_pending.append((event, block))
        self._pending_events = still_pending

    def _ensure_event_pool(self, runtime):
        if self._event_pool_ready:
            return
        cdef int i
        for i in range(64):
            try:
                event = runtime.create_event(False, False, False)
                self._event_pool.append(event)
            except Exception:
                break
        self._event_pool_ready = True

    def _record_event(self, stream):
        from candle._backends.npu import runtime as npu_runtime
        from candle._backends.npu import state as npu_state
        runtime = npu_runtime.get_runtime(self.device_id)
        if stream is None:
            stream = npu_state.current_stream(self.device_id).stream
        try:
            self._ensure_event_pool(runtime)
            if self._event_pool:
                event = self._event_pool.pop()
            else:
                event = runtime.create_event(False, False, False)
            runtime.record_event(event, stream)
            return event
        except Exception:
            return None

    def _event_complete(self, event):
        if event is None:
            return True
        from candle._backends.npu import runtime as npu_runtime
        runtime = npu_runtime.get_runtime(self.device_id)
        try:
            return runtime.query_event(event)
        except Exception:
            return True

    def _sync_device(self):
        from candle._backends.npu import runtime as npu_runtime
        runtime = npu_runtime.get_runtime(self.device_id)
        try:
            runtime.synchronize_device()
        except Exception:
            pass

    def _raw_free(self, ptr):
        from candle._backends.npu import runtime as npu_runtime
        runtime = npu_runtime.get_runtime(self.device_id)
        runtime.activate()
        ret = npu_runtime.acl.rt.free(ptr)
        if ret != 0:
            raise RuntimeError(f"acl.rt.free failed: {ret}")
        self._stats_arr[IDX_NUM_DEVICE_FREE] += 1

    def _raw_malloc(self, size):
        from candle._backends.npu import runtime as npu_runtime
        runtime = npu_runtime.get_runtime(self.device_id)
        runtime.activate()
        ptr, ret = npu_runtime.acl.rt.malloc(size, npu_runtime.ACL_MEM_MALLOC_HUGE_FIRST)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        return int(ptr), int(size)

    def _mem_get_info(self):
        from candle._backends.npu import runtime as npu_runtime
        return npu_runtime.mem_get_info(self.device_id)

    def _maybe_collect_garbage(self):
        if self.gc_threshold is None:
            return
        if not self._cached["small_pool"] and not self._cached["large_pool"]:
            return
        try:
            _, total = self._mem_get_info()
        except Exception:
            return
        if total <= 0:
            return
        cdef int64_t reserved = self._stats_arr[_NAME_TO_IDX["reserved_bytes.all.current"]]
        if reserved / float(total) <= self.gc_threshold:
            return
        self._drain_pending()
        self.empty_cache()

    def _oom_retry(self, allocated):
        self._stats_arr[IDX_NUM_OOMS] += 1
        self._maybe_collect_garbage()
        self.synchronize()
        self.empty_cache()
        ptr, _ = self._raw_malloc(allocated)
        self._stats_arr[IDX_NUM_ALLOC_RETRIES] += 1
        return int(ptr)

    cpdef int64_t malloc(self, int64_t size, object stream=None):
        cdef int64_t requested = size
        cdef int64_t allocated = _round_size(requested)
        cdef str pool = self._pool_for_size(allocated)
        self._drain_pending()
        self._maybe_collect_garbage()
        cdef FastBlock block = self._find_cached(allocated, pool)
        if block is None:
            from candle import npu
            npu._enforce_memory_fraction(allocated, device=self.device_id)
            try:
                ptr, _ = self._raw_malloc(allocated)
            except RuntimeError:
                ptr = self._oom_retry(allocated)
            block = FastBlock(ptr, allocated, requested, pool, stream)
            self._track_alloc(requested, allocated, pool)
        else:
            cached_size = block.size
            block, remainder = self._split_block(block, allocated)
            self._bump_fast("inactive_split_bytes", pool, -cached_size, 0, cached_size)
            self._bump_fast("inactive_split", pool, -1, 0, 1)
            if remainder is not None:
                self._cached[pool].append(remainder)
                self._bump_fast("inactive_split_bytes", pool, remainder.size, remainder.size, 0)
                self._bump_fast("inactive_split", pool, 1, 1, 0)
            self._track_reuse(requested, block.size, pool)
        block.requested = requested
        block.stream = stream
        self._active[block.ptr] = block
        return block.ptr

    cpdef void free(self, int64_t ptr, object stream=None):
        # Peek at block size before popping, for cache invalidation
        cdef object block_peek = self._active.get(ptr)
        if block_peek is not None:
            try:
                from candle._cython._aclnn_ffi import get_tensor_desc_cache  # pylint: disable=import-error,no-name-in-module
                get_tensor_desc_cache().invalidate_range(ptr, block_peek.size)
            except (ImportError, Exception):
                pass
        block = self._active.pop(ptr, None)
        if block is None:
            return
        if stream is None:
            stream = block.stream
        self._track_free(block)
        if not block.stream_uses:
            self._cached[block.pool].append(block)
            self._bump_fast("inactive_split_bytes", block.pool,
                            block.size, block.size, 0)
            self._bump_fast("inactive_split", block.pool, 1, 1, 0)
            return
        self._insert_events(block)

    def _insert_events(self, block):
        streams = block.stream_uses
        block.stream_uses = set()
        for s in streams:
            event = self._record_event(s)
            if event is None:
                self._sync_device()
                self._stats_arr[IDX_NUM_SYNC_ALL_STREAMS] += 1
                block.event_count = 0
                self._cached[block.pool].append(block)
                self._bump_fast("inactive_split_bytes", block.pool,
                                block.size, block.size, 0)
                self._bump_fast("inactive_split", block.pool, 1, 1, 0)
                return
            block.event_count += 1
            self._pending_events.append((event, block))
        event = self._record_event(block.stream)
        if event is not None:
            block.event_count += 1
            self._pending_events.append((event, block))

    def synchronize(self):
        self._sync_device()
        self._stats_arr[IDX_NUM_SYNC_ALL_STREAMS] += 1
        self._drain_pending()

        # Also drain runtime-level deferred frees so alloc.synchronize()
        # preserves the historical contract used by tests and internal callers.
        from candle._backends.npu import runtime as npu_runtime
        runtime = npu_runtime.get_runtime(self.device_id)

        frees = runtime._deferred_frees
        if frees:
            runtime._deferred_frees = []
            for ptr in frees:
                self.free(ptr, None)

        raw_frees = runtime._deferred_raw_frees
        if raw_frees:
            runtime._deferred_raw_frees = []
            for ptr in raw_frees:
                ret = npu_runtime.acl.rt.free(ptr)
                if ret != 0:
                    raise RuntimeError(f"acl.rt.free failed: {ret}")

        host_frees = runtime._deferred_host_frees
        if host_frees:
            runtime._deferred_host_frees = []
            for ptr in host_frees:
                ret = npu_runtime.acl.rt.free_host(ptr)
                if ret != 0:
                    raise RuntimeError(f"acl.rt.free_host failed: {ret}")


    def record_stream(self, ptr, stream):
        block = self._active.get(int(ptr))
        if block is None:
            return
        if stream != block.stream:
            block.stream_uses.add(stream)

    def empty_cache(self):
        for pool_name, blocks in self._cached.items():
            for block in blocks:
                self._raw_free(block.ptr)
                self._bump_fast("reserved_bytes", block.pool, -block.size, 0, block.size)
                self._bump_fast("segment", block.pool, -1, 0, 1)
                self._bump_fast("inactive_split_bytes", block.pool, -block.size, 0, block.size)
                self._bump_fast("inactive_split", block.pool, -1, 0, 1)
            blocks.clear()

    def reset_peak_memory_stats(self):
        cdef int i
        for i in range(len(_STAT_NAMES)):
            name = _STAT_NAMES[i]
            if name.endswith('.peak'):
                current_name = name.replace('.peak', '.current')
                ci = _NAME_TO_IDX.get(current_name, -1)
                if ci >= 0:
                    self._stats_arr[i] = self._stats_arr[ci]

    def reset_accumulated_memory_stats(self):
        cdef int i
        for i in range(len(_STAT_NAMES)):
            name = _STAT_NAMES[i]
            if name.endswith('.allocated') or name.endswith('.freed'):
                self._stats_arr[i] = 0
        self._stats_arr[IDX_NUM_ALLOC_RETRIES] = 0
        self._stats_arr[IDX_NUM_OOMS] = 0
        self._stats_arr[IDX_NUM_DEVICE_ALLOC] = 0
        self._stats_arr[IDX_NUM_DEVICE_FREE] = 0
        self._stats_arr[IDX_NUM_SYNC_ALL_STREAMS] = 0

    def memory_stats(self):
        cdef dict result = {}
        cdef int i
        for i in range(len(_STAT_NAMES)):
            result[_STAT_NAMES[i]] = self._stats_arr[i]
        if self.max_split_size is not None:
            result["max_split_size"] = self.max_split_size
        else:
            result["max_split_size"] = 0
        return result

    # Compatibility: expose _stats as a property that returns the dict
    @property
    def _stats(self):
        return self.memory_stats()

    @_stats.setter
    def _stats(self, value):
        # Allow setting from dict (for tests)
        if isinstance(value, dict):
            for k, v in value.items():
                idx = _NAME_TO_IDX.get(k, -1)
                if idx >= 0:
                    self._stats_arr[idx] = v
