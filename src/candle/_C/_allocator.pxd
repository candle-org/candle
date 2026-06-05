from libc.stdint cimport int64_t


cdef class FastBlock:
    cdef public int64_t ptr
    cdef public int64_t size
    cdef public int64_t requested
    cdef public int64_t seg_base
    cdef public str pool
    cdef public object stream
    cdef public object event
    cdef public set stream_uses
    cdef public int event_count


cdef class FastNpuAllocator:
    cdef public int device_id
    cdef int64_t _stats_arr[128]
    cdef public dict _active
    cdef public dict _cached
    cdef public dict _segments
    cdef public list _pending_events
    cdef public list _event_pool
    cdef public bint _event_pool_ready
    cdef public object max_split_size
    cdef public object gc_threshold
    cdef object _desc_cache
    cdef dict __dict__

    cdef object _get_desc_cache(self)
    cdef void _bump_fast(self, str prefix, str pool,
                         int64_t current, int64_t allocated,
                         int64_t freed)
    cdef void _track_alloc(self, int64_t requested, int64_t allocated, str pool)
    cdef void _track_reuse(self, int64_t requested, int64_t allocated, str pool)
    cdef void _track_free(self, FastBlock block)
    cdef str _pool_for_size(self, int64_t size)
    cdef FastBlock _find_cached(self, int64_t size, str pool)
    cdef tuple _split_block(self, FastBlock block, int64_t size)
    cpdef int64_t malloc(self, int64_t size, object stream=*)
    cpdef int64_t malloc_large_cached(self, int64_t size, object stream=*)
    cpdef void free(self, int64_t ptr, object stream=*)
    cpdef void free_large_cached(self, int64_t ptr, object stream=*)
    cpdef void free_synchronized(self, int64_t ptr)
