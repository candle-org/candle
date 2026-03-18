# cython: language_level=3
"""
Public cdef declarations for _c10d.pyx.

Allows _c10d_gloo.pyx and _c10d_hccl.pyx to ``cimport`` the extension
types defined here so they can subclass ProcessGroup / Store and access
cdef fields at C speed.
"""


# ---------------------------------------------------------------------------
# ReduceOp
# ---------------------------------------------------------------------------

cdef class ReduceOp:
    cdef public int _op


# ---------------------------------------------------------------------------
# Options structs
# ---------------------------------------------------------------------------

cdef class AllreduceOptions:
    cdef public int reduceOp
    cdef public double timeout
    cdef public bint asyncOp


cdef class BroadcastOptions:
    cdef public int rootRank
    cdef public int rootTensor
    cdef public double timeout
    cdef public bint asyncOp


cdef class ReduceOptions:
    cdef public int reduceOp
    cdef public int rootRank
    cdef public int rootTensor
    cdef public double timeout
    cdef public bint asyncOp


cdef class AllgatherOptions:
    cdef public double timeout
    cdef public bint asyncOp


cdef class GatherOptions:
    cdef public int rootRank
    cdef public double timeout
    cdef public bint asyncOp


cdef class ScatterOptions:
    cdef public int rootRank
    cdef public double timeout
    cdef public bint asyncOp


cdef class ReduceScatterOptions:
    cdef public int reduceOp
    cdef public double timeout
    cdef public bint asyncOp


cdef class BarrierOptions:
    cdef public double timeout
    cdef public bint asyncOp
    cdef public object device_ids


cdef class AllToAllOptions:
    cdef public double timeout
    cdef public bint asyncOp


# ---------------------------------------------------------------------------
# Work
# ---------------------------------------------------------------------------

cdef class Work:
    cdef public bint _completed
    cdef public object _stream
    cdef public object _device_id
    cdef public object _exception
    cdef public int _source_rank
    cdef public object _on_wait

    cpdef bint wait(self, timeout=*)
    cpdef bint is_completed(self)
    cpdef bint is_success(self)


# ---------------------------------------------------------------------------
# Store hierarchy
# ---------------------------------------------------------------------------

cdef class Store:
    cpdef set(self, str key, bytes value)
    cpdef bytes get(self, str key)
    cpdef int add(self, str key, int amount)
    cpdef bint delete_key(self, str key)
    cpdef int num_keys(self)
    cpdef bint check(self, list keys)
    cpdef bytes compare_set(self, str key, str expected, str desired)
    cpdef wait(self, list keys, timeout=*)
    cpdef close(self)


cdef class TCPStore(Store):
    cdef str _host
    cdef int _port
    cdef int _world_size
    cdef double _timeout
    cdef object _server_inst
    cdef object _lock
    cdef object _sock

    cdef object _connect(self, str host, int port, double timeout)


cdef class PrefixStore(Store):
    cdef str _prefix
    cdef object _store


cdef class HashStore(Store):
    cdef dict _data
    cdef object _lock
    cdef object _cond


# ---------------------------------------------------------------------------
# ProcessGroup
# ---------------------------------------------------------------------------

cdef class ProcessGroup:
    cdef public int _rank
    cdef public int _size
    cdef public str _group_name
    cdef public str _group_desc
    cdef public object _ranks
    cdef public object _bound_device_id

    cpdef int rank(self)
    cpdef int size(self)
    cpdef str name(self)
