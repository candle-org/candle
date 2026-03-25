from libc.stdint cimport int64_t

DEF MAX_NDIM = 64

cdef class TensorImpl:
    # -- shape / stride (C arrays + cached Python tuples) --
    cdef int64_t _c_shape[MAX_NDIM]
    cdef int64_t _c_stride[MAX_NDIM]
    cdef public int _ndim
    cdef public int64_t _c_numel
    cdef public int64_t _c_offset

    # -- device (int enum + cached object) --
    cdef public int _device_type
    cdef public int _device_index
    cdef public object _device_obj

    # -- dtype (int code + cached object) --
    cdef public int _dtype_code
    cdef public int _itemsize
    cdef public object _dtype_obj

    # -- storage --
    cdef public object _storage

    # -- autograd --
    cdef public bint requires_grad
    cdef public object grad
    cdef public object grad_fn
    cdef public int64_t _version_value
    cdef public object _base
    cdef public object _view_meta
    cdef public bint _pending
    cdef public bint _retain_grad
    cdef public int _output_nr

    cdef public tuple _shape_tuple
    cdef public object _stride_tuple

    # -- allow dynamic attrs (__dict__) --
    cdef dict __dict__

    # -- cached version counter proxy --
    cdef public object _vc_proxy

    # -- precomputed dispatch key bits --
    cdef public unsigned int _dispatch_keys

    # -- inline methods --
    cdef inline void _set_shape(self, tuple shape)
    cdef inline void _set_stride(self, object stride)
    cdef inline void _set_device_from_obj(self, object dev)
    cdef inline void _recompute_dispatch_keys(self)
    cdef inline void _set_dtype_from_obj(self, object dtype)

    # -- view operations --
    cpdef object cy_view(self, tuple new_shape)
    cpdef object cy_as_strided(self, tuple size, tuple stride, int64_t storage_offset)
    cpdef object cy_transpose(self, int dim0, int dim1)


cdef class _VersionCounterProxy:
    cdef TensorImpl _impl

    cpdef void bump(self)
