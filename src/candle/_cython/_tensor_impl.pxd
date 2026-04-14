from libc.stdint cimport int64_t

DEF MAX_NDIM = 64

cdef class TensorImpl:
    # -- runtime-owned tensor metadata --
    cdef int64_t _c_shape[MAX_NDIM]
    cdef int64_t _c_stride[MAX_NDIM]
    cdef public int _ndim
    cdef public int64_t _c_numel
    cdef public int64_t _c_offset
    cdef public tuple _shape_tuple
    cdef public object _stride_tuple

    # -- runtime-owned device / dtype caches --
    cdef public int _device_type
    cdef public int _device_index
    cdef public object _device_obj
    cdef public int _dtype_code
    cdef public int _itemsize
    cdef public object _dtype_obj
    cdef public unsigned int _dispatch_keys

    # -- runtime-owned storage / autograd attachment --
    cdef public object _storage
    cdef public bint requires_grad
    cdef public object grad
    cdef public object grad_fn
    cdef public int64_t _version_value
    cdef public object _base
    cdef public object _view_meta
    cdef public bint _pending
    cdef public bint _retain_grad
    cdef public int _output_nr
    cdef public object _backward_hooks
    cdef public object _vc_proxy

    # -- allow dynamic attrs (__dict__) --
    cdef dict __dict__

    # -- inline methods --
    cdef inline void _set_shape(self, tuple shape)
    cdef inline void _set_stride(self, object stride)
    cdef inline void _set_device_from_obj(self, object dev)
    cdef inline void _recompute_dispatch_keys(self)
    cdef inline void _set_dtype_from_obj(self, object dtype)
    cdef inline void _attach_view_runtime_truth(self, TensorImpl view)

    # -- view operations --
    cpdef object cy_detach(self)
    cpdef object cy_view(self, tuple new_shape)
    cpdef object cy_as_strided(self, tuple size, tuple stride, int64_t storage_offset)
    cpdef object cy_transpose(self, int dim0, int dim1)
    cpdef object cy_set_runtime_truth(
        self,
        object typed_storage,
        tuple size,
        object stride,
        int64_t storage_offset,
    )
    # Internal helper: callers must already pass a validated Tensor/TensorImpl.
    # This is not a public validation boundary.
    cpdef object cy_set_data_runtime_truth_from(self, object other)


cdef class _VersionCounterProxy:
    cdef TensorImpl _impl

    cpdef void bump(self)


cpdef void cy_init_tensor_fields(
    TensorImpl t,
    object storage,
    tuple shape,
    object stride,
    int64_t offset,
    bint requires_grad,
    object grad,
    object grad_fn,
    object base,
    object view_meta,
    bint pending,
    bint retain_grad,
    object backward_hooks,
    int64_t version_value,
    object vc_proxy,
)


cpdef object cy_make_tensor_from_storage(
    object storage,
    tuple shape,
    object stride,
    int64_t offset=*,
    bint requires_grad=*,
)

cpdef object cy_make_view_tensor(
    object base,
    object storage,
    tuple shape,
    object stride,
    int64_t offset=*,
)
