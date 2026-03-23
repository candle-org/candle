# cython: language_level=3
"""Public declarations for DTensor / TP bookkeeping fastpath."""

cpdef tuple compute_local_shape_and_global_offset_cy(
    tuple global_shape,
    object mesh,
    object placements,
)
cpdef tuple compute_global_shape_cy(
    tuple local_shape,
    object mesh,
    object placements,
)
cpdef tuple compute_global_stride_cy(object local_stride)
cpdef int normalize_shard_dim_cy(int dim, int ndim)
cpdef tuple compute_gather_sizes_cy(
    tuple local_shape,
    int shard_dim,
    int mesh_size,
)
cpdef tuple compute_scatter_sizes_cy(
    tuple local_shape,
    int shard_dim,
    int mesh_size,
)
