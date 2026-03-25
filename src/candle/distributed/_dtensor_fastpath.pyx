# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot-path for DTensor / TP shard bookkeeping.

Moves the following out of Python:
  - compute_local_shape_and_global_offset: shard/offset math for
    Shard placements (remainder-aware chunk split)
  - compute_global_shape: reconstruct global shape from local shard
  - compute_global_stride: identity stride pass-through
  - normalize_shard_dim: clamp negative dim to canonical form
  - compute_gather_sizes: full-tensor sizes for Shard->Replicate
  - compute_scatter_sizes: local shard sizes for Replicate->Shard

All Python-level orchestration (collective calls, DTensor wrapping,
error handling) remains in the Python modules.  Only the bookkeeping
inner loops live here.
"""


# ---------------------------------------------------------------------------
# Dimension normalisation
# ---------------------------------------------------------------------------

cpdef int normalize_shard_dim_cy(int dim, int ndim):
    """Return the canonical (non-negative) index for *dim* in a tensor
    of *ndim* dimensions.  Raises ``IndexError`` for out-of-range dims.
    """
    if dim < 0:
        dim = dim + ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(
            f"Dimension {dim} out of range for tensor with {ndim} dims"
        )
    return dim


# ---------------------------------------------------------------------------
# Global shape / stride helpers
# ---------------------------------------------------------------------------

cpdef tuple compute_global_shape_cy(
    tuple local_shape,
    object mesh,
    object placements,
):
    """Compute the global (unsharded) shape from a local shard shape.

    For each Shard placement the corresponding dimension is multiplied
    by the mesh size; Replicate / Partial dimensions are unchanged.

    Parameters
    ----------
    local_shape:
        Shape of the local shard as a tuple of ints.
    mesh:
        DeviceMesh instance (only ``mesh.size()`` is called).
    placements:
        Sequence of Placement objects.

    Returns
    -------
    tuple of int
    """
    cdef list gs = list(local_shape)
    cdef int mesh_sz = mesh.size()
    cdef object p

    for p in placements:
        if type(p).__name__ == 'Shard':
            gs[p.dim] = gs[p.dim] * mesh_sz
    return tuple(gs)


cpdef tuple compute_global_stride_cy(object local_stride):
    """Return local_stride as a plain tuple (identity pass-through for MVP).

    Converts any tuple subclass (e.g. _StrideTuple named-tuple) to a plain
    ``tuple`` so Cython's return-type check is satisfied.
    """
    return tuple(local_stride)


# ---------------------------------------------------------------------------
# Local shape + global offset (core shard arithmetic)
# ---------------------------------------------------------------------------

cpdef tuple compute_local_shape_and_global_offset_cy(
    tuple global_shape,
    object mesh,
    object placements,
):
    """Compute local shard shape and global offset from a distribution spec.

    Implements the standard PyTorch DTensor chunk split:
      chunk_size  = global_dim // num_chunks
      remainder   = global_dim % num_chunks
      - ranks 0..remainder-1 get (chunk_size + 1) elements
      - ranks remainder..num_chunks-1 get chunk_size elements

    Parameters
    ----------
    global_shape:
        The global (unsharded) tensor shape as a tuple of ints.
    mesh:
        DeviceMesh instance.
    placements:
        Sequence of Placement objects aligned with mesh dimensions.

    Returns
    -------
    (local_shape, global_offset) : (tuple of int, tuple of int)
    """
    cdef list local_shape = list(global_shape)
    cdef list global_offset = [0] * len(global_shape)
    cdef int mesh_dim
    cdef object placement
    cdef int dim
    cdef int num_chunks
    cdef int local_rank
    cdef Py_ssize_t chunk_size
    cdef Py_ssize_t remainder
    cdef Py_ssize_t gs_dim

    for mesh_dim, placement in enumerate(placements):
        if type(placement).__name__ == 'Shard':
            dim = placement.dim
            num_chunks = mesh.size(mesh_dim)
            local_rank = mesh.get_local_rank(mesh_dim)
            gs_dim = <Py_ssize_t>global_shape[dim]
            chunk_size = gs_dim // num_chunks
            remainder = gs_dim % num_chunks
            if local_rank < remainder:
                local_shape[dim] = chunk_size + 1
                global_offset[dim] = local_rank * (chunk_size + 1)
            else:
                local_shape[dim] = chunk_size
                global_offset[dim] = (
                    remainder * (chunk_size + 1)
                    + (local_rank - remainder) * chunk_size
                )
    return tuple(local_shape), tuple(global_offset)


# ---------------------------------------------------------------------------
# Gather / scatter size helpers (used in redistribute hot path)
# ---------------------------------------------------------------------------

cpdef tuple compute_gather_sizes_cy(
    tuple local_shape,
    int shard_dim,
    int mesh_size,
):
    """Compute the full-tensor sizes for a Shard->Replicate all_gather.

    Returns a new shape tuple where dimension *shard_dim* is scaled by
    *mesh_size*; all other dimensions are unchanged.

    Parameters
    ----------
    local_shape:
        Shape of the local shard.
    shard_dim:
        Dimension along which the tensor is sharded.
    mesh_size:
        Number of ranks in the mesh group.

    Returns
    -------
    tuple of int
    """
    cdef list sizes = list(local_shape)
    sizes[shard_dim] = sizes[shard_dim] * mesh_size
    return tuple(sizes)


cpdef tuple compute_scatter_sizes_cy(
    tuple local_shape,
    int shard_dim,
    int mesh_size,
):
    """Compute the local shard sizes for a Replicate->Shard reduce_scatter.

    Returns a new shape tuple where dimension *shard_dim* is divided by
    *mesh_size*; all other dimensions are unchanged.

    Parameters
    ----------
    local_shape:
        Shape of the full (replicated) tensor.
    shard_dim:
        Dimension along which to scatter.
    mesh_size:
        Number of ranks in the mesh group.

    Returns
    -------
    tuple of int
    """
    cdef list sizes = list(local_shape)
    sizes[shard_dim] = sizes[shard_dim] // mesh_size
    return tuple(sizes)
