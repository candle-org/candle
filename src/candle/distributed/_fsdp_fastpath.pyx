# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot-path for FSDP shard bookkeeping.

Moves the following hot inner loops out of Python:
  - shard offset calculation (flat-buffer pack/unpack index arithmetic)
  - flat-buffer pack and unpack loops
  - owner/leaf-parameter mapping construction
  - shard padding arithmetic (chunk_size, padded_size)
  - local shard writeback grad-flag extraction

All Python-level orchestration (hook registration, collective calls, DTensor
wrapping, error handling) remains in the Python FSDP modules.  Only the
bookkeeping inner loops live here.
"""


# ---------------------------------------------------------------------------
# Shard offset arithmetic
# ---------------------------------------------------------------------------

cpdef Py_ssize_t compute_shard_offset(list shapes, int idx):
    """Return the flat-buffer start offset for shard *idx*.

    *shapes* is a list of shape tuples (or any sequence of ints).
    The offset is the sum of ``numel(shapes[j])`` for j < idx.
    """
    cdef Py_ssize_t offset = 0
    cdef Py_ssize_t n
    cdef int j
    cdef object shape
    cdef int k

    for j in range(idx):
        shape = shapes[j]
        n = 1
        for k in range(len(shape)):
            n = n * <Py_ssize_t>shape[k]
        offset += n
    return offset


cpdef list build_flat_shard_offsets(list shapes):
    """Build a list of (start, end) offset pairs for all shards.

    Parameters
    ----------
    shapes:
        List of shape tuples, one per shard parameter.

    Returns
    -------
    list of (int, int)
        ``offsets[i] = (start, end)`` where ``end - start == numel(shapes[i])``.
    """
    cdef list offsets = []
    cdef Py_ssize_t cursor = 0
    cdef Py_ssize_t n
    cdef object shape
    cdef int k

    for shape in shapes:
        n = 1
        for k in range(len(shape)):
            n = n * <Py_ssize_t>shape[k]
        offsets.append((cursor, cursor + n))
        cursor += n
    return offsets


# ---------------------------------------------------------------------------
# Flat-buffer pack / unpack
# ---------------------------------------------------------------------------

cpdef void pack_shards_to_flat(list shards, list flat, list offsets):
    """Copy shard data into *flat* at the positions given by *offsets*.

    Parameters
    ----------
    shards:
        List of flat iterables (each shard's data as a 1-D sequence).
    flat:
        Destination flat list; mutated in-place.
    offsets:
        List of (start, end) pairs matching *shards*.
    """
    cdef Py_ssize_t start, end, pos, j
    cdef object shard

    for j in range(len(shards)):
        start, end = offsets[j]
        shard = shards[j]
        pos = start
        for val in shard:
            flat[pos] = val
            pos += 1


cpdef void unpack_flat_to_shards(list flat, list shards, list offsets):
    """Copy slices of *flat* back into *shards* in-place.

    Parameters
    ----------
    flat:
        Source flat list.
    shards:
        List of mutable flat lists; each is mutated in-place.
    offsets:
        List of (start, end) pairs matching *shards*.
    """
    cdef Py_ssize_t start, end, pos, j, i
    cdef object shard

    for j in range(len(shards)):
        start, end = offsets[j]
        shard = shards[j]
        i = 0
        for pos in range(start, end):
            shard[i] = flat[pos]
            i += 1


# ---------------------------------------------------------------------------
# Owner / leaf-parameter mapping
# ---------------------------------------------------------------------------

cpdef dict build_param_owner_map(list groups):
    """Build a mapping from param name to group index.

    Parameters
    ----------
    groups:
        List of lists; each inner list contains param names (str) belonging
        to that group.

    Returns
    -------
    dict mapping param_name -> group_index (int)
    """
    cdef dict mapping = {}
    cdef int group_idx
    cdef object name
    cdef list group

    for group_idx in range(len(groups)):
        group = groups[group_idx]
        for name in group:
            mapping[name] = group_idx
    return mapping


# ---------------------------------------------------------------------------
# Padding arithmetic
# ---------------------------------------------------------------------------

cpdef Py_ssize_t compute_chunk_size(Py_ssize_t dim_size, int world_size):
    """Return ceil(dim_size / world_size) — the per-rank shard size."""
    return (dim_size + world_size - 1) // world_size


cpdef tuple compute_padded_size(Py_ssize_t dim_size, int world_size):
    """Return (padded_dim_size, padding) where padded is the smallest
    multiple of *world_size* that is >= *dim_size*.

    Returns
    -------
    (padded_dim_size, padding) : (int, int)
    """
    cdef Py_ssize_t chunk = compute_chunk_size(dim_size, world_size)
    cdef Py_ssize_t padded = chunk * world_size
    return (padded, padded - dim_size)


# ---------------------------------------------------------------------------
# Writeback grad-flag extraction
# ---------------------------------------------------------------------------

cpdef list writeback_shard_grad_flags(list params):
    """Extract ``requires_grad`` flag from each param in *params*.

    This is the hot loop that runs once per reshard to decide which shards
    need gradient writeback.  Keeping it in Cython avoids repeated Python
    attribute lookups.

    Parameters
    ----------
    params:
        List of objects with a boolean ``requires_grad`` attribute.

    Returns
    -------
    list of bool
    """
    cdef list flags = []
    cdef object p

    for p in params:
        flags.append(<bint>p.requires_grad)
    return flags
