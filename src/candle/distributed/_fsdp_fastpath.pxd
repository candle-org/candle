# cython: language_level=3
"""Public declarations for FSDP shard bookkeeping fastpath."""

cpdef Py_ssize_t compute_shard_offset(list shapes, int idx)
cpdef list build_flat_shard_offsets(list shapes)
cpdef void pack_shards_to_flat(list shards, list flat, list offsets)
cpdef void unpack_flat_to_shards(list flat, list shards, list offsets)
cpdef dict build_param_owner_map(list groups)
cpdef Py_ssize_t compute_chunk_size(Py_ssize_t dim_size, int world_size)
cpdef tuple compute_padded_size(Py_ssize_t dim_size, int world_size)
cpdef list writeback_shard_grad_flags(list params)
