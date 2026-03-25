# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot-path for DDP bucket bookkeeping.

Provides dtype-aware bucket sizing and fast pending-count management.
All Python-level orchestration (hook registration, allreduce calls) stays
in distributed.py; only the inner bookkeeping loops live here.
"""


cdef dict _DTYPE_ITEMSIZE = {
    'float32': 4,
    'float64': 8,
    'float16': 2,
    'bfloat16': 2,
    'int8': 1,
    'int16': 2,
    'int32': 4,
    'int64': 8,
    'uint8': 1,
    'bool': 1,
}


def dtype_itemsize(dtype_obj) -> int:
    """Return the byte width of a candle dtype object.

    Falls back to ``dtype_obj.itemsize`` if the name is not in the
    compile-time table so that future dtypes degrade gracefully.
    """
    cdef str name
    name = str(dtype_obj)
    # candle dtype __str__ returns e.g. 'torch.float32' or 'float32'
    if '.' in name:
        name = name.rsplit('.', 1)[1]
    if name in _DTYPE_ITEMSIZE:
        return _DTYPE_ITEMSIZE[name]
    # fallback: ask the object itself
    return getattr(dtype_obj, 'itemsize', 4)


def build_bucket_mapping(
    list grad_params,
    Py_ssize_t bucket_cap_bytes,
):
    """Group grad_params into buckets (reversed order = backward order).

    Parameters
    ----------
    grad_params:
        List of (param_idx, param) tuples for params that require grad.
    bucket_cap_bytes:
        Maximum bucket size in bytes.

    Returns
    -------
    buckets : list of list of (int, param)
        Each inner list is one bucket.
    param_to_bucket : dict mapping param_idx -> bucket_idx
    """
    cdef list buckets = []
    cdef list cur_bucket = []
    cdef Py_ssize_t cur_size = 0
    cdef Py_ssize_t param_bytes
    cdef int idx
    cdef dict param_to_bucket = {}
    cdef int bucket_idx

    for idx, param in reversed(grad_params):
        param_bytes = <Py_ssize_t>param.numel() * dtype_itemsize(param.dtype)
        if cur_size + param_bytes > bucket_cap_bytes and cur_bucket:
            buckets.append(list(cur_bucket))
            cur_bucket = []
            cur_size = 0
        cur_bucket.append((idx, param))
        cur_size += param_bytes

    if cur_bucket:
        buckets.append(list(cur_bucket))

    for bucket_idx in range(len(buckets)):
        for idx, _ in buckets[bucket_idx]:
            param_to_bucket[idx] = bucket_idx

    return buckets, param_to_bucket


def make_bucket_pending_counts(list buckets):
    """Return a list of pending-grad counts, one per bucket."""
    cdef list result = []
    for bucket in buckets:
        result.append(len(bucket))
    return result


def decrement_pending(list bucket_pending, int bucket_idx):
    """Decrement bucket_pending[bucket_idx] in-place; return new value."""
    cdef int val
    val = bucket_pending[bucket_idx] - 1
    bucket_pending[bucket_idx] = val
    return val
