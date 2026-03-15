import math
import ctypes
import struct
import numpy as np

from ._helpers import (
    _can_use_gpu, _metal_buf, _kernel_suffix, _scalar_fmt, _itemsize,
    _alloc_output_buf, _metal_buf_to_bytes, _from_metal_buffer,
    _get_dispatcher, _dispatch_unary_gpu, _dispatch_unary_predicate_gpu,
    _scalar_value, _dispatch_binary_gpu,
    _to_numpy, _from_numpy,
    _compute_reduce_dims, _reduce_shape, _gpu_reduce_single_dim,
    _normalize_tensor_sequence_args,
    _can_use_blas, _blas_gemm,
    float32_dtype, float16_dtype, float64_dtype,
    int32_dtype, int64_dtype, bool_dtype,
    to_numpy_dtype, Tensor,
    mps_typed_storage_from_numpy, _MPSUntypedStorage, TypedStorage,
    _accel,
)

def contiguous(a):
    if a.device.type != "mps":
        raise ValueError("MPS contiguous expects MPS tensors")
    if a.is_contiguous():
        return a
    return _dispatch_unary_gpu(a, "identity")

def flip(a, dims):
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype)
            and len(a.shape) <= 8):
        import struct
        ndim = len(a.shape)
        if isinstance(dims, int):
            dims = (dims,)
        norm_dims = set()
        for d in dims:
            norm_dims.add(d if d >= 0 else ndim + d)
        flip_mask = [1 if d in norm_dims else 0 for d in range(ndim)]
        total = a.numel()
        if total > 0:
            sfx = _kernel_suffix(a.dtype)
            d = _get_dispatcher()
            out_buf = _alloc_output_buf(total, a.dtype)
            shape_packed = struct.pack(f"{ndim}I", *a.shape)
            flip_packed = struct.pack(f"{ndim}I", *flip_mask)
            d.dispatch_flip(f"flip_{sfx}", _metal_buf(a), out_buf,
                            shape_packed, flip_packed, ndim, total)
            from ...._tensor import _compute_strides
            out_shape = tuple(a.shape)
            return _from_metal_buffer(out_buf, out_shape,
                                      _compute_strides(out_shape),
                                      a.dtype, a.device)
    # Non-contiguous GPU: make contiguous and retry
    if _can_use_gpu(a):
        return flip(a.contiguous(), dims)
    arr = _to_numpy(a)
    out = np.flip(arr, axis=dims)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def roll(a, shifts, dims=None):
    # GPU path for dims=None: flatten → roll(flat, shifts, 0) → reshape
    if _can_use_gpu(a) and dims is None:
        orig_shape = tuple(a.shape)
        flat = flatten(a)
        rolled = roll(flat, shifts, dims=0)
        return rolled.reshape(orig_shape)
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype)
            and len(a.shape) <= 8 and dims is not None):
        import struct
        ndim = len(a.shape)
        if isinstance(shifts, int):
            shifts = (shifts,)
        if isinstance(dims, int):
            dims = (dims,)
        # Build per-dim shift array (0 for unshifted dims)
        shift_arr = [0] * ndim
        for s, d in zip(shifts, dims):
            dd = d if d >= 0 else ndim + d
            shift_arr[dd] = int(s)
        total = a.numel()
        if total > 0:
            sfx = _kernel_suffix(a.dtype)
            d = _get_dispatcher()
            out_buf = _alloc_output_buf(total, a.dtype)
            shape_packed = struct.pack(f"{ndim}I", *a.shape)
            shifts_packed = struct.pack(f"{ndim}i", *shift_arr)
            d.dispatch_roll(f"roll_{sfx}", _metal_buf(a), out_buf,
                            shape_packed, shifts_packed, ndim, total)
            from ...._tensor import _compute_strides
            out_shape = tuple(a.shape)
            return _from_metal_buffer(out_buf, out_shape,
                                      _compute_strides(out_shape),
                                      a.dtype, a.device)
    # Non-contiguous GPU or dims=None: make contiguous and retry
    if _can_use_gpu(a):
        return roll(a.contiguous(), shifts, dims)
    arr = _to_numpy(a)
    out = np.roll(arr, shift=shifts, axis=dims)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def rot90(a, k=1, dims=(0, 1)):
    # GPU composite: rot90 = combinations of flip + transpose
    if _can_use_gpu(a):
        k = k % 4
        if k == 0:
            return a
        d0, d1 = dims[0], dims[1]
        if k == 1:
            return flip(a.transpose(d0, d1), (d0,))
        elif k == 2:
            return flip(flip(a, (d0,)), (d1,))
        else:  # k == 3
            return flip(a, (d0,)).transpose(d0, d1)
    # Non-contiguous GPU: make contiguous and retry
    if _can_use_gpu(a):
        return rot90(a.contiguous(), k=k, dims=dims)
    arr = _to_numpy(a)
    out = np.rot90(arr, k=k, axes=dims)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def repeat(a, repeats):
    if _can_use_gpu(a):
        ndim = len(a.shape)
        reps = tuple(repeats)
        # Pad shape with leading 1s if repeats has more dims
        if len(reps) > ndim:
            a = a.reshape((1,) * (len(reps) - ndim) + a.shape)
            ndim = len(a.shape)
        # Pad repeats with leading 1s if needed
        reps = (1,) * (ndim - len(reps)) + reps
        # If all repeats are 1, return contiguous copy
        if all(r == 1 for r in reps):
            return a.contiguous()
        # Build interleaved shape: (1, s0, 1, s1, ...)
        inter_shape = []
        for s in a.shape:
            inter_shape.extend([1, s])
        a_inter = a.reshape(tuple(inter_shape))
        # Expand: (r0, s0, r1, s1, ...)
        exp_shape = []
        for r, s in zip(reps, a.shape):
            exp_shape.extend([r, s])
        a_exp = expand(a_inter, tuple(exp_shape))
        # Reshape to final: (r0*s0, r1*s1, ...)
        final_shape = tuple(r * s for r, s in zip(reps, a.shape))
        return a_exp.contiguous().reshape(final_shape)
    # Non-contiguous: make contiguous and retry
    if a.device.type == 'mps':
        return repeat(a.contiguous(), repeats)
    arr = _to_numpy(a)
    out = np.tile(arr, repeats)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def repeat_interleave(a, repeats, dim=None):
    # GPU composite for integer repeats: unsqueeze → expand → reshape
    if _can_use_gpu(a) and isinstance(repeats, int):
        if dim is None:
            # Flatten first, then repeat along dim=0
            flat = flatten(a)
            n = flat.shape[0]
            # unsqueeze to (n, 1), expand to (n, repeats), reshape to (n*repeats,)
            return expand(flat.unsqueeze(1), (n, repeats)).contiguous().reshape(-1)
        d = dim if dim >= 0 else dim + len(a.shape)
        shape = list(a.shape)
        n = shape[d]
        # Insert a new dim after d: unsqueeze at d+1
        unsq = a.unsqueeze(d + 1)
        # Expand that new dim to repeats
        exp_shape = list(unsq.shape)
        exp_shape[d + 1] = repeats
        expanded = expand(unsq, tuple(exp_shape))
        # Reshape: merge dims d and d+1
        new_shape = shape[:d] + [n * repeats] + shape[d + 1:]
        return expanded.contiguous().reshape(new_shape)
    # CPU/numpy fallback (also handles tensor repeats)
    arr = _to_numpy(a)
    axis = None if dim is None else dim
    out = np.repeat(arr, repeats, axis=axis)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def tile(a, dims):
    # tile is repeat with dim padding
    ndim = len(a.shape)
    reps = tuple(dims)
    if len(reps) < ndim:
        reps = (1,) * (ndim - len(reps)) + reps
    return repeat(a, reps)

def nonzero(a, as_tuple=False):
    idx = np.nonzero(_to_numpy(a))
    if as_tuple:
        return tuple(
            _from_numpy(
                np.ascontiguousarray(dim_idx, dtype=np.int64), int64_dtype, a.device
            )
            for dim_idx in idx
        )
    stacked = np.stack(idx, axis=1).astype(np.int64, copy=False)
    return _from_numpy(np.ascontiguousarray(stacked), int64_dtype, a.device)

def stack(tensors, dim=0):
    # GPU path: composite via unsqueeze + cat
    if (tensors and all(_can_use_gpu(t) for t in tensors)
            and all(t.dtype == tensors[0].dtype for t in tensors)):
        unsqueezed = [t.contiguous().unsqueeze(dim) if not t.is_contiguous()
                      else t.unsqueeze(dim) for t in tensors]
        return cat(unsqueezed, dim=dim)
    arrays = [_to_numpy(t) for t in tensors]
    return _from_numpy(np.stack(arrays, axis=dim), tensors[0].dtype, tensors[0].device)

def cat(tensors, dim=0):
    # GPU path: all tensors GPU+contiguous+same dtype+ndim>0
    if (tensors and len(tensors) > 0
            and all(_can_use_gpu(t) and t.is_contiguous() and t.dim() > 0
                    for t in tensors)
            and all(t.dtype == tensors[0].dtype for t in tensors)):
        dtype = tensors[0].dtype
        ndim = tensors[0].dim()
        d_pos = dim if dim >= 0 else dim + ndim
        if 0 <= d_pos < ndim:
            d = _get_dispatcher()
            sfx = _kernel_suffix(dtype)
            # Compute output shape
            out_shape = list(tensors[0].shape)
            total_dim = tensors[0].shape[d_pos]
            for t in tensors[1:]:
                total_dim += t.shape[d_pos]
            out_shape[d_pos] = total_dim
            out_numel = 1
            for s in out_shape:
                out_numel *= s
            out_buf = _alloc_output_buf(out_numel, dtype)
            # Compute outer_size and inner_size
            outer_size = 1
            for i in range(d_pos):
                outer_size *= out_shape[i]
            inner_size = 1
            for i in range(d_pos + 1, ndim):
                inner_size *= out_shape[i]
            dst_dim = total_dim
            offset = 0
            for t in tensors:
                src_dim = t.shape[d_pos]
                src_numel = t.numel()
                if src_numel > 0:
                    d.dispatch_cat_copy(f"cat_copy_{sfx}", _metal_buf(t),
                                        out_buf, outer_size, src_dim,
                                        inner_size, dst_dim, offset,
                                        src_numel)
                offset += src_dim
            out_stride = []
            stride = 1
            for s in reversed(out_shape):
                out_stride.append(stride)
                stride *= s
            out_stride.reverse()
            return _from_metal_buffer(out_buf, tuple(out_shape),
                                      tuple(out_stride), dtype,
                                      tensors[0].device)
    # Non-contiguous GPU: make contiguous and retry
    if (tensors and all(_can_use_gpu(t) for t in tensors)
            and all(t.dtype == tensors[0].dtype for t in tensors)):
        return cat([t.contiguous() for t in tensors], dim=dim)
    arrays = [_to_numpy(t) for t in tensors]
    return _from_numpy(np.concatenate(arrays, axis=dim), tensors[0].dtype, tensors[0].device)

def concatenate(tensors, dim=0):
    return cat(tensors, dim=dim)

def hstack(tensors):
    if tensors[0].dim() == 1:
        return cat(tensors, dim=0)
    return cat(tensors, dim=1)

def vstack(tensors):
    if tensors[0].dim() == 1:
        expanded = [t.reshape((1, t.shape[0])) for t in tensors]
        return cat(expanded, dim=0)
    return cat(tensors, dim=0)

def row_stack(tensors):
    return vstack(tensors)

def dstack(tensors):
    expanded = []
    for t in tensors:
        if t.dim() == 1:
            expanded.append(t.reshape((1, t.shape[0], 1)))
        elif t.dim() == 2:
            expanded.append(t.reshape((t.shape[0], t.shape[1], 1)))
        else:
            expanded.append(t)
    return cat(expanded, dim=2)

def column_stack(tensors):
    if tensors[0].dim() == 1:
        expanded = [t.reshape((t.shape[0], 1)) for t in tensors]
        return cat(expanded, dim=1)
    return cat(tensors, dim=1)

def pad_sequence(seqs, batch_first=False, padding_value=0.0, padding_side="right"):
    arrays = [_to_numpy(t) for t in seqs]
    max_len = max(a.shape[0] for a in arrays)
    batch = len(arrays)
    trailing = arrays[0].shape[1:]
    out_shape = (batch, max_len, *trailing) if batch_first else (max_len, batch, *trailing)
    out = np.full(out_shape, padding_value, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        length = a.shape[0]
        start = max_len - length if padding_side == "left" else 0
        if batch_first:
            out[i, start:start + length, ...] = a
        else:
            out[start:start + length, i, ...] = a
    return _from_numpy(out, seqs[0].dtype, seqs[0].device)

def block_diag(*tensors):
    tensors = _normalize_tensor_sequence_args(tensors)
    arrays = [_to_numpy(t) for t in tensors]
    rows = sum(a.shape[0] for a in arrays)
    cols = sum(a.shape[1] for a in arrays)
    out = np.zeros((rows, cols), dtype=arrays[0].dtype)
    r = c = 0
    for a in arrays:
        out[r:r + a.shape[0], c:c + a.shape[1]] = a
        r += a.shape[0]
        c += a.shape[1]
    return _from_numpy(out, tensors[0].dtype, tensors[0].device)

def tril(a, diagonal=0):
    # GPU path: a is GPU+contiguous+ndim>=2
    if _can_use_gpu(a) and a.is_contiguous() and a.dim() >= 2:
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        rows = a.shape[-2]
        cols = a.shape[-1]
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_tril_triu(f"tril_{sfx}", _metal_buf(a), out_buf,
                             rows, cols, diagonal, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    # Non-contiguous GPU: make contiguous and retry
    if _can_use_gpu(a) and a.dim() >= 2:
        return tril(a.contiguous(), diagonal=diagonal)
    out = np.tril(_to_numpy(a), k=diagonal)
    return _from_numpy(out, a.dtype, a.device)

def triu(a, diagonal=0):
    # GPU path: a is GPU+contiguous+ndim>=2
    if _can_use_gpu(a) and a.is_contiguous() and a.dim() >= 2:
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        rows = a.shape[-2]
        cols = a.shape[-1]
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_tril_triu(f"triu_{sfx}", _metal_buf(a), out_buf,
                             rows, cols, diagonal, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    # Non-contiguous GPU: make contiguous and retry
    if _can_use_gpu(a) and a.dim() >= 2:
        return triu(a.contiguous(), diagonal=diagonal)
    out = np.triu(_to_numpy(a), k=diagonal)
    return _from_numpy(out, a.dtype, a.device)

def diag(a, diagonal_offset=0):
    # GPU composite for 2D input: delegate to diagonal()
    if _can_use_gpu(a) and len(a.shape) == 2:
        return diagonal(a, offset=diagonal_offset)
    arr = _to_numpy(a)
    if arr.ndim not in (1, 2):
        raise ValueError("diag expects 1D or 2D tensor")
    out = np.diag(arr, k=diagonal_offset).copy()
    return _from_numpy(out, a.dtype, a.device)

def cartesian_prod(*tensors):
    tensors = _normalize_tensor_sequence_args(tensors)
    arrays = [_to_numpy(t) for t in tensors]
    grids = np.meshgrid(*arrays, indexing="ij")
    stacked = np.stack([g.reshape(-1) for g in grids], axis=1)
    return _from_numpy(stacked, tensors[0].dtype, tensors[0].device)

def chunk(a, chunks, dim=0):
    if dim < 0:
        dim += len(a.shape)
    dim_size = a.shape[dim]
    if chunks <= 0:
        raise ValueError("chunks must be > 0")
    actual_chunks = min(chunks, dim_size) if dim_size > 0 else chunks
    if actual_chunks == 0:
        return tuple()
    chunk_size = (dim_size + actual_chunks - 1) // actual_chunks
    outputs = []
    for i in range(actual_chunks):
        start = i * chunk_size
        length = min(chunk_size, dim_size - start)
        if length <= 0:
            break
        outputs.append(narrow(a, dim, start, length))
    return tuple(outputs)

def split(a, split_size_or_sections, dim=0):
    if dim < 0:
        dim += len(a.shape)
    dim_size = a.shape[dim]
    if isinstance(split_size_or_sections, int):
        if split_size_or_sections <= 0:
            raise ValueError("split_size must be > 0")
        step = split_size_or_sections
        return tuple(narrow(a, dim, s, min(step, dim_size - s))
                     for s in range(0, dim_size, step))
    sizes = list(split_size_or_sections)
    if sum(sizes) != dim_size:
        raise ValueError("split sections must sum to dim size")
    outputs = []
    start = 0
    for size in sizes:
        outputs.append(narrow(a, dim, start, size))
        start += size
    return tuple(outputs)

def _split_sections_from_count(dim_size, sections):
    if sections <= 0:
        raise ValueError("sections must be > 0")
    size, extra = divmod(dim_size, sections)
    return [size + 1] * extra + [size] * (sections - extra)

def vsplit(a, split_size_or_sections):
    if isinstance(split_size_or_sections, int):
        sizes = _split_sections_from_count(a.shape[0], split_size_or_sections)
        return split(a, sizes, dim=0)
    return split(a, split_size_or_sections, dim=0)

def hsplit(a, split_size_or_sections):
    dim = 0 if a.dim() == 1 else 1
    if isinstance(split_size_or_sections, int):
        sizes = _split_sections_from_count(a.shape[dim], split_size_or_sections)
        return split(a, sizes, dim=dim)
    return split(a, split_size_or_sections, dim=dim)

def dsplit(a, split_size_or_sections):
    if a.dim() < 3:
        raise ValueError("dsplit expects input with at least 3 dimensions")
    if isinstance(split_size_or_sections, int):
        sizes = _split_sections_from_count(a.shape[2], split_size_or_sections)
        return split(a, sizes, dim=2)
    return split(a, split_size_or_sections, dim=2)

def unbind(a, dim=0):
    from ...common import view as view_backend

    d = dim if dim >= 0 else dim + a.dim()
    dim_size = a.shape[d]
    outputs = []
    for i in range(dim_size):
        outputs.append(view_backend.select(a, d, i, creation_kind="multi_view"))
    return tuple(outputs)

def masked_select(a, mask):
    arr = _to_numpy(a)
    mask_arr = _to_numpy(mask).astype(bool)
    out = arr[mask_arr]
    return _from_numpy(out, a.dtype, a.device)

def _check_indices_layout(layout):
    if layout is None:
        return
    if isinstance(layout, str):
        if layout != "strided":
            raise ValueError("layout must be strided")
        return
    raise ValueError("layout must be strided")

def _indices_device(device):
    if device is None:
        return None
    if isinstance(device, str):
        return device
    return str(device)

def _ensure_integer_indices(arr, name):
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must be integer dtype")
    return arr

def take(a, index):
    arr = _to_numpy(a).reshape(-1)
    idx = _ensure_integer_indices(_to_numpy(index), "index").astype(np.int64, copy=False)
    out = np.take(arr, idx)
    return _from_numpy(out, a.dtype, a.device)

def take_along_dim(a, indices, dim):
    arr = _to_numpy(a)
    idx = _ensure_integer_indices(_to_numpy(indices), "indices").astype(np.int64, copy=False)
    if dim < 0:
        dim += arr.ndim
    if dim < 0 or dim >= arr.ndim:
        raise ValueError("dim out of range")
    if idx.ndim != arr.ndim:
        raise ValueError("indices shape mismatch")
    for i, size in enumerate(idx.shape):
        if i != dim and size != arr.shape[i]:
            raise ValueError("indices shape mismatch")
    out = np.take_along_axis(arr, idx, axis=dim)
    return _from_numpy(out, a.dtype, a.device)

def index_select(a, dim, index):
    arr = _to_numpy(a)
    idx = _ensure_integer_indices(_to_numpy(index), "index").astype(np.int64, copy=False)
    if idx.ndim != 1:
        raise ValueError("index must be 1D")
    if dim < 0:
        dim += arr.ndim
    if dim < 0 or dim >= arr.ndim:
        raise ValueError("dim out of range")
    # GPU path: a is GPU+contiguous
    if _can_use_gpu(a) and a.is_contiguous():
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        idx_size = len(idx)
        input_dim_size = a.shape[dim]
        # Compute outer/inner sizes
        outer_size = 1
        for i in range(dim):
            outer_size *= a.shape[i]
        inner_size = 1
        for i in range(dim + 1, a.dim()):
            inner_size *= a.shape[i]
        out_numel = outer_size * idx_size * inner_size
        out_buf = _alloc_output_buf(out_numel, a.dtype)
        # Cast indices to int32 for Metal
        idx_i32 = idx.astype(np.int32)
        idx_tensor = _from_numpy(idx_i32, int32_dtype, a.device)
        d.dispatch_index_gather(f"index_select_{sfx}", _metal_buf(a),
                                _metal_buf(idx_tensor), out_buf,
                                outer_size, idx_size, inner_size,
                                input_dim_size, out_numel)
        out_shape = list(a.shape)
        out_shape[dim] = idx_size
        out_stride = []
        stride = 1
        for s in reversed(out_shape):
            out_stride.append(stride)
            stride *= s
        out_stride.reverse()
        return _from_metal_buffer(out_buf, tuple(out_shape), tuple(out_stride),
                                  a.dtype, a.device)
    out = np.take(arr, idx, axis=dim)
    return _from_numpy(out, a.dtype, a.device)

def _check_index_range(index, dim_size):
    if (index < 0).any() or (index >= dim_size).any():
        raise IndexError("index out of range")

def gather(a, dim, index):
    arr = _to_numpy(a)
    idx = _ensure_integer_indices(_to_numpy(index), "index").astype(np.int64, copy=False)
    if dim < 0:
        dim += arr.ndim
    if dim < 0 or dim >= arr.ndim:
        raise ValueError("dim out of range")
    if idx.ndim != arr.ndim:
        raise ValueError("index shape mismatch")
    for i, size in enumerate(idx.shape):
        if i != dim and size != arr.shape[i]:
            raise ValueError("index shape mismatch")
    _check_index_range(idx, arr.shape[dim])
    # GPU path: a and index are GPU+contiguous
    if (_can_use_gpu(a) and a.is_contiguous()
            and isinstance(index, Tensor) and _can_use_gpu(index)
            and index.is_contiguous()):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        idx_size = idx.shape[dim]
        input_dim_size = a.shape[dim]
        outer_size = 1
        for i in range(dim):
            outer_size *= idx.shape[i]
        inner_size = 1
        for i in range(dim + 1, idx.ndim):
            inner_size *= idx.shape[i]
        out_numel = outer_size * idx_size * inner_size
        out_buf = _alloc_output_buf(out_numel, a.dtype)
        # Cast indices to int32 for Metal
        idx_i32 = idx.astype(np.int32)
        idx_tensor = _from_numpy(idx_i32, int32_dtype, a.device)
        d.dispatch_index_gather(f"gather_{sfx}", _metal_buf(a),
                                _metal_buf(idx_tensor), out_buf,
                                outer_size, idx_size, inner_size,
                                input_dim_size, out_numel)
        out_shape = tuple(idx.shape)
        out_stride = []
        stride = 1
        for s in reversed(out_shape):
            out_stride.append(stride)
            stride *= s
        out_stride.reverse()
        return _from_metal_buffer(out_buf, out_shape, tuple(out_stride),
                                  a.dtype, a.device)
    out = np.take_along_axis(arr, idx, axis=dim)
    return _from_numpy(out, a.dtype, a.device)

def scatter(a, dim, index, src):
    arr = _to_numpy(a).copy()
    idx = _ensure_integer_indices(_to_numpy(index), "index").astype(np.int64, copy=False)
    if dim < 0:
        dim += arr.ndim
    if dim < 0 or dim >= arr.ndim:
        raise ValueError("dim out of range")
    if idx.ndim != arr.ndim:
        raise ValueError("index shape mismatch")
    for i, size in enumerate(idx.shape):
        if i != dim and size != arr.shape[i]:
            raise ValueError("index shape mismatch")
    _check_index_range(idx, arr.shape[dim])
    if hasattr(src, "shape"):
        src_arr = _to_numpy(src)
    else:
        src_arr = np.array(src, dtype=arr.dtype)
    src_arr = np.broadcast_to(src_arr, idx.shape)
    np.put_along_axis(arr, idx, src_arr, axis=dim)
    return _from_numpy(arr, a.dtype, a.device)

def tril_indices(row, col, offset=0, dtype=None, device=None, layout=None):
    _check_indices_layout(layout)
    if row < 0 or col < 0:
        raise ValueError("row and col must be non-negative")
    if dtype is None:
        dtype = int64_dtype
    dev = _indices_device(device)
    r, c = np.tril_indices(row, k=offset, m=col)
    out = np.stack([r, c], axis=0).astype(to_numpy_dtype(dtype), copy=False)
    return _from_numpy(out, dtype, dev)

def triu_indices(row, col, offset=0, dtype=None, device=None, layout=None):
    _check_indices_layout(layout)
    if row < 0 or col < 0:
        raise ValueError("row and col must be non-negative")
    if dtype is None:
        dtype = int64_dtype
    dev = _indices_device(device)
    r, c = np.triu_indices(row, k=offset, m=col)
    out = np.stack([r, c], axis=0).astype(to_numpy_dtype(dtype), copy=False)
    return _from_numpy(out, dtype, dev)

def _normalize_index_key(key):
    if isinstance(key, Tensor):
        arr = _to_numpy(key)
        if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
            return arr
        raise IndexError("tensors used as indices must be integer or boolean")
    if isinstance(key, tuple):
        return tuple(_normalize_index_key(k) for k in key)
    if isinstance(key, list):
        return [_normalize_index_key(k) for k in key]
    return key

def _is_int_index(key):
    return isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_))

def _is_basic_index_key(key):
    keys = key if isinstance(key, tuple) else (key,)
    for item in keys:
        if item is Ellipsis or item is None:
            continue
        if isinstance(item, slice):
            continue
        if _is_int_index(item):
            continue
        return False
    return True

def _expand_ellipsis(keys, ndim):
    ellipsis_count = sum(1 for item in keys if item is Ellipsis)
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if ellipsis_count == 0:
        return keys

    specified_dims = sum(1 for item in keys if item is not None and item is not Ellipsis)
    fill = ndim - specified_dims
    if fill < 0:
        raise IndexError("too many indices for tensor")

    expanded = []
    for item in keys:
        if item is Ellipsis:
            expanded.extend([slice(None)] * fill)
        else:
            expanded.append(item)
    return expanded

def _basic_getitem_view(tensor, key):
    keys = list(key) if isinstance(key, tuple) else [key]
    keys = _expand_ellipsis(keys, tensor.dim())

    in_dim = 0
    out_shape = []
    out_stride = []
    out_offset = tensor.offset

    for item in keys:
        if item is None:
            out_shape.append(1)
            out_stride.append(0)
            continue

        if in_dim >= tensor.dim():
            raise IndexError("too many indices for tensor")

        dim_size = tensor.shape[in_dim]
        dim_stride = tensor.stride[in_dim]

        if _is_int_index(item):
            idx = int(item)
            if idx < 0:
                idx += dim_size
            if idx < 0 or idx >= dim_size:
                raise IndexError("index out of range")
            out_offset += idx * dim_stride
            in_dim += 1
            continue

        if isinstance(item, slice):
            start, stop, step = item.indices(dim_size)
            out_offset += start * dim_stride
            out_shape.append(len(range(start, stop, step)))
            out_stride.append(dim_stride * step)
            in_dim += 1
            continue

        return None

    while in_dim < tensor.dim():
        out_shape.append(tensor.shape[in_dim])
        out_stride.append(tensor.stride[in_dim])
        in_dim += 1

    # Keep fallback copy path for negative strides until Tensor._numpy_view
    # supports them safely.
    if any(s < 0 for s in out_stride):
        return None

    return Tensor(tensor.storage(), tuple(out_shape), tuple(out_stride), out_offset)

def getitem(tensor, key):
    norm_key = _normalize_index_key(key)
    arr = _to_numpy(tensor)
    result = arr[norm_key]

    if _is_basic_index_key(norm_key):
        view = _basic_getitem_view(tensor, norm_key)
        if view is not None:
            return view

    if isinstance(result, np.generic) or (isinstance(result, np.ndarray) and result.ndim == 0):
        # Return 0-dim tensor (matches PyTorch behavior)
        scalar_arr = np.array(result)
        return _from_numpy(scalar_arr, tensor.dtype, tensor.device)
    return _from_numpy(np.ascontiguousarray(result), tensor.dtype, tensor.device)

def setitem(tensor, key, value):
    arr = _to_numpy(tensor)
    norm_key = _normalize_index_key(key)
    if hasattr(value, 'numpy'):
        arr[norm_key] = value.numpy()
    else:
        arr[norm_key] = value
    return tensor

def narrow(a, dim, start, length):
    from ...._tensor import Tensor
    d = dim if dim >= 0 else dim + a.dim()
    new_shape = list(a.shape)
    new_shape[d] = int(length)
    new_offset = a.offset + int(start) * a.stride[d]
    return Tensor(a.storage(), tuple(new_shape), a.stride, new_offset)

def select(a, dim, index):
    from ...._tensor import Tensor
    d = dim if dim >= 0 else dim + a.dim()
    idx = int(index)
    if idx < 0:
        idx += a.shape[d]
    new_shape = list(a.shape)
    del new_shape[d]
    new_stride = list(a.stride)
    new_offset = a.offset + idx * a.stride[d]
    del new_stride[d]
    return Tensor(a.storage(), tuple(new_shape), tuple(new_stride), new_offset)

def expand(a, sizes):
    from ...._tensor import Tensor
    sizes = tuple(sizes)
    ndiff = len(sizes) - a.dim()
    if ndiff < 0:
        raise RuntimeError("expand: number of sizes must be >= tensor dim")
    src_shape = (1,) * ndiff + a.shape
    src_stride = (0,) * ndiff + a.stride
    out_shape = []
    out_stride = []
    for i, sz in enumerate(sizes):
        if sz == -1:
            out_shape.append(src_shape[i])
            out_stride.append(src_stride[i])
        elif src_shape[i] == 1:
            out_shape.append(sz)
            out_stride.append(0)
        elif src_shape[i] == sz:
            out_shape.append(sz)
            out_stride.append(src_stride[i])
        else:
            raise RuntimeError(
                f"expand: size {sz} not compatible with dim size {src_shape[i]}"
            )
    return Tensor(a.storage(), tuple(out_shape), tuple(out_stride), a.offset)

def masked_fill(a, mask, value):
    # GPU path: a and mask are GPU+contiguous+same shape
    if (_can_use_gpu(a) and a.is_contiguous()
            and isinstance(mask, Tensor) and _can_use_gpu(mask)
            and mask.is_contiguous() and mask.shape == a.shape):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        # Ensure mask is uint8 (uchar) for the shader
        if mask.dtype != bool_dtype:
            mask_u8 = _from_numpy(
                _to_numpy(mask).astype(np.bool_).astype(np.uint8),
                bool_dtype, mask.device)
        else:
            mask_u8 = mask
        scalar_val = float(value) if a.dtype in (float32_dtype, float16_dtype) else int(value)
        d.dispatch_masked_fill(f"masked_fill_{sfx}", _metal_buf(a),
                               _metal_buf(mask_u8), scalar_val, out_buf,
                               numel, scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    arr = _to_numpy(a).copy()
    m = _to_numpy(mask).astype(bool)
    arr[m] = value
    return _from_numpy(arr, a.dtype, a.device)

def masked_fill_(a, mask, value):
    # GPU path: use non-inplace masked_fill (has Metal kernel), then copy back
    if (_can_use_gpu(a) and a.is_contiguous()
            and isinstance(mask, Tensor) and _can_use_gpu(mask)):
        # Broadcast mask to a's shape if needed
        if mask.shape != a.shape:
            mask = broadcast_to(mask, a.shape).contiguous()
        result = masked_fill(a, mask, value)
        # Copy result buffer back into a's storage via identity kernel
        if result.numel() > 0:
            d = _get_dispatcher()
            sfx = _kernel_suffix(a.dtype)
            d.dispatch_unary(f"identity_{sfx}", _metal_buf(result),
                             _metal_buf(a), a.numel())
        return a
    # CPU/numpy fallback
    arr = _to_numpy(a)
    m = _to_numpy(mask).astype(bool)
    arr[m] = value
    return a

def index_put_(a, indices, values, accumulate=False):
    arr = _to_numpy(a)
    idx = tuple(_to_numpy(t) if hasattr(t, '_numpy_view') else t for t in indices)
    vals = _to_numpy(values) if hasattr(values, '_numpy_view') else values
    if accumulate:
        np.add.at(arr, idx, vals)
    else:
        arr[idx] = vals
    return a

def index_put(a, indices, values, accumulate=False):
    arr = _to_numpy(a).copy()
    idx = tuple(_to_numpy(t) if hasattr(t, '_numpy_view') else t for t in indices)
    vals = _to_numpy(values) if hasattr(values, '_numpy_view') else values
    if accumulate:
        np.add.at(arr, idx, vals)
    else:
        arr[idx] = vals
    return _from_numpy(arr, a.dtype, a.device)

def index_copy_(a, dim, index, source):
    arr = _to_numpy(a)
    idx = _to_numpy(index).ravel().astype(np.intp)
    src = _to_numpy(source)
    d = dim if dim >= 0 else dim + a.dim()
    for j, i in enumerate(idx):
        slices_dst = [slice(None)] * arr.ndim
        slices_dst[d] = int(i)
        slices_src = [slice(None)] * arr.ndim
        slices_src[d] = j
        arr[tuple(slices_dst)] = src[tuple(slices_src)]
    return a

def index_fill_(a, dim, index, value):
    arr = _to_numpy(a)
    idx = _to_numpy(index).ravel().astype(np.intp)
    d = dim if dim >= 0 else dim + a.dim()
    for i in idx:
        slices = [slice(None)] * arr.ndim
        slices[d] = int(i)
        arr[tuple(slices)] = value
    return a

def index_add_(a, dim, index, source, alpha=1.0):
    arr = _to_numpy(a)
    idx = _to_numpy(index).ravel().astype(np.intp)
    src = _to_numpy(source)
    d = dim if dim >= 0 else dim + a.dim()
    for j, i in enumerate(idx):
        slices_dst = [slice(None)] * arr.ndim
        slices_dst[d] = int(i)
        slices_src = [slice(None)] * arr.ndim
        slices_src[d] = j
        arr[tuple(slices_dst)] += float(alpha) * src[tuple(slices_src)]
    return a

def scatter_(a, dim, index, src):
    arr = _to_numpy(a)
    idx = _to_numpy(index).astype(np.intp)
    d = dim if dim >= 0 else dim + a.dim()
    if hasattr(src, '_numpy_view'):
        src_arr = _to_numpy(src)
    else:
        src_arr = src
    it = np.nditer(idx, flags=['multi_index'])
    while not it.finished:
        mi = it.multi_index
        dst_idx = list(mi)
        dst_idx[d] = int(it[0])
        if hasattr(src, '_numpy_view'):
            arr[tuple(dst_idx)] = src_arr[mi]
        else:
            arr[tuple(dst_idx)] = src_arr
        it.iternext()
    return a

def scatter_add_(a, dim, index, src):
    arr = _to_numpy(a)
    idx = _to_numpy(index).astype(np.intp)
    src_arr = _to_numpy(src)
    d = dim if dim >= 0 else dim + a.dim()
    it = np.nditer(idx, flags=['multi_index'])
    while not it.finished:
        mi = it.multi_index
        dst_idx = list(mi)
        dst_idx[d] = int(it[0])
        arr[tuple(dst_idx)] += src_arr[mi]
        it.iternext()
    return a

def masked_scatter_(a, mask, source):
    arr = _to_numpy(a)
    m = _to_numpy(mask).astype(bool)
    src = _to_numpy(source)
    arr[m] = src.ravel()[:m.sum()]
    return a

def unfold(a, dimension, size, step):
    # GPU composite: narrow windows → stack
    if _can_use_gpu(a):
        ndim = len(a.shape)
        d = dimension if dimension >= 0 else dimension + ndim
        dim_size = a.shape[d]
        n_windows = max(0, (dim_size - size) // step + 1)
        if n_windows == 0:
            new_shape = list(a.shape)
            new_shape[d] = 0
            new_shape.append(size)
            return _from_numpy(np.empty(new_shape, dtype=to_numpy_dtype(a.dtype)),
                               a.dtype, a.device)
        windows = []
        for i in range(n_windows):
            # narrow along dim d: start=i*step, length=size
            win = narrow(a, d, i * step, size)
            # Move the window dim to the end: transpose d with last, but we
            # need to keep other dims intact. Use movedim equivalent.
            # Actually, we need to move dim d to the last position.
            # permute: move d to end
            perm = list(range(ndim))
            perm.pop(d)
            perm.append(d)
            win = win.permute(*perm)
            windows.append(win)
        return stack(windows, dim=d)
    # CPU/numpy fallback
    arr = _to_numpy(a)
    d = dimension if dimension >= 0 else dimension + arr.ndim
    dim_size = arr.shape[d]
    n_windows = max(0, (dim_size - size) // step + 1)
    if n_windows == 0:
        new_shape = list(arr.shape)
        new_shape[d] = 0
        new_shape.append(size)
        return _from_numpy(np.empty(new_shape, dtype=arr.dtype), a.dtype, a.device)
    out_shape = list(arr.shape)
    out_shape[d] = n_windows
    out_shape.append(size)
    out = np.empty(out_shape, dtype=arr.dtype)
    for i in range(n_windows):
        src_s = [slice(None)] * arr.ndim
        src_s[d] = slice(i * step, i * step + size)
        chunk = np.moveaxis(arr[tuple(src_s)], d, -1)
        dst_s = [slice(None)] * (arr.ndim + 1)
        dst_s[d] = i
        out[tuple(dst_s)] = chunk
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def flatten(a, start_dim=0, end_dim=-1):
    ndim = len(a.shape)
    start = start_dim if start_dim >= 0 else start_dim + ndim
    end = end_dim if end_dim >= 0 else end_dim + ndim
    new_shape = a.shape[:start] + (-1,) + a.shape[end + 1:]
    # Compute actual -1 size
    known = 1
    for i, s in enumerate(new_shape):
        if s != -1:
            known *= s
    total = 1
    for s in a.shape:
        total *= s
    new_shape = tuple(s if s != -1 else total // known for s in new_shape)
    if _can_use_gpu(a) and a.is_contiguous():
        from ...._tensor import _compute_strides
        return _from_metal_buffer(_metal_buf(a), new_shape, _compute_strides(new_shape), a.dtype, a.device)
    # Non-contiguous GPU: make contiguous and retry
    if _can_use_gpu(a):
        return flatten(a.contiguous(), start_dim=start_dim, end_dim=end_dim)
    arr = _to_numpy(a)
    out = arr.reshape(new_shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def unflatten(a, dim, sizes):
    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    new_shape = a.shape[:d] + tuple(sizes) + a.shape[d + 1:]
    if _can_use_gpu(a) and a.is_contiguous():
        from ...._tensor import _compute_strides
        return _from_metal_buffer(_metal_buf(a), new_shape, _compute_strides(new_shape), a.dtype, a.device)
    # Non-contiguous GPU: make contiguous and retry
    if _can_use_gpu(a):
        return unflatten(a.contiguous(), dim, sizes)
    arr = _to_numpy(a)
    out = arr.reshape(new_shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def broadcast_to(a, shape):
    if _can_use_gpu(a) and a.is_contiguous():
        # Compute broadcast strides: 0 for expanded dims, original stride otherwise
        a_shape = a.shape
        a_stride = list(a.stride)
        ndim_out = len(shape)
        ndim_a = len(a_shape)
        # Pad a_shape/a_stride on the left
        padded_shape = [1] * (ndim_out - ndim_a) + list(a_shape)
        padded_stride = [0] * (ndim_out - ndim_a) + a_stride
        out_stride = []
        for i in range(ndim_out):
            if padded_shape[i] == shape[i]:
                out_stride.append(padded_stride[i])
            elif padded_shape[i] == 1:
                out_stride.append(0)
            else:
                break
        else:
            return _from_metal_buffer(_metal_buf(a), tuple(shape), tuple(out_stride), a.dtype, a.device)
    # Non-contiguous GPU: make contiguous and retry
    if _can_use_gpu(a):
        return broadcast_to(a.contiguous(), shape)
    arr = _to_numpy(a)
    out = np.broadcast_to(arr, shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def movedim(a, source, destination):
    # GPU composite: compute permutation and use permute (pure view op)
    ndim = len(a.shape)
    if isinstance(source, int):
        source = (source,)
    if isinstance(destination, int):
        destination = (destination,)
    source = tuple(s % ndim for s in source)
    destination = tuple(d % ndim for d in destination)
    # Build permutation: same logic as np.moveaxis
    order = [n for n in range(ndim) if n not in source]
    dst_order = sorted(range(len(destination)), key=lambda i: destination[i])
    for dst_idx in dst_order:
        order.insert(destination[dst_idx], source[dst_idx])
    return a.permute(*order)

def diagonal(a, offset=0, dim1=0, dim2=1):
    # GPU composite: select along dim1 and dim2 in loop, then stack
    if _can_use_gpu(a):
        ndim = len(a.shape)
        d1 = dim1 if dim1 >= 0 else dim1 + ndim
        d2 = dim2 if dim2 >= 0 else dim2 + ndim
        size1 = a.shape[d1]
        size2 = a.shape[d2]
        if offset >= 0:
            diag_len = max(0, min(size1, size2 - offset))
        else:
            diag_len = max(0, min(size1 + offset, size2))
        if diag_len == 0:
            # Empty diagonal
            out_shape = [s for i, s in enumerate(a.shape) if i not in (d1, d2)] + [0]
            return _from_numpy(np.empty(out_shape, dtype=to_numpy_dtype(a.dtype)),
                               a.dtype, a.device)
        elements = []
        for i in range(diag_len):
            row = i if offset >= 0 else i - offset
            col = i + offset if offset >= 0 else i
            # Select along the higher dim first to keep lower dim index valid
            if d1 < d2:
                t = select(a, d1, row)
                t = select(t, d2 - 1, col)  # d2 shifts by -1 after removing d1
            else:
                t = select(a, d2, col)
                t = select(t, d1 - 1, row)  # d1 shifts by -1 after removing d2
            elements.append(t)
        # Stack along last dim (PyTorch convention: diagonal goes to last dim)
        return stack(elements, dim=-1)
    # CPU/numpy fallback
    arr = _to_numpy(a)
    out = np.diagonal(arr, offset=offset, axis1=dim1, axis2=dim2)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


# ---------------------------------------------------------------------------
# Group 6: Search ops
# ---------------------------------------------------------------------------

def one_hot(a, num_classes=-1):
    arr = _to_numpy(a)
    # Check logical dtype — may be a string or DType object
    dtype_str = str(a.dtype).replace('torch.', '')
    if dtype_str not in ('int8', 'int16', 'int32', 'int64', 'uint8', 'bool'):
        raise TypeError("one_hot is only applicable to index tensor")
    flat = arr.astype(np.int64, copy=False).reshape(-1)
    if num_classes is None or int(num_classes) < 0:
        num_classes = int(flat.max()) + 1 if flat.size > 0 else 0
    num_classes = int(num_classes)
    if (flat < 0).any():
        raise ValueError("one_hot indices must be non-negative")
    if (flat >= num_classes).any() and flat.size > 0:
        raise ValueError("one_hot indices out of range")
    out = np.eye(num_classes, dtype=np.int64)[flat]
    out = out.reshape(arr.shape + (num_classes,))
    return _from_numpy(np.ascontiguousarray(out), int64_dtype, a.device)
