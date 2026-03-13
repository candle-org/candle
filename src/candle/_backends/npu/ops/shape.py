"""Shape manipulation, view, and indexing operations for NPU."""
import ctypes
from ._helpers import (
    _unwrap_storage, _wrap_tensor, _unary_op, _binary_op,
    _broadcast_shape, _broadcast_shape_checked,
    _numel, _dtype_itemsize, _use_soc_fallback,
    _scalar_to_npu_tensor, _scalar_to_npu_tensor_no_add,
    _npu_broadcast_to, _npu_arange_1d, _npu_linear_index,
    _npu_add_scalar_, npu_index_put_impl,
    _normalize_reduction_dims, _reduce_out_shape,
    _cast_tensor_dtype, _normalize_tensor_sequence_args,
    _normalize_dim,
    bool_dtype, int32_dtype, int64_dtype, float_dtype,
    npu_typed_storage_from_ptr, reshape,
    aclnn, npu_runtime, npu_state, ops_soc,
)
from ...common import view as view_backend


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalize_dims_tuple(dims, ndim, name):
    if isinstance(dims, int):
        raise TypeError(f"{name} dims must be tuple/list of ints")
    if not isinstance(dims, (tuple, list)):
        raise TypeError(f"{name} dims must be tuple/list of ints")
    norm = []
    seen = set()
    for d in dims:
        if not isinstance(d, int):
            raise TypeError(f"{name} dims must contain ints")
        d = _normalize_dim(d, ndim)
        if d in seen:
            raise RuntimeError(f"dim {d} appears multiple times in the list of dims")
        seen.add(d)
        norm.append(d)
    return tuple(norm)


def _normalize_roll_args(shifts, dims, ndim):
    if isinstance(shifts, int):
        shifts_tuple = (int(shifts),)
    elif isinstance(shifts, (tuple, list)):
        shifts_tuple = tuple(int(s) for s in shifts)
    else:
        raise TypeError("roll shifts must be int/tuple/list")
    if len(shifts_tuple) == 0:
        raise RuntimeError("`shifts` required")

    if isinstance(dims, int):
        dims_tuple = (int(dims),)
    elif isinstance(dims, (tuple, list)):
        dims_tuple = tuple(int(d) for d in dims)
    else:
        raise TypeError("roll dims must be int/tuple/list/None")

    if len(shifts_tuple) != len(dims_tuple):
        raise RuntimeError(f"shifts and dimensions must align. shifts: {len(shifts_tuple)}, dims:{len(dims_tuple)}")
    return shifts_tuple, tuple(_normalize_dim(d, ndim) for d in dims_tuple)


def _normalize_repeats_tuple(repeats, ndim, name):
    if isinstance(repeats, int):
        repeats = (int(repeats),)
    elif isinstance(repeats, (tuple, list)):
        repeats = tuple(int(r) for r in repeats)
    else:
        raise TypeError(f"{name} repeats must be int/tuple/list")
    if len(repeats) < ndim:
        repeats = (1,) * (ndim - len(repeats)) + repeats
    if len(repeats) < ndim:
        raise RuntimeError("Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor")
    return repeats


def _build_repeat_interleave_indices(dim_size, repeats, device):
    from ..creation import zeros_create

    if isinstance(repeats, int):
        if repeats < 0:
            raise ValueError("repeats must be non-negative")
        output_size = int(dim_size) * int(repeats)
        if output_size == 0:
            return zeros_create((0,), dtype=int64_dtype, device=device), output_size
        base = _npu_arange_1d(int(dim_size), device)
        idx = repeat(base, (int(repeats),))
        return idx, output_size

    if not hasattr(repeats, "shape"):
        raise TypeError("repeats must be int or Tensor")
    if repeats.device.type != "npu":
        raise ValueError("repeats tensor must be on NPU")
    if repeats.dtype != int64_dtype:
        raise TypeError("repeats tensor must be int64")
    if repeats.dim() != 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")
    if repeats.shape[0] not in (1, int(dim_size)):
        raise RuntimeError(
            f"repeats must have the same size as input along dim, but got repeats.size(0) = {repeats.shape[0]} and input.size(0) = {dim_size}"
        )

    rep_list = repeats.to("cpu").tolist()
    if any(int(v) < 0 for v in rep_list):
        raise ValueError("repeats must be non-negative")

    if repeats.shape[0] == 1:
        rep = int(rep_list[0])
        return _build_repeat_interleave_indices(dim_size, rep, device)

    output_size = int(sum(int(v) for v in rep_list))
    if output_size == 0:
        return zeros_create((0,), dtype=int64_dtype, device=device), output_size

    idx_chunks = []
    for i, rep in enumerate(rep_list):
        rep = int(rep)
        if rep == 0:
            continue
        scalar = _npu_arange_1d(int(rep), device)
        from .math import add as math_add
        scalar = math_add(scalar, _scalar_to_npu_tensor(int(i), scalar))
        idx_chunks.append(scalar)

    if not idx_chunks:
        return zeros_create((0,), dtype=int64_dtype, device=device), 0
    if len(idx_chunks) == 1:
        return idx_chunks[0], output_size
    return cat(idx_chunks, dim=0), output_size


def _read_bool_scalar(tensor):
    runtime = npu_runtime.get_runtime((tensor.device.index or 0))
    stream = npu_state.current_stream((tensor.device.index or 0))
    runtime.activate()
    if hasattr(runtime, "synchronize_stream"):
        runtime.synchronize_stream(stream.stream)
    buf = (ctypes.c_uint8 * 1)()
    npu_runtime.memcpy_d2h(
        ctypes.addressof(buf),
        1,
        _unwrap_storage(tensor).data_ptr(),
        runtime=runtime,
    )
    return bool(buf[0])


def _read_int64_scalar(tensor):
    runtime = npu_runtime.get_runtime((tensor.device.index or 0))
    stream = npu_state.current_stream((tensor.device.index or 0))
    runtime.activate()
    if hasattr(runtime, "synchronize_stream"):
        runtime.synchronize_stream(stream.stream)
    buf = ctypes.c_int64()
    size = ctypes.sizeof(buf)
    npu_runtime.memcpy_d2h(
        ctypes.addressof(buf),
        size,
        _unwrap_storage(tensor).data_ptr(),
        runtime=runtime,
    )
    return int(buf.value)


def _require_int64_indices(indices, name):
    if indices.dtype != int64_dtype:
        raise ValueError(f"{name} indices must be int64")
    if indices.device.type != "npu":
        raise ValueError(f"{name} indices must be on NPU")
    return indices


def _validate_index_bounds(indices, dim_size, allow_negative, name):
    from .comparison import lt, gt
    from .reduce import any_
    if indices.numel() == 0:
        return
    if allow_negative:
        min_ok = _scalar_to_npu_tensor(-int(dim_size), indices)
        max_ok = _scalar_to_npu_tensor(int(dim_size - 1), indices)
    else:
        min_ok = _scalar_to_npu_tensor(0, indices)
        max_ok = _scalar_to_npu_tensor(int(dim_size - 1), indices)
    below_min = lt(indices, min_ok)
    above_max = gt(indices, max_ok)
    if _read_bool_scalar(any_(below_min)) or _read_bool_scalar(any_(above_max)):
        raise IndexError(f"{name} indices out of range")


def _normalize_negative_indices(indices, dim_size):
    from .comparison import lt
    from .reduce import any_
    from .math import add, mul
    neg_mask = lt(indices, _scalar_to_npu_tensor(0, indices))
    if not _read_bool_scalar(any_(neg_mask)):
        return indices

    # 310B static path: avoid SWhere by converting mask to int64 and blending arithmetically.
    if _use_soc_fallback("take_along_dim"):
        if not aclnn.cast_symbols_ok():
            raise RuntimeError("aclnnCast symbols not available")
        runtime = npu_runtime.get_runtime((indices.device.index or 0))
        stream = npu_state.current_stream((indices.device.index or 0))

        shape = tuple(indices.shape)
        stride = tuple(indices.stride)
        numel = max(_numel(shape), 1)

        mask_i64_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(int64_dtype), runtime=runtime)
        aclnn.cast(
            _unwrap_storage(neg_mask).data_ptr(),
            mask_i64_ptr,
            shape,
            stride,
            bool_dtype,
            int64_dtype,
            runtime,
            stream=stream.stream,
        )
        mask_i64_storage = npu_typed_storage_from_ptr(mask_i64_ptr, numel, int64_dtype, device=indices.device)
        mask_i64 = _wrap_tensor(mask_i64_storage, shape, stride)

        offset = _scalar_to_npu_tensor(int(dim_size), indices)
        return add(indices, mul(mask_i64, offset))

    from . import where
    return where(neg_mask, add(indices, _scalar_to_npu_tensor(int(dim_size), indices)), indices)


def _npu_data_ptr(tensor):
    """Return the effective data pointer for *tensor* (base + offset)."""
    itemsize = _dtype_itemsize(tensor.dtype)
    return int(_unwrap_storage(tensor).data_ptr()) + tensor.offset * itemsize


def _split_sections_from_count(dim_size, sections):
    if sections <= 0:
        raise ValueError("sections must be > 0")
    size, extra = divmod(dim_size, sections)
    return [size + 1] * extra + [size] * (sections - extra)


def _move_dim_to_last(a, dim):
    dim = _normalize_dim(dim, a.dim())
    out = a
    for i in range(dim, a.dim() - 1):
        out = view_backend.transpose(out, i, i + 1)
    return out


def _slice_along_dim(a, start, end, dim):
    if a.device.type != "npu":
        raise ValueError("NPU slice expects NPU tensors")
    dim = _normalize_dim(dim, a.dim())
    if not a.is_contiguous():
        raise NotImplementedError("NPU split only supports contiguous input")
    dim_size = a.shape[dim]
    length = max(0, end - start)
    out_shape = list(a.shape)
    out_shape[dim] = length
    out_shape = tuple(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    runtime = npu_runtime.get_runtime((a.device.index or 0))

    if out_numel == 0:
        out_stride = npu_runtime._contiguous_stride(out_shape) if out_shape else ()
        out_ptr = npu_runtime._alloc_device(max(itemsize, 1), runtime=runtime)
        storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), a.dtype, device=a.device)
        return _wrap_tensor(storage, out_shape, out_stride)

    inner = 1
    for d in a.shape[dim + 1:]:
        inner *= d
    outer = 1
    for d in a.shape[:dim]:
        outer *= d

    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    src_base = int(_unwrap_storage(a).data_ptr()) + a.offset * itemsize
    dst_base = int(out_ptr)

    if inner == 1:
        block_bytes = length * itemsize
        for outer_idx in range(outer):
            src_ptr = src_base + (outer_idx * dim_size + start) * itemsize
            dst_ptr = dst_base + outer_idx * length * itemsize
            npu_runtime.memcpy_d2d(dst_ptr, block_bytes, src_ptr, runtime=runtime)
    else:
        block_bytes = inner * itemsize
        for outer_idx in range(outer):
            src_outer = src_base + outer_idx * dim_size * inner * itemsize
            dst_outer = dst_base + outer_idx * length * inner * itemsize
            for i in range(length):
                src_ptr = src_outer + (start + i) * inner * itemsize
                dst_ptr = dst_outer + i * inner * itemsize
                npu_runtime.memcpy_d2d(dst_ptr, block_bytes, src_ptr, runtime=runtime)

    storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(storage, out_shape, out_stride)


def _is_int_index(key):
    """True for integer indices (not bool). Handles numpy.integer too."""
    import numpy as np
    return isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_))


def _is_basic_index_key(keys):
    """True when *keys* (a tuple) contains only int/slice/None/Ellipsis/bool."""
    import numpy as np
    for item in keys:
        if item is Ellipsis or item is None:
            continue
        if isinstance(item, slice):
            continue
        if _is_int_index(item):
            continue
        # Python bool / numpy.bool_ treated as basic index
        # (True → unsqueeze+keep, False → unsqueeze+empty)
        if isinstance(item, (bool, np.bool_)):
            continue
        return False
    return True


def _expand_ellipsis(keys, ndim):
    """Expand Ellipsis into the right number of ``slice(None)``."""
    import numpy as np
    ellipsis_count = sum(1 for item in keys if item is Ellipsis)
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if ellipsis_count == 0:
        return list(keys)

    # Count dims that consume real tensor dimensions
    # None, Ellipsis, and bool don't consume tensor dims
    specified_dims = 0
    for item in keys:
        if item is None or item is Ellipsis:
            continue
        if isinstance(item, (bool, np.bool_)):
            continue
        specified_dims += 1
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


def _npu_basic_getitem_view(tensor, key):
    """Create a view for basic indexing (int, slice, None, Ellipsis, bool).

    Returns a Tensor sharing the same storage, or None if we need to fall back
    to a copy (e.g. negative-step slices that require aclnnSlice).
    """
    from ...._tensor import Tensor
    import numpy as np

    keys = list(key) if isinstance(key, tuple) else [key]
    keys = _expand_ellipsis(keys, tensor.dim())

    in_dim = 0
    out_shape = []
    out_stride = []
    out_offset = tensor.offset

    needs_aclnn_slice = False

    for item in keys:
        if item is None:
            out_shape.append(1)
            if in_dim < tensor.dim():
                out_stride.append(tensor.stride[in_dim] * tensor.shape[in_dim])
            else:
                out_stride.append(1)
            continue

        # Python bool / np.bool_: True → unsqueeze (size 1), False → empty dim (size 0)
        # Does NOT consume a tensor dimension (same as None).
        if isinstance(item, (bool, np.bool_)):
            if item:
                out_shape.append(1)
            else:
                out_shape.append(0)
            if in_dim < tensor.dim():
                out_stride.append(tensor.stride[in_dim] * tensor.shape[in_dim])
            else:
                out_stride.append(1)
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
                raise IndexError(
                    f"index {item} is out of bounds for dimension {in_dim} with size {dim_size}"
                )
            out_offset += idx * dim_stride
            in_dim += 1
            continue

        if isinstance(item, slice):
            start, stop, step = item.indices(dim_size)
            if step < 0:
                # Negative step requires data reversal — fall back to aclnnSlice
                needs_aclnn_slice = True
                break
            length = len(range(start, stop, step))
            out_offset += start * dim_stride
            out_shape.append(length)
            out_stride.append(dim_stride * step)
            in_dim += 1
            continue

        # Non-basic element — shouldn't reach here due to _is_basic_index_key check
        return None

    if needs_aclnn_slice:
        return _npu_basic_getitem_with_strided_slices(tensor, keys)

    # Append remaining dims
    while in_dim < tensor.dim():
        out_shape.append(tensor.shape[in_dim])
        out_stride.append(tensor.stride[in_dim])
        in_dim += 1

    out_shape = tuple(out_shape)
    out_stride = tuple(out_stride)

    return Tensor(tensor.storage(), out_shape, out_stride, out_offset)


def _npu_basic_getitem_with_strided_slices(tensor, keys):
    """Handle basic indexing when one or more slices have step != 1.

    Process left-to-right: step==1 slices and ints become view ops;
    step!=1 slices use aclnnSlice which produces a contiguous copy.
    """
    from ...._tensor import Tensor

    cur = tensor
    in_dim = 0
    pending_none_count = 0

    for item in keys:
        if item is None:
            pending_none_count += 1
            continue

        if in_dim >= cur.dim():
            raise IndexError("too many indices for tensor")

        # Insert pending None (unsqueeze) dimensions before this real dim
        for _ in range(pending_none_count):
            cur = _npu_unsqueeze_view(cur, in_dim)
            in_dim += 1
        pending_none_count = 0

        dim_size = cur.shape[in_dim]

        if _is_int_index(item):
            idx = int(item)
            if idx < 0:
                idx += dim_size
            cur = _npu_select_view(cur, in_dim, idx)
            # select removes the dim, so in_dim stays the same
            continue

        if isinstance(item, slice):
            start, stop, step = item.indices(dim_size)
            if step > 0:
                # Positive step: strided view (no copy)
                cur = _npu_strided_slice_view(cur, in_dim, start, stop, step)
            else:
                # Negative step: requires data reversal, use aclnnSlice kernel
                cur = _npu_aclnn_slice(cur, in_dim, start, stop, step)
            in_dim += 1
            continue

    # Insert any trailing None dims
    for _ in range(pending_none_count):
        cur = _npu_unsqueeze_view(cur, cur.dim())

    return cur


def _npu_select_view(tensor, dim, idx):
    """Select a single element along *dim* — returns a view with dim removed."""
    from ...._tensor import Tensor
    new_offset = tensor.offset + idx * tensor.stride[dim]
    new_shape = tensor.shape[:dim] + tensor.shape[dim + 1:]
    new_stride = tensor.stride[:dim] + tensor.stride[dim + 1:]
    return Tensor(tensor.storage(), new_shape, new_stride, new_offset)


def _npu_slice_view(tensor, dim, start, stop):
    """Step-1 slice as a view — adjust offset and shape[dim]."""
    from ...._tensor import Tensor
    length = max(0, stop - start)
    new_offset = tensor.offset + start * tensor.stride[dim]
    new_shape = tensor.shape[:dim] + (length,) + tensor.shape[dim + 1:]
    new_stride = tensor.stride  # stride unchanged for step==1
    return Tensor(tensor.storage(), new_shape, new_stride, new_offset)


def _npu_strided_slice_view(tensor, dim, start, stop, step):
    """Strided slice as a view — adjust offset, shape, and stride. step must be > 0."""
    from ...._tensor import Tensor
    length = len(range(start, stop, step))
    new_offset = tensor.offset + start * tensor.stride[dim]
    new_shape = tensor.shape[:dim] + (length,) + tensor.shape[dim + 1:]
    new_stride = tensor.stride[:dim] + (tensor.stride[dim] * step,) + tensor.stride[dim + 1:]
    return Tensor(tensor.storage(), new_shape, new_stride, new_offset)


def _npu_unsqueeze_view(tensor, dim):
    """Insert a size-1 dimension at *dim* — pure view."""
    from ...._tensor import Tensor
    new_shape = tensor.shape[:dim] + (1,) + tensor.shape[dim:]
    # Compute a stride that keeps the tensor contiguous-looking
    if dim < len(tensor.stride):
        new_s = tensor.stride[dim] * tensor.shape[dim]
    elif len(tensor.stride) > 0:
        new_s = 1
    else:
        new_s = 1
    new_stride = tensor.stride[:dim] + (new_s,) + tensor.stride[dim:]
    return Tensor(tensor.storage(), new_shape, new_stride, tensor.offset)


def _npu_aclnn_slice(tensor, dim, start, stop, step):
    """Strided slice via aclnnSlice kernel — returns a new contiguous tensor."""
    length = len(range(start, stop, step))
    out_shape = tensor.shape[:dim] + (length,) + tensor.shape[dim + 1:]
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(tensor.dtype)
    runtime = npu_runtime.get_runtime((tensor.device.index or 0))
    stream = npu_state.current_stream((tensor.device.index or 0))

    if out_numel == 0:
        out_stride = npu_runtime._contiguous_stride(out_shape) if out_shape else ()
        out_ptr = npu_runtime._alloc_device(max(itemsize, 1), runtime=runtime)
        storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), tensor.dtype, device=tensor.device)
        return _wrap_tensor(storage, out_shape, out_stride)

    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    # Compute the data pointer including any offset from prior view ops
    src_ptr = int(_unwrap_storage(tensor).data_ptr()) + tensor.offset * itemsize

    aclnn.slice_op(
        src_ptr,
        tensor.shape,
        tensor.stride,
        tensor.dtype,
        dim,
        start,
        stop,
        step,
        out_ptr,
        out_shape,
        out_stride,
        tensor.dtype,
        runtime,
        stream=stream.stream,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, out_numel, tensor.dtype, device=tensor.device)
    return _wrap_tensor(storage, out_shape, out_stride)


def _npu_assign_to_view(view, value):
    """Write *value* into a view tensor (which shares storage with the original).

    For contiguous views, use D2D memcpy.  Otherwise, use aclnnInplaceCopy.
    """
    runtime = npu_runtime.get_runtime((view.device.index or 0))
    stream = npu_state.current_stream((view.device.index or 0))
    itemsize = _dtype_itemsize(view.dtype)

    if isinstance(value, (int, float)):
        # Create a filled tensor matching the view shape, then copy
        from ..creation import zeros_create
        temp = zeros_create(view.shape, dtype=view.dtype, device=view.device)
        temp = _scalar_to_npu_tensor(value, temp)
        value = temp

    if hasattr(value, 'storage'):
        if view.is_contiguous() and value.is_contiguous() and view.shape == value.shape:
            dst_ptr = _npu_data_ptr(view)
            numel = view.numel()
            copy_size = numel * itemsize
            if value.device.type != "npu":
                src_ptr = value.storage().data_ptr()
                npu_runtime.memcpy_h2d(dst_ptr, copy_size, src_ptr, runtime=runtime)
            else:
                src_ptr = _npu_data_ptr(value)
                npu_runtime.memcpy_d2d(dst_ptr, copy_size, src_ptr, runtime=runtime)
        else:
            # Non-contiguous: use aclnnInplaceCopy
            dst_ptr = _npu_data_ptr(view)
            if value.device.type != "npu":
                # Move value to NPU first
                from ..creation import tensor_create
                import numpy as np
                value = tensor_create(value._numpy_view().copy(), dtype=value.dtype, device=view.device)
            src_ptr = _npu_data_ptr(value)
            aclnn.inplace_copy(
                dst_ptr,
                src_ptr,
                view.shape,
                view.stride,
                view.dtype,
                value.shape,
                value.stride,
                value.dtype,
                runtime,
                stream=stream.stream,
            )
    else:
        raise TypeError(f"Cannot assign {type(value)} to NPU tensor view")


# ---------------------------------------------------------------------------
# Advanced indexing (Tensor, bool mask, list, mixed)
# ---------------------------------------------------------------------------


def _is_advanced_index(item):
    """True if *item* is a Tensor, list, or other advanced index."""
    from ...._tensor import Tensor
    if isinstance(item, Tensor):
        return True
    if isinstance(item, (list, tuple)):
        # A list/tuple of numbers is advanced indexing
        return True
    return False


def _to_npu_index_tensor(key, device, dtype_hint=None):
    """Convert a Python int/list/Tensor to an NPU int64 tensor for indexing."""
    from ...._tensor import Tensor
    from ..creation import tensor_create
    import numpy as np

    if isinstance(key, Tensor):
        if key.dtype.name == 'bool':
            # Bool tensor → nonzero indices
            return _expand_bool_tensor(key)
        if key.device.type == "npu":
            if key.dtype == int64_dtype:
                return key
            # Cast to int64
            return _cast_to_int64(key)
        # CPU tensor → move to NPU
        arr = key._numpy_view().copy()
        return tensor_create(arr.astype(np.int64), dtype=int64_dtype, device=device)

    if isinstance(key, (list, tuple)):
        arr = np.array(key, dtype=np.int64)
        return tensor_create(arr, dtype=int64_dtype, device=device)

    if isinstance(key, (int, np.integer)):
        arr = np.array([int(key)], dtype=np.int64)
        t = tensor_create(arr, dtype=int64_dtype, device=device)
        return reshape(t, ())

    if isinstance(key, (bool, np.bool_)):
        arr = np.array([int(key)], dtype=np.int64)
        t = tensor_create(arr, dtype=int64_dtype, device=device)
        return reshape(t, ())

    raise TypeError(f"Cannot convert {type(key)} to index tensor")


def _cast_to_int64(tensor):
    """Cast an NPU tensor to int64 dtype."""
    runtime = npu_runtime.get_runtime((tensor.device.index or 0))
    stream = npu_state.current_stream((tensor.device.index or 0))
    out_numel = _numel(tensor.shape)
    out_ptr = npu_runtime._alloc_device(out_numel * 8, runtime=runtime)  # int64 = 8 bytes
    src_ptr = _npu_data_ptr(tensor)
    aclnn.cast(
        src_ptr,
        out_ptr,
        tensor.shape,
        tensor.stride,
        tensor.dtype,
        int64_dtype,
        runtime,
        stream=stream.stream,
    )
    out_stride = npu_runtime._contiguous_stride(tensor.shape)
    storage = npu_typed_storage_from_ptr(out_ptr, out_numel, int64_dtype, device=tensor.device)
    return _wrap_tensor(storage, tensor.shape, out_stride)


def _expand_bool_tensor(mask):
    """Convert a bool mask tensor to a tuple of int64 index tensors via nonzero."""
    result = nonzero(mask, as_tuple=True)
    return result


def _compute_broadcast_shape(shapes):
    """Broadcast multiple shapes together following NumPy rules."""
    if not shapes:
        return ()
    result = list(shapes[0])
    for shape in shapes[1:]:
        if len(shape) > len(result):
            result = [1] * (len(shape) - len(result)) + result
        elif len(result) > len(shape):
            shape = (1,) * (len(result) - len(shape)) + tuple(shape)
        new_result = []
        for a, b in zip(result, shape):
            if a == 1:
                new_result.append(b)
            elif b == 1:
                new_result.append(a)
            elif a == b:
                new_result.append(a)
            else:
                raise ValueError(f"Cannot broadcast shapes")
        result = new_result
    return tuple(result)


def _npu_advanced_getitem(tensor, key):
    """Full getitem supporting mixed basic + advanced indexing.

    Phase 1: Process basic indices (int, slice, None, Ellipsis) via views.
    Phase 2: Process advanced indices (Tensor, list, bool) via aclnnIndex.
    """
    from ...._tensor import Tensor

    keys = list(key) if isinstance(key, tuple) else [key]

    if any(isinstance(item, Tensor) and item.dtype.name == 'bool' for item in keys):
        # TODO: re-enable native kernel when CANN fixes aclnnIndex bool-mask advanced indexing (161001)
        raise RuntimeError('NPU boolean mask indexing is not supported')

    # Step 1: Expand bool Tensor indices BEFORE expanding Ellipsis.
    # A bool tensor of N dims consumes N real dims, and we need the correct
    # dim count for Ellipsis expansion.
    expanded_keys = []
    for item in keys:
        if isinstance(item, Tensor) and item.dtype.name == 'bool':
            nz_indices = nonzero(item, as_tuple=True)
            for idx_t in nz_indices:
                expanded_keys.append(idx_t)
        else:
            expanded_keys.append(item)
    keys = expanded_keys

    # Step 2: Expand Ellipsis (now with the correct real dim count)
    ellipsis_count = sum(1 for item in keys if item is Ellipsis)
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if ellipsis_count == 1:
        keys = _expand_ellipsis(keys, tensor.dim())

    # Check if there are any remaining advanced indices
    has_advanced = any(isinstance(item, (Tensor, list)) for item in keys)

    if not has_advanced:
        # All basic — use view path (with potential aclnnSlice)
        return _npu_basic_getitem_with_strided_slices(tensor, keys)

    # Pad keys to ndim with slice(None)
    ndim = tensor.dim()
    real_keys = [k for k in keys if k is not None]
    while len(real_keys) < ndim:
        real_keys.append(slice(None))
        keys.append(slice(None))

    # Separate real-dim actions from None (newaxis) positions
    dim_idx = 0
    dim_actions = []  # (position_in_dim_actions, key_item)
    none_positions = []

    pos = 0
    for item in keys:
        if item is None:
            none_positions.append(pos)
            pos += 1
            continue
        dim_actions.append((dim_idx, item))
        dim_idx += 1
        pos += 1

    # Find which positions in dim_actions have advanced indices
    adv_dims = [i for i, (d, item) in enumerate(dim_actions) if isinstance(item, (Tensor, list))]

    if not adv_dims:
        return _npu_basic_getitem_with_strided_slices(tensor, keys)

    # Pre-apply basic indices on non-advanced dims (high → low to avoid shift)
    prepared = tensor
    dim_remap = list(range(len(dim_actions)))

    for i in range(len(dim_actions) - 1, -1, -1):
        d_orig, item = dim_actions[i]
        if i in adv_dims:
            continue
        cur_dim = dim_remap[i]
        if _is_int_index(item):
            idx = int(item)
            if idx < 0:
                idx += prepared.shape[cur_dim]
            prepared = _npu_select_view(prepared, cur_dim, idx)
            for j in range(len(dim_remap)):
                if dim_remap[j] > cur_dim:
                    dim_remap[j] -= 1
            dim_remap[i] = -1  # removed
        elif isinstance(item, slice):
            start, stop, step = item.indices(prepared.shape[cur_dim])
            if step == 1:
                prepared = _npu_slice_view(prepared, cur_dim, start, stop)
            else:
                prepared = _npu_aclnn_slice(prepared, cur_dim, start, stop, step)

    # Build advanced index tensors and their current dim positions
    adv_index_tensors = []
    adv_current_dims = []
    for i in adv_dims:
        cur_dim = dim_remap[i]
        if cur_dim < 0:
            continue
        adv_current_dims.append(cur_dim)
        idx_tensor = _to_npu_index_tensor(dim_actions[i][1], prepared.device)
        adv_index_tensors.append(idx_tensor)

    if not adv_index_tensors:
        result = prepared
        for pos in none_positions:
            result = _npu_unsqueeze_view(result, pos)
        return result

    # Broadcast all advanced index tensors
    idx_shapes = [t.shape for t in adv_index_tensors]
    broadcast_shape = _compute_broadcast_shape(idx_shapes)

    expanded_idx_tensors = []
    for t in adv_index_tensors:
        if t.shape != broadcast_shape:
            t = _npu_expand(t, broadcast_shape)
        expanded_idx_tensors.append(t)

    # Build entries list for aclnnIndex (None for dims not indexed)
    entries = [None] * prepared.dim()
    for dim_pos, idx_t in zip(adv_current_dims, expanded_idx_tensors):
        entries[dim_pos] = (
            _npu_data_ptr(idx_t),
            idx_t.shape,
            idx_t.stride,
            idx_t.dtype,
        )

    # Compute output shape following PyTorch advanced indexing rules:
    # - If advanced dims are contiguous: broadcast_shape replaces them in-place
    # - If advanced dims are non-contiguous: broadcast_shape goes to the front
    adv_dim_positions = sorted(adv_current_dims)
    are_contiguous = all(
        adv_dim_positions[j] == adv_dim_positions[j - 1] + 1
        for j in range(1, len(adv_dim_positions))
    )

    out_shape_parts = []
    if are_contiguous:
        adv_inserted = False
        for i in range(prepared.dim()):
            if entries[i] is not None:
                if not adv_inserted:
                    out_shape_parts.extend(broadcast_shape)
                    adv_inserted = True
            else:
                out_shape_parts.append(prepared.shape[i])
    else:
        # Non-contiguous: broadcast shape goes to the front
        out_shape_parts.extend(broadcast_shape)
        for i in range(prepared.dim()):
            if entries[i] is None:
                out_shape_parts.append(prepared.shape[i])

    out_shape = tuple(out_shape_parts)
    out_numel = _numel(out_shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    itemsize = _dtype_itemsize(prepared.dtype)

    runtime = npu_runtime.get_runtime((prepared.device.index or 0))
    stream = npu_state.current_stream((prepared.device.index or 0))

    if out_numel == 0:
        out_ptr = npu_runtime._alloc_device(max(itemsize, 1), runtime=runtime)
        storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), prepared.dtype, device=prepared.device)
        return _wrap_tensor(storage, out_shape, out_stride)

    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)
    src_ptr = _npu_data_ptr(prepared)

    aclnn.index(
        src_ptr,
        prepared.shape,
        prepared.stride,
        prepared.dtype,
        entries,
        out_ptr,
        out_shape,
        out_stride,
        prepared.dtype,
        runtime,
        stream=stream.stream,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, out_numel, prepared.dtype, device=prepared.device)
    result = _wrap_tensor(storage, out_shape, out_stride)

    # Apply None insertions
    for pos in none_positions:
        result = _npu_unsqueeze_view(result, pos)

    return result


def _npu_expand(tensor, target_shape):
    """Expand tensor to target shape (broadcast — no data copy, just stride manipulation)."""
    from ...._tensor import Tensor

    src_shape = tensor.shape
    src_stride = tensor.stride
    ndiff = len(target_shape) - len(src_shape)

    # Pad shape/stride on the left with 1s/0s
    padded_shape = (1,) * ndiff + src_shape
    padded_stride = (0,) * ndiff + src_stride

    new_stride = []
    for i, (ts, ps, pst) in enumerate(zip(target_shape, padded_shape, padded_stride)):
        if ps == ts:
            new_stride.append(pst)
        elif ps == 1:
            new_stride.append(0)
        else:
            raise RuntimeError(f"Cannot expand dim {i} from {ps} to {ts}")

    return Tensor(tensor.storage(), tuple(target_shape), tuple(new_stride), tensor.offset)


def _npu_advanced_setitem(tensor, key, value):
    """Full setitem for advanced indexing using aclnnIndexPutImpl."""
    from ...._tensor import Tensor
    import numpy as np

    keys = list(key) if isinstance(key, tuple) else [key]

    # Step 1: Expand bool tensors BEFORE Ellipsis (same reason as getitem)
    expanded_keys = []
    for item in keys:
        if isinstance(item, Tensor) and item.dtype.name == 'bool':
            nz_indices = nonzero(item, as_tuple=True)
            for idx_t in nz_indices:
                expanded_keys.append(idx_t)
        else:
            expanded_keys.append(item)
    keys = expanded_keys

    # Step 2: Expand Ellipsis
    ellipsis_count = sum(1 for item in keys if item is Ellipsis)
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if ellipsis_count == 1:
        keys = _expand_ellipsis(keys, tensor.dim())

    # Remove None entries (newaxis doesn't apply to setitem destination)
    keys = [k for k in keys if k is not None]

    # Pad to ndim
    while len(keys) < tensor.dim():
        keys.append(slice(None))

    # Separate basic and advanced indices
    # For setitem, we apply basic slices as views on the target tensor,
    # then use index_put_impl for the advanced indices.

    prepared = tensor
    adv_dims = []

    dim_remap = list(range(len(keys)))

    for i, item in enumerate(keys):
        if isinstance(item, (Tensor, list)):
            adv_dims.append(i)

    if not adv_dims:
        # All basic — use view + assign
        view = _npu_basic_getitem_with_strided_slices(prepared, keys)
        if view is not None:
            _npu_assign_to_view(view, value)
            return

    # Apply basic slices on non-advanced dims (from high to low)
    dim_actions = list(enumerate(keys))
    for i in range(len(dim_actions) - 1, -1, -1):
        orig_i, item = dim_actions[i]
        if i in adv_dims:
            continue
        cur_dim = dim_remap[i]
        if _is_int_index(item):
            idx = int(item)
            if idx < 0:
                idx += prepared.shape[cur_dim]
            prepared = _npu_select_view(prepared, cur_dim, idx)
            for j in range(len(dim_remap)):
                if dim_remap[j] > cur_dim:
                    dim_remap[j] -= 1
            dim_remap[i] = -1
        elif isinstance(item, slice):
            start, stop, step = item.indices(prepared.shape[cur_dim])
            if step == 1:
                prepared = _npu_slice_view(prepared, cur_dim, start, stop)
            else:
                # For setitem with step!=1, we can't slice as view.
                # Keep the full dim and let index_put_impl handle it.
                # Convert slice to an index tensor
                import numpy as np
                from ..creation import tensor_create
                indices = list(range(start, stop, step))
                idx_t = tensor_create(np.array(indices, dtype=np.int64), dtype=int64_dtype, device=prepared.device)
                adv_dims.append(i)
                keys[i] = idx_t

    # Build index tensors for the advanced dims
    adv_index_tensors = []
    for i in adv_dims:
        cur_dim = dim_remap[i]
        if cur_dim < 0:
            continue
        item = keys[i]
        idx_tensor = _to_npu_index_tensor(item, prepared.device)
        if isinstance(idx_tensor, tuple):
            for t in idx_tensor:
                adv_index_tensors.append(t)
        else:
            adv_index_tensors.append(idx_tensor)

    if not adv_index_tensors:
        return

    # Prepare value tensor
    if isinstance(value, (int, float)):
        from ..creation import tensor_create
        import numpy as np
        val_arr = np.full((1,), value, dtype=npu_runtime._dtype_to_numpy(prepared.dtype))
        value_tensor = tensor_create(val_arr, dtype=prepared.dtype, device=prepared.device)
    elif hasattr(value, 'storage'):
        value_tensor = value
        if value_tensor.device.type != "npu":
            from ..creation import tensor_create
            import numpy as np
            value_tensor = tensor_create(
                value_tensor._numpy_view().copy(),
                dtype=value_tensor.dtype,
                device=prepared.device,
            )
    else:
        from ..creation import tensor_create
        import numpy as np
        value_tensor = tensor_create(
            np.array(value, dtype=npu_runtime._dtype_to_numpy(prepared.dtype)),
            dtype=prepared.dtype,
            device=prepared.device,
        )

    runtime = npu_runtime.get_runtime((prepared.device.index or 0))
    stream = npu_state.current_stream((prepared.device.index or 0))

    index_ptrs = [_npu_data_ptr(t) for t in adv_index_tensors]
    index_shapes = [t.shape for t in adv_index_tensors]
    index_strides = [t.stride for t in adv_index_tensors]
    index_dtypes = [t.dtype for t in adv_index_tensors]

    aclnn.index_put_impl(
        _npu_data_ptr(prepared),
        prepared.shape,
        prepared.stride,
        prepared.dtype,
        index_ptrs,
        index_shapes,
        index_strides,
        index_dtypes,
        _npu_data_ptr(value_tensor),
        value_tensor.shape,
        value_tensor.stride,
        value_tensor.dtype,
        False,  # accumulate
        False,  # unsafe
        runtime,
        stream=stream.stream,
    )


def _flip_310b_fallback(a, dims):
    out = a
    for d in dims:
        size = int(out.shape[d])
        if size <= 1:
            continue
        parts = split(out, 1, dim=d)
        out = cat(tuple(reversed(parts)), dim=d)
    return out


def _diag_310b_fallback(a, diagonal=0):
    from ..creation import empty_create, zeros_create
    from .math import add, mul

    diagonal = int(diagonal)

    if a.dim() == 1:
        n = int(a.shape[0])
        size = n + (diagonal if diagonal >= 0 else -diagonal)
        out = zeros_create((size, size), dtype=a.dtype, device=a.device)
        if n == 0:
            return out

        idx = _npu_arange_1d(n, a.device)
        if diagonal >= 0:
            rows = idx
            cols = idx if diagonal == 0 else add(idx, diagonal)
        else:
            rows = add(idx, -diagonal)
            cols = idx
        return index_put_(out, (rows, cols), a, accumulate=False)

    m = int(a.shape[0])
    n = int(a.shape[1])
    if diagonal >= 0:
        length = max(0, min(m, n - diagonal))
    else:
        length = max(0, min(m + diagonal, n))

    if length == 0:
        return empty_create((0,), dtype=a.dtype, device=a.device)

    idx = _npu_arange_1d(length, a.device)
    if diagonal >= 0:
        rows = idx
        cols = idx if diagonal == 0 else add(idx, diagonal)
    else:
        rows = add(idx, -diagonal)
        cols = idx

    linear = add(mul(rows, n), cols)
    flat = view_backend.reshape(a, (a.numel(),))
    return take(flat, linear)


def _gather_310b_fallback(a, dim, index):
    from ..creation import ones_create, zeros_create
    from .math import mul
    from .reduce import sum_

    dim = _normalize_dim(dim, a.dim())
    dim_size = int(a.shape[dim])
    flat_idx = view_backend.reshape(index, (index.numel(),))
    n = int(flat_idx.shape[0])

    # Build one-hot(index) on NPU via scatter to avoid aclnnGather.
    base = zeros_create((n, dim_size), dtype=a.dtype, device=a.device)
    idx2d = view_backend.reshape(flat_idx, (n, 1))
    src = ones_create((n, 1), dtype=a.dtype, device=a.device)
    one_hot_2d = scatter(base, 1, idx2d, src)
    one_hot = view_backend.reshape(one_hot_2d, tuple(index.shape) + (dim_size,))

    # Move gather dim to last and broadcast over index dim.
    moved = _move_dim_to_last(a, dim)
    moved_shape = list(moved.shape)
    moved_shape.insert(dim, 1)
    moved = view_backend.reshape(moved, tuple(moved_shape))
    moved = _npu_broadcast_to(moved, one_hot.shape)

    weighted = mul(one_hot, moved)
    return sum_(weighted, dim=weighted.dim() - 1, keepdim=False)


def flatten_op(a, start_dim=0, end_dim=-1):
    """Flatten tensor dimensions using reshape."""
    from ...common import view as view_backend
    ndim = len(a.shape)
    if start_dim < 0:
        start_dim += ndim
    if end_dim < 0:
        end_dim += ndim
    if start_dim == end_dim:
        return a
    if start_dim > end_dim:
        raise ValueError(f"flatten: start_dim ({start_dim}) > end_dim ({end_dim})")
    flat_size = 1
    for i in range(start_dim, end_dim + 1):
        flat_size *= a.shape[i]
    new_shape = a.shape[:start_dim] + (flat_size,) + a.shape[end_dim+1:]
    return view_backend.reshape(a, new_shape)


def contiguous(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU contiguous expects NPU tensors")

    a_storage = _unwrap_storage(a)
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    npu_runtime.memcpy_d2d(
        out_ptr,
        out_size,
        a_storage.data_ptr(),
        runtime=runtime,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(storage, a.shape, npu_runtime._contiguous_stride(a.shape))


def flip(a, dims):
    if a.device.type != "npu":
        raise ValueError("NPU flip expects NPU tensors")
    dims = _normalize_dims_tuple(dims, a.dim(), "flip")
    if len(dims) == 0:
        return a

    if _use_soc_fallback("flip"):
        return _flip_310b_fallback(a, dims)

    if not aclnn.flip_symbols_ok():
        raise RuntimeError("aclnnFlip symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.flip(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dims,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def roll(a, shifts, dims=None):
    if a.device.type != "npu":
        raise ValueError("NPU roll expects NPU tensors")
    if dims is None:
        flat = view_backend.reshape(a, (a.numel(),))
        rolled = roll(flat, shifts, dims=0)
        return view_backend.reshape(rolled, a.shape)
    shifts_tuple, dims_tuple = _normalize_roll_args(shifts, dims, a.dim())
    if not aclnn.roll_symbols_ok():
        raise RuntimeError("aclnnRoll symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.roll(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        shifts_tuple,
        dims_tuple,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def rot90(a, k=1, dims=(0, 1)):
    if a.device.type != "npu":
        raise ValueError("NPU rot90 expects NPU tensors")
    if a.dim() < 2:
        raise RuntimeError(f"expected total dims >= 2, but got total dims = {a.dim()}")
    if not isinstance(dims, (tuple, list)) or len(dims) != 2:
        raise RuntimeError("rot90 expects dims to be a tuple of length 2")

    dim0 = _normalize_dim(int(dims[0]), a.dim())
    dim1 = _normalize_dim(int(dims[1]), a.dim())
    if dim0 == dim1:
        raise RuntimeError(f"expected rotation dims to be different, but got dim0 = {dim0} and dim1 = {dim1}")

    k = int(k) % 4
    if k == 0:
        return a
    if k == 1:
        return view_backend.transpose(flip(a, dims=(dim1,)), dim0, dim1)
    if k == 2:
        return flip(flip(a, dims=(dim0,)), dims=(dim1,))
    return view_backend.transpose(flip(a, dims=(dim0,)), dim0, dim1)


def repeat(a, repeats):
    if a.device.type != "npu":
        raise ValueError("NPU repeat expects NPU tensors")
    repeats = _normalize_repeats_tuple(repeats, a.dim(), "repeat")
    if any(int(r) < 0 for r in repeats):
        raise RuntimeError(f"Trying to create tensor with negative dimension {tuple(int(s) * int(r) for s, r in zip(a.shape, repeats))}")
    if not aclnn.repeat_symbols_ok():
        raise RuntimeError("aclnnRepeat symbols not available")

    out_shape = tuple(int(s) * int(r) for s, r in zip(a.shape, repeats))
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = max(_numel(out_shape), 1)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_ptr = npu_runtime._alloc_device(out_numel * _dtype_itemsize(a.dtype), runtime=runtime)

    aclnn.repeat(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        repeats,
        out_shape,
        out_stride,
        runtime,
        stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def tile(a, dims):
    if isinstance(dims, int):
        raise TypeError("tile(): argument 'dims' (position 2) must be tuple of ints, not int")
    return repeat(a, dims)


def repeat_interleave(a, repeats, dim=None):
    if a.device.type != "npu":
        raise ValueError("NPU repeat_interleave expects NPU tensors")

    from ..creation import zeros_create

    if hasattr(repeats, "shape"):
        raise RuntimeError("NPU repeat_interleave with tensor repeats is not implemented without CPU fallback")

    if isinstance(repeats, int) and aclnn.repeat_interleave_int_symbols_ok():
        rep = int(repeats)
        if rep < 0:
            raise ValueError("repeats must be non-negative")
        runtime = npu_runtime.get_runtime((a.device.index or 0))
        stream = npu_state.current_stream((a.device.index or 0))
        if dim is None:
            in_tensor = view_backend.reshape(a, (a.numel(),))
            output_size = in_tensor.numel() * rep
            out_shape = (output_size,)
        else:
            dim = _normalize_dim(dim, a.dim())
            in_tensor = a
            output_size = in_tensor.shape[dim] * rep
            out_shape = list(in_tensor.shape)
            out_shape[dim] = output_size
            out_shape = tuple(out_shape)

        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_ptr = npu_runtime._alloc_device(max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype), runtime=runtime)
        aclnn.repeat_interleave_int(
            _unwrap_storage(in_tensor).data_ptr(),
            out_ptr,
            in_tensor.shape,
            in_tensor.stride,
            in_tensor.dtype,
            rep,
            None if dim is None else int(dim),
            int(output_size),
            out_shape,
            out_stride,
            runtime,
            stream=stream.stream,
        )
        out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
        return _wrap_tensor(out_storage, out_shape, out_stride)

    if dim is None:
        flat = view_backend.reshape(a, (a.numel(),))
        idx, out_size = _build_repeat_interleave_indices(flat.shape[0], repeats, a.device)
        if out_size == 0:
            return zeros_create((0,), dtype=a.dtype, device=a.device)
        return index_select(flat, 0, idx)

    dim = _normalize_dim(dim, a.dim())
    idx, out_size = _build_repeat_interleave_indices(a.shape[dim], repeats, a.device)
    out_shape = list(a.shape)
    out_shape[dim] = out_size
    out_shape = tuple(out_shape)
    if out_size == 0:
        return zeros_create(out_shape, dtype=a.dtype, device=a.device)
    return index_select(a, dim, idx)


def tril(a, diagonal=0):
    if a.device.type != "npu":
        raise ValueError("NPU tril expects NPU tensors")
    if not aclnn.tril_symbols_ok():
        raise RuntimeError("aclnnTril symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.tril(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        int(diagonal),
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def triu(a, diagonal=0):
    if a.device.type != "npu":
        raise ValueError("NPU triu expects NPU tensors")
    if not aclnn.triu_symbols_ok():
        raise RuntimeError("aclnnTriu symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.triu(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        int(diagonal),
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def tril_indices(row, col, offset=0, dtype=None, device=None, layout=None):
    if layout is not None and layout != "strided":
        raise ValueError("layout must be strided")
    if row < 0 or col < 0:
        raise ValueError("row and col must be non-negative")
    if dtype is None:
        dtype = int64_dtype

    from ..creation import tensor_create
    from ...._device import device as Device

    dev = Device("cpu") if device is None else (Device(device) if isinstance(device, str) else device)
    row = int(row)
    col = int(col)
    offset = int(offset)

    rows = []
    cols = []
    for r in range(row):
        upper = min(col - 1, r + offset)
        if upper < 0:
            continue
        for c in range(upper + 1):
            rows.append(r)
            cols.append(c)

    return tensor_create([rows, cols], dtype=dtype, device=dev)


def triu_indices(row, col, offset=0, dtype=None, device=None, layout=None):
    if layout is not None and layout != "strided":
        raise ValueError("layout must be strided")
    if row < 0 or col < 0:
        raise ValueError("row and col must be non-negative")
    if dtype is None:
        dtype = int64_dtype

    from ..creation import tensor_create
    from ...._device import device as Device

    dev = Device("cpu") if device is None else (Device(device) if isinstance(device, str) else device)
    row = int(row)
    col = int(col)
    offset = int(offset)

    rows = []
    cols = []
    for r in range(row):
        start = max(0, r + offset)
        if start >= col:
            continue
        for c in range(start, col):
            rows.append(r)
            cols.append(c)

    return tensor_create([rows, cols], dtype=dtype, device=dev)


def diag(a, diagonal=0):
    if a.device.type != "npu":
        raise ValueError("NPU diag expects NPU tensors")
    if a.dim() not in (1, 2):
        raise ValueError("diag expects 1D or 2D tensor")

    if _use_soc_fallback("diag"):
        return _diag_310b_fallback(a, diagonal=diagonal)

    if not aclnn.diag_symbols_ok():
        raise RuntimeError("aclnnDiag symbols not available")

    from ...meta import infer as meta_infer
    spec = meta_infer.infer_diag(a, diagonal=diagonal)
    out_shape = spec.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_ptr = npu_runtime._alloc_device(max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype), runtime=runtime)

    aclnn.diag(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        int(diagonal),
        out_shape,
        out_stride,
        runtime,
        stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def cartesian_prod(*tensors):
    tensors = _normalize_tensor_sequence_args(tensors)
    if len(tensors) == 0:
        raise RuntimeError("cartesian_prod expects at least one tensor")
    first = tensors[0]
    for t in tensors:
        if t.device.type != "npu":
            raise ValueError("NPU cartesian_prod expects NPU tensors")
        if t.dim() != 1:
            raise ValueError("cartesian_prod expects 1D tensors")
        if t.dtype != first.dtype:
            raise RuntimeError("meshgrid expects all tensors to have the same dtype")

    raise RuntimeError("NPU cartesian_prod is not implemented without CPU fallback")


def block_diag(*tensors):
    tensors = _normalize_tensor_sequence_args(tensors)

    if len(tensors) == 0:
        raise RuntimeError("NPU block_diag is not implemented without CPU fallback")

    first = tensors[0]
    for t in tensors:
        if t.device.type != "npu":
            raise ValueError("NPU block_diag expects NPU tensors")
        if t.dim() != 2:
            raise ValueError("block_diag expects 2D tensors")
        if t.dtype != first.dtype:
            raise ValueError("block_diag expects tensors with the same dtype")

    raise RuntimeError("NPU block_diag is not implemented without CPU fallback")


def broadcast_to_op(a, shape):
    """Tensor.broadcast_to — delegates to expand."""
    return expand(a, shape)


def movedim_op(a, source, destination):
    """torch.movedim — compute permutation then delegate to permute."""
    from ...common import view as view_backend
    ndim = a.dim()
    if isinstance(source, int):
        source = [source]
    if isinstance(destination, int):
        destination = [destination]
    source = [s % ndim for s in source]
    destination = [d % ndim for d in destination]
    order = [i for i in range(ndim) if i not in source]
    dst_src = sorted(zip(destination, source))
    for dst, src in dst_src:
        order.insert(dst, src)
    return view_backend.permute(a, order)


def moveaxis_op(a, source, destination):
    """Move axes of tensor to new positions (alias for movedim)."""
    from ...._dispatch.dispatcher import dispatch
    return dispatch("movedim", "npu", a, source, destination)


# ===========================================================================
# Phase 3: 1D pooling composites (unsqueeze → 2D pool → squeeze)
# ===========================================================================


def unflatten_op(a, dim, sizes):
    """Tensor.unflatten — reshape one dim into multiple dims."""
    from ...common import view as view_backend
    ndim = a.dim()
    d = dim if dim >= 0 else dim + ndim
    new_shape = a.shape[:d] + tuple(sizes) + a.shape[d + 1:]
    return view_backend.reshape(a, new_shape)


def diagonal_op(a, offset=0, dim1=0, dim2=1):
    """torch.diagonal — permute + flatten + gather.

    Uses gather with pre-expanded numpy indices to avoid ACLNN offset bug
    (select creates views with non-zero offset that _create_tensor ignores).
    """
    from ...common import view as view_backend
    import numpy as _np

    ndim = a.dim()
    d1 = dim1 % ndim
    d2 = dim2 % ndim
    if d1 == d2:
        raise RuntimeError("diagonal: dim1 and dim2 cannot be equal")

    # Move d1, d2 to the last two dims
    dims = [i for i in range(ndim) if i != d1 and i != d2] + [d1, d2]
    t = view_backend.permute(a, dims)

    rows = t.shape[-2]
    cols = t.shape[-1]
    if offset >= 0:
        diag_len = max(0, min(rows, cols - offset))
    else:
        diag_len = max(0, min(rows + offset, cols))

    if diag_len == 0:
        batch_shape = t.shape[:-2]
        out_shape = batch_shape + (0,)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        runtime = npu_runtime.get_runtime((a.device.index or 0))
        out_ptr = npu_runtime._alloc_device(
            max(1, _numel(out_shape) * _dtype_itemsize(a.dtype)), runtime=runtime
        )
        out_storage = npu_typed_storage_from_ptr(
            out_ptr, max(1, _numel(out_shape)), a.dtype, device=a.device
        )
        return _wrap_tensor(out_storage, out_shape, out_stride)

    batch_shape = t.shape[:-2]
    flat_shape = batch_shape + (rows * cols,)
    t_flat = view_backend.reshape(contiguous(t), flat_shape)

    if offset >= 0:
        flat_idx = [(i * cols + i + offset) for i in range(diag_len)]
    else:
        flat_idx = [((i - offset) * cols + i) for i in range(diag_len)]

    idx_1d = _np.array(flat_idx, dtype=_np.int64)
    idx_expanded = _np.broadcast_to(idx_1d, batch_shape + (diag_len,)).copy()

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    idx_ptr, _ = npu_runtime._copy_cpu_to_npu(idx_expanded, runtime=runtime)
    idx_shape = batch_shape + (diag_len,)
    idx_stride = npu_runtime._contiguous_stride(idx_shape)
    idx_storage = npu_typed_storage_from_ptr(
        idx_ptr, _numel(idx_shape), int64_dtype, device=a.device
    )
    idx_tensor = _wrap_tensor(idx_storage, idx_shape, idx_stride)

    result = gather(t_flat, -1, idx_tensor)
    return result


# ===========================================================================
# Missing forward ops — composite implementations
# ===========================================================================


def one_hot(indices, num_classes=-1):
    runtime = npu_runtime.get_runtime((indices.device.index or 0))
    stream = npu_state.current_stream((indices.device.index or 0))

    from ...._dtype import float32 as f32, int64 as i64
    from .reduce import amax

    if num_classes < 0:
        max_val = amax(indices)
        import numpy as np
        storage = _unwrap_storage(max_val)
        nbytes = _numel(max_val.shape) * _dtype_itemsize(max_val.dtype)
        buf = (ctypes.c_uint8 * max(nbytes, 1))()
        npu_runtime._memcpy_d2h(ctypes.addressof(buf), nbytes, storage.data_ptr(), runtime=runtime)
        arr = np.frombuffer(buf, dtype=np.int64 if max_val.dtype == i64 else np.int32)
        num_classes = int(arr[0]) + 1

    import numpy as np
    on_data = np.array([1.0], dtype=np.float32)
    off_data = np.array([0.0], dtype=np.float32)
    on_ptr = npu_runtime._alloc_device(4, runtime=runtime)
    off_ptr = npu_runtime._alloc_device(4, runtime=runtime)
    npu_runtime._memcpy_h2d(on_ptr, 4, on_data.ctypes.data, runtime=runtime)
    npu_runtime._memcpy_h2d(off_ptr, 4, off_data.ctypes.data, runtime=runtime)

    out_shape = tuple(indices.shape) + (num_classes,)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * 4, runtime=runtime)

    aclnn.one_hot(
        _unwrap_storage(indices).data_ptr(), on_ptr, off_ptr, out_ptr,
        indices.shape, indices.stride, indices.dtype,
        (1,), (1,), f32,
        (1,), (1,), f32,
        out_shape, out_stride, f32,
        num_classes, -1,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), f32, device=indices.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def scatter(a, dim, index, src):
    if a.device.type != "npu":
        raise ValueError("NPU scatter expects NPU tensors")
    dim = _normalize_dim(dim, a.dim())
    _require_int64_indices(index, "scatter")
    if index.dim() != a.dim():
        raise ValueError("index shape mismatch")
    for i, size in enumerate(index.shape):
        if i != dim and size != a.shape[i]:
            raise ValueError("index shape mismatch")
    _validate_index_bounds(index, a.shape[dim], allow_negative=False, name="scatter")

    if hasattr(src, "shape"):
        if src.device.type != "npu":
            raise ValueError("scatter src tensor must be on NPU")
        src_tensor = src
    else:
        src_tensor = _scalar_to_npu_tensor(src, a)

    if src_tensor.shape != index.shape:
        src_tensor = _npu_broadcast_to(src_tensor, index.shape)

    if not aclnn.scatter_symbols_ok():
        raise RuntimeError("aclnnScatter symbols not available")

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype), runtime=runtime)

    aclnn.scatter(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(index).data_ptr(),
        _unwrap_storage(src_tensor).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        index.shape,
        index.stride,
        index.dtype,
        src_tensor.shape,
        src_tensor.stride,
        src_tensor.dtype,
        dim,
        0,
        runtime,
        stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def nonzero(a, as_tuple=False):
    if a.device.type != "npu":
        raise ValueError("NPU nonzero expects NPU tensors")
    if not aclnn.nonzero_symbols_ok():
        raise RuntimeError("aclnnNonzero symbols not available")
    from .reduce import count_nonzero

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    out_shape = (max(a.numel(), 1), a.dim())
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(max(_numel(out_shape), 1) * _dtype_itemsize(int64_dtype), runtime=runtime)

    aclnn.nonzero(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        out_shape,
        out_stride,
        runtime,
        stream=stream.stream,
    )

    full_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), int64_dtype, device=a.device)
    full = _wrap_tensor(full_storage, out_shape, out_stride)

    nonzero_count = count_nonzero(a, dim=None, keepdim=False)
    rows = _read_int64_scalar(nonzero_count)

    if rows < out_shape[0]:
        full = _slice_along_dim(full, 0, rows, 0)

    if not as_tuple:
        return full

    from ..creation import zeros_create
    from ...common import view as view_backend

    if a.dim() == 0:
        if rows == 0:
            return (zeros_create((0,), dtype=int64_dtype, device=a.device),)
        return (zeros_create((1,), dtype=int64_dtype, device=a.device),)

    outputs = []
    for dim_idx in range(a.dim()):
        col = _slice_along_dim(full, dim_idx, dim_idx + 1, 1)
        outputs.append(view_backend.reshape(col, (rows,)))
    return tuple(outputs)


def cat(tensors, dim=0):
    """Concatenate tensors along an existing dimension using aclnnCat."""
    if not tensors:
        raise RuntimeError("cat requires at least one tensor")
    if len(tensors) == 1:
        return contiguous(tensors[0])

    first = tensors[0]
    runtime = npu_runtime.get_runtime((first.device.index or 0))
    stream = npu_state.current_stream((first.device.index or 0))

    if not aclnn.cat_symbols_ok():
        raise RuntimeError("aclnnCat not available")

    ndim = len(first.shape)
    if dim < 0:
        dim += ndim

    # Validate shapes and compute output shape
    out_shape = list(first.shape)
    for t in tensors[1:]:
        if len(t.shape) != ndim:
            raise RuntimeError("cat: tensors must have the same number of dimensions")
        for d in range(ndim):
            if d != dim and t.shape[d] != first.shape[d]:
                raise RuntimeError(f"cat: dimension {d} size mismatch")
        out_shape[dim] += t.shape[dim]
    out_shape = tuple(out_shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(first.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    # Prepare inputs for aclnn
    tensor_ptrs = [_unwrap_storage(t).data_ptr() for t in tensors]
    shapes = [t.shape for t in tensors]
    strides = [t.stride for t in tensors]
    dtypes = [t.dtype for t in tensors]

    aclnn.cat(
        tensor_ptrs, shapes, strides, dtypes,
        dim, out_ptr, out_shape, out_stride, first.dtype,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, first.dtype, device=first.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def concatenate(tensors, dim=0):
    return cat(tensors, dim=dim)


def stack(tensors, dim=0):
    """Stack tensors along a new dimension using aclnnStack."""
    if not tensors:
        raise RuntimeError("stack requires at least one tensor")

    first = tensors[0]
    runtime = npu_runtime.get_runtime((first.device.index or 0))
    stream = npu_state.current_stream((first.device.index or 0))

    if not aclnn.stack_symbols_ok():
        raise RuntimeError("aclnnStack not available")

    ndim = len(first.shape)
    if dim < 0:
        dim += ndim + 1

    # Validate shapes
    for t in tensors[1:]:
        if t.shape != first.shape:
            raise RuntimeError("stack: all tensors must have the same shape")

    # Compute output shape: insert new dimension with size = len(tensors)
    out_shape = list(first.shape[:dim]) + [len(tensors)] + list(first.shape[dim:])
    out_shape = tuple(out_shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(first.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    # Prepare inputs for aclnn
    tensor_ptrs = [_unwrap_storage(t).data_ptr() for t in tensors]
    shapes = [t.shape for t in tensors]
    strides = [t.stride for t in tensors]
    dtypes = [t.dtype for t in tensors]

    aclnn.stack(
        tensor_ptrs, shapes, strides, dtypes,
        dim, out_ptr, out_shape, out_stride, first.dtype,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, first.dtype, device=first.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def pad_sequence(seqs, batch_first=False, padding_value=0.0, padding_side="right"):
    if not seqs:
        raise ValueError("pad_sequence expects a non-empty list of tensors")

    first = seqs[0]
    if first.device.type != "npu":
        raise ValueError("NPU pad_sequence expects NPU tensors")
    if padding_side not in ("left", "right"):
        raise ValueError("padding_side must be 'left' or 'right'")

    from ..creation import full_create

    max_len = max(int(t.shape[0]) for t in seqs)
    batch = len(seqs)
    trailing = tuple(first.shape[1:])
    trailing_numel = 1
    for d in trailing:
        trailing_numel *= int(d)

    if batch_first:
        out_shape = (batch, max_len) + trailing
    else:
        out_shape = (max_len, batch) + trailing
    out = full_create(out_shape, padding_value, dtype=first.dtype, device=first.device)

    itemsize = _dtype_itemsize(first.dtype)
    dst_base = int(_unwrap_storage(out).data_ptr())
    out_stride = out.stride

    for i, t in enumerate(seqs):
        if t.device.type != "npu":
            raise ValueError("all tensors must be NPU tensors")
        if t.dtype != first.dtype:
            raise ValueError("all tensors must have the same dtype")
        if tuple(t.shape[1:]) != trailing:
            raise ValueError("all tensors must have the same trailing dimensions")

        src = t if t.is_contiguous() else contiguous(t)
        length = int(src.shape[0])
        start_idx = max_len - length if padding_side == "left" else 0

        src_base = int(_unwrap_storage(src).data_ptr())
        if batch_first:
            dst_elem_offset = int(i) * int(out_stride[0]) + int(start_idx) * int(out_stride[1])
            copy_bytes = int(length * trailing_numel * itemsize)
            npu_runtime.memcpy_d2d(
                dst_base + dst_elem_offset * itemsize,
                copy_bytes,
                src_base,
                runtime=runtime,
            )
        else:
            block_bytes = int(trailing_numel * itemsize)
            for step in range(length):
                dst_elem_offset = (int(start_idx + step) * int(out_stride[0])) + (int(i) * int(out_stride[1]))
                src_elem_offset = int(step) * int(src.stride[0])
                npu_runtime.memcpy_d2d(
                    dst_base + dst_elem_offset * itemsize,
                    block_bytes,
                    src_base + src_elem_offset * itemsize,
                    runtime=runtime,
                )
    return out


def chunk(a, chunks, dim=0):
    dim = _normalize_dim(dim, a.dim())
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
        end = min(start + chunk_size, dim_size)
        if start >= end:
            break
        outputs.append(_slice_along_dim(a, start, end, dim))
    return tuple(outputs)


def split(a, split_size_or_sections, dim=0):
    dim = _normalize_dim(dim, a.dim())
    dim_size = a.shape[dim]
    outputs = []
    if isinstance(split_size_or_sections, int):
        if split_size_or_sections <= 0:
            raise ValueError("split_size must be > 0")
        step = split_size_or_sections
        for start in range(0, dim_size, step):
            end = min(start + step, dim_size)
            outputs.append(_slice_along_dim(a, start, end, dim))
    else:
        sizes = list(split_size_or_sections)
        if sum(sizes) != dim_size:
            raise ValueError("split sections must sum to dim size")
        start = 0
        for size in sizes:
            end = start + size
            outputs.append(_slice_along_dim(a, start, end, dim))
            start = end
    return tuple(outputs)


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
    dim = _normalize_dim(dim, a.dim())
    dim_size = a.shape[dim]
    outputs = []
    from ...common import view as view_backend
    for i in range(dim_size):
        sliced = _slice_along_dim(a, i, i + 1, dim)
        out_shape = a.shape[:dim] + a.shape[dim + 1:]
        outputs.append(view_backend.reshape(sliced, out_shape))
    return tuple(outputs)


def hstack(tensors):
    if tensors[0].dim() == 1:
        return cat(tensors, dim=0)
    return cat(tensors, dim=1)


def vstack(tensors):
    from ...common import view as view_backend
    if tensors[0].dim() == 1:
        expanded = [view_backend.reshape(t, (1, t.shape[0])) for t in tensors]
        return cat(expanded, dim=0)
    return cat(tensors, dim=0)


def row_stack(tensors):
    return vstack(tensors)


def dstack(tensors):
    from ...common import view as view_backend
    expanded = []
    for t in tensors:
        if t.dim() == 1:
            expanded.append(view_backend.reshape(t, (1, t.shape[0], 1)))
        elif t.dim() == 2:
            expanded.append(view_backend.reshape(t, (t.shape[0], t.shape[1], 1)))
        else:
            expanded.append(t)
    return cat(expanded, dim=2)


def column_stack(tensors):
    from ...common import view as view_backend
    if tensors[0].dim() == 1:
        expanded = [view_backend.reshape(t, (t.shape[0], 1)) for t in tensors]
        return cat(expanded, dim=1)
    return cat(tensors, dim=1)


def getitem(tensor, key):
    """NPU tensor indexing — full support for basic and advanced indexing."""
    if not isinstance(key, tuple):
        key = (key,)

    if _is_basic_index_key(key):
        view = _npu_basic_getitem_view(tensor, key)
        if view is not None:
            return view

    return _npu_advanced_getitem(tensor, key)


def setitem(tensor, key, value):
    """NPU tensor index assignment — full support for basic and advanced indexing."""
    if not isinstance(key, tuple):
        key = (key,)

    if _is_basic_index_key(key):
        view = _npu_basic_getitem_view(tensor, key)
        if view is not None:
            _npu_assign_to_view(view, value)
            return tensor

    _npu_advanced_setitem(tensor, key, value)
    return tensor


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------


def gather(a, dim, index):
    dim = _normalize_dim(dim, a.dim())
    _require_int64_indices(index, "gather")
    if index.dim() != a.dim():
        raise ValueError("index shape mismatch")
    for i, size in enumerate(index.shape):
        if i != dim and size != a.shape[i]:
            raise ValueError("index shape mismatch")
    _validate_index_bounds(index, a.shape[dim], allow_negative=False, name="gather")

    if _use_soc_fallback("gather"):
        return _gather_310b_fallback(a, dim, index)

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = index.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(a.dtype), runtime=runtime)
    aclnn.gather(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(index).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        index.shape,
        index.stride,
        index.dtype,
        out_shape,
        out_stride,
        a.dtype,
        dim,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def index_select(a, dim, index):
    dim = _normalize_dim(dim, a.dim())
    _require_int64_indices(index, "index_select")
    if index.dim() != 1:
        raise ValueError("index must be 1D")
    dim_size = a.shape[dim]
    _validate_index_bounds(index, dim_size, allow_negative=True, name="index_select")
    norm_index = _normalize_negative_indices(index, dim_size)
    index_shape = list(a.shape)
    index_shape[dim] = norm_index.shape[0]
    index_shape = tuple(index_shape)
    expand_shape = [1] * a.dim()
    expand_shape[dim] = norm_index.shape[0]
    expanded = view_backend.reshape(norm_index, tuple(expand_shape))
    expanded = _npu_broadcast_to(expanded, index_shape)
    return gather(a, dim, expanded)


def take(a, index):
    _require_int64_indices(index, "take")
    flat = view_backend.reshape(a, (a.numel(),))
    dim_size = flat.shape[0]
    _validate_index_bounds(index, dim_size, allow_negative=True, name="take")
    norm_index = _normalize_negative_indices(index, dim_size)
    index_shape = norm_index.shape
    gather_index = norm_index
    if gather_index.dim() == 0:
        gather_index = gather_index.reshape((1,))
    if gather_index.dim() != 1:
        gather_index = gather_index.reshape((gather_index.numel(),))
    out = gather(flat, 0, gather_index)
    return out.reshape(index_shape)


def take_along_dim(a, indices, dim):
    dim = _normalize_dim(dim, a.dim())
    _require_int64_indices(indices, "take_along_dim")
    if indices.dim() != a.dim():
        raise ValueError("indices shape mismatch")
    for i, size in enumerate(indices.shape):
        if i != dim and size != a.shape[i]:
            raise ValueError("indices shape mismatch")
    dim_size = a.shape[dim]
    _validate_index_bounds(indices, dim_size, allow_negative=True, name="take_along_dim")
    norm_indices = _normalize_negative_indices(indices, dim_size)
    return gather(a, dim, norm_indices)


def masked_select(a, mask):
    from .comparison import ne
    from .reduce import count_nonzero
    if mask.dtype != bool_dtype:
        mask = ne(mask, _scalar_to_npu_tensor(0, mask))
    if mask.shape != a.shape:
        raise ValueError("mask shape mismatch")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    mask_count = count_nonzero(mask, dim=None, keepdim=False)
    out_numel = _read_int64_scalar(mask_count)
    out_shape = (a.numel(),)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(max(a.numel(), 1) * _dtype_itemsize(a.dtype), runtime=runtime)
    aclnn.masked_select(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(mask).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        mask.shape,
        mask.stride,
        mask.dtype,
        out_shape,
        out_stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    full_storage = npu_typed_storage_from_ptr(out_ptr, max(a.numel(), 1), a.dtype, device=a.device)
    full = _wrap_tensor(full_storage, out_shape, out_stride)
    if out_numel == out_shape[0]:
        return full
    return _slice_along_dim(full, 0, out_numel, 0)


def narrow(a, dim, start, length):
    """Narrow: return a view of tensor along dim from start to start+length."""
    from ...._tensor import Tensor
    d = dim if dim >= 0 else dim + a.dim()
    new_shape = list(a.shape)
    new_shape[d] = int(length)
    new_offset = a.offset + int(start) * a.stride[d]
    return Tensor(a.storage(), tuple(new_shape), a.stride, new_offset)


def select(a, dim, index):
    """Select: remove dim by indexing a single element along it (view op)."""
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
    """Expand: broadcast tensor to larger sizes (view op, no copy)."""
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
    """Non-inplace masked fill — returns a copy with mask applied."""
    result = a.clone()
    masked_fill_(result, mask, value)
    return result


def masked_fill_(a, mask, value):
    """In-place masked fill using aclnnInplaceMaskedFillScalar."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.inplace_masked_fill_scalar(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        _npu_data_ptr(mask), mask.shape, mask.stride, mask.dtype,
        value, runtime, stream=stream.stream,
    )
    return a


def index_put_(a, indices, values, accumulate=False):
    """In-place index_put_ using aclnnIndexPutImpl."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    index_ptrs = [_npu_data_ptr(t) for t in indices]
    index_shapes = [t.shape for t in indices]
    index_strides = [t.stride for t in indices]
    index_dtypes = [t.dtype for t in indices]
    val_ptr = _npu_data_ptr(values)
    aclnn.index_put_impl(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        index_ptrs, index_shapes, index_strides, index_dtypes,
        val_ptr, values.shape, values.stride, values.dtype,
        accumulate, False, runtime, stream=stream.stream,
    )
    return a


def index_put(a, indices, values, accumulate=False):
    """Non-inplace index_put — returns a copy."""
    result = a.clone()
    index_put_(result, indices, values, accumulate)
    return result


def index_copy_(a, dim, index, source):
    """In-place index_copy_ using aclnnInplaceIndexCopy."""
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.inplace_index_copy(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        d,
        _npu_data_ptr(index), index.shape, index.stride, index.dtype,
        _npu_data_ptr(source), source.shape, source.stride, source.dtype,
        runtime, stream=stream.stream,
    )
    return a


def index_fill_(a, dim, index, value):
    """In-place index_fill_ using aclnnInplaceIndexFill."""
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.inplace_index_fill(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        d,
        _npu_data_ptr(index), index.shape, index.stride, index.dtype,
        value, runtime, stream=stream.stream,
    )
    return a


def index_add_(a, dim, index, source, alpha=1.0):
    """In-place index_add_ using aclnnIndexAdd (writes to self as out)."""
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.index_add(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        d,
        _npu_data_ptr(index), index.shape, index.stride, index.dtype,
        _npu_data_ptr(source), source.shape, source.stride, source.dtype,
        float(alpha),
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        runtime, stream=stream.stream,
    )
    return a


def scatter_(a, dim, index, src):
    """In-place scatter_ — delegates to existing scatter with self as out."""
    from ...._tensor import Tensor
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if isinstance(src, Tensor):
        src_ptr = _npu_data_ptr(src)
        src_shape = src.shape
        src_stride = src.stride
        src_dtype = src.dtype
    else:
        # Scalar src — create a filled tensor
        from ..creation import tensor_create
        import numpy as np
        src_arr = np.full(a.shape, src, dtype=npu_runtime._dtype_to_numpy(a.dtype))
        src_t = tensor_create(src_arr, dtype=a.dtype, device=a.device)
        src_ptr = _npu_data_ptr(src_t)
        src_shape = src_t.shape
        src_stride = src_t.stride
        src_dtype = src_t.dtype
    aclnn.scatter(
        _npu_data_ptr(a),
        _npu_data_ptr(a),
        a.shape, a.stride, a.dtype,
        index.shape, index.stride, index.dtype,
        src_shape, src_stride, src_dtype,
        d, 0, runtime, stream=stream.stream,
    )
    return a


def scatter_add_(a, dim, index, src):
    """In-place scatter_add_ using aclnnScatterAdd."""
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.scatter_add_op(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        d,
        _npu_data_ptr(index), index.shape, index.stride, index.dtype,
        _npu_data_ptr(src), src.shape, src.stride, src.dtype,
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        runtime, stream=stream.stream,
    )
    return a


def masked_scatter_(a, mask, source):
    """In-place masked_scatter_ using aclnnInplaceMaskedScatter."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.inplace_masked_scatter(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        _npu_data_ptr(mask), mask.shape, mask.stride, mask.dtype,
        _npu_data_ptr(source), source.shape, source.stride, source.dtype,
        runtime, stream=stream.stream,
    )
    return a


def unfold(a, dimension, size, step):
    """Unfold along a dimension — returns a higher-dimensional view/copy."""
    d = dimension if dimension >= 0 else dimension + a.dim()
    dim_size = a.shape[d]
    n_windows = max(0, (dim_size - size) // step + 1)
    if n_windows == 0:
        new_shape = list(a.shape)
        new_shape[d] = 0
        new_shape.append(size)
        out_shape = tuple(new_shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_numel = 0
        itemsize = _dtype_itemsize(a.dtype)
        runtime = npu_runtime.get_runtime((a.device.index or 0))
        out_ptr = npu_runtime._alloc_device(max(itemsize, 1), runtime=runtime)
        storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), a.dtype, device=a.device)
        return _wrap_tensor(storage, out_shape, out_stride)
    # Build result by gathering slices
    # Each window is a[..., i*step:i*step+size, ...] along dim d
    # Output shape: a.shape with a.shape[d] replaced by n_windows, plus trailing `size`
    slices = []
    for i in range(n_windows):
        start = i * step
        # Use the existing view-based slice
        sliced = _npu_slice_view(a, d, start, start + size)
        slices.append(sliced)
    # Stack slices along dim d, then the window elements are along d+1
    # Actually, unfold should have shape [..., n_windows, ..., size] with size at end
    # The simplest correct approach: use contiguous + D2D copies
    out_shape = list(a.shape)
    out_shape[d] = n_windows
    out_shape.append(size)
    out_shape = tuple(out_shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_ptr = npu_runtime._alloc_device(max(out_numel * itemsize, 1), runtime=runtime)
    storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), a.dtype, device=a.device)
    result = _wrap_tensor(storage, out_shape, out_stride)
    # Fill each window slot
    for i in range(n_windows):
        start = i * step
        sliced = _npu_slice_view(a, d, start, start + size)
        # sliced has shape [..., size, ...] with size at dim d
        # We need to copy this into result[..., i, ..., :] where i is at dim d and : is at the end
        # Build destination view
        dst = _npu_select_view(result, d, i)
        # dst has shape [..., ...rest..., size] — the original dims except d, plus trailing size
        # sliced needs to be transposed: move dim d to the end
        # Use contiguous copy approach
        sliced_contig = sliced.contiguous()
        dst_shape_flat = _numel(dst.shape)
        src_ptr = _npu_data_ptr(sliced_contig)
        dst_ptr_val = _npu_data_ptr(dst)
        copy_bytes = dst_shape_flat * itemsize
        if copy_bytes > 0:
            npu_runtime.memcpy_d2d(dst_ptr_val, copy_bytes, src_ptr, runtime=runtime)
    return result

