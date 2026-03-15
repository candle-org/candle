from ..._tensor import Tensor
from ...autograd.grad_mode import current_creation_mode
from ... import _dtype as _dtype


def _get_base(tensor):
    return tensor._base if tensor._base is not None else tensor


def _make_view(base, shape, stride, offset, op, source=None, *, creation_kind=None):
    source_view_meta = getattr(source, "_view_meta", None) or {}
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    inherited_kind = source_view_meta.get("creation_kind")
    if creation_kind is None:
        creation_kind = inherited_kind
    if creation_mode is not None:
        if getattr(source, "_is_view", lambda: False)():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    view = Tensor(base.storage(), shape, stride, offset, requires_grad=base.requires_grad)
    view._base = base
    view._version_counter = base._version_counter
    view._view_meta = {
        "op": op,
        "shape": tuple(shape),
        "stride": tuple(stride),
        "offset": int(offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return view


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def reshape(a, shape):
    shape = tuple(shape)
    size = 1
    for d in a.shape:
        size *= d
    # Handle -1 dimension inference (same as view)
    infer_idx = None
    known_size = 1
    shape_list = list(shape)
    for idx, dim in enumerate(shape_list):
        if dim == -1:
            if infer_idx is not None:
                raise RuntimeError("only one dimension can be inferred")
            infer_idx = idx
            continue
        known_size *= dim
    if infer_idx is not None:
        if known_size == 0 or size % known_size != 0:
            raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {size}")
        shape_list[infer_idx] = size // known_size
    shape = tuple(shape_list)
    new_size = 1
    for d in shape:
        new_size *= d
    if size != new_size:
        raise ValueError("reshape size mismatch")
    if not a.is_contiguous():
        a = a.contiguous()
    stride = _contiguous_stride(shape)
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "reshape", source=a)


def view(a, shape):
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)

    if not a.is_contiguous():
        raise RuntimeError(
            "view size is not compatible with input tensor's size and stride "
            "(at least one dimension spans across two contiguous subspaces). "
            "Use .reshape(...) instead."
        )

    size = 1
    for d in a.shape:
        size *= d

    infer_idx = None
    known_size = 1
    shape_list = list(shape)
    for idx, dim in enumerate(shape_list):
        if dim == -1:
            if infer_idx is not None:
                raise RuntimeError("only one dimension can be inferred")
            infer_idx = idx
            continue
        known_size *= dim

    if infer_idx is not None:
        if known_size == 0 or size % known_size != 0:
            raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {size}")
        shape_list[infer_idx] = size // known_size

    shape = tuple(shape_list)
    new_size = 1
    for d in shape:
        new_size *= d
    if size != new_size:
        raise ValueError("view size mismatch")

    stride = _contiguous_stride(shape)
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "view", source=a)


def transpose(a, dim0, dim1):
    shape = list(a.shape)
    stride = list(a.stride)
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "transpose", source=a)


def squeeze(a, dim=None):
    shape = list(a.shape)
    stride = list(a.stride)
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            if dim:
                ndim = len(shape)
                targets = set()
                for item in dim:
                    d = item if item >= 0 else item + ndim
                    targets.add(d)
                pairs = [
                    (s, st)
                    for idx, (s, st) in enumerate(zip(shape, stride))
                    if idx not in targets or s != 1
                ]
                shape = [p[0] for p in pairs]
                stride = [p[1] for p in pairs]
        else:
            d = dim if dim >= 0 else dim + len(shape)
            if 0 <= d < len(shape) and shape[d] == 1:
                del shape[d]
                del stride[d]
    else:
        pairs = [(s, st) for s, st in zip(shape, stride) if s != 1]
        shape = [p[0] for p in pairs]
        stride = [p[1] for p in pairs]
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "squeeze", source=a)


def unsqueeze(a, dim):
    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim + 1
    shape = list(a.shape)
    stride = list(a.stride)
    new_stride = stride[d] * shape[d] if d < ndim else 1
    shape.insert(d, 1)
    stride.insert(d, new_stride)
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "unsqueeze", source=a)


def narrow(a, dim, start, length, *, creation_kind=None):
    d = dim if dim >= 0 else dim + len(a.shape)
    new_shape = list(a.shape)
    new_shape[d] = int(length)
    new_offset = a.offset + int(start) * a.stride[d]
    base = _get_base(a)
    return _make_view(base, tuple(new_shape), a.stride, new_offset, "narrow", source=a, creation_kind=creation_kind)


def select(a, dim, index, *, creation_kind=None):
    d = dim if dim >= 0 else dim + len(a.shape)
    idx = int(index)
    if idx < 0:
        idx += a.shape[d]
    new_shape = list(a.shape)
    del new_shape[d]
    new_stride = list(a.stride)
    new_offset = a.offset + idx * a.stride[d]
    del new_stride[d]
    base = _get_base(a)
    return _make_view(base, tuple(new_shape), tuple(new_stride), new_offset, "select", source=a, creation_kind=creation_kind)


def permute(a, dims):
    shape = [a.shape[d] for d in dims]
    stride = [a.stride[d] for d in dims]
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "permute", source=a)


def view_as_real(a):
    if not a.is_complex():
        raise RuntimeError("view_as_real expects a complex tensor")
    if a.dtype.itemsize % 2 != 0:
        raise RuntimeError("view_as_real expects complex dtype with even itemsize")
    if a.dtype == _dtype.complex64:
        out_dtype = _dtype.float32
    elif a.dtype == _dtype.complex128:
        out_dtype = _dtype.float64
    elif a.dtype == _dtype.complex32:
        out_dtype = _dtype.float16
    else:
        raise RuntimeError("view_as_real expects a supported complex dtype")
    shape = tuple(a.shape) + (2,)
    stride = tuple(s * 2 for s in a.stride) + (1,)
    base = _get_base(a)
    view = _make_view(base, shape, stride, a.offset * 2, "view_as_real", source=a)
    view._storage = base._storage._reinterpret(out_dtype)
    return view


def view_as_complex(a):
    if a.is_complex():
        raise RuntimeError("view_as_complex expects a non-complex tensor")
    if len(a.shape) == 0 or a.shape[-1] != 2:
        raise RuntimeError("view_as_complex expects last dimension of size 2")
    if a.dtype == _dtype.float16:
        out_dtype = _dtype.complex32
    elif a.dtype == _dtype.float32:
        out_dtype = _dtype.complex64
    elif a.dtype == _dtype.float64:
        out_dtype = _dtype.complex128
    else:
        raise RuntimeError("view_as_complex is only supported for half, float and double tensors")
    shape = tuple(a.shape[:-1])
    stride = tuple(s // 2 for s in a.stride[:-1])
    base = _get_base(a)
    view = _make_view(base, shape, stride, a.offset // 2, "view_as_complex", source=a)
    view._storage = base._storage._reinterpret(out_dtype)
    return view
