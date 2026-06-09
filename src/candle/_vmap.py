"""Small torch.vmap compatibility wrapper."""

import builtins as _builtins

from ._functional import movedim, select, stack
from ._tensor import Tensor


def _normalize_in_dims(in_dims, arg_count):
    if isinstance(in_dims, tuple):
        if len(in_dims) != arg_count:
            raise ValueError(
                f"vmap in_dims must have one entry per argument; got {len(in_dims)} "
                f"entries for {arg_count} arguments"
            )
        return in_dims
    return tuple(in_dims for _ in _builtins.range(arg_count))


def _normalize_dim(dim, ndim):
    if dim is None:
        return None
    if not isinstance(dim, _builtins.int):
        raise TypeError("vmap in_dims entries must be int or None")
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"vmap in_dim {dim} is out of bounds for tensor with {ndim} dims")
    return dim


def _mapped_size(args, in_dims):
    size = None
    normalized = []
    for arg, dim in zip(args, in_dims):
        if dim is None:
            normalized.append(None)
            continue
        if not isinstance(arg, Tensor):
            raise TypeError("vmap can only map tensor positional arguments")
        normalized_dim = _normalize_dim(dim, len(arg.shape))
        current = arg.shape[normalized_dim]
        if size is None:
            size = current
        elif current != size:
            raise ValueError("vmap mapped dimensions must all have the same size")
        normalized.append(normalized_dim)
    if size is None:
        raise ValueError("vmap requires at least one mapped positional argument")
    return size, tuple(normalized)


def _stack_outputs(outputs, out_dims):
    first = outputs[0]
    if isinstance(first, Tensor):
        result = stack(outputs, dim=0)
        if out_dims != 0:
            result = movedim(result, 0, out_dims)
        return result
    if isinstance(first, tuple):
        return tuple(
            _stack_outputs([output[index] for output in outputs], out_dims)
            for index in _builtins.range(len(first))
        )
    if isinstance(first, list):
        return [
            _stack_outputs([output[index] for output in outputs], out_dims)
            for index in _builtins.range(len(first))
        ]
    return outputs


def vmap(func, in_dims=0, out_dims=0, randomness="error", *, chunk_size=None):
    """Vectorize ``func`` over tensor dimensions.

    This implements the eager semantics needed by PyTorch-compatible library code:
    mapped tensor inputs are sliced along ``in_dims``, ``func`` is called for each
    slice, and tensor outputs are stacked along ``out_dims``.  It intentionally
    does not implement PyTorch's batching-rule performance machinery yet.
    """
    if randomness != "error":
        raise NotImplementedError("vmap randomness modes other than 'error' are not implemented")
    if chunk_size is not None:
        raise NotImplementedError("vmap chunk_size is not implemented")

    def wrapper(*args, **kwargs):
        dims = _normalize_in_dims(in_dims, len(args))
        size, normalized_dims = _mapped_size(args, dims)
        outputs = []
        for index in _builtins.range(size):
            sliced_args = []
            for arg, dim in zip(args, normalized_dims):
                if dim is None:
                    sliced_args.append(arg)
                else:
                    sliced_args.append(select(arg, dim, index))
            outputs.append(func(*sliced_args, **kwargs))
        return _stack_outputs(outputs, out_dims)

    return wrapper
