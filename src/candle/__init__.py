__version__ = "0.1.0"

import os
import math

from ._dtype import (
    DType,
    float8_e4m3fn, float8_e5m2, float8_e8m0fnu,
    float16, float32, float64, bfloat16,
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    bool,
    complex32, complex64, complex128,
    quint8, qint8, qint32, quint4x2,
    # aliases
    half, double, short, long, byte, cfloat, cdouble,
    # info classes
    finfo, iinfo,
)
from ._dtype import float as float  # noqa: F811
from ._dtype import int as int  # noqa: F811
from ._dtype import DType as dtype  # torch.dtype compatibility
from ._dtype import DType as Dtype  # schema/type alias compatibility
from ._device import device as Device, _default_device, get_default_device, set_default_device
from ._device import device
from ._tensor import Tensor

# Torch-level numeric constants
import builtins as _builtins
inf = _builtins.float("inf")
nan = _builtins.float("nan")

# math constants
e = _builtins.float(math.e)
pi = _builtins.float(math.pi)

_vitals = {}
# Default dtype handling (torch.get_default_dtype / set_default_dtype)
_DEFAULT_DTYPE = float32

def set_default_dtype(dtype):
    if not isinstance(dtype, DType):
        raise TypeError("dtype must be a candle DType")
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = dtype


def get_default_dtype():
    return _DEFAULT_DTYPE


# Vitals (minimal compatibility)
def vitals_enabled():
    return os.environ.get("TORCH_VITAL", "").upper() == "ON"


def set_vital(category, name, value):
    if not vitals_enabled():
        return False
    _vitals.setdefault(category, {})[name] = value
    return True


def read_vitals():
    if not vitals_enabled():
        return ""
    lines = []
    if "Dataloader" in _vitals:
        lines.append("Dataloader.enabled\t\t True")
    lines.append("CUDA.used\t\t true")
    for category, entries in _vitals.items():
        for key, val in entries.items():
            lines.append(str(val))
    return "\n".join(lines)


def quantize_per_tensor(*_args, **_kwargs):
    raise RuntimeError("quantized tensors are not supported")

# Tensor type aliases for torch API compatibility
FloatTensor = Tensor
DoubleTensor = Tensor
HalfTensor = Tensor
BFloat16Tensor = Tensor
ByteTensor = Tensor
CharTensor = Tensor
ShortTensor = Tensor
IntTensor = Tensor
LongTensor = Tensor
BoolTensor = Tensor
ComplexFloatTensor = Tensor
ComplexDoubleTensor = Tensor
Size = tuple
from ._creation import tensor, zeros, ones, empty, arange, linspace, full, logspace, eye, range, randn, rand, randint, randperm, from_numpy, as_tensor, normal
from ._functional import zeros_like
from ._functional import ones_like, empty_like, full_like, randn_like, rand_like, randint_like
from ._storage import UntypedStorage, TypedStorage
from ._functional import add, mul, matmul, relu, sum, all, any, argmax, argmin, count_nonzero, masked_select, flip, roll, rot90, repeat, repeat_interleave, tile, nonzero, allclose, isclose, equal, cumsum, cumprod, cummax, argsort, sort, topk, stack, cat, concat, concatenate, hstack, vstack, row_stack, dstack, column_stack, pad_sequence, block_diag, tril, triu, diag, cartesian_prod, chunk, split, vsplit, hsplit, dsplit, unbind, tril_indices, triu_indices, take, take_along_dim, index_select, gather, scatter, abs, neg, exp, log, sqrt, div, true_divide, mean, std
from ._functional import sin, cos, tan, tanh, sigmoid, floor, ceil, round, trunc, frac
from ._functional import pow, log2, log10, exp2, rsqrt
from ._functional import sign, signbit, isnan, isinf, isfinite
from ._functional import sinh, cosh, asinh, acosh, atanh, erf, erfc, softplus
from ._functional import clamp, clamp_min, clamp_max, relu6, hardtanh
from ._functional import add_, mul_, relu_, zero_, clamp_, copy_
from ._functional import min, max, amin, amax, fmin, fmax, where
from ._functional import atan, atan2, asin, acos, lerp, addcmul, addcdiv
from ._functional import reshape, transpose, view_as_real, view_as_complex
from ._functional import logaddexp, logaddexp2, hypot, remainder, fmod
from ._functional import squeeze, unsqueeze, permute
from ._functional import var, var_mean, norm, prod
from ._functional import reciprocal, addmm, einsum
from ._functional import mm, bmm
from ._functional import floor_divide
from ._functional import narrow, flatten
from ._functional import logical_and, logical_or, logical_not
from ._functional import sub, log1p, expm1, maximum, minimum
from ._functional import dot, outer, inner, mv, cross, tensordot
from ._functional import logical_xor
from ._functional import baddbmm, trace, cummin, logsumexp, renorm
from ._functional import bitwise_and, bitwise_or, bitwise_xor, bitwise_not
from ._functional import unflatten, broadcast_to, movedim, moveaxis, diagonal
from ._functional import unique, searchsorted, kthvalue, median
# Category A: Export existing functions
from ._functional import eq, ne, lt, le, gt, ge
from ._functional import select, expand, masked_fill, unfold
from ._functional import sum_to_size
from ._functional import slice, slice_copy, slice_scatter, expand_copy
from ._functional import as_strided_, as_strided_copy, as_strided_scatter
from ._functional import scatter_, scatter_add_, scatter_reduce
from ._functional import index_add_, index_copy_, index_fill_
from ._functional import index_put, index_put_
from ._functional import masked_fill_, masked_scatter_
# Category B: Wrapper + export
from ._functional import nansum, nanmean, det, dist, matrix_power, argwhere
# Category C1: Pure-Python functions
from ._functional import meshgrid, atleast_1d, atleast_2d, atleast_3d
from ._functional import broadcast_tensors, broadcast_shapes
from ._functional import complex, polar
# Category C2: Dispatch-based functions
from ._functional import diff, bincount, cdist, aminmax
from ._functional import quantile, nanquantile, nanmedian
from ._functional import histc, histogram, bucketize
from ._functional import isneginf, isposinf, isreal, isin, heaviside
# P0 dtype utilities & query functions
from ._functional import is_tensor, is_floating_point, is_complex, numel, square
from ._functional import clone, detach, contiguous
from ._functional import index_add, index_copy, index_fill, scatter_add
from ._functional import tensor_split, split_with_sizes
from ._functional import hann_window, hamming_window, bartlett_window, blackman_window
# Aliases matching torch top-level names
absolute = abs
arccos = acos
arccosh = acosh
arcsin = asin
arcsinh = asinh
arctan = atan
arctan2 = atan2
arctanh = atanh
clip = clamp
clip_ = clamp_
divide = div
multiply = mul
subtract = sub
negative = neg
greater = gt
greater_equal = ge
less = lt
less_equal = le
not_equal = ne
swapaxes = transpose
swapdims = transpose
fix = trunc
def msort(input):
    """Sort along dim 0, returning values only (no indices)."""
    return sort(input, dim=0)[0]
vdot = dot
ger = outer


def t(input):
    """2-D transpose (alias for transpose(input, 0, 1))."""
    if input.ndim != 2:
        raise RuntimeError(f"t() expects a 2-D tensor, got {input.ndim}-D")
    return transpose(input, 0, 1)


def fliplr(input):
    """Flip tensor left-right (along dim 1)."""
    if input.ndim < 2:
        raise RuntimeError("fliplr requires at least 2-D input")
    return flip(input, [1])


def flipud(input):
    """Flip tensor upside-down (along dim 0)."""
    return flip(input, [0])


def std_mean(input, dim=None, *, unbiased=True, keepdim=False):
    """Return (std, mean) tuple."""
    s = std(input, dim=dim, unbiased=unbiased, keepdim=keepdim)
    m = mean(input, dim=dim, keepdim=keepdim)
    return s, m


def rsub(input, other, alpha=1):
    """Subtract input from other: other - alpha * input."""
    return sub(other, input if alpha == 1 else mul(input, alpha))


def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    """Replace NaN/inf/-inf with finite values."""
    import builtins as _b
    out = where(isnan(input), full(input.shape, nan, dtype=input.dtype, device=input.device), input)
    if posinf is not None:
        out = where(isposinf(out), full(input.shape, posinf, dtype=input.dtype, device=input.device), out)
    if neginf is not None:
        out = where(isneginf(out), full(input.shape, neginf, dtype=input.dtype, device=input.device), out)
    return out


def nan_to_num_(input, nan=0.0, posinf=None, neginf=None):
    """In-place nan_to_num."""
    result = nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf)
    copy_(input, result)
    return input


def diag_embed(input, offset=0, dim1=-2, dim2=-1):
    """Embed input as diagonals of a new tensor."""
    import builtins as _b
    n = input.shape[-1]
    out_size = n + _b.abs(offset)
    shape = list(input.shape[:-1]) + [out_size, out_size]
    out = zeros(shape, dtype=input.dtype, device=input.device)
    row = _b.max(0, -offset)
    col = _b.max(0, offset)
    for i in _b.range(n):
        out[..., row + i, col + i] = input[..., i]
    return out


def diagflat(input, offset=0):
    """Flatten input and create a 2-D diagonal matrix."""
    return diag(flatten(input), offset)


# Inplace unary aliases (delegate to Tensor method)
def abs_(input): return input.abs_()
def acos_(input): return input.acos_() if hasattr(input, 'acos_') else copy_(input, acos(input))
def acosh_(input): return copy_(input, acosh(input))
def asin_(input): return copy_(input, asin(input))
def asinh_(input): return copy_(input, asinh(input))
def atan_(input): return copy_(input, atan(input))
def atanh_(input): return copy_(input, atanh(input))
def ceil_(input): return input.ceil_()
def cos_(input): return input.cos_()
def cosh_(input): return copy_(input, cosh(input))
def erf_(input): return copy_(input, erf(input))
def erfc_(input): return copy_(input, erfc(input))
def exp_(input): return input.exp_()
def exp2_(input): return copy_(input, exp2(input))
def expm1_(input): return copy_(input, expm1(input))
def floor_(input): return input.floor_()
def frac_(input): return copy_(input, frac(input))
def log_(input): return input.log_()
def log2_(input): return input.log2_()
def log10_(input): return input.log10_()
def log1p_(input): return copy_(input, log1p(input))
def neg_(input): return input.neg_()
negative_ = neg_
def reciprocal_(input): return input.reciprocal_()
def round_(input): return input.round_()
def rsqrt_(input): return copy_(input, rsqrt(input))
def sigmoid_(input): return input.sigmoid_()
def sin_(input): return input.sin_()
def sinh_(input): return copy_(input, sinh(input))
def sqrt_(input): return input.sqrt_()
def square_(input): return copy_(input, square(input))
def tan_(input): return input.tan_()
def tanh_(input): return input.tanh_()
def trunc_(input): return input.trunc_()
fix_ = trunc_
def clamp_min_(input, min): return copy_(input, clamp(input, min_val=min))
def clamp_max_(input, max): return copy_(input, clamp(input, max_val=max))
def detach_(input): return input.detach_()
arccos_ = acos_
arccosh_ = acosh_
arcsin_ = asin_
arcsinh_ = asinh_
arctan_ = atan_
arctanh_ = atanh_

# NN functional ops exposed at top level
from .nn.functional import (
    softmax, log_softmax, dropout, embedding,
    layer_norm, group_norm, instance_norm, batch_norm,
    hardshrink, selu, celu, threshold,
)
from ._functional import rms_norm

def selu_(input): return copy_(input, selu(input))
def celu_(input, alpha=1.0): return copy_(input, celu(input, alpha=alpha))
def threshold_(input, threshold_val, value): return copy_(input, threshold(input, threshold_val, value))

# as_strided (view op)
def as_strided(input, size, stride, storage_offset=None):
    from ._functional import dispatch
    return dispatch("as_strided", input.device.type, input, size, stride, storage_offset)

# erfinv top-level
def erfinv(input):
    out = clone(input)
    return out.erfinv_()

# masked_scatter top-level (out-of-place)
def masked_scatter(input, mask, source):
    from ._functional import dispatch
    return dispatch("masked_scatter", input.device.type, input, mask, source)

# bitwise shifts
def bitwise_left_shift(input, other):
    from ._functional import dispatch
    return dispatch("bitwise_left_shift", input.device.type, input, other)

def bitwise_right_shift(input, other):
    from ._functional import dispatch
    return dispatch("bitwise_right_shift", input.device.type, input, other)

# constant_pad_nd
def constant_pad_nd(input, pad, value=0):
    from ._functional import dispatch
    return dispatch("constant_pad_nd", input.device.type, input, pad, value)

# mode: returns (values, indices) namedtuple-like
def mode(input, dim=-1, keepdim=False):
    from ._functional import dispatch
    return dispatch("mode", input.device.type, input, dim, keepdim)

# conv ops
from .nn.functional import conv1d, conv2d, conv3d

# dropout_ (in-place)
def dropout_(input, p=0.5, training=True):
    result = dropout(input, p=p, training=training)
    copy_(input, result)
    return input

# embedding alias (already imported above)

# inverse / solve / svd / qr — linalg wrappers
def inverse(input):
    return linalg.inv(input)

def solve(input, other):
    return linalg.solve(other, input)

def svd(input, some=True, compute_uv=True):
    return linalg.svd(input, full_matrices=not some)

def qr(input, some=True):
    return linalg.qr(input)

from . import linalg

# binomial
def binomial(count, prob, *, generator=None):
    from ._functional import dispatch
    return dispatch("binomial", count.device.type, count, prob)

# cond (matrix condition number)
def cond(input, p=None):
    return linalg.cond(input, p)


from ._printing import set_printoptions, get_printoptions
from ._dispatch import (
    pipeline_context,
    functionalize_context,
    set_pipeline_config,
    get_pipeline_config,
)
from ._backends import cpu
from .autograd.grad_mode import is_grad_enabled, set_grad_enabled, no_grad, enable_grad, inference_mode
from . import autograd
from ._backends import autograd as _autograd_kernels
from . import backends
from . import cuda
from . import npu
from . import mps
from . import _C
from . import distributed
from . import onnx
from . import futures
from . import amp
from . import compiler
from . import _dynamo
from . import quasirandom
from .ops import ops
from . import library
from . import optim
from . import nn
from . import jit
from . import fx
from . import profiler
from . import multiprocessing
from . import linalg
from . import fft
from . import special
from . import testing
from ._random import (
    manual_seed, seed, initial_seed, get_rng_state, set_rng_state,
    Generator, default_generator,
    bernoulli, multinomial, poisson,
    fork_rng,
)
from . import _random as random
from .serialization import save, load
from .amp.state import (
    is_autocast_enabled,
    set_autocast_enabled,
    get_autocast_dtype,
    set_autocast_dtype,
    is_autocast_cache_enabled,
    set_autocast_cache_enabled,
)


def pipeline(**kwargs):
    return pipeline_context(**kwargs)


def pipeline_config(**kwargs):
    return set_pipeline_config(**kwargs)


def functionalize():
    return functionalize_context()


def compile(model=None, *args, **kwargs):
    backend = kwargs.get("backend", None)
    mode = kwargs.get("mode", None)
    if backend is not None or mode is not None:
        raise NotImplementedError(
            "compile backend selection is outside 0.1 NPU-first scope"
        )
    if args:
        raise NotImplementedError(
            "compile advanced options are outside 0.1 NPU-first scope"
        )
    if callable(model):
        return model

    def decorator(fn):
        return fn

    return decorator


__all__ = [
    "Device",
    "device",
    "cuda",
    "backends",
    "Tensor",
    "Size",
    "FloatTensor", "DoubleTensor", "HalfTensor", "BFloat16Tensor",
    "ByteTensor", "CharTensor", "ShortTensor", "IntTensor", "LongTensor",
    "BoolTensor", "ComplexFloatTensor", "ComplexDoubleTensor",
    "DType",
    "dtype",
    "Dtype",
    # constants
    "inf",
    "nan",
    "e",
    "pi",
    # default dtype
    "set_default_dtype",
    "get_default_dtype",
    "vitals_enabled",
    "set_vital",
    "read_vitals",
    # quantization stubs
    "quantize_per_tensor",
    # dtypes
    "float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu",
    "float16", "float32", "float64", "bfloat16",
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "bool",
    "complex32", "complex64", "complex128",
    "quint8", "qint8", "qint32", "quint4x2",
    # dtype aliases
    "half", "float", "double",
    "short", "int", "long", "byte",
    "cfloat", "cdouble",
    # creation
    "tensor",
    "zeros",
    "ones",
    "empty",
    "randn",
    "rand",
    "arange",
    "linspace",
    "full",
    "logspace",
    "eye",
    "range",
    "zeros_like",
    "ones_like",
    "empty_like",
    "full_like",
    "randn_like",
    "rand_like",
    "randint_like",
    # ops
    "add",
    "mul",
    "matmul",
    "relu",
    "abs",
    "neg",
    "exp",
    "log",
    "sqrt",
    "all",
    "any",
    "argmax",
    "argmin",
    "count_nonzero",
    "masked_select",
    "flip",
    "roll",
    "rot90",
    "repeat",
    "repeat_interleave",
    "tile",
    "nonzero",
    "cumsum",
    "cumprod",
    "cummax",
    "argsort",
    "sort",
    "topk",
    "stack",
    "cat",
    "concat",
    "concatenate",
    "hstack",
    "vstack",
    "row_stack",
    "dstack",
    "column_stack",
    "pad_sequence",
    "block_diag",
    "tril",
    "triu",
    "diag",
    "cartesian_prod",
    "chunk",
    "split",
    "vsplit",
    "hsplit",
    "dsplit",
    "unbind",
    "tril_indices",
    "triu_indices",
    "take",
    "take_along_dim",
    "index_select",
    "gather",
    "scatter",
    "allclose",
    "isclose",
    "equal",
    "sin",
    "cos",
    "tan",
    "tanh",
    "sigmoid",
    "floor",
    "ceil",
    "round",
    "trunc",
    "frac",
    "pow",
    "log2",
    "log10",
    "exp2",
    "rsqrt",
    "sign",
    "signbit",
    "isnan",
    "isinf",
    "isfinite",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "erfc",
    "softplus",
    "clamp",
    "clamp_min",
    "clamp_max",
    "relu6",
    "hardtanh",
    "add_",
    "mul_",
    "relu_",
    "zero_",
    "clamp_",
    "copy_",
    "min",
    "max",
    "amin",
    "amax",
    "fmin",
    "fmax",
    "where",
    "atan",
    "atan2",
    "asin",
    "acos",
    "lerp",
    "addcmul",
    "addcdiv",
    "logaddexp",
    "logaddexp2",
    "hypot",
    "remainder",
    "fmod",
    "sum",
    "std",
    "reshape",
    "transpose",
    "view_as_real",
    "view_as_complex",
    "squeeze",
    "unsqueeze",
    "permute",
    "var",
    "var_mean",
    "norm",
    "prod",
    "mm",
    "bmm",
    "floor_divide",
    # P1 ops
    "reciprocal",
    "addmm",
    "einsum",
    "randint",
    "randperm",
    "from_numpy",
    "as_tensor",
    # Batch 1 ops
    "narrow",
    "flatten",
    "logical_and",
    "logical_or",
    "logical_not",
    # printing
    "set_printoptions",
    "get_printoptions",
    # pipeline
    "pipeline",
    "pipeline_context",
    "pipeline_config",
    "get_pipeline_config",
    "functionalize",
    "functionalize_context",
    # device
    "get_default_device",
    "set_default_device",
    "npu",
    # autograd
    "autograd",
    "is_grad_enabled",
    "set_grad_enabled",
    "no_grad",
    "enable_grad",
    "inference_mode",
    # distributed
    "distributed",
    "multiprocessing",
    "onnx",
    # amp
    "amp",
    "is_autocast_enabled",
    "set_autocast_enabled",
    "get_autocast_dtype",
    "set_autocast_dtype",
    "is_autocast_cache_enabled",
    "set_autocast_cache_enabled",
    "ops",
    "library",
    "compiler",
    "_dynamo",
    "quasirandom",
    "optim",
    "nn",
    "jit",
    "profiler",
    "compile",
    "save",
    "load",
    # new creation ops
    "randint",
    "randperm",
    # new math ops
    "sub",
    "log1p",
    "expm1",
    "reciprocal",
    "maximum",
    "minimum",
    "dot",
    "outer",
    "inner",
    "mv",
    "cross",
    "tensordot",
    "einsum",
    # new logical ops
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    # new bitwise ops
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    # new shape ops
    "flatten",
    "unflatten",
    "broadcast_to",
    "movedim",
    "moveaxis",
    "diagonal",
    # new search ops
    "unique",
    "searchsorted",
    "kthvalue",
    "median",
    # P1 new ops
    "baddbmm",
    "trace",
    "cummin",
    "logsumexp",
    "renorm",
    # new random ops
    "bernoulli",
    "multinomial",
    # Category A: Comparison ops
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    # Category A: Indexing/mutation ops
    "select",
    "expand",
    "masked_fill",
    "masked_fill_",
    "unfold",
    "scatter_",
    "scatter_add_",
    "scatter_reduce",
    "index_add_",
    "index_copy_",
    "index_fill_",
    "index_put",
    "index_put_",
    "masked_scatter_",
    # Category B: Wrapper ops
    "nansum",
    "nanmean",
    "det",
    "dist",
    "matrix_power",
    "argwhere",
    # Category C1: Pure-Python ops
    "meshgrid",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_tensors",
    "broadcast_shapes",
    "complex",
    "polar",
    # Category C2: Dispatch-based ops
    "diff",
    "bincount",
    "cdist",
    "aminmax",
    "quantile",
    "nanquantile",
    "nanmedian",
    "histc",
    "histogram",
    "bucketize",
    "isneginf",
    "isposinf",
    "isreal",
    "isin",
    "heaviside",
    # P0 dtype utilities & query functions
    "finfo",
    "iinfo",
    "is_tensor",
    "is_floating_point",
    "is_complex",
    "numel",
    "square",
    # submodules
    "linalg",
    "fft",
    "special",
    "testing",
    "fx",
    "strided",
    "sparse_coo",
    "sparse_csr",
    "sparse_csc",
    "sparse_bsr",
    "sparse_bsc",
]


strided = "strided"


def sparse_coo(*args, **kwargs):  # noqa: ARG001
    raise NotImplementedError("sparse_coo is not implemented in candle")


def sparse_csr(*args, **kwargs):  # noqa: ARG001
    raise NotImplementedError("sparse_csr is not implemented in candle")


def sparse_csc(*args, **kwargs):  # noqa: ARG001
    raise NotImplementedError("sparse_csc is not implemented in candle")


def sparse_bsr(*args, **kwargs):  # noqa: ARG001
    raise NotImplementedError("sparse_bsr is not implemented in candle")


def sparse_bsc(*args, **kwargs):  # noqa: ARG001
    raise NotImplementedError("sparse_bsc is not implemented in candle")


def sparse_coo_tensor(indices, values, size=None, *, dtype=None, device=None, requires_grad=False):
    idx = indices._numpy_view()
    vals = values._numpy_view()
    if size is None:
        if idx.ndim == 0:
            inferred_size = ()
        elif idx.size == 0:
            inferred_size = (0,) if idx.ndim == 1 else tuple(0 for _ in range(idx.shape[0]))
        elif idx.ndim == 1:
            inferred_size = (_builtins.int(idx.max()) + 1,)
        else:
            inferred_size = tuple(_builtins.int(idx[dim].max()) + 1 for dim in _builtins.range(idx.shape[0]))
    else:
        inferred_size = tuple(size)

    dense = zeros(*inferred_size, dtype=dtype or values.dtype, device=device or values.device)
    dense_arr = dense._numpy_view()
    if idx.ndim <= 1:
        flat_idx = idx.reshape(-1)
        flat_vals = vals.reshape(-1)
        for i, v in zip(flat_idx, flat_vals):
            dense_arr[_builtins.int(i)] += v
    else:
        for coords, v in zip(idx.T, vals.reshape(-1)):
            dense_arr[tuple(_builtins.int(c) for c in coords)] += v
    dense._is_sparse = True
    dense.layout = sparse_coo
    if requires_grad:
        dense.requires_grad_(True)
    return dense
