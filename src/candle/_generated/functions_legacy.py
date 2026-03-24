"""Legacy backward Node classes — hand-maintained.

These classes are NOT generated from derivatives.yaml.
"""
from __future__ import annotations

import math as _math

from ..autograd.node import Node
from .._backends.autograd import (
    _grad_context,
    _strip_autograd_keys,
    _backward_dispatch_keyset,
    _scalar_tensor_like,
)
from .._dispatch.dispatcher import redispatch
from ..autograd.utils import reduce_grad


def _inverse_permutation(dims):
    inv = [0] * len(dims)
    for i, d in enumerate(dims):
        inv[d] = i
    return inv


# ---------------------------------------------------------------------------
# Activation backward helpers (called from generated formulas)
# ---------------------------------------------------------------------------

def _sqrt_backward_helper(grad, result, keyset):
    two = _scalar_tensor_like(result, 2.0)
    return redispatch("div", keyset, grad, redispatch("mul", keyset, two, result))


def _rsqrt_backward_helper(grad, result, keyset):
    half = _scalar_tensor_like(result, -0.5)
    return redispatch("mul", keyset, redispatch("mul", keyset, half, grad), redispatch("pow", keyset, result, 3))


def _exp2_backward_helper(grad, result, keyset):
    ln2 = _scalar_tensor_like(result, _math.log(2.0))
    return redispatch("mul", keyset, redispatch("mul", keyset, grad, result), ln2)


def _sigmoid_backward_helper(grad, result, keyset):
    ones = result._ones_like()
    return redispatch("mul", keyset, grad, redispatch("mul", keyset, result, redispatch("add", keyset, ones, redispatch("neg", keyset, result))))


def _tanh_backward_helper(grad, result, keyset):
    ones = result._ones_like()
    return redispatch("mul", keyset, grad, redispatch("add", keyset, ones, redispatch("neg", keyset, redispatch("mul", keyset, result, result))))


def _threshold_backward_helper(grad, result, threshold, keyset):
    ones = result._ones_like()
    zero = _scalar_tensor_like(result, 0.0)
    mask = redispatch("where", keyset, redispatch("gt", keyset, result, threshold), ones, zero)
    return redispatch("mul", keyset, grad, mask)


def _softplus_backward_helper(grad, input_, beta, threshold, keyset):
    del threshold
    beta_t = _scalar_tensor_like(input_, float(beta))
    return redispatch("mul", keyset, grad, redispatch("sigmoid", keyset, redispatch("mul", keyset, input_, beta_t)))


def _mul_tensor_backward_helper(grad, other, dtype, keyset):
    del dtype
    return redispatch("mul", keyset, grad, other)


def _pow_backward_helper(grad, self_, exponent, keyset):
    """grad * exponent * self**(exponent-1)"""
    if hasattr(exponent, 'shape'):  # tensor exponent
        return redispatch("mul", keyset, grad,
               redispatch("mul", keyset, exponent,
               redispatch("pow", keyset, self_, redispatch("sub", keyset, exponent, 1))))
    exp = float(exponent)
    return redispatch("mul", keyset, grad,
           redispatch("mul", keyset,
           redispatch("pow", keyset, self_, exp - 1), exp))


def _pow_backward_self_helper(grad, self_, exponent, keyset):
    """Backward for pow.Tensor_Tensor w.r.t. self."""
    return redispatch("mul", keyset, grad,
           redispatch("mul", keyset, exponent,
           redispatch("pow", keyset, self_, redispatch("sub", keyset, exponent, 1))))


def _pow_backward_exponent_helper(grad, self_, exponent, result, keyset):
    """Backward for pow w.r.t. exponent: grad * result * log(self)"""
    return redispatch("mul", keyset, grad,
           redispatch("mul", keyset, result,
           redispatch("log", keyset, self_)))


def _div_tensor_self_backward_helper(grad, other, dtype, *extra_and_keyset):
    del dtype
    keyset = extra_and_keyset[-1]
    return redispatch("div", keyset, grad, other)


def _div_tensor_other_backward_helper(grad, self_, other, *extra_and_keyset):
    keyset = extra_and_keyset[-1]
    if hasattr(other, 'shape'):
        denom = redispatch("mul", keyset, other, other)
    else:
        denom = other * other
    num = redispatch("mul", keyset, grad, self_)
    return redispatch("neg", keyset, redispatch("div", keyset, num, denom))


def _matmul_backward_helper(grad, self_, other, grad_input_mask, keyset):
    grad_self = reduce_grad(redispatch("matmul", keyset, grad, redispatch("transpose", keyset, other, -1, -2)), self_.shape) if grad_input_mask[0] else None
    grad_other = reduce_grad(redispatch("matmul", keyset, redispatch("transpose", keyset, self_, -1, -2), grad), other.shape) if grad_input_mask[1] else None
    return grad_self, grad_other


def _gelu_backward_helper(grad, self_, approximate, keyset):
    del approximate
    return _gelu_grad(grad, self_, keyset)


def _unsqueeze_to_backward_helper(grad, dim, input_sizes, keyset):
    del dim
    return redispatch("reshape", keyset, grad, input_sizes)


def _sum_to_backward_helper(grad, size, keyset):
    target_shape = tuple(size)
    if tuple(grad.shape) == target_shape:
        return grad
    result = grad
    # Reduce extra leading dimensions
    n_extra = len(result.shape) - len(target_shape)
    if n_extra > 0:
        dims = tuple(range(n_extra))
        result = redispatch("sum", keyset, result, dim=dims, keepdim=False)
    # Reduce broadcast dimensions (size-1 dims)
    for i, (g_dim, s_dim) in enumerate(zip(result.shape, target_shape)):
        if s_dim == 1 and g_dim != 1:
            result = redispatch("sum", keyset, result, dim=i, keepdim=True)
    if tuple(result.shape) != target_shape:
        result = redispatch("reshape", keyset, result, target_shape)
    return result


def _permute_backward_helper(grad, dims, keyset):
    inv = [0] * len(dims)
    for i, d in enumerate(dims):
        inv[d] = i
    return redispatch("permute", keyset, grad, inv)


def _select_backward_symint_helper(grad, input_sizes, dim, index, keyset):
    del dim, index
    return redispatch("reshape", keyset, grad, input_sizes)


def _cat_tensors_backward_helper(grad, *args):
    keyset = args[-1]
    del keyset
    # Placeholder: preserve runtime by returning the upstream grad for tensor-list cat cases.
    return grad


def _clamp_backward_helper(grad, self_, min_val, max_val, keyset):
    ones = self_._ones_like()
    zero = _scalar_tensor_like(self_, 0.0)
    mask = ones
    if min_val is not None:
        mask = redispatch("mul", keyset, mask, redispatch("where", keyset, redispatch("ge", keyset, self_, min_val), ones, zero))
    if max_val is not None:
        mask = redispatch("mul", keyset, mask, redispatch("where", keyset, redispatch("le", keyset, self_, max_val), ones, zero))
    return redispatch("mul", keyset, grad, mask)


    one = _scalar_tensor_like(self_, 1.0)
    denom = redispatch("add", keyset, one, redispatch("abs", keyset, self_))
    denom_sq = redispatch("mul", keyset, denom, denom)
    return redispatch("div", keyset, grad, denom_sq)


def _silu_grad(grad, self_, keyset):
    sig = redispatch("sigmoid", keyset, self_)
    ones = self_._ones_like()
    one_minus_sig = redispatch("add", keyset, ones, redispatch("neg", keyset, sig))
    x_mul = redispatch("mul", keyset, self_, one_minus_sig)
    factor = redispatch("mul", keyset, sig, redispatch("add", keyset, ones, x_mul))
    return redispatch("mul", keyset, grad, factor)


def _gelu_grad(grad, self_, keyset):
    sqrt2 = _scalar_tensor_like(self_, _math.sqrt(2.0))
    x_over_sqrt2 = redispatch("div", keyset, self_, sqrt2)
    erf_val = redispatch("erf", keyset, x_over_sqrt2)
    ones = self_._ones_like()
    half = _scalar_tensor_like(self_, 0.5)
    cdf = redispatch("mul", keyset, half, redispatch("add", keyset, ones, erf_val))
    coeff = _scalar_tensor_like(self_, 1.0 / _math.sqrt(2.0 * _math.pi))
    x_sq = redispatch("mul", keyset, self_, self_)
    neg_half_x_sq = redispatch("mul", keyset, _scalar_tensor_like(self_, -0.5), x_sq)
    pdf = redispatch("mul", keyset, coeff, redispatch("exp", keyset, neg_half_x_sq))
    x_pdf = redispatch("mul", keyset, self_, pdf)
    factor = redispatch("add", keyset, cdf, x_pdf)
    return redispatch("mul", keyset, grad, factor)


def _mish_grad(grad, self_, keyset):
    ones = self_._ones_like()
    sp = redispatch("softplus", keyset, self_)
    tanh_sp = redispatch("tanh", keyset, sp)
    tanh_sp_sq = redispatch("mul", keyset, tanh_sp, tanh_sp)
    sech2 = redispatch("add", keyset, ones, redispatch("neg", keyset, tanh_sp_sq))
    sig = redispatch("sigmoid", keyset, self_)
    tail = redispatch("mul", keyset, self_, redispatch("mul", keyset, sech2, sig))
    factor = redispatch("add", keyset, tanh_sp, tail)
    return redispatch("mul", keyset, grad, factor)


def _leaky_relu_grad(grad, self_, negative_slope, keyset):
    pos_mask = redispatch("sign", keyset, redispatch("relu", keyset, self_))
    ones = self_._ones_like()
    nonpos_mask = redispatch("add", keyset, ones, redispatch("neg", keyset, pos_mask))
    slope = _scalar_tensor_like(self_, negative_slope)
    factor = redispatch("add", keyset, pos_mask, redispatch("mul", keyset, nonpos_mask, slope))
    return redispatch("mul", keyset, grad, factor)


def _elu_grad(grad, self_, alpha, keyset):
    pos_mask = redispatch("sign", keyset, redispatch("relu", keyset, self_))
    ones = self_._ones_like()
    nonpos_mask = redispatch("add", keyset, ones, redispatch("neg", keyset, pos_mask))
    alpha_t = _scalar_tensor_like(self_, float(alpha))
    exp_x = redispatch("exp", keyset, self_)
    neg_branch = redispatch("mul", keyset, alpha_t, exp_x)
    factor = redispatch("add", keyset, pos_mask, redispatch("mul", keyset, nonpos_mask, neg_branch))
    return redispatch("mul", keyset, grad, factor)


def _celu_grad(grad, self_, alpha, keyset):
    alpha_t = _scalar_tensor_like(self_, float(alpha))
    small_eps = _scalar_tensor_like(self_, 1e-7)
    ones = self_._ones_like()
    pos_mask = redispatch("sign", keyset, redispatch("relu", keyset,
        redispatch("add", keyset, self_, small_eps)))
    neg_mask = redispatch("add", keyset, ones, redispatch("neg", keyset, pos_mask))
    exp_x_alpha = redispatch("exp", keyset, redispatch("div", keyset, self_, alpha_t))
    deriv = redispatch("add", keyset, pos_mask, redispatch("mul", keyset, neg_mask, exp_x_alpha))
    return redispatch("mul", keyset, grad, deriv)


def _hardtanh_grad(grad, self_, min_val, max_val, keyset):
    min_t = _scalar_tensor_like(self_, float(min_val))
    max_t = _scalar_tensor_like(self_, float(max_val))
    ge_min = redispatch("ge", keyset, self_, min_t)
    le_max = redispatch("le", keyset, self_, max_t)
    mask = redispatch("mul", keyset,
        redispatch("where", keyset, ge_min, self_._ones_like(), _scalar_tensor_like(self_, 0.0)),
        redispatch("where", keyset, le_max, self_._ones_like(), _scalar_tensor_like(self_, 0.0)))
    return redispatch("mul", keyset, grad, mask)


def _hardswish_grad(grad, self_, keyset):
    three = _scalar_tensor_like(self_, 3.0)
    sixth = _scalar_tensor_like(self_, 1.0 / 6.0)
    two = _scalar_tensor_like(self_, 2.0)
    ones = self_._ones_like()
    zero = _scalar_tensor_like(self_, 0.0)
    gt_neg3 = redispatch("gt", keyset, self_, redispatch("neg", keyset, three))
    lt_3 = redispatch("lt", keyset, self_, three)
    inner_mask = redispatch("mul", keyset,
        redispatch("where", keyset, gt_neg3, ones, zero),
        redispatch("where", keyset, lt_3, ones, zero))
    ge_3_mask = redispatch("where", keyset, redispatch("ge", keyset, self_, three), ones, zero)
    two_x_plus_3 = redispatch("add", keyset, redispatch("mul", keyset, two, self_), three)
    inner_grad = redispatch("mul", keyset, two_x_plus_3, sixth)
    dout = redispatch("add", keyset,
        redispatch("mul", keyset, inner_grad, inner_mask), ge_3_mask)
    return redispatch("mul", keyset, grad, dout)


def _hardsigmoid_grad(grad, self_, keyset):
    three = _scalar_tensor_like(self_, 3.0)
    sixth = _scalar_tensor_like(self_, 1.0 / 6.0)
    ones = self_._ones_like()
    zero = _scalar_tensor_like(self_, 0.0)
    gt_neg3 = redispatch("gt", keyset, self_, redispatch("neg", keyset, three))
    lt_3 = redispatch("lt", keyset, self_, three)
    inner_mask = redispatch("mul", keyset,
        redispatch("where", keyset, gt_neg3, ones, zero),
        redispatch("where", keyset, lt_3, ones, zero))
    return redispatch("mul", keyset, grad, redispatch("mul", keyset, inner_mask, sixth))


# Aliases for backward helpers (match _SPECIAL_CALLS transpiler names)
_hardtanh_backward_helper = _hardtanh_grad
_hardswish_backward_helper = _hardswish_grad
_hardsigmoid_backward_helper = _hardsigmoid_grad


def _clamp_backward_min_max_helper(grad, self_, min_val, max_val, grad_input_mask, keyset):
    """Backward for clamp with both min and max tensor."""
    return _clamp_backward_helper(grad, self_, min_val, max_val, keyset), None, None


def _to_padded_tensor_backward_helper(grad, self_):
    raise NotImplementedError("to_padded_tensor backward is not supported for nested tensors")


def _selu_grad(grad, self_, keyset):
    SCALE = 1.0507009873554804934193349852946
    ALPHA = 1.6732631921893986195596513061800
    scale_t = _scalar_tensor_like(self_, SCALE)
    alpha_scale_t = _scalar_tensor_like(self_, SCALE * ALPHA)
    small_eps = _scalar_tensor_like(self_, 1e-7)
    ones = self_._ones_like()
    pos_mask = redispatch("sign", keyset, redispatch("relu", keyset,
        redispatch("add", keyset, self_, small_eps)))
    neg_mask = redispatch("add", keyset, ones, redispatch("neg", keyset, pos_mask))
    exp_x = redispatch("exp", keyset, self_)
    neg_deriv = redispatch("mul", keyset, alpha_scale_t, exp_x)
    deriv = redispatch("add", keyset,
        redispatch("mul", keyset, pos_mask, scale_t),
        redispatch("mul", keyset, neg_mask, neg_deriv))
    return redispatch("mul", keyset, grad, deriv)


def _softmax_grad(grad, self_, dim, keyset):
    s = redispatch("softmax", keyset, self_, dim)
    gs = redispatch("mul", keyset, grad, s)
    gs_sum = redispatch("sum", keyset, gs, dim=dim, keepdim=True)
    return redispatch("mul", keyset, s, redispatch("add", keyset, grad,
        redispatch("neg", keyset, gs_sum)))


def _log_softmax_grad(grad, self_, dim, keyset):
    log_s = redispatch("log_softmax", keyset, self_, dim)
    s = redispatch("exp", keyset, log_s)
    grad_sum = redispatch("sum", keyset, grad, dim=dim, keepdim=True)
    return redispatch("add", keyset, grad,
        redispatch("neg", keyset, redispatch("mul", keyset, s, grad_sum)))


def _prelu_grad_input(grad, self_, weight, keyset):
    pos_mask = redispatch("sign", keyset, redispatch("relu", keyset, self_))
    ones = self_._ones_like()
    nonpos_mask = redispatch("add", keyset, ones, redispatch("neg", keyset, pos_mask))
    factor = redispatch("add", keyset, pos_mask, redispatch("mul", keyset, nonpos_mask, weight))
    return redispatch("mul", keyset, grad, factor)


def _prelu_grad_weight(grad, self_, weight, keyset):
    pos_mask = redispatch("sign", keyset, redispatch("relu", keyset, self_))
    ones = self_._ones_like()
    nonpos_mask = redispatch("add", keyset, ones, redispatch("neg", keyset, pos_mask))
    w_input = redispatch("mul", keyset, nonpos_mask, self_)
    return redispatch("mul", keyset, grad, w_input)


# ---------------------------------------------------------------------------
# Norm backward helpers (called from generated formulas)
# ---------------------------------------------------------------------------

def _norm_stats(x, axes, eps, keyset):
    """Shared: compute mean, diff, inv_std, x_hat for norm backward."""
    mean = redispatch("mean", keyset, x, dim=axes, keepdim=True)
    diff = redispatch("add", keyset, x, redispatch("neg", keyset, mean))
    var = redispatch("mean", keyset, redispatch("mul", keyset, diff, diff), dim=axes, keepdim=True)
    eps_t = _scalar_tensor_like(x, eps)
    inv_std = redispatch("rsqrt", keyset, redispatch("add", keyset, var, eps_t))
    x_hat = redispatch("mul", keyset, diff, inv_std)
    return inv_std, x_hat


def _norm_grad_core(dl_dxhat, x_hat, inv_std, axes, n, keyset):
    """Shared: compute grad_input from dl_dxhat, x_hat, inv_std."""
    n_t = _scalar_tensor_like(x_hat, float(n))
    mean_dl = redispatch("div", keyset,
        redispatch("sum", keyset, dl_dxhat, dim=axes, keepdim=True), n_t)
    mean_dl_xhat = redispatch("div", keyset,
        redispatch("sum", keyset, redispatch("mul", keyset, dl_dxhat, x_hat),
                   dim=axes, keepdim=True), n_t)
    return redispatch("mul", keyset, inv_std,
        redispatch("add", keyset,
            redispatch("add", keyset, dl_dxhat, redispatch("neg", keyset, mean_dl)),
            redispatch("neg", keyset, redispatch("mul", keyset, x_hat, mean_dl_xhat))))


def _layer_norm_grad_input(grad, input_, normalized_shape, weight, eps, keyset):
    norm_shape = tuple(normalized_shape)
    ndim = len(input_.shape)
    n_norm = len(norm_shape)
    axes = tuple(range(ndim - n_norm, ndim))
    inv_std, x_hat = _norm_stats(input_, axes, eps, keyset)
    dl_dxhat = redispatch("mul", keyset, grad, weight) if weight is not None else grad
    n = 1
    for d in axes:
        n *= input_.shape[d]
    return _norm_grad_core(dl_dxhat, x_hat, inv_std, axes, n, keyset)


def _layer_norm_backward_all(grad, input_, normalized_shape, weight, bias, eps, keyset):
    norm_shape = tuple(normalized_shape)
    ndim = len(input_.shape)
    n_norm = len(norm_shape)
    axes = tuple(range(ndim - n_norm, ndim))
    inv_std, x_hat = _norm_stats(input_, axes, eps, keyset)
    dl_dxhat = redispatch("mul", keyset, grad, weight) if weight is not None else grad
    n = 1
    for d in axes:
        n *= input_.shape[d]
    grad_input = _norm_grad_core(dl_dxhat, x_hat, inv_std, axes, n, keyset)
    grad_weight = None
    if weight is not None:
        batch_dims = tuple(range(ndim - n_norm))
        grad_weight = redispatch("sum", keyset, redispatch("mul", keyset, grad, x_hat), dim=batch_dims)
    grad_bias = None
    if bias is not None:
        batch_dims = tuple(range(ndim - n_norm))
        grad_bias = redispatch("sum", keyset, grad, dim=batch_dims)
    return grad_input, grad_weight, grad_bias


def _batch_norm_grad_input(grad, input_, weight, eps, keyset):
    ndim = len(input_.shape)
    axes = (0,) + tuple(range(2, ndim))
    shape_for_stats = [1, input_.shape[1]] + [1] * (ndim - 2)
    inv_std, x_hat = _norm_stats(input_, axes, eps, keyset)
    if weight is not None:
        dl_dxhat = redispatch("mul", keyset, grad, weight.reshape(shape_for_stats))
    else:
        dl_dxhat = grad
    n = 1
    for ax in axes:
        n *= input_.shape[ax]
    return _norm_grad_core(dl_dxhat, x_hat, inv_std, axes, n, keyset)


def _group_norm_grad_input(grad, input_, num_groups, weight, eps, keyset):
    N = input_.shape[0]
    C = input_.shape[1]
    spatial = input_.shape[2:]
    channels_per_group = C // num_groups
    group_size = channels_per_group
    for s in spatial:
        group_size *= s
    reshaped = redispatch("reshape", keyset, input_,
                          (N, num_groups, channels_per_group, *spatial))
    axes = tuple(range(2, len(reshaped.shape)))
    inv_std, x_hat = _norm_stats(reshaped, axes, eps, keyset)
    grad_reshaped = redispatch("reshape", keyset, grad,
                               (N, num_groups, channels_per_group, *spatial))
    if weight is not None:
        w_shape = [1, num_groups, channels_per_group] + [1] * len(spatial)
        w_reshaped = redispatch("reshape", keyset, weight, tuple(w_shape))
        dl_dxhat = redispatch("mul", keyset, grad_reshaped, w_reshaped)
    else:
        dl_dxhat = grad_reshaped
    grad_out = _norm_grad_core(dl_dxhat, x_hat, inv_std, axes, group_size, keyset)
    return redispatch("reshape", keyset, grad_out, input_.shape)


def _rms_norm_grad_input(grad, input_, normalized_shape, weight, eps, keyset):
    norm_shape = tuple(normalized_shape)
    ndim = len(input_.shape)
    n_norm = len(norm_shape)
    axes = tuple(range(ndim - n_norm, ndim))
    x_sq = redispatch("mul", keyset, input_, input_)
    variance = redispatch("mean", keyset, x_sq, dim=axes, keepdim=True)
    eps_t = _scalar_tensor_like(input_, eps)
    rms = redispatch("sqrt", keyset, redispatch("add", keyset, variance, eps_t))
    x_hat = redispatch("div", keyset, input_, rms)
    dl_dxhat = redispatch("mul", keyset, grad, weight) if weight is not None else grad
    n = 1
    for d in axes:
        n *= input_.shape[d]
    n_t = _scalar_tensor_like(input_, float(n))
    dot = redispatch("sum", keyset,
        redispatch("mul", keyset, dl_dxhat, x_hat), dim=axes, keepdim=True)
    return redispatch("div", keyset,
        redispatch("add", keyset, dl_dxhat,
            redispatch("neg", keyset, redispatch("mul", keyset, x_hat,
                redispatch("div", keyset, dot, n_t)))),
        rms)


# ---------------------------------------------------------------------------
# Conv backward helpers (Phase 2)
# ---------------------------------------------------------------------------

class _ConvGradCache:
    """Cache conv backward results to avoid recomputation across grad formulas."""
    _cache = {}

    @classmethod
    def get(cls, key, compute_fn):
        if key not in cls._cache:
            cls._cache[key] = compute_fn()
        return cls._cache[key]

    @classmethod
    def clear(cls):
        cls._cache.clear()


def _conv_backward_all(grad, input_, weight, bias, stride, padding, dilation, groups, name, keyset):
    """Compute all conv gradients, returning (grad_input, grad_weight, grad_bias).

    Delegates to the existing _conv_backward implementation.
    Always returns a 3-tuple (grad_bias is None when bias is None).
    """
    from .._backends.autograd import _conv_backward
    args = (stride, padding, dilation, groups)
    result = _conv_backward(name, grad, input_, weight, bias,
                            input_, weight, bias, keyset, args, {})
    if len(result) == 2:
        return result[0], result[1], None
    return result


# ---------------------------------------------------------------------------
# Pool backward helpers (Phase 2)
# ---------------------------------------------------------------------------

def _max_pool1d_backward_helper(grad, self_, result, kernel_size, stride, padding, dilation, ceil_mode, keyset):
    from .._backends.autograd import _max_pool1d_backward
    args = (kernel_size, stride, padding, dilation, ceil_mode)
    return _max_pool1d_backward(grad, self_, self_, result, keyset, args, {})[0]


def _max_pool2d_backward_helper(grad, self_, result, kernel_size, stride, padding, dilation, ceil_mode, keyset):
    from .._backends.autograd import _max_pool2d_backward
    args = (kernel_size, stride, padding, dilation, ceil_mode)
    return _max_pool2d_backward(grad, self_, self_, result, keyset, args, {})[0]


def _max_pool3d_backward_helper(grad, self_, result, kernel_size, stride, padding, dilation, ceil_mode, keyset):
    from .._backends.autograd import _max_pool3d_backward
    args = (kernel_size, stride, padding, dilation, ceil_mode)
    return _max_pool3d_backward(grad, self_, self_, result, keyset, args, {})[0]


def _avg_pool1d_backward_helper(grad, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, keyset):
    from .._backends.autograd import _avg_pool1d_backward
    args = (kernel_size, stride, padding, ceil_mode, count_include_pad)
    return _avg_pool1d_backward(grad, self_, self_, None, keyset, args, {})[0]


def _avg_pool2d_backward_helper(grad, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, keyset):
    from .._backends.autograd import _avg_pool2d_backward
    args = (kernel_size, stride, padding, ceil_mode, count_include_pad)
    return _avg_pool2d_backward(grad, self_, self_, None, keyset, args, {})[0]


def _avg_pool3d_backward_helper(grad, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, keyset):
    from .._backends.autograd import _avg_pool3d_backward
    args = (kernel_size, stride, padding, ceil_mode, count_include_pad)
    return _avg_pool3d_backward(grad, self_, self_, None, keyset, args, {})[0]


def _adaptive_avg_pool1d_backward_helper(grad, self_, output_size, keyset):
    from .._backends.autograd import _adaptive_avg_pool1d_backward
    args = (output_size,)
    return _adaptive_avg_pool1d_backward(grad, self_, self_, None, keyset, args, {})[0]


def _adaptive_avg_pool2d_backward_helper(grad, self_, output_size, keyset):
    from .._backends.autograd import _adaptive_avg_pool2d_backward
    args = (output_size,)
    return _adaptive_avg_pool2d_backward(grad, self_, self_, None, keyset, args, {})[0]


def _adaptive_avg_pool3d_backward_helper(grad, self_, output_size, keyset):
    from .._backends.autograd import _adaptive_avg_pool3d_backward
    args = (output_size,)
    return _adaptive_avg_pool3d_backward(grad, self_, self_, None, keyset, args, {})[0]


def _adaptive_max_pool1d_backward_helper(grad, self_, result, output_size, keyset):
    from .._backends.autograd import _adaptive_max_pool1d_backward
    args = (output_size,)
    return _adaptive_max_pool1d_backward(grad, self_, self_, result, keyset, args, {})[0]


def _adaptive_max_pool2d_backward_helper(grad, self_, result, output_size, keyset):
    from .._backends.autograd import _adaptive_max_pool2d_backward
    args = (output_size,)
    return _adaptive_max_pool2d_backward(grad, self_, self_, result, keyset, args, {})[0]


# ---------------------------------------------------------------------------
# Upsample backward helpers (Phase 2)
# ---------------------------------------------------------------------------

def _upsample_nearest1d_backward_helper(grad, self_, output_size, keyset):
    from .._backends.autograd import _upsample_nearest1d_backward
    args = (output_size,)
    return _upsample_nearest1d_backward(grad, self_, self_, None, keyset, args, {})[0]


def _upsample_nearest2d_backward_helper(grad, self_, output_size, keyset):
    from .._backends.autograd import _upsample_nearest2d_backward
    args = (output_size,)
    return _upsample_nearest2d_backward(grad, self_, self_, None, keyset, args, {})[0]


def _upsample_bilinear2d_backward_helper(grad, self_, output_size, align_corners, keyset):
    from .._backends.autograd import _upsample_bilinear2d_backward
    args = (output_size, align_corners)
    return _upsample_bilinear2d_backward(grad, self_, self_, None, keyset, args, {})[0]


def _upsample_linear1d_backward_helper(grad, self_, output_size, align_corners, keyset):
    from .._backends.autograd import _upsample_linear1d_backward
    args = (output_size, align_corners)
    return _upsample_linear1d_backward(grad, self_, self_, None, keyset, args, {})[0]


def _upsample_bicubic2d_backward_helper(grad, self_, output_size, align_corners, keyset):
    from .._backends.autograd import _upsample_bicubic2d_backward
    args = (output_size, align_corners)
    return _upsample_bicubic2d_backward(grad, self_, self_, None, keyset, args, {})[0]


# ---------------------------------------------------------------------------
# Binary backward helpers (Batch 2)
# ---------------------------------------------------------------------------

def _add_backward_all(grad, self_, other, keyset):
    grad_self = reduce_grad(grad, self_.shape) if hasattr(self_, 'shape') else None
    grad_other = reduce_grad(grad, other.shape) if hasattr(other, 'shape') else None
    return grad_self, grad_other


def _sub_backward_all(grad, self_, other, keyset):
    grad_self = reduce_grad(grad, self_.shape) if hasattr(self_, 'shape') else None
    grad_other = None
    if hasattr(other, 'shape'):
        grad_other = reduce_grad(redispatch("neg", keyset, grad), other.shape)
    return grad_self, grad_other


def _mul_backward_all(grad, self_, other, keyset):
    if hasattr(other, 'shape'):
        g = redispatch("mul", keyset, grad, other)
        grad_self = reduce_grad(g, self_.shape)
    else:
        g = redispatch("mul", keyset, grad, _scalar_tensor_like(self_, float(other)))
        grad_self = reduce_grad(g, self_.shape)
    if hasattr(other, 'shape'):
        g = redispatch("mul", keyset, grad, self_)
        grad_other = reduce_grad(g, other.shape)
    else:
        grad_other = None
    return grad_self, grad_other


def _div_backward_all(grad, self_, other, keyset):
    if hasattr(other, 'shape'):
        grad_self = reduce_grad(redispatch("div", keyset, grad, other), self_.shape)
        other_sq = redispatch("mul", keyset, other, other)
        g = redispatch("neg", keyset, redispatch("div", keyset, redispatch("mul", keyset, grad, self_), other_sq))
        grad_other = reduce_grad(g, other.shape)
    else:
        # other is a scalar
        grad_self = reduce_grad(redispatch("div", keyset, grad, _scalar_tensor_like(self_, float(other))), self_.shape)
        grad_other = None
    return grad_self, grad_other


def _pow_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _pow_backward
    return _pow_backward(grad, self_, other, self_, other, keyset)


def _maximum_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _maximum_backward
    return _maximum_backward(grad, self_, other, self_, other, keyset)


def _minimum_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _minimum_backward
    return _minimum_backward(grad, self_, other, self_, other, keyset)


def _fmin_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _fmin_backward
    return _fmin_backward(grad, self_, other, self_, other, keyset)


def _fmax_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _fmax_backward
    return _fmax_backward(grad, self_, other, self_, other, keyset)


def _hypot_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _hypot_backward
    return _hypot_backward(grad, self_, other, self_, other, keyset)


def _logaddexp_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _logaddexp_backward
    return _logaddexp_backward(grad, self_, other, self_, other, keyset)


def _logaddexp2_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _logaddexp2_backward
    return _logaddexp2_backward(grad, self_, other, self_, other, keyset)


def _atan2_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _atan2_backward
    return _atan2_backward(grad, self_, other, self_, other, keyset)


def _inner_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _inner_backward
    return _inner_backward(grad, self_, other, self_, other, keyset)


# ---------------------------------------------------------------------------
# Unary-args backward helpers (Batch 3)
# ---------------------------------------------------------------------------

def _clamp_backward_helper(grad, input_, min_val, max_val, keyset):
    from .._backends.autograd import _clamp_backward
    args = (min_val, max_val) if min_val is not None or max_val is not None else ()
    kwargs = {}
    if not args:
        kwargs['min_val'] = min_val
        kwargs['max_val'] = max_val
    return _clamp_backward(grad, input_, input_, keyset, args, kwargs)[0]


def _clamp_min_backward_helper(grad, input_, min_val, keyset):
    from .._backends.autograd import _clamp_min_backward
    return _clamp_min_backward(grad, input_, input_, keyset, (min_val,), {})[0]


def _clamp_max_backward_helper(grad, input_, max_val, keyset):
    from .._backends.autograd import _clamp_max_backward
    return _clamp_max_backward(grad, input_, input_, keyset, (max_val,), {})[0]


def _gather_backward_helper(grad, input_, dim, index, keyset):
    from .._backends.autograd import _gather_backward
    return _gather_backward(grad, input_, input_, keyset, (dim, index), {})[0]


def _index_select_backward_helper(grad, input_, dim, index, keyset):
    from .._backends.autograd import _index_select_backward
    return _index_select_backward(grad, input_, input_, keyset, (dim, index), {})[0]


def _repeat_backward_helper(grad, input_, repeats, keyset):
    from .._backends.autograd import _repeat_backward
    return _repeat_backward(grad, input_, input_, keyset, (repeats,), {})[0]


def _norm_backward_helper(grad, input_, p, dim, keepdim, keyset):
    from .._backends.autograd import _norm_backward
    return _norm_backward(grad, input_, input_, keyset, (p,), {"dim": dim, "keepdim": keepdim})[0]


def _prod_backward_helper(grad, input_, dim, keepdim, keyset):
    from .._backends.autograd import _prod_backward
    return _prod_backward(grad, input_, input_, keyset, (), {"dim": dim, "keepdim": keepdim})[0]


def _cumsum_backward_helper(grad, input_, dim, keyset):
    from .._backends.autograd import _cumsum_backward
    return _cumsum_backward(grad, input_, input_, keyset, (dim,), {})[0]


def _pad_backward_helper(grad, input_, pad, keyset):
    from .._backends.autograd import _pad_backward
    return _pad_backward(grad, input_, input_, keyset, (pad,), {})[0]


def _roll_backward_helper(grad, input_, shifts, dims, keyset):
    from .._backends.autograd import _roll_backward
    return _roll_backward(grad, input_, input_, keyset, (shifts, dims) if dims is not None else (shifts,), {"dims": dims} if dims is not None else {})[0]


def _tile_backward_helper(grad, input_, dims, keyset):
    from .._backends.autograd import _tile_backward
    return _tile_backward(grad, input_, input_, keyset, (dims,), {})[0]


def _movedim_backward_helper(grad, input_, source, destination, keyset):
    from .._backends.autograd import _movedim_backward
    return _movedim_backward(grad, input_, input_, keyset, (source, destination), {})[0]


def _diagonal_backward_helper(grad, input_, offset, dim1, dim2, keyset):
    from .._backends.autograd import _diagonal_backward
    return _diagonal_backward(grad, input_, input_, keyset, (offset, dim1, dim2), {})[0]


def _threshold_backward_helper(grad, input_, threshold, keyset):
    from .._backends.autograd import _threshold_backward
    return _threshold_backward(grad, input_, input_, keyset, (threshold,), {})[0]


def _instance_norm_backward_helper(grad, input_, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled, keyset):
    from .._backends.autograd import _instance_norm_backward
    args = (weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled)
    return _instance_norm_backward(grad, input_, input_, keyset, args, {})[0]


def _normalize_backward_helper(grad, input_, p, dim, eps, keyset):
    from .._backends.autograd import _normalize_backward
    return _normalize_backward(grad, input_, input_, keyset, (p, dim, eps), {})[0]


def _cumprod_backward_helper(grad, input_, dim, keyset):
    from .._backends.autograd import _cumprod_backward
    return _cumprod_backward(grad, input_, input_, keyset, (dim,), {})[0]


def _repeat_interleave_backward_helper(grad, input_, repeats, dim, keyset):
    from .._backends.autograd import _repeat_interleave_backward
    return _repeat_interleave_backward(grad, input_, input_, keyset, (repeats,), {"dim": dim})[0]


def _rot90_backward_helper(grad, input_, k, dims, keyset):
    from .._backends.autograd import _rot90_backward
    return _rot90_backward(grad, input_, input_, keyset, (k, dims), {})[0]


def _take_backward_helper(grad, input_, index, keyset):
    from .._backends.autograd import _take_backward
    return _take_backward(grad, input_, input_, keyset, (index,), {})[0]


def _take_along_dim_backward_helper(grad, input_, indices, dim, keyset):
    from .._backends.autograd import _take_along_dim_backward
    return _take_along_dim_backward(grad, input_, input_, keyset, (indices, dim), {})[0]


def _hardshrink_backward_helper(grad, input_, lambd, keyset):
    from .._backends.autograd import _hardshrink_backward
    return _hardshrink_backward(grad, input_, input_, keyset, (lambd,), {})[0]


def _softshrink_backward_helper(grad, input_, lambd, keyset):
    from .._backends.autograd import _softshrink_backward
    return _softshrink_backward(grad, input_, input_, keyset, (lambd,), {})[0]


def _logsumexp_backward_helper(grad, input_, dim, keepdim, keyset):
    from .._backends.autograd import _logsumexp_backward
    return _logsumexp_backward(grad, input_, input_, keyset, (dim,), {"keepdim": keepdim})[0]


def _renorm_backward_helper(grad, input_, p, dim, maxnorm, keyset):
    from .._backends.autograd import _renorm_backward
    return _renorm_backward(grad, input_, input_, keyset, (p, dim, maxnorm), {})[0]


def _unfold_backward_helper(grad, input_, dimension, size, step, keyset):
    from .._backends.autograd import _unfold_backward
    return _unfold_backward(grad, input_, input_, keyset, (dimension, size, step), {})[0]


def _var_backward_helper(grad, input_, dim, unbiased, keepdim, keyset):
    from .._backends.autograd import _var_backward
    return _var_backward(grad, input_, input_, keyset, (), {"dim": dim, "unbiased": unbiased, "keepdim": keepdim})[0]


def _std_backward_helper(grad, input_, dim, keepdim, unbiased, keyset):
    from .._backends.autograd import _std_backward
    return _std_backward(grad, input_, input_, keyset, (), {"dim": dim, "keepdim": keepdim, "unbiased": unbiased})[0]


def _amax_backward_helper(grad, input_, dim, keepdim, keyset):
    from .._backends.autograd import _amax_backward
    return _amax_backward(grad, input_, input_, keyset, (dim,) if dim is not None else (), {"dim": dim, "keepdim": keepdim})[0]


def _amin_backward_helper(grad, input_, dim, keepdim, keyset):
    from .._backends.autograd import _amin_backward
    return _amin_backward(grad, input_, input_, keyset, (dim,) if dim is not None else (), {"dim": dim, "keepdim": keepdim})[0]


def _narrow_backward_helper(grad, input_, dim, start, length, keyset):
    from .._backends.autograd import _narrow_backward
    return _narrow_backward(grad, input_, input_, keyset, (dim, start, length), {})[0]


def _select_backward_helper(grad, input_, dim, index, keyset):
    from .._backends.autograd import _select_backward
    return _select_backward(grad, input_, input_, keyset, (dim, index), {})[0]


def _masked_fill_backward_helper(grad, input_, mask, keyset):
    from .._backends.autograd import _masked_fill_backward
    return _masked_fill_backward(grad, input_, input_, keyset, (mask,), {})[0]


def _diff_backward_helper(grad, input_, n, dim, keyset):
    from .._backends.autograd import _diff_backward
    return _diff_backward(grad, input_, input_, keyset, (n, dim), {})[0]


def _trace_backward_helper(grad, input_, keyset):
    from .._backends.autograd import _trace_backward
    return _trace_backward(grad, input_, input_, keyset, (), {})[0]


def _det_backward_helper(grad, input_, keyset):
    from .._backends.autograd import _det_backward
    return _det_backward(grad, input_, input_, keyset, (), {})[0]


def _diag_backward_helper(grad, input_, diagonal, keyset):
    from .._backends.autograd import _diag_backward
    return _diag_backward(grad, input_, input_, keyset, (diagonal,), {})[0]


def _im2col_backward_helper(grad, input_, kernel_size, dilation, padding, stride, keyset):
    from .._backends.autograd import _im2col_backward
    return _im2col_backward(grad, input_, input_, keyset, (kernel_size, dilation, padding, stride), {})[0]


def _col2im_backward_helper(grad, input_, output_size, kernel_size, dilation, padding, stride, keyset):
    from .._backends.autograd import _col2im_backward
    return _col2im_backward(grad, input_, input_, keyset, (output_size, kernel_size, dilation, padding, stride), {})[0]


def _nansum_backward_helper(grad, input_, dim, keepdim, keyset):
    from .._backends.autograd import _nansum_backward
    return _nansum_backward(grad, input_, input_, keyset, (), {"dim": dim, "keepdim": keepdim})[0]


def _nanmean_backward_helper(grad, input_, dim, keepdim, keyset):
    from .._backends.autograd import _nanmean_backward
    return _nanmean_backward(grad, input_, input_, keyset, (), {"dim": dim, "keepdim": keepdim})[0]


def _matrix_power_backward_helper(grad, input_, n, keyset):
    from .._backends.autograd import _matrix_power_backward
    return _matrix_power_backward(grad, input_, input_, keyset, (n,), {})[0]


def _special_logit_backward_helper(grad, input_, eps, keyset):
    from .._backends.autograd import _special_logit_backward
    return _special_logit_backward(grad, input_, input_, keyset, (eps,) if eps is not None else (), {"eps": eps} if eps is not None else {})[0]


def _linalg_norm_backward_helper(grad, input_, ord, dim, keepdim, keyset):
    from .._backends.autograd import _linalg_norm_backward
    return _linalg_norm_backward(grad, input_, input_, keyset, (ord,), {"dim": dim, "keepdim": keepdim})[0]


def _linalg_vector_norm_backward_helper(grad, input_, ord, dim, keepdim, keyset):
    from .._backends.autograd import _linalg_vector_norm_backward
    return _linalg_vector_norm_backward(grad, input_, input_, keyset, (ord,), {"dim": dim, "keepdim": keepdim})[0]


def _linalg_matrix_norm_backward_helper(grad, input_, ord, dim, keepdim, keyset):
    from .._backends.autograd import _linalg_matrix_norm_backward
    return _linalg_matrix_norm_backward(grad, input_, input_, keyset, (ord,), {"dim": dim, "keepdim": keepdim})[0]


def _linalg_pinv_backward_helper(grad, input_, atol, rtol, hermitian, keyset):
    from .._backends.autograd import _linalg_pinv_backward
    return _linalg_pinv_backward(grad, input_, input_, keyset, (atol, rtol, hermitian), {})[0]


def _linalg_cond_backward_helper(grad, input_, p, keyset):
    from .._backends.autograd import _linalg_cond_backward
    return _linalg_cond_backward(grad, input_, input_, keyset, (p,) if p is not None else (), {})[0]


def _linalg_matrix_exp_backward_helper(grad, input_, keyset):
    from .._backends.autograd import _linalg_matrix_exp_backward
    return _linalg_matrix_exp_backward(grad, input_, input_, keyset, (), {})[0]


def _linalg_tensorinv_backward_helper(grad, input_, ind, keyset):
    from .._backends.autograd import _linalg_tensorinv_backward
    return _linalg_tensorinv_backward(grad, input_, input_, keyset, (ind,), {})[0]


def _linalg_vander_backward_helper(grad, x, N, keyset):
    from .._backends.autograd import _linalg_vander_backward
    return _linalg_vander_backward(grad, x, x, keyset, (N,) if N is not None else (), {"N": N})[0]


def _linalg_eigvalsh_backward_helper(grad, input_, UPLO, keyset):
    from .._backends.autograd import _linalg_eigvalsh_backward
    return _linalg_eigvalsh_backward(grad, input_, input_, keyset, (UPLO,), {})[0]


def _sum_to_size_backward_helper(grad, input_, size, keyset):
    from .._backends.autograd import _sum_to_size_backward
    return _sum_to_size_backward(grad, input_, input_, keyset, (size,), {})[0]


def _sum_backward_helper(grad, input_sizes, dim, keepdim, keyset):
    # Expand reduced gradient back to input shape using Candle's existing expand op.
    if dim is None:
        return redispatch("expand", keyset, grad, input_sizes)
    expanded = grad
    dims = dim if isinstance(dim, (list, tuple)) else [dim]
    if not keepdim:
        for d in sorted(dims):
            expanded = redispatch("unsqueeze", keyset, expanded, d)
    return redispatch("expand", keyset, expanded, input_sizes)


def _mean_backward_helper(grad, input_sizes, dim, numel, keepdim, keyset):
    sum_grad = _sum_backward_helper(grad, input_sizes, dim, keepdim, keyset)
    scale = 1.0 / numel if numel else 0.0
    return redispatch("mul", keyset, sum_grad, scale)


def _slice_backward_helper(grad, input_, dim, start, end, step, keyset):
    from .._backends.autograd import _slice_backward
    return _slice_backward(grad, input_, input_, keyset, (dim, start, end, step), {})[0]


def _slice_backward_wrapper_helper(grad, input_sizes, dim, start, end, step, keyset):
    """Backward for slice using input sizes (shape) rather than the input tensor."""
    from .._backends.autograd import _slice_backward
    placeholder = redispatch('zeros', keyset, tuple(input_sizes), dtype=grad.dtype, device=grad.device)
    return _slice_backward(grad, placeholder, placeholder, keyset, (dim, start, end, step), {})[0]


def _as_strided_backward_helper(grad, self_, size, stride, storage_offset, keyset):
    """Backward for as_strided: scatter grad into zeros of self's shape."""
    grad_out = redispatch('zeros', keyset, self_.shape, dtype=grad.dtype, device=grad.device)
    grad_view = redispatch('as_strided', keyset, grad_out, size, stride, storage_offset)
    redispatch('add_', keyset, grad_view, grad)
    return grad_out


def _as_strided_scatter_backward_helper(grad, self_, src, size, stride, storage_offset, keyset):
    """Backward for as_strided_scatter (TensorGeometry args are just self/src)."""
    from .._backends.autograd import _as_strided_scatter_backward
    return _as_strided_scatter_backward(grad, self_, src, self_, src, keyset, (size, stride, storage_offset), {})


def _getitem_backward_helper(grad, self_, key, keyset):
    from .._backends.autograd import _getitem_backward
    return _getitem_backward(grad, self_, self_, keyset, (key,), {})[0]


def _to_backward_helper(grad, input_, keyset):
    from .._backends.autograd import _to_backward
    return _to_backward(grad, input_, input_, keyset, (), {})[0]


def _quantile_backward_helper(grad, input_, q, dim, keepdim, keyset):
    from .._backends.autograd import _quantile_backward
    args = (q,)
    if dim is not None:
        args = (q, dim)
    return _quantile_backward(grad, input_, input_, keyset, args, {"dim": dim, "keepdim": keepdim})[0]


# ---------------------------------------------------------------------------
# Special function backward helpers (Batch 4)
# ---------------------------------------------------------------------------

def _special_erfinv_grad(grad, self_, keyset):
    from .._backends.autograd import _special_erfinv_backward
    return _special_erfinv_backward(grad, self_, self_, keyset)[0]


def _special_erfcx_grad(grad, self_, keyset):
    from .._backends.autograd import _special_erfcx_backward
    return _special_erfcx_backward(grad, self_, self_, keyset)[0]


def _special_ndtr_grad(grad, self_, keyset):
    from .._backends.autograd import _special_ndtr_backward
    return _special_ndtr_backward(grad, self_, self_, keyset)[0]


def _special_ndtri_grad(grad, self_, keyset):
    from .._backends.autograd import _special_ndtri_backward
    return _special_ndtri_backward(grad, self_, self_, keyset)[0]


def _special_sinc_grad(grad, self_, keyset):
    from .._backends.autograd import _special_sinc_backward
    return _special_sinc_backward(grad, self_, self_, keyset)[0]


def _special_entr_grad(grad, self_, keyset):
    from .._backends.autograd import _special_entr_backward
    return _special_entr_backward(grad, self_, self_, keyset)[0]


def _special_log_ndtr_grad(grad, self_, keyset):
    from .._backends.autograd import _special_log_ndtr_backward
    return _special_log_ndtr_backward(grad, self_, self_, keyset)[0]


def _special_i0e_grad(grad, self_, keyset):
    from .._backends.autograd import _special_i0e_backward
    return _special_i0e_backward(grad, self_, self_, keyset)[0]


def _special_i1_grad(grad, self_, keyset):
    from .._backends.autograd import _special_i1_backward
    return _special_i1_backward(grad, self_, self_, keyset)[0]


def _special_i1e_grad(grad, self_, keyset):
    from .._backends.autograd import _special_i1e_backward
    return _special_i1e_backward(grad, self_, self_, keyset)[0]


def _special_xlogy_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _special_xlogy_backward
    return _special_xlogy_backward(grad, self_, other, self_, other, keyset)


def _special_xlog1py_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _special_xlog1py_backward
    return _special_xlog1py_backward(grad, self_, other, self_, other, keyset)


def _special_zeta_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _special_zeta_backward
    return _special_zeta_backward(grad, self_, other, self_, other, keyset)


def _special_gammainc_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _special_gammainc_backward
    return _special_gammainc_backward(grad, self_, other, self_, other, keyset)


def _special_gammaincc_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _special_gammaincc_backward
    return _special_gammaincc_backward(grad, self_, other, self_, other, keyset)


def _linalg_inv_grad(grad, self_, keyset):
    from .._backends.autograd import _linalg_inv_backward
    return _linalg_inv_backward(grad, self_, self_, keyset)[0]


def _linalg_svdvals_grad(grad, self_, keyset):
    from .._backends.autograd import _linalg_svdvals_backward
    return _linalg_svdvals_backward(grad, self_, self_, keyset)[0]


# ---------------------------------------------------------------------------
# FFT backward helpers (Batch 5)
# ---------------------------------------------------------------------------

def _fft_c2c_backward_helper(grad, input_, n, dim, norm, inverse_op, keyset):
    return redispatch(inverse_op, keyset, grad, n=n, dim=dim, norm=norm)


def _fft_r2c_backward_helper(grad, input_, n, dim, norm, inverse_op, keyset):
    result = redispatch(inverse_op, keyset, grad, n=n if n is not None else input_.shape[dim if isinstance(dim, int) else -1], dim=dim, norm=norm)
    return result


def _fft_c2r_backward_helper(grad, input_, n, dim, norm, inverse_op, keyset):
    return redispatch(inverse_op, keyset, grad, dim=dim, norm=norm)


def _fft_shift_backward_helper(grad, input_, dim, inverse_op, keyset):
    return redispatch(inverse_op, keyset, grad, dim=dim)


# ---------------------------------------------------------------------------
# Linalg custom backward helpers (Batch 5)
# ---------------------------------------------------------------------------

def _linalg_det_backward_helper(grad, self_, keyset):
    from .._backends.autograd import _linalg_det_backward
    return _linalg_det_backward(grad, self_, self_, keyset)[0]


def _linalg_slogdet_backward_helper(grad, self_, keyset):
    from .._backends.autograd import _linalg_slogdet_backward
    return _linalg_slogdet_backward(grad, self_, self_, keyset)[0]


def _linalg_cholesky_backward_helper(grad, self_, upper, keyset):
    from .._backends.autograd import _linalg_cholesky_backward
    return _linalg_cholesky_backward(grad, self_, self_, keyset, (upper,))[0]


def _linalg_solve_backward_all(grad, self_, other, left, keyset):
    from .._backends.autograd import _linalg_solve_backward
    return _linalg_solve_backward(grad, self_, other, self_, other, keyset, (left,))


def _linalg_solve_triangular_backward_all(grad, self_, B, upper, left, unitriangular, keyset):
    from .._backends.autograd import _linalg_solve_triangular_backward
    return _linalg_solve_triangular_backward(grad, self_, B, self_, B, keyset, (upper, left, unitriangular))


def _linalg_qr_backward_helper(grad, self_, mode, keyset):
    from .._backends.autograd import _linalg_qr_backward
    return _linalg_qr_backward(grad, self_, self_, keyset, (mode,))[0]


def _linalg_svd_backward_helper(grad, self_, full_matrices, keyset):
    from .._backends.autograd import _linalg_svd_backward
    return _linalg_svd_backward(grad, self_, self_, keyset, (full_matrices,))[0]


def _linalg_eigh_backward_helper(grad, self_, UPLO, keyset):
    from .._backends.autograd import _linalg_eigh_backward
    return _linalg_eigh_backward(grad, self_, self_, keyset, (UPLO,))[0]


def _linalg_eig_backward_helper(grad, self_, keyset):
    from .._backends.autograd import _linalg_eig_backward
    return _linalg_eig_backward(grad, self_, self_, keyset)[0]


def _linalg_lu_backward_helper(grad, self_, pivot, keyset):
    from .._backends.autograd import _linalg_lu_backward
    return _linalg_lu_backward(grad, self_, self_, keyset, (pivot,))[0]


def _linalg_lu_factor_backward_helper(grad, self_, pivot, keyset):
    from .._backends.autograd import _linalg_lu_factor_backward
    return _linalg_lu_factor_backward(grad, self_, self_, keyset, (pivot,))[0]


def _linalg_lu_solve_backward_all(grad, LU, pivots, B, left, adjoint, keyset):
    from .._backends.autograd import _linalg_lu_solve_backward
    return _linalg_lu_solve_backward(grad, LU, pivots, B, LU, pivots, B, keyset, (left, adjoint))


def _linalg_lu_solve_backward_helper(grad, LU, pivots, B, left, adjoint, keyset):
    from .._backends.autograd import _linalg_lu_solve_backward
    return _linalg_lu_solve_backward(grad, LU, pivots, B, LU, pivots, B, keyset, (left, adjoint))[2]


def _linalg_multi_dot_backward_helper(grad, tensors, keyset):
    from .._backends.autograd import _linalg_multi_dot_backward
    return _linalg_multi_dot_backward(grad, tensors, keyset)


# ---------------------------------------------------------------------------
# Special custom backward helpers (Batch 5)
# ---------------------------------------------------------------------------

def _special_polygamma_backward_helper(grad, n, self_, keyset):
    return redispatch("mul", keyset, grad, redispatch("special_polygamma", keyset, n + 1, self_))


def _special_multigammaln_backward_helper(grad, self_, p, keyset):
    from .._backends.autograd import _special_multigammaln_backward
    return _special_multigammaln_backward(grad, self_, self_, keyset, (p,))[0]


# ---------------------------------------------------------------------------
# Multi-tensor backward helpers (Batch 6)
# ---------------------------------------------------------------------------

def _lerp_backward_all(grad, self_, other, weight, keyset):
    from .._backends.autograd import _lerp_backward
    return _lerp_backward(grad, self_, other, weight, keyset)


def _addcmul_backward_all(grad, self_, tensor1, tensor2, value, keyset):
    from .._backends.autograd import _addcmul_backward
    return _addcmul_backward(grad, self_, tensor1, tensor2, value, keyset)


def _addcdiv_backward_all(grad, self_, tensor1, tensor2, value, keyset):
    from .._backends.autograd import _addcdiv_backward
    return _addcdiv_backward(grad, self_, tensor1, tensor2, value, keyset)


def _addmm_backward_all(grad, self_, mat1, mat2, beta, alpha, keyset):
    from .._backends.autograd import _addmm_backward
    return _addmm_backward(grad, self_, mat1, mat2, beta, alpha, keyset)


def _baddbmm_backward_all(grad, self_, batch1, batch2, beta, alpha, keyset):
    from .._backends.autograd import _baddbmm_backward
    return _baddbmm_backward(grad, self_, batch1, batch2, beta, alpha, keyset)


def _where_backward_all(grad, condition, self_, other, keyset):
    from .._backends.autograd import _where_backward
    return _where_backward(grad, condition, self_, other, condition, keyset)[1:]


def _embedding_backward_helper(grad, weight, indices, padding_idx, scale_grad_by_freq, keyset):
    from .._backends.autograd import _embedding_backward
    return _embedding_backward(grad, weight, indices, weight, indices, keyset,
                               padding_idx=padding_idx, scale_grad_by_freq=scale_grad_by_freq)[0]


def _cross_backward_all(grad, self_, other, dim, keyset):
    from .._backends.autograd import _cross_backward
    return _cross_backward(grad, self_, other, self_, other, dim, keyset)


def _min_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _min_backward
    return _min_backward(grad, self_, other, self_, other, keyset)


def _max_backward_all(grad, self_, other, keyset):
    from .._backends.autograd import _max_backward
    return _max_backward(grad, self_, other, self_, other, keyset)


def _dist_backward_all(grad, self_, other, p, keyset):
    from .._backends.autograd import _dist_backward
    return _dist_backward(grad, self_, other, self_, other, keyset, (p,))


def _cdist_backward_all(grad, self_, other, p, keyset):
    from .._backends.autograd import _cdist_backward
    return _cdist_backward(grad, self_, other, self_, other, keyset, (p,))


def _as_strided_scatter_backward_all(grad, self_, src, size, stride, storage_offset, keyset):
    from .._backends.autograd import _as_strided_scatter_backward
    return _as_strided_scatter_backward(grad, self_, src, self_, src, keyset, (size, stride, storage_offset))


def _masked_select_backward_helper(grad, self_, mask, keyset):
    from .._backends.autograd import _masked_select_backward
    return _masked_select_backward(grad, self_, mask, keyset)[0]


def _tensordot_backward_all(grad, self_, other, dims, keyset):
    from .._backends.autograd import _tensordot_backward
    return _tensordot_backward(grad, self_, other, dims, keyset)


def _grid_sample_backward_all(grad, self_, grid, mode, padding_mode, align_corners, keyset):
    from .._backends.autograd import _grid_sample_backward
    return _grid_sample_backward(grad, self_, grid, self_, grid, keyset,
                                 (mode, padding_mode, align_corners), {})


def _affine_grid_backward_helper(grad, theta, size, align_corners, keyset):
    from .._backends.autograd import _affine_grid_backward
    return _affine_grid_backward(grad, theta, size, align_corners, keyset)


def _rrelu_backward_helper(grad, self_, lower, upper, training, keyset):
    from .._backends.autograd import _rrelu_backward
    return _rrelu_backward(grad, self_, self_, keyset, (lower, upper, training), {})[0]


def _ctc_loss_backward_helper(grad, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity, keyset):
    from .._backends.autograd import _ctc_loss_backward
    return _ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths,
                              blank, reduction, zero_infinity, keyset)


def _sort_backward_helper(grad, self_, result1, dim, keyset):
    from .._backends.autograd import _sort_backward
    return _sort_backward(grad, self_, result1, keyset, (dim,), {})[0]


def _topk_backward_helper(grad, self_, result1, k, dim, keyset):
    from .._backends.autograd import _topk_backward
    return _topk_backward(grad, self_, result1, keyset, (k, dim), {})[0]


def _kthvalue_backward_helper(grad, self_, result1, k, dim, keepdim, keyset):
    from .._backends.autograd import _kthvalue_backward
    return _kthvalue_backward(grad, self_, result1, keyset, (k, dim), {"keepdim": keepdim})[0]


def _cummax_backward_helper(grad, self_, result1, dim, keyset):
    from .._backends.autograd import _cummax_backward
    return _cummax_backward(grad, self_, result1, keyset, (dim,), {})[0]


def _cummin_backward_helper(grad, self_, result1, dim, keyset):
    from .._backends.autograd import _cummin_backward
    return _cummin_backward(grad, self_, result1, keyset, (dim,), {})[0]


# ---------------------------------------------------------------------------
# Tensor[] backward helpers (called from generated formulas)
# ---------------------------------------------------------------------------

def _select_backward_symint_helper(grad, input_sizes, dim, index, keyset):
    from .._creation import zeros as _zeros
    out = _zeros(*input_sizes, dtype=grad.dtype, device=grad.device)
    indexer = [slice(None)] * len(input_sizes)
    indexer[dim] = index
    out[tuple(indexer)] = grad
    return out


def _cat_backward_helper(grad, tensors, dim, keyset):
    from .._backends.autograd import _cat_backward
    return _cat_backward(grad, tensors, None, keyset, (dim,), {})


def _stack_backward_helper(grad, tensors, dim, keyset):
    from .._backends.autograd import _stack_backward
    return _stack_backward(grad, tensors, None, keyset, (dim,), {})


def _hstack_backward_helper(grad, tensors, keyset):
    from .._backends.autograd import _hstack_backward
    return _hstack_backward(grad, tensors, None, keyset, (), {})


def _vstack_backward_helper(grad, tensors, keyset):
    from .._backends.autograd import _vstack_backward
    return _vstack_backward(grad, tensors, None, keyset, (), {})


def _dstack_backward_helper(grad, tensors, keyset):
    from .._backends.autograd import _dstack_backward
    return _dstack_backward(grad, tensors, None, keyset, (), {})


def _column_stack_backward_helper(grad, tensors, keyset):
    from .._backends.autograd import _column_stack_backward
    return _column_stack_backward(grad, tensors, None, keyset, (), {})


def _concat_backward_helper(grad, tensors, dim, keyset):
    from .._backends.autograd import _cat_backward
    return _cat_backward(grad, tensors, None, keyset, (dim,), {})


def _concatenate_backward_helper(grad, tensors, dim, keyset):
    from .._backends.autograd import _cat_backward
    return _cat_backward(grad, tensors, None, keyset, (dim,), {})


def _block_diag_backward_helper(grad, tensors, keyset):
    from .._backends.autograd import _block_diag_backward
    return _block_diag_backward(grad, tensors, keyset)


def _pad_sequence_backward_helper(grad, sequences, batch_first, padding_value, padding_side, keyset):
    from .._backends.autograd import _pad_sequence_backward
    return _pad_sequence_backward(grad, sequences, keyset, batch_first, padding_value, padding_side)


class SumToSizeBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='SumToSizeBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._self_shape = None

    def _save(self, *, self_=None):
        if self_ is not None:
            self._self_shape = self_.shape

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        with _grad_context(keyset):
            grad_self = redispatch("add", keyset,
                redispatch("zeros", keyset, self._self_shape, dtype=grad.dtype, device=grad.device),
                grad)
        return (grad_self,)


class Adaptive_avg_pool1dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Adaptive_avg_pool1dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._output_size = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        output_size = self._output_size
        with _grad_context(keyset):
            grad_self = _adaptive_avg_pool1d_backward_helper(grad, self_, output_size, keyset)
        return (grad_self,)


class Adaptive_avg_pool2dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Adaptive_avg_pool2dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._output_size = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        output_size = self._output_size
        with _grad_context(keyset):
            grad_self = _adaptive_avg_pool2d_backward_helper(grad, self_, output_size, keyset)
        return (grad_self,)


class Adaptive_avg_pool3dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Adaptive_avg_pool3dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._output_size = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        output_size = self._output_size
        with _grad_context(keyset):
            grad_self = _adaptive_avg_pool3d_backward_helper(grad, self_, output_size, keyset)
        return (grad_self,)


class Adaptive_max_pool1dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Adaptive_max_pool1dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._saved_result_idx = None
        self._output_size = None
        self._return_indices = None

    def _save(self, *, self_=None, result=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if result is not None:
            self._saved_result_idx = len(tensors)
            tensors.append(result)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]
        if self._saved_result_idx is not None:
            self._saved_fields['result'] = self._saved_tensors_list[self._saved_result_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        result = _saved[self._saved_result_idx]
        output_size = self._output_size
        return_indices = self._return_indices
        with _grad_context(keyset):
            grad_self = _adaptive_max_pool1d_backward_helper(grad, self_, result, output_size, keyset)
        return (grad_self,)


class Affine_gridBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Affine_gridBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_theta_idx = None
        self._size = None
        self._align_corners = None

    def _save(self, *, theta=None):
        tensors = []
        if theta is not None:
            self._saved_theta_idx = len(tensors)
            tensors.append(theta)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_theta_idx is not None:
            self._saved_fields['theta'] = self._saved_tensors_list[self._saved_theta_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        theta = _saved[self._saved_theta_idx]
        size = self._size
        align_corners = self._align_corners
        with _grad_context(keyset):
            grad_theta = _affine_grid_backward_helper(grad, theta, size, align_corners, keyset)
        return (grad_theta,)


class Avg_pool1dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Avg_pool1dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._kernel_size = None
        self._stride = None
        self._padding = None
        self._ceil_mode = None
        self._count_include_pad = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        kernel_size = self._kernel_size
        stride = self._stride
        padding = self._padding
        ceil_mode = self._ceil_mode
        count_include_pad = self._count_include_pad
        with _grad_context(keyset):
            grad_self = _avg_pool1d_backward_helper(grad, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, keyset)
        return (grad_self,)


class Batch_normBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Batch_normBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._saved_weight_idx = None
        self._training = None
        self._momentum = None
        self._eps = None

    def _save(self, *, input_=None, weight=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if weight is not None:
            self._saved_weight_idx = len(tensors)
            tensors.append(weight)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]
        if self._saved_weight_idx is not None:
            self._saved_fields['weight'] = self._saved_tensors_list[self._saved_weight_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        weight = _saved[self._saved_weight_idx] if self._saved_weight_idx is not None else None
        training = self._training
        momentum = self._momentum
        eps = self._eps
        with _grad_context(keyset):
            grad_input = _batch_norm_grad_input(grad, input_, weight, eps, keyset)
        return (grad_input,)


class Broadcast_toBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Broadcast_toBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._shape = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        shape = self._shape
        with _grad_context(keyset):
            grad_input = reduce_grad(grad, input_.shape)
        return (grad_input,)


class CdistBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='CdistBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_x1_idx = None
        self._saved_x2_idx = None
        self._p = None

    def _save(self, *, x1=None, x2=None):
        tensors = []
        if x1 is not None:
            self._saved_x1_idx = len(tensors)
            tensors.append(x1)
        if x2 is not None:
            self._saved_x2_idx = len(tensors)
            tensors.append(x2)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_x1_idx is not None:
            self._saved_fields['x1'] = self._saved_tensors_list[self._saved_x1_idx]
        if self._saved_x2_idx is not None:
            self._saved_fields['x2'] = self._saved_tensors_list[self._saved_x2_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        x1 = _saved[self._saved_x1_idx]
        x2 = _saved[self._saved_x2_idx]
        p = self._p
        with _grad_context(keyset):
            grad_x1, grad_x2 = _cdist_backward_all(grad, x1, x2, p, keyset)
        return (grad_x1, grad_x2,)


class Column_stackBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Column_stackBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        tensors = list(self.inputs)
        with _grad_context(keyset):
            grad_tensors = _column_stack_backward_helper(grad, tensors, keyset)
        return grad_tensors


class ConcatBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='ConcatBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._dim = None

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        tensors = list(self.inputs)
        dim = self._dim
        with _grad_context(keyset):
            grad_tensors = _concat_backward_helper(grad, tensors, dim, keyset)
        return grad_tensors


class ConcatenateBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='ConcatenateBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._dim = None

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        tensors = list(self.inputs)
        dim = self._dim
        with _grad_context(keyset):
            grad_tensors = _concatenate_backward_helper(grad, tensors, dim, keyset)
        return grad_tensors


class ContiguousBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='ContiguousBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        with _grad_context(keyset):
            grad_self = grad
        return (grad_self,)


class Conv1dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Conv1dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_bias_idx = None
        self._saved_input_idx = None
        self._saved_weight_idx = None
        self._stride = None
        self._padding = None
        self._dilation = None
        self._groups = None

    def _save(self, *, bias=None, input_=None, weight=None):
        tensors = []
        if bias is not None:
            self._saved_bias_idx = len(tensors)
            tensors.append(bias)
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if weight is not None:
            self._saved_weight_idx = len(tensors)
            tensors.append(weight)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_bias_idx is not None:
            self._saved_fields['bias'] = self._saved_tensors_list[self._saved_bias_idx]
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]
        if self._saved_weight_idx is not None:
            self._saved_fields['weight'] = self._saved_tensors_list[self._saved_weight_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        bias = _saved[self._saved_bias_idx] if self._saved_bias_idx is not None else None
        input_ = _saved[self._saved_input_idx]
        weight = _saved[self._saved_weight_idx]
        stride = self._stride
        padding = self._padding
        dilation = self._dilation
        groups = self._groups
        with _grad_context(keyset):
            grad_input, grad_weight, grad_bias = _conv_backward_all(grad, input_, weight, bias, stride, padding, dilation, groups, 'conv1d', keyset)
        return (grad_input, grad_weight, grad_bias,)


class Conv2dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Conv2dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_bias_idx = None
        self._saved_input_idx = None
        self._saved_weight_idx = None
        self._stride = None
        self._padding = None
        self._dilation = None
        self._groups = None

    def _save(self, *, bias=None, input_=None, weight=None):
        tensors = []
        if bias is not None:
            self._saved_bias_idx = len(tensors)
            tensors.append(bias)
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if weight is not None:
            self._saved_weight_idx = len(tensors)
            tensors.append(weight)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_bias_idx is not None:
            self._saved_fields['bias'] = self._saved_tensors_list[self._saved_bias_idx]
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]
        if self._saved_weight_idx is not None:
            self._saved_fields['weight'] = self._saved_tensors_list[self._saved_weight_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        bias = _saved[self._saved_bias_idx] if self._saved_bias_idx is not None else None
        input_ = _saved[self._saved_input_idx]
        weight = _saved[self._saved_weight_idx]
        stride = self._stride
        padding = self._padding
        dilation = self._dilation
        groups = self._groups
        with _grad_context(keyset):
            grad_input, grad_weight, grad_bias = _conv_backward_all(grad, input_, weight, bias, stride, padding, dilation, groups, 'conv2d', keyset)
        return (grad_input, grad_weight, grad_bias,)


class Conv3dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Conv3dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_bias_idx = None
        self._saved_input_idx = None
        self._saved_weight_idx = None
        self._stride = None
        self._padding = None
        self._dilation = None
        self._groups = None

    def _save(self, *, bias=None, input_=None, weight=None):
        tensors = []
        if bias is not None:
            self._saved_bias_idx = len(tensors)
            tensors.append(bias)
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if weight is not None:
            self._saved_weight_idx = len(tensors)
            tensors.append(weight)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_bias_idx is not None:
            self._saved_fields['bias'] = self._saved_tensors_list[self._saved_bias_idx]
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]
        if self._saved_weight_idx is not None:
            self._saved_fields['weight'] = self._saved_tensors_list[self._saved_weight_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        bias = _saved[self._saved_bias_idx] if self._saved_bias_idx is not None else None
        input_ = _saved[self._saved_input_idx]
        weight = _saved[self._saved_weight_idx]
        stride = self._stride
        padding = self._padding
        dilation = self._dilation
        groups = self._groups
        with _grad_context(keyset):
            grad_input, grad_weight, grad_bias = _conv_backward_all(grad, input_, weight, bias, stride, padding, dilation, groups, 'conv3d', keyset)
        return (grad_input, grad_weight, grad_bias,)


class Conv_transpose1dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Conv_transpose1dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_bias_idx = None
        self._saved_input_idx = None
        self._saved_weight_idx = None
        self._stride = None
        self._padding = None
        self._output_padding = None
        self._groups = None
        self._dilation = None

    def _save(self, *, bias=None, input_=None, weight=None):
        tensors = []
        if bias is not None:
            self._saved_bias_idx = len(tensors)
            tensors.append(bias)
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if weight is not None:
            self._saved_weight_idx = len(tensors)
            tensors.append(weight)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_bias_idx is not None:
            self._saved_fields['bias'] = self._saved_tensors_list[self._saved_bias_idx]
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]
        if self._saved_weight_idx is not None:
            self._saved_fields['weight'] = self._saved_tensors_list[self._saved_weight_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        bias = _saved[self._saved_bias_idx] if self._saved_bias_idx is not None else None
        input_ = _saved[self._saved_input_idx]
        weight = _saved[self._saved_weight_idx]
        stride = self._stride
        padding = self._padding
        output_padding = self._output_padding
        groups = self._groups
        dilation = self._dilation
        with _grad_context(keyset):
            grad_input, grad_weight, grad_bias = _conv_backward_all(grad, input_, weight, bias, stride, padding, dilation, groups, 'conv_transpose1d', keyset)
        return (grad_input, grad_weight, grad_bias,)


class Conv_transpose2dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Conv_transpose2dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_bias_idx = None
        self._saved_input_idx = None
        self._saved_weight_idx = None
        self._stride = None
        self._padding = None
        self._output_padding = None
        self._groups = None
        self._dilation = None

    def _save(self, *, bias=None, input_=None, weight=None):
        tensors = []
        if bias is not None:
            self._saved_bias_idx = len(tensors)
            tensors.append(bias)
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if weight is not None:
            self._saved_weight_idx = len(tensors)
            tensors.append(weight)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_bias_idx is not None:
            self._saved_fields['bias'] = self._saved_tensors_list[self._saved_bias_idx]
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]
        if self._saved_weight_idx is not None:
            self._saved_fields['weight'] = self._saved_tensors_list[self._saved_weight_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        bias = _saved[self._saved_bias_idx] if self._saved_bias_idx is not None else None
        input_ = _saved[self._saved_input_idx]
        weight = _saved[self._saved_weight_idx]
        stride = self._stride
        padding = self._padding
        output_padding = self._output_padding
        groups = self._groups
        dilation = self._dilation
        with _grad_context(keyset):
            grad_input, grad_weight, grad_bias = _conv_backward_all(grad, input_, weight, bias, stride, padding, dilation, groups, 'conv_transpose2d', keyset)
        return (grad_input, grad_weight, grad_bias,)


class Conv_transpose3dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Conv_transpose3dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_bias_idx = None
        self._saved_input_idx = None
        self._saved_weight_idx = None
        self._stride = None
        self._padding = None
        self._output_padding = None
        self._groups = None
        self._dilation = None

    def _save(self, *, bias=None, input_=None, weight=None):
        tensors = []
        if bias is not None:
            self._saved_bias_idx = len(tensors)
            tensors.append(bias)
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if weight is not None:
            self._saved_weight_idx = len(tensors)
            tensors.append(weight)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_bias_idx is not None:
            self._saved_fields['bias'] = self._saved_tensors_list[self._saved_bias_idx]
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]
        if self._saved_weight_idx is not None:
            self._saved_fields['weight'] = self._saved_tensors_list[self._saved_weight_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        bias = _saved[self._saved_bias_idx] if self._saved_bias_idx is not None else None
        input_ = _saved[self._saved_input_idx]
        weight = _saved[self._saved_weight_idx]
        stride = self._stride
        padding = self._padding
        output_padding = self._output_padding
        groups = self._groups
        dilation = self._dilation
        with _grad_context(keyset):
            grad_input, grad_weight, grad_bias = _conv_backward_all(grad, input_, weight, bias, stride, padding, dilation, groups, 'conv_transpose3d', keyset)
        return (grad_input, grad_weight, grad_bias,)


class CrossBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='CrossBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_other_idx = None
        self._saved_self_idx = None
        self._dim = None

    def _save(self, *, other=None, self_=None):
        tensors = []
        if other is not None:
            self._saved_other_idx = len(tensors)
            tensors.append(other)
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_other_idx is not None:
            self._saved_fields['other'] = self._saved_tensors_list[self._saved_other_idx]
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        other = _saved[self._saved_other_idx]
        self_ = _saved[self._saved_self_idx]
        dim = self._dim
        with _grad_context(keyset):
            grad_self, grad_other = _cross_backward_all(grad, self_, other, dim, keyset)
        return (grad_self, grad_other,)


class Ctc_lossBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Ctc_lossBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._targets = None
        self._input_lengths = None
        self._target_lengths = None
        self._blank = None
        self._reduction = None
        self._zero_infinity = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        targets = self._targets
        input_lengths = self._input_lengths
        target_lengths = self._target_lengths
        blank = self._blank
        reduction = self._reduction
        zero_infinity = self._zero_infinity
        with _grad_context(keyset):
            grad_self = _ctc_loss_backward_helper(grad, self_, targets, input_lengths, target_lengths, blank, reduction, zero_infinity, keyset)
        return (grad_self,)


class DetBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='DetBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        with _grad_context(keyset):
            grad_input = _det_backward_helper(grad, input_, keyset)
        return (grad_input,)


class DiagBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='DiagBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._diagonal = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        diagonal = self._diagonal
        with _grad_context(keyset):
            grad_input = _diag_backward_helper(grad, input_, diagonal, keyset)
        return (grad_input,)


class DiffBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='DiffBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._n = None
        self._dim = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        n = self._n
        dim = self._dim
        with _grad_context(keyset):
            grad_input = _diff_backward_helper(grad, input_, n, dim, keyset)
        return (grad_input,)


class DstackBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='DstackBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        tensors = list(self.inputs)
        with _grad_context(keyset):
            grad_tensors = _dstack_backward_helper(grad, tensors, keyset)
        return grad_tensors


class Fft_fft2Backward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_fft2Backward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._s = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        s = self._s
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_c2c_backward_helper(grad, input_, s, dim, norm, 'fft_ifft2', keyset)
        return (grad_input,)


class Fft_fftBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_fftBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._n = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        n = self._n
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_c2c_backward_helper(grad, input_, n, dim, norm, 'fft_ifft', keyset)
        return (grad_input,)


class Fft_fftnBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_fftnBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._s = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        s = self._s
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_c2c_backward_helper(grad, input_, s, dim, norm, 'fft_ifftn', keyset)
        return (grad_input,)


class Fft_fftshiftBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_fftshiftBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._dim = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        dim = self._dim
        with _grad_context(keyset):
            grad_input = _fft_shift_backward_helper(grad, input_, dim, 'fft_ifftshift', keyset)
        return (grad_input,)


class Fft_hfftBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_hfftBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._n = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        n = self._n
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_c2r_backward_helper(grad, input_, n, dim, norm, 'fft_ihfft', keyset)
        return (grad_input,)


class Fft_ifft2Backward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_ifft2Backward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._s = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        s = self._s
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_c2c_backward_helper(grad, input_, s, dim, norm, 'fft_fft2', keyset)
        return (grad_input,)


class Fft_ifftBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_ifftBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._n = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        n = self._n
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_c2c_backward_helper(grad, input_, n, dim, norm, 'fft_fft', keyset)
        return (grad_input,)


class Fft_ifftnBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_ifftnBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._s = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        s = self._s
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_c2c_backward_helper(grad, input_, s, dim, norm, 'fft_fftn', keyset)
        return (grad_input,)


class Fft_ifftshiftBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_ifftshiftBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._dim = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        dim = self._dim
        with _grad_context(keyset):
            grad_input = _fft_shift_backward_helper(grad, input_, dim, 'fft_fftshift', keyset)
        return (grad_input,)


class Fft_ihfftBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_ihfftBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._n = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        n = self._n
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_r2c_backward_helper(grad, input_, n, dim, norm, 'fft_hfft', keyset)
        return (grad_input,)


class Fft_irfft2Backward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_irfft2Backward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._s = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        s = self._s
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_c2r_backward_helper(grad, input_, s, dim, norm, 'fft_rfft2', keyset)
        return (grad_input,)


class Fft_irfftBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_irfftBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._n = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        n = self._n
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_c2r_backward_helper(grad, input_, n, dim, norm, 'fft_rfft', keyset)
        return (grad_input,)


class Fft_irfftnBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_irfftnBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._s = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        s = self._s
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_c2r_backward_helper(grad, input_, s, dim, norm, 'fft_rfftn', keyset)
        return (grad_input,)


class Fft_rfft2Backward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_rfft2Backward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._s = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        s = self._s
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_r2c_backward_helper(grad, input_, s, dim, norm, 'fft_irfft2', keyset)
        return (grad_input,)


class Fft_rfftBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_rfftBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._n = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        n = self._n
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_r2c_backward_helper(grad, input_, n, dim, norm, 'fft_irfft', keyset)
        return (grad_input,)


class Fft_rfftnBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Fft_rfftnBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._s = None
        self._dim = None
        self._norm = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        s = self._s
        dim = self._dim
        norm = self._norm
        with _grad_context(keyset):
            grad_input = _fft_r2c_backward_helper(grad, input_, s, dim, norm, 'fft_irfftn', keyset)
        return (grad_input,)


class FlattenBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='FlattenBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._start_dim = None
        self._end_dim = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        start_dim = self._start_dim
        end_dim = self._end_dim
        with _grad_context(keyset):
            grad_input = redispatch("reshape", keyset, grad, input_.shape)
        return (grad_input,)


class Floor_divideBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Floor_divideBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._saved_other_idx = None

    def _save(self, *, self_=None, other=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if other is not None:
            self._saved_other_idx = len(tensors)
            tensors.append(other)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]
        if self._saved_other_idx is not None:
            self._saved_fields['other'] = self._saved_tensors_list[self._saved_other_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        other = _saved[self._saved_other_idx]
        with _grad_context(keyset):
            grad_self = redispatch("mul", keyset, grad, _scalar_tensor_like(self_, 0.0))
            grad_other = redispatch("mul", keyset, grad, _scalar_tensor_like(other, 0.0))
        return (grad_self, grad_other,)


class GetitemBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='GetitemBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._key = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        key = self._key
        with _grad_context(keyset):
            grad_self = _getitem_backward_helper(grad, self_, key, keyset)
        return (grad_self,)


class Grid_sampleBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Grid_sampleBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_grid_idx = None
        self._saved_self_idx = None
        self._mode = None
        self._padding_mode = None
        self._align_corners = None

    def _save(self, *, grid=None, self_=None):
        tensors = []
        if grid is not None:
            self._saved_grid_idx = len(tensors)
            tensors.append(grid)
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_grid_idx is not None:
            self._saved_fields['grid'] = self._saved_tensors_list[self._saved_grid_idx]
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        grid = _saved[self._saved_grid_idx]
        self_ = _saved[self._saved_self_idx]
        mode = self._mode
        padding_mode = self._padding_mode
        align_corners = self._align_corners
        with _grad_context(keyset):
            grad_self, grad_grid = _grid_sample_backward_all(grad, self_, grid, mode, padding_mode, align_corners, keyset)
        return (grad_self, grad_grid,)


class Group_normBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Group_normBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._saved_weight_idx = None
        self._num_groups = None
        self._eps = None

    def _save(self, *, input_=None, weight=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if weight is not None:
            self._saved_weight_idx = len(tensors)
            tensors.append(weight)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]
        if self._saved_weight_idx is not None:
            self._saved_fields['weight'] = self._saved_tensors_list[self._saved_weight_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        weight = _saved[self._saved_weight_idx] if self._saved_weight_idx is not None else None
        num_groups = self._num_groups
        eps = self._eps
        with _grad_context(keyset):
            grad_input = _group_norm_grad_input(grad, input_, num_groups, weight, eps, keyset)
        return (grad_input,)


class HeavisideBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='HeavisideBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._saved_other_idx = None

    def _save(self, *, self_=None, other=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if other is not None:
            self._saved_other_idx = len(tensors)
            tensors.append(other)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]
        if self._saved_other_idx is not None:
            self._saved_fields['other'] = self._saved_tensors_list[self._saved_other_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        other = _saved[self._saved_other_idx]
        with _grad_context(keyset):
            grad_self = redispatch("mul", keyset, grad, _scalar_tensor_like(self_, 0.0))
            grad_other = redispatch("mul", keyset, grad, _scalar_tensor_like(other, 0.0))
        return (grad_self, grad_other,)


class HstackBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='HstackBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        tensors = list(self.inputs)
        with _grad_context(keyset):
            grad_tensors = _hstack_backward_helper(grad, tensors, keyset)
        return grad_tensors


class InnerBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='InnerBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_other_idx = None
        self._saved_self_idx = None

    def _save(self, *, other=None, self_=None):
        tensors = []
        if other is not None:
            self._saved_other_idx = len(tensors)
            tensors.append(other)
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_other_idx is not None:
            self._saved_fields['other'] = self._saved_tensors_list[self._saved_other_idx]
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        other = _saved[self._saved_other_idx]
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self, grad_other = _inner_backward_all(grad, self_, other, keyset)
        return (grad_self, grad_other,)


class Instance_normBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Instance_normBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._weight = None
        self._bias = None
        self._running_mean = None
        self._running_var = None
        self._use_input_stats = None
        self._momentum = None
        self._eps = None
        self._cudnn_enabled = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        weight = self._weight
        bias = self._bias
        running_mean = self._running_mean
        running_var = self._running_var
        use_input_stats = self._use_input_stats
        momentum = self._momentum
        eps = self._eps
        cudnn_enabled = self._cudnn_enabled
        with _grad_context(keyset):
            grad_input = _instance_norm_backward_helper(grad, input_, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled, keyset)
        return (grad_input,)


class Layer_normBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Layer_normBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_bias_idx = None
        self._saved_input_idx = None
        self._saved_weight_idx = None
        self._normalized_shape = None
        self._eps = None

    def _save(self, *, bias=None, input_=None, weight=None):
        tensors = []
        if bias is not None:
            self._saved_bias_idx = len(tensors)
            tensors.append(bias)
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if weight is not None:
            self._saved_weight_idx = len(tensors)
            tensors.append(weight)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_bias_idx is not None:
            self._saved_fields['bias'] = self._saved_tensors_list[self._saved_bias_idx]
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]
        if self._saved_weight_idx is not None:
            self._saved_fields['weight'] = self._saved_tensors_list[self._saved_weight_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        bias = _saved[self._saved_bias_idx] if self._saved_bias_idx is not None else None
        input_ = _saved[self._saved_input_idx]
        weight = _saved[self._saved_weight_idx] if self._saved_weight_idx is not None else None
        normalized_shape = self._normalized_shape
        eps = self._eps
        with _grad_context(keyset):
            grad_input, grad_weight, grad_bias = _layer_norm_backward_all(grad, input_, normalized_shape, weight, bias, eps, keyset)
        return (grad_input, grad_weight, grad_bias,)


class Linalg_choleskyBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_choleskyBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._upper = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        upper = self._upper
        with _grad_context(keyset):
            grad_self = _linalg_cholesky_backward_helper(grad, self_, upper, keyset)
        return (grad_self,)


class Linalg_condBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_condBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._p = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        p = self._p
        with _grad_context(keyset):
            grad_input = _linalg_cond_backward_helper(grad, input_, p, keyset)
        return (grad_input,)


class Linalg_detBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_detBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = _linalg_det_backward_helper(grad, self_, keyset)
        return (grad_self,)


class Linalg_eigvalsBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_eigvalsBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = redispatch("mul", keyset, grad, _scalar_tensor_like(self_, 0.0))
        return (grad_self,)


class Linalg_eigvalshBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_eigvalshBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._UPLO = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        UPLO = self._UPLO
        with _grad_context(keyset):
            grad_input = _linalg_eigvalsh_backward_helper(grad, input_, UPLO, keyset)
        return (grad_input,)


class Linalg_invBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_invBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = _linalg_inv_grad(grad, self_, keyset)
        return (grad_self,)


class Linalg_matrix_normBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_matrix_normBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._ord = None
        self._dim = None
        self._keepdim = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        ord = self._ord
        dim = self._dim
        keepdim = self._keepdim
        with _grad_context(keyset):
            grad_input = _linalg_matrix_norm_backward_helper(grad, input_, ord, dim, keepdim, keyset)
        return (grad_input,)


class Linalg_matrix_powerBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_matrix_powerBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._n = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        n = self._n
        with _grad_context(keyset):
            grad_input = _matrix_power_backward_helper(grad, input_, n, keyset)
        return (grad_input,)


class Linalg_matrix_rankBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_matrix_rankBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._atol = None
        self._rtol = None
        self._hermitian = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        atol = self._atol
        rtol = self._rtol
        hermitian = self._hermitian
        with _grad_context(keyset):
            grad_input = redispatch("mul", keyset, grad, _scalar_tensor_like(input_, 0.0))
        return (grad_input,)


class Linalg_normBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_normBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._ord = None
        self._dim = None
        self._keepdim = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        ord = self._ord
        dim = self._dim
        keepdim = self._keepdim
        with _grad_context(keyset):
            grad_input = _linalg_norm_backward_helper(grad, input_, ord, dim, keepdim, keyset)
        return (grad_input,)


class Linalg_slogdetBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_slogdetBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = _linalg_slogdet_backward_helper(grad, self_, keyset)
        return (grad_self,)


class Linalg_solveBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_solveBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_other_idx = None
        self._saved_self_idx = None
        self._left = None

    def _save(self, *, other=None, self_=None):
        tensors = []
        if other is not None:
            self._saved_other_idx = len(tensors)
            tensors.append(other)
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_other_idx is not None:
            self._saved_fields['other'] = self._saved_tensors_list[self._saved_other_idx]
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        other = _saved[self._saved_other_idx]
        self_ = _saved[self._saved_self_idx]
        left = self._left
        with _grad_context(keyset):
            grad_self, grad_other = _linalg_solve_backward_all(grad, self_, other, left, keyset)
        return (grad_self, grad_other,)


class Linalg_svdvalsBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_svdvalsBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = _linalg_svdvals_grad(grad, self_, keyset)
        return (grad_self,)


class Linalg_tensorinvBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_tensorinvBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._ind = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        ind = self._ind
        with _grad_context(keyset):
            grad_input = _linalg_tensorinv_backward_helper(grad, input_, ind, keyset)
        return (grad_input,)


class Linalg_tensorsolveBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_tensorsolveBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._dims = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        dims = self._dims
        with _grad_context(keyset):
            grad_input = redispatch("mul", keyset, grad, _scalar_tensor_like(input_, 0.0))
        return (grad_input,)


class Linalg_vanderBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Linalg_vanderBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_x_idx = None
        self._N = None

    def _save(self, *, x=None):
        tensors = []
        if x is not None:
            self._saved_x_idx = len(tensors)
            tensors.append(x)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_x_idx is not None:
            self._saved_fields['x'] = self._saved_tensors_list[self._saved_x_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        x = _saved[self._saved_x_idx]
        N = self._N
        with _grad_context(keyset):
            grad_x = _linalg_vander_backward_helper(grad, x, N, keyset)
        return (grad_x,)


class Log_softmaxBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Log_softmaxBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._dim = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        dim = self._dim
        with _grad_context(keyset):
            grad_self = _log_softmax_grad(grad, self_, dim, keyset)
        return (grad_self,)


class Matrix_powerBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Matrix_powerBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._n = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        n = self._n
        with _grad_context(keyset):
            grad_input = _matrix_power_backward_helper(grad, input_, n, keyset)
        return (grad_input,)


class Max_pool1dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Max_pool1dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._saved_result_idx = None
        self._kernel_size = None
        self._stride = None
        self._padding = None
        self._dilation = None
        self._ceil_mode = None
        self._return_indices = None

    def _save(self, *, self_=None, result=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if result is not None:
            self._saved_result_idx = len(tensors)
            tensors.append(result)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]
        if self._saved_result_idx is not None:
            self._saved_fields['result'] = self._saved_tensors_list[self._saved_result_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        result = _saved[self._saved_result_idx]
        kernel_size = self._kernel_size
        stride = self._stride
        padding = self._padding
        dilation = self._dilation
        ceil_mode = self._ceil_mode
        return_indices = self._return_indices
        with _grad_context(keyset):
            grad_self = _max_pool1d_backward_helper(grad, self_, result, kernel_size, stride, padding, dilation, ceil_mode, keyset)
        return (grad_self,)


class Max_pool3dBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Max_pool3dBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._saved_result_idx = None
        self._kernel_size = None
        self._stride = None
        self._padding = None
        self._dilation = None
        self._ceil_mode = None
        self._return_indices = None

    def _save(self, *, self_=None, result=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if result is not None:
            self._saved_result_idx = len(tensors)
            tensors.append(result)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]
        if self._saved_result_idx is not None:
            self._saved_fields['result'] = self._saved_tensors_list[self._saved_result_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        result = _saved[self._saved_result_idx]
        kernel_size = self._kernel_size
        stride = self._stride
        padding = self._padding
        dilation = self._dilation
        ceil_mode = self._ceil_mode
        return_indices = self._return_indices
        with _grad_context(keyset):
            grad_self = _max_pool3d_backward_helper(grad, self_, result, kernel_size, stride, padding, dilation, ceil_mode, keyset)
        return (grad_self,)


class MoveaxisBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='MoveaxisBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._source = None
        self._destination = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        source = self._source
        destination = self._destination
        with _grad_context(keyset):
            grad_input = _movedim_backward_helper(grad, input_, source, destination, keyset)
        return (grad_input,)


class MovedimBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='MovedimBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._source = None
        self._destination = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        source = self._source
        destination = self._destination
        with _grad_context(keyset):
            grad_input = _movedim_backward_helper(grad, input_, source, destination, keyset)
        return (grad_input,)


class NanmeanBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='NanmeanBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._dim = None
        self._keepdim = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        dim = self._dim
        keepdim = self._keepdim
        with _grad_context(keyset):
            grad_input = _nanmean_backward_helper(grad, input_, dim, keepdim, keyset)
        return (grad_input,)


class NanquantileBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='NanquantileBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._q = None
        self._dim = None
        self._keepdim = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        q = self._q
        dim = self._dim
        keepdim = self._keepdim
        with _grad_context(keyset):
            grad_input = _quantile_backward_helper(grad, input_, q, dim, keepdim, keyset)
        return (grad_input,)


class NarrowBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='NarrowBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._dim = None
        self._start = None
        self._length = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        dim = self._dim
        start = self._start
        length = self._length
        with _grad_context(keyset):
            grad_input = _narrow_backward_helper(grad, input_, dim, start, length, keyset)
        return (grad_input,)


class NormalizeBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='NormalizeBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._p = None
        self._dim = None
        self._eps = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        p = self._p
        dim = self._dim
        eps = self._eps
        with _grad_context(keyset):
            grad_input = _normalize_backward_helper(grad, input_, p, dim, eps, keyset)
        return (grad_input,)


class OuterBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='OuterBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_other_idx = None
        self._saved_self_idx = None

    def _save(self, *, other=None, self_=None):
        tensors = []
        if other is not None:
            self._saved_other_idx = len(tensors)
            tensors.append(other)
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_other_idx is not None:
            self._saved_fields['other'] = self._saved_tensors_list[self._saved_other_idx]
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        other = _saved[self._saved_other_idx]
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = redispatch("matmul", keyset, grad, other)
            grad_other = redispatch("matmul", keyset, redispatch("transpose", keyset, grad, 0, 1), self_)
        return (grad_self, grad_other,)


class PadBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='PadBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._pad = None
        self._mode = None
        self._value = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        pad = self._pad
        mode = self._mode
        value = self._value
        with _grad_context(keyset):
            grad_input = _pad_backward_helper(grad, input_, pad, keyset)
        return (grad_input,)


class Pad_sequenceBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Pad_sequenceBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._batch_first = None
        self._padding_value = None
        self._padding_side = None

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        sequences = list(self.inputs)
        batch_first = self._batch_first
        padding_value = self._padding_value
        padding_side = self._padding_side
        with _grad_context(keyset):
            grad_sequences = _pad_sequence_backward_helper(grad, sequences, batch_first, padding_value, padding_side, keyset)
        return grad_sequences


class PreluBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='PreluBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._saved_weight_idx = None

    def _save(self, *, self_=None, weight=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if weight is not None:
            self._saved_weight_idx = len(tensors)
            tensors.append(weight)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]
        if self._saved_weight_idx is not None:
            self._saved_fields['weight'] = self._saved_tensors_list[self._saved_weight_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        weight = _saved[self._saved_weight_idx]
        with _grad_context(keyset):
            grad_self = _prelu_grad_input(grad, self_, weight, keyset)
            grad_weight = reduce_grad(_prelu_grad_weight(grad, self_, weight, keyset), weight.shape)
        return (grad_self, grad_weight,)


class QuantileBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='QuantileBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._q = None
        self._dim = None
        self._keepdim = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        q = self._q
        dim = self._dim
        keepdim = self._keepdim
        with _grad_context(keyset):
            grad_input = _quantile_backward_helper(grad, input_, q, dim, keepdim, keyset)
        return (grad_input,)


class Relu6Backward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Relu6Backward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = _hardtanh_grad(grad, self_, 0.0, 6.0, keyset)
        return (grad_self,)


class Repeat_interleaveBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Repeat_interleaveBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._repeats = None
        self._dim = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        repeats = self._repeats
        dim = self._dim
        with _grad_context(keyset):
            grad_input = _repeat_interleave_backward_helper(grad, input_, repeats, dim, keyset)
        return (grad_input,)


class Rms_normBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Rms_normBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._saved_weight_idx = None
        self._normalized_shape = None
        self._eps = None

    def _save(self, *, input_=None, weight=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if weight is not None:
            self._saved_weight_idx = len(tensors)
            tensors.append(weight)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]
        if self._saved_weight_idx is not None:
            self._saved_fields['weight'] = self._saved_tensors_list[self._saved_weight_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        weight = _saved[self._saved_weight_idx] if self._saved_weight_idx is not None else None
        normalized_shape = self._normalized_shape
        eps = self._eps
        with _grad_context(keyset):
            grad_input = _rms_norm_grad_input(grad, input_, normalized_shape, weight, eps, keyset)
        return (grad_input,)


class Row_stackBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Row_stackBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        tensors = list(self.inputs)
        with _grad_context(keyset):
            grad_tensors = _vstack_backward_helper(grad, tensors, keyset)
        return grad_tensors


class SeluBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='SeluBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = _selu_grad(grad, self_, keyset)
        return (grad_self,)


class SignbitBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='SignbitBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = redispatch("mul", keyset, grad, _scalar_tensor_like(self_, 0.0))
        return (grad_self,)


class SoftsignBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='SoftsignBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = _softsign_grad(grad, self_, keyset)
        return (grad_self,)


class Special_digammaBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Special_digammaBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = redispatch("mul", keyset, grad, redispatch("special_polygamma", keyset, 1, self_))
        return (grad_self,)


class Special_erfinvBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Special_erfinvBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = _special_erfinv_grad(grad, self_, keyset)
        return (grad_self,)


class Special_gammaincBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Special_gammaincBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_other_idx = None
        self._saved_self_idx = None

    def _save(self, *, other=None, self_=None):
        tensors = []
        if other is not None:
            self._saved_other_idx = len(tensors)
            tensors.append(other)
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_other_idx is not None:
            self._saved_fields['other'] = self._saved_tensors_list[self._saved_other_idx]
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        other = _saved[self._saved_other_idx]
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self, grad_other = _special_gammainc_backward_all(grad, self_, other, keyset)
        return (grad_self, grad_other,)


class Special_gammainccBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Special_gammainccBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_other_idx = None
        self._saved_self_idx = None

    def _save(self, *, other=None, self_=None):
        tensors = []
        if other is not None:
            self._saved_other_idx = len(tensors)
            tensors.append(other)
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_other_idx is not None:
            self._saved_fields['other'] = self._saved_tensors_list[self._saved_other_idx]
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        other = _saved[self._saved_other_idx]
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self, grad_other = _special_gammaincc_backward_all(grad, self_, other, keyset)
        return (grad_self, grad_other,)


class Special_gammalnBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Special_gammalnBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = redispatch("mul", keyset, grad, redispatch("special_digamma", keyset, self_))
        return (grad_self,)


class Special_i0Backward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Special_i0Backward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = redispatch("mul", keyset, grad, redispatch("special_i1", keyset, self_))
        return (grad_self,)


class Special_logitBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Special_logitBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._eps = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        eps = self._eps
        with _grad_context(keyset):
            grad_input = _special_logit_backward_helper(grad, input_, eps, keyset)
        return (grad_input,)


class Special_multigammalnBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Special_multigammalnBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._p = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        p = self._p
        with _grad_context(keyset):
            grad_self = _special_multigammaln_backward_helper(grad, self_, p, keyset)
        return (grad_self,)


class Special_ndtrBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Special_ndtrBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = _special_ndtr_grad(grad, self_, keyset)
        return (grad_self,)


class Special_polygammaBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Special_polygammaBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None
        self._n = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        n = self._n
        with _grad_context(keyset):
            grad_self = _special_polygamma_backward_helper(grad, n, self_, keyset)
        return (grad_self,)


class Special_sincBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Special_sincBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = _special_sinc_grad(grad, self_, keyset)
        return (grad_self,)


class Special_xlogyBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Special_xlogyBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_other_idx = None
        self._saved_self_idx = None

    def _save(self, *, other=None, self_=None):
        tensors = []
        if other is not None:
            self._saved_other_idx = len(tensors)
            tensors.append(other)
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_other_idx is not None:
            self._saved_fields['other'] = self._saved_tensors_list[self._saved_other_idx]
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        other = _saved[self._saved_other_idx]
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self, grad_other = _special_xlogy_backward_all(grad, self_, other, keyset)
        return (grad_self, grad_other,)


class SquareBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='SquareBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_self_idx = None

    def _save(self, *, self_=None):
        tensors = []
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self = redispatch("mul", keyset, grad, redispatch("mul", keyset, _scalar_tensor_like(self_, 2.0), self_))
        return (grad_self,)


class Take_along_dimBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='Take_along_dimBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_indices_idx = None
        self._saved_input_idx = None
        self._dim = None

    def _save(self, *, indices=None, input_=None):
        tensors = []
        if indices is not None:
            self._saved_indices_idx = len(tensors)
            tensors.append(indices)
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_indices_idx is not None:
            self._saved_fields['indices'] = self._saved_tensors_list[self._saved_indices_idx]
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        indices = _saved[self._saved_indices_idx]
        input_ = _saved[self._saved_input_idx]
        dim = self._dim
        with _grad_context(keyset):
            grad_input = _take_along_dim_backward_helper(grad, input_, indices, dim, keyset)
        return (grad_input,)


class TensordotBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='TensordotBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_other_idx = None
        self._saved_self_idx = None
        self._dims = None

    def _save(self, *, other=None, self_=None):
        tensors = []
        if other is not None:
            self._saved_other_idx = len(tensors)
            tensors.append(other)
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_other_idx is not None:
            self._saved_fields['other'] = self._saved_tensors_list[self._saved_other_idx]
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        other = _saved[self._saved_other_idx]
        self_ = _saved[self._saved_self_idx]
        dims = self._dims
        with _grad_context(keyset):
            grad_self, grad_other = _tensordot_backward_all(grad, self_, other, dims, keyset)
        return (grad_self, grad_other,)


class TileBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='TileBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._dims = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        dims = self._dims
        with _grad_context(keyset):
            grad_input = _tile_backward_helper(grad, input_, dims, keyset)
        return (grad_input,)


class True_divideBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='True_divideBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_other_idx = None
        self._saved_self_idx = None

    def _save(self, *, other=None, self_=None):
        tensors = []
        if other is not None:
            self._saved_other_idx = len(tensors)
            tensors.append(other)
        if self_ is not None:
            self._saved_self_idx = len(tensors)
            tensors.append(self_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_other_idx is not None:
            self._saved_fields['other'] = self._saved_tensors_list[self._saved_other_idx]
        if self._saved_self_idx is not None:
            self._saved_fields['self'] = self._saved_tensors_list[self._saved_self_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        other = _saved[self._saved_other_idx]
        self_ = _saved[self._saved_self_idx]
        with _grad_context(keyset):
            grad_self, grad_other = _div_backward_all(grad, self_, other, keyset)
        return (grad_self, grad_other,)


class UnflattenBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='UnflattenBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset
        self._saved_input_idx = None
        self._dim = None
        self._sizes = None

    def _save(self, *, input_=None):
        tensors = []
        if input_ is not None:
            self._saved_input_idx = len(tensors)
            tensors.append(input_)
        if tensors:
            super().save_for_backward(*tensors)
        if self._saved_input_idx is not None:
            self._saved_fields['input'] = self._saved_tensors_list[self._saved_input_idx]

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        _saved = self.saved_tensors()
        input_ = _saved[self._saved_input_idx]
        dim = self._dim
        sizes = self._sizes
        with _grad_context(keyset):
            grad_input = redispatch("reshape", keyset, grad, input_.shape)
        return (grad_input,)


class VstackBackward0(Node):
    def __init__(self, inputs, *, raw_keyset=None, active_keyset=None):
        super().__init__(None, inputs, name='VstackBackward0')
        self._raw_keyset = raw_keyset
        self._active_keyset = active_keyset

    def backward(self, grad):
        from .._dispatch.dispatcher import current_dispatch_keyset
        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)
        tensors = list(self.inputs)
        with _grad_context(keyset):
            grad_tensors = _vstack_backward_helper(grad, tensors, keyset)
        return grad_tensors
