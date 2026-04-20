"""Legacy forward autograd wrappers — hand-maintained.

These wrappers are NOT generated from derivatives.yaml.
"""
from __future__ import annotations

from ..autograd.grad_mode import GradMode
from ..autograd.anomaly_mode import annotate_node_creation
from .._backends.autograd import _strip_autograd_keys
from .._dispatch.dispatcher import current_dispatch_keyset, redispatch
from . import functions as _F
from . import functions_legacy as _FL


def sum_to_size_autograd_post(result, self_, size, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self_.requires_grad):
        grad_fn = _FL.SumToSizeBackward0((self_,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self_)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def pow_autograd_post(result, self_, exponent, *, raw_keyset, active_keyset, **_kwargs):
    from .._tensor import Tensor as _Tensor
    if isinstance(exponent, _Tensor):
        return pow_tensor_tensor_autograd_post(result, self_, exponent, raw_keyset=raw_keyset, active_keyset=active_keyset, **_kwargs)
    return pow_tensor_scalar_autograd_post(result, self_, exponent, raw_keyset=raw_keyset, active_keyset=active_keyset, **_kwargs)

def adaptive_avg_pool1d_autograd(self, output_size, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("adaptive_avg_pool1d", raw_keyset, self, output_size, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Adaptive_avg_pool1dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._output_size = output_size
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def adaptive_avg_pool1d_autograd_post(result, self, output_size, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Adaptive_avg_pool1dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._output_size = output_size
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def adaptive_avg_pool2d_autograd(self, output_size, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("adaptive_avg_pool2d", raw_keyset, self, output_size, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Adaptive_avg_pool2dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._output_size = output_size
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def adaptive_avg_pool2d_autograd_post(result, self, output_size, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Adaptive_avg_pool2dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._output_size = output_size
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def adaptive_avg_pool3d_autograd(self, output_size, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("adaptive_avg_pool3d", raw_keyset, self, output_size, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Adaptive_avg_pool3dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._output_size = output_size
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def adaptive_avg_pool3d_autograd_post(result, self, output_size, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Adaptive_avg_pool3dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._output_size = output_size
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def adaptive_max_pool1d_autograd(self, output_size, return_indices=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("adaptive_max_pool1d", raw_keyset, self, output_size, return_indices, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Adaptive_max_pool1dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self, result=result)
        grad_fn._output_size = output_size
        grad_fn._return_indices = return_indices
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def adaptive_max_pool1d_autograd_post(result, self, output_size, return_indices=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Adaptive_max_pool1dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self, result=result)
        grad_fn._output_size = output_size
        grad_fn._return_indices = return_indices
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def affine_grid_autograd(theta, size, align_corners=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("affine_grid", raw_keyset, theta, size, align_corners, **_kwargs)
    if GradMode.enabled and (theta.requires_grad):
        grad_fn = _FL.Affine_gridBackward0((theta,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(theta=theta)
        grad_fn._size = size
        grad_fn._align_corners = align_corners
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def affine_grid_autograd_post(result, theta, size, align_corners=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (theta.requires_grad):
        grad_fn = _FL.Affine_gridBackward0((theta,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(theta=theta)
        grad_fn._size = size
        grad_fn._align_corners = align_corners
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def avg_pool1d_autograd(self, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("avg_pool1d", raw_keyset, self, kernel_size, stride, padding, ceil_mode, count_include_pad, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Avg_pool1dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._kernel_size = kernel_size
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._ceil_mode = ceil_mode
        grad_fn._count_include_pad = count_include_pad
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def avg_pool1d_autograd_post(result, self, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Avg_pool1dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._kernel_size = kernel_size
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._ceil_mode = ceil_mode
        grad_fn._count_include_pad = count_include_pad
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def batch_norm_autograd(input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("batch_norm", raw_keyset, input, running_mean, running_var, weight, bias, training, momentum, eps, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Batch_normBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input, weight=weight)
        grad_fn._training = training
        grad_fn._momentum = momentum
        grad_fn._eps = eps
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def batch_norm_autograd_post(result, input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Batch_normBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input, weight=weight)
        grad_fn._training = training
        grad_fn._momentum = momentum
        grad_fn._eps = eps
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def broadcast_to_autograd(input, shape, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("broadcast_to", raw_keyset, input, shape, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Broadcast_toBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._shape = shape
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def broadcast_to_autograd_post(result, input, shape, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Broadcast_toBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._shape = shape
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def cdist_autograd(x1, x2, p=2.0, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("cdist", raw_keyset, x1, x2, p, **_kwargs)
    if GradMode.enabled and (getattr(x1, 'requires_grad', False) or getattr(x2, 'requires_grad', False)):
        grad_fn = _FL.CdistBackward0((x1, x2,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(x1=x1, x2=x2)
        grad_fn._p = p
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def cdist_autograd_post(result, x1, x2, p=2.0, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(x1, 'requires_grad', False) or getattr(x2, 'requires_grad', False)):
        grad_fn = _FL.CdistBackward0((x1, x2,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(x1=x1, x2=x2)
        grad_fn._p = p
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def column_stack_autograd(tensors, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("column_stack", raw_keyset, tensors, **_kwargs)
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.Column_stackBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def column_stack_autograd_post(result, tensors, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.Column_stackBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def concat_autograd(tensors, dim=0, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("concat", raw_keyset, tensors, dim, **_kwargs)
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.ConcatBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def concat_autograd_post(result, tensors, dim=0, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.ConcatBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def concatenate_autograd(tensors, dim=0, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("concatenate", raw_keyset, tensors, dim, **_kwargs)
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.ConcatenateBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def concatenate_autograd_post(result, tensors, dim=0, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.ConcatenateBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def contiguous_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("contiguous", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.ContiguousBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def contiguous_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.ContiguousBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def conv1d_autograd(input, weight, bias=None, stride=None, padding=None, dilation=None, groups=1, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("conv1d", raw_keyset, input, weight, bias, stride, padding, dilation, groups, **_kwargs)
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or getattr(weight, 'requires_grad', False) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Conv1dBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._dilation = dilation
        grad_fn._groups = groups
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def conv1d_autograd_post(result, input, weight, bias=None, stride=None, padding=None, dilation=None, groups=1, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or getattr(weight, 'requires_grad', False) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Conv1dBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._dilation = dilation
        grad_fn._groups = groups
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def conv2d_autograd(input, weight, bias=None, stride=None, padding=None, dilation=None, groups=1, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("conv2d", raw_keyset, input, weight, bias, stride, padding, dilation, groups, **_kwargs)
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or getattr(weight, 'requires_grad', False) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Conv2dBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._dilation = dilation
        grad_fn._groups = groups
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def conv2d_autograd_post(result, input, weight, bias=None, stride=None, padding=None, dilation=None, groups=1, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or getattr(weight, 'requires_grad', False) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Conv2dBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._dilation = dilation
        grad_fn._groups = groups
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def conv3d_autograd(input, weight, bias, stride, padding, dilation, groups=1, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("conv3d", raw_keyset, input, weight, bias, stride, padding, dilation, groups, **_kwargs)
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or getattr(weight, 'requires_grad', False) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Conv3dBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._dilation = dilation
        grad_fn._groups = groups
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def conv3d_autograd_post(result, input, weight, bias, stride, padding, dilation, groups=1, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or getattr(weight, 'requires_grad', False) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Conv3dBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._dilation = dilation
        grad_fn._groups = groups
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def conv_transpose1d_autograd(input, weight, bias=None, stride=None, padding=None, output_padding=None, groups=1, dilation=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("conv_transpose1d", raw_keyset, input, weight, bias, stride, padding, output_padding, groups, dilation, **_kwargs)
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or getattr(weight, 'requires_grad', False) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Conv_transpose1dBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._output_padding = output_padding
        grad_fn._groups = groups
        grad_fn._dilation = dilation
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def conv_transpose1d_autograd_post(result, input, weight, bias=None, stride=None, padding=None, output_padding=None, groups=1, dilation=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or getattr(weight, 'requires_grad', False) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Conv_transpose1dBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._output_padding = output_padding
        grad_fn._groups = groups
        grad_fn._dilation = dilation
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def conv_transpose2d_autograd(input, weight, bias=None, stride=None, padding=None, output_padding=None, groups=1, dilation=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("conv_transpose2d", raw_keyset, input, weight, bias, stride, padding, output_padding, groups, dilation, **_kwargs)
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or getattr(weight, 'requires_grad', False) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Conv_transpose2dBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._output_padding = output_padding
        grad_fn._groups = groups
        grad_fn._dilation = dilation
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def conv_transpose2d_autograd_post(result, input, weight, bias=None, stride=None, padding=None, output_padding=None, groups=1, dilation=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or getattr(weight, 'requires_grad', False) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Conv_transpose2dBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._output_padding = output_padding
        grad_fn._groups = groups
        grad_fn._dilation = dilation
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def conv_transpose3d_autograd(input, weight, bias=None, stride=None, padding=None, output_padding=None, groups=1, dilation=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("conv_transpose3d", raw_keyset, input, weight, bias, stride, padding, output_padding, groups, dilation, **_kwargs)
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or getattr(weight, 'requires_grad', False) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Conv_transpose3dBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._output_padding = output_padding
        grad_fn._groups = groups
        grad_fn._dilation = dilation
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def conv_transpose3d_autograd_post(result, input, weight, bias=None, stride=None, padding=None, output_padding=None, groups=1, dilation=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or getattr(weight, 'requires_grad', False) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Conv_transpose3dBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._output_padding = output_padding
        grad_fn._groups = groups
        grad_fn._dilation = dilation
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def cross_autograd(self, other, dim=-1, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("cross", raw_keyset, self, other, dim, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.CrossBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def cross_autograd_post(result, self, other, dim=-1, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.CrossBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def ctc_loss_autograd(self, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("ctc_loss", raw_keyset, self, targets, input_lengths, target_lengths, blank, reduction, zero_infinity, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Ctc_lossBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._targets = targets
        grad_fn._input_lengths = input_lengths
        grad_fn._target_lengths = target_lengths
        grad_fn._blank = blank
        grad_fn._reduction = reduction
        grad_fn._zero_infinity = zero_infinity
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def ctc_loss_autograd_post(result, self, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Ctc_lossBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._targets = targets
        grad_fn._input_lengths = input_lengths
        grad_fn._target_lengths = target_lengths
        grad_fn._blank = blank
        grad_fn._reduction = reduction
        grad_fn._zero_infinity = zero_infinity
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def det_autograd(input, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("det", raw_keyset, input, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.DetBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def det_autograd_post(result, input, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.DetBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def diag_autograd(input, diagonal=0, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("diag", raw_keyset, input, diagonal, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.DiagBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._diagonal = diagonal
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def diag_autograd_post(result, input, diagonal=0, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.DiagBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._diagonal = diagonal
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def diff_autograd(input, n=1, dim=-1, prepend=None, append=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("diff", raw_keyset, input, n, dim, prepend, append, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.DiffBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def diff_autograd_post(result, input, n=1, dim=-1, prepend=None, append=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.DiffBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def dstack_autograd(tensors, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("dstack", raw_keyset, tensors, **_kwargs)
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.DstackBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def dstack_autograd_post(result, tensors, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.DstackBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_fft2_autograd(input, s=None, dim=None, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_fft2", raw_keyset, input, s, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_fft2Backward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_fft2_autograd_post(result, input, s=None, dim=None, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_fft2Backward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_fft_autograd(input, n=None, dim=-1, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_fft", raw_keyset, input, n, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_fftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_fft_autograd_post(result, input, n=None, dim=-1, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_fftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_fftn_autograd(input, s=None, dim=None, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_fftn", raw_keyset, input, s, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_fftnBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_fftn_autograd_post(result, input, s=None, dim=None, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_fftnBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_fftshift_autograd(input, dim=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_fftshift", raw_keyset, input, dim, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_fftshiftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_fftshift_autograd_post(result, input, dim=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_fftshiftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_hfft_autograd(input, n=None, dim=-1, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_hfft", raw_keyset, input, n, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_hfftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_hfft_autograd_post(result, input, n=None, dim=-1, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_hfftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_ifft2_autograd(input, s=None, dim=None, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_ifft2", raw_keyset, input, s, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_ifft2Backward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_ifft2_autograd_post(result, input, s=None, dim=None, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_ifft2Backward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_ifft_autograd(input, n=None, dim=-1, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_ifft", raw_keyset, input, n, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_ifftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_ifft_autograd_post(result, input, n=None, dim=-1, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_ifftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_ifftn_autograd(input, s=None, dim=None, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_ifftn", raw_keyset, input, s, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_ifftnBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_ifftn_autograd_post(result, input, s=None, dim=None, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_ifftnBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_ifftshift_autograd(input, dim=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_ifftshift", raw_keyset, input, dim, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_ifftshiftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_ifftshift_autograd_post(result, input, dim=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_ifftshiftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_ihfft_autograd(input, n=None, dim=-1, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_ihfft", raw_keyset, input, n, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_ihfftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_ihfft_autograd_post(result, input, n=None, dim=-1, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_ihfftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_irfft2_autograd(input, s=None, dim=None, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_irfft2", raw_keyset, input, s, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_irfft2Backward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_irfft2_autograd_post(result, input, s=None, dim=None, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_irfft2Backward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_irfft_autograd(input, n=None, dim=-1, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_irfft", raw_keyset, input, n, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_irfftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_irfft_autograd_post(result, input, n=None, dim=-1, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_irfftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_irfftn_autograd(input, s=None, dim=None, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_irfftn", raw_keyset, input, s, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_irfftnBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_irfftn_autograd_post(result, input, s=None, dim=None, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_irfftnBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_rfft2_autograd(input, s=None, dim=None, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_rfft2", raw_keyset, input, s, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_rfft2Backward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_rfft2_autograd_post(result, input, s=None, dim=None, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_rfft2Backward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_rfft_autograd(input, n=None, dim=-1, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_rfft", raw_keyset, input, n, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_rfftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_rfft_autograd_post(result, input, n=None, dim=-1, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_rfftBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_rfftn_autograd(input, s=None, dim=None, norm=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("fft_rfftn", raw_keyset, input, s, dim, norm, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_rfftnBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def fft_rfftn_autograd_post(result, input, s=None, dim=None, norm=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Fft_rfftnBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._s = s
        grad_fn._dim = dim
        grad_fn._norm = norm
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def flatten_autograd(input, start_dim=0, end_dim=-1, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("flatten", raw_keyset, input, start_dim, end_dim, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.FlattenBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._start_dim = start_dim
        grad_fn._end_dim = end_dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def flatten_autograd_post(result, input, start_dim=0, end_dim=-1, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.FlattenBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._start_dim = start_dim
        grad_fn._end_dim = end_dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def floor_divide_autograd(self, other, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("floor_divide", raw_keyset, self, other, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.Floor_divideBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self, other=other)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def floor_divide_autograd_post(result, self, other, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.Floor_divideBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self, other=other)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def getitem_autograd(self, key, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("getitem", raw_keyset, self, key, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.GetitemBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._key = key
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def getitem_autograd_post(result, self, key, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.GetitemBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._key = key
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def grid_sample_autograd(self, grid, mode='bilinear', padding_mode='zeros', align_corners=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("grid_sample", raw_keyset, self, grid, mode, padding_mode, align_corners, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(grid, 'requires_grad', False)):
        grad_fn = _FL.Grid_sampleBackward0((self, grid,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(grid=grid, self_=self)
        grad_fn._mode = mode
        grad_fn._padding_mode = padding_mode
        grad_fn._align_corners = align_corners
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def grid_sample_autograd_post(result, self, grid, mode='bilinear', padding_mode='zeros', align_corners=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(grid, 'requires_grad', False)):
        grad_fn = _FL.Grid_sampleBackward0((self, grid,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(grid=grid, self_=self)
        grad_fn._mode = mode
        grad_fn._padding_mode = padding_mode
        grad_fn._align_corners = align_corners
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def group_norm_autograd(input, num_groups, weight, bias, eps=1e-5, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("group_norm", raw_keyset, input, num_groups, weight, bias, eps, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Group_normBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input, weight=weight)
        grad_fn._num_groups = num_groups
        grad_fn._eps = eps
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def group_norm_autograd_post(result, input, num_groups, weight, bias, eps=1e-5, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Group_normBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input, weight=weight)
        grad_fn._num_groups = num_groups
        grad_fn._eps = eps
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def heaviside_autograd(self, other, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("heaviside", raw_keyset, self, other, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.HeavisideBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self, other=other)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def heaviside_autograd_post(result, self, other, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.HeavisideBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self, other=other)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def hstack_autograd(tensors, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("hstack", raw_keyset, tensors, **_kwargs)
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.HstackBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def hstack_autograd_post(result, tensors, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.HstackBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def inner_autograd(self, other, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("inner", raw_keyset, self, other, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.InnerBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def inner_autograd_post(result, self, other, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.InnerBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def instance_norm_autograd(input, weight=None, bias=None, running_mean=None, running_var=None, use_input_stats=True, momentum=0.1, eps=1e-5, cudnn_enabled=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("instance_norm", raw_keyset, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Instance_normBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._weight = weight
        grad_fn._bias = bias
        grad_fn._running_mean = running_mean
        grad_fn._running_var = running_var
        grad_fn._use_input_stats = use_input_stats
        grad_fn._momentum = momentum
        grad_fn._eps = eps
        grad_fn._cudnn_enabled = cudnn_enabled
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def instance_norm_autograd_post(result, input, weight=None, bias=None, running_mean=None, running_var=None, use_input_stats=True, momentum=0.1, eps=1e-5, cudnn_enabled=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Instance_normBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._weight = weight
        grad_fn._bias = bias
        grad_fn._running_mean = running_mean
        grad_fn._running_var = running_var
        grad_fn._use_input_stats = use_input_stats
        grad_fn._momentum = momentum
        grad_fn._eps = eps
        grad_fn._cudnn_enabled = cudnn_enabled
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def layer_norm_autograd(input, normalized_shape, weight, bias, eps=1e-5, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("layer_norm", raw_keyset, input, normalized_shape, weight, bias, eps, **_kwargs)
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or (weight is not None and getattr(weight, 'requires_grad', False)) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Layer_normBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._normalized_shape = normalized_shape
        grad_fn._eps = eps
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def layer_norm_autograd_post(result, input, normalized_shape, weight, bias, eps=1e-5, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(input, 'requires_grad', False) or (weight is not None and getattr(weight, 'requires_grad', False)) or (bias is not None and getattr(bias, 'requires_grad', False))):
        _inputs = [x for x in (input, weight, bias,) if x is not None]
        grad_fn = _FL.Layer_normBackward0(_inputs, raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(bias=bias, input_=input, weight=weight)
        grad_fn._normalized_shape = normalized_shape
        grad_fn._eps = eps
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_cholesky_autograd(self, upper=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_cholesky", raw_keyset, self, upper, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Linalg_choleskyBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._upper = upper
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_cholesky_autograd_post(result, self, upper=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Linalg_choleskyBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._upper = upper
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_cond_autograd(input, p=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_cond", raw_keyset, input, p, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_condBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._p = p
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_cond_autograd_post(result, input, p=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_condBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._p = p
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_det_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_det", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Linalg_detBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_det_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Linalg_detBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_eigvals_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_eigvals", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Linalg_eigvalsBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_eigvals_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Linalg_eigvalsBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_eigvalsh_autograd(input, UPLO="L", **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_eigvalsh", raw_keyset, input, UPLO, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_eigvalshBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._UPLO = UPLO
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_eigvalsh_autograd_post(result, input, UPLO="L", *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_eigvalshBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._UPLO = UPLO
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_inv_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_inv", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Linalg_invBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_inv_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Linalg_invBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_matrix_norm_autograd(input, ord="fro", dim=None, keepdim=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_matrix_norm", raw_keyset, input, ord, dim, keepdim, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_matrix_normBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._ord = ord
        grad_fn._dim = dim
        grad_fn._keepdim = keepdim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_matrix_norm_autograd_post(result, input, ord="fro", dim=None, keepdim=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_matrix_normBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._ord = ord
        grad_fn._dim = dim
        grad_fn._keepdim = keepdim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_matrix_power_autograd(input, n, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_matrix_power", raw_keyset, input, n, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_matrix_powerBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_matrix_power_autograd_post(result, input, n, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_matrix_powerBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_matrix_rank_autograd(input, atol=None, rtol=None, hermitian=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_matrix_rank", raw_keyset, input, atol, rtol, hermitian, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_matrix_rankBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._atol = atol
        grad_fn._rtol = rtol
        grad_fn._hermitian = hermitian
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_matrix_rank_autograd_post(result, input, atol=None, rtol=None, hermitian=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_matrix_rankBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._atol = atol
        grad_fn._rtol = rtol
        grad_fn._hermitian = hermitian
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_norm_autograd(input, ord=None, dim=None, keepdim=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_norm", raw_keyset, input, ord, dim, keepdim, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_normBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._ord = ord
        grad_fn._dim = dim
        grad_fn._keepdim = keepdim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_norm_autograd_post(result, input, ord=None, dim=None, keepdim=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_normBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._ord = ord
        grad_fn._dim = dim
        grad_fn._keepdim = keepdim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_slogdet_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_slogdet", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Linalg_slogdetBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result[1].grad_fn = grad_fn
        result[1].requires_grad = True
    return result


def linalg_slogdet_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Linalg_slogdetBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result[1].grad_fn = grad_fn
        result[1].requires_grad = True
    return result


def linalg_solve_autograd(self, other, left=True, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_solve", raw_keyset, self, other, left, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.Linalg_solveBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        grad_fn._left = left
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_solve_autograd_post(result, self, other, left=True, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.Linalg_solveBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        grad_fn._left = left
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_svdvals_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_svdvals", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Linalg_svdvalsBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_svdvals_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Linalg_svdvalsBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_tensorinv_autograd(input, ind=2, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_tensorinv", raw_keyset, input, ind, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_tensorinvBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._ind = ind
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_tensorinv_autograd_post(result, input, ind=2, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_tensorinvBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._ind = ind
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_tensorsolve_autograd(input, B, dims=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_tensorsolve", raw_keyset, input, B, dims, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_tensorsolveBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dims = dims
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_tensorsolve_autograd_post(result, input, B, dims=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Linalg_tensorsolveBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dims = dims
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_vander_autograd(x, N=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("linalg_vander", raw_keyset, x, N, **_kwargs)
    if GradMode.enabled and (x.requires_grad):
        grad_fn = _FL.Linalg_vanderBackward0((x,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(x=x)
        grad_fn._N = N
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def linalg_vander_autograd_post(result, x, N=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (x.requires_grad):
        grad_fn = _FL.Linalg_vanderBackward0((x,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(x=x)
        grad_fn._N = N
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def log_softmax_autograd(self, dim, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("log_softmax", raw_keyset, self, dim, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Log_softmaxBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def log_softmax_autograd_post(result, self, dim, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Log_softmaxBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def matrix_power_autograd(input, n, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("matrix_power", raw_keyset, input, n, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Matrix_powerBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def matrix_power_autograd_post(result, input, n, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Matrix_powerBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._n = n
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def max_pool1d_autograd(self, kernel_size, stride, padding, dilation, ceil_mode=False, return_indices=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("max_pool1d", raw_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, return_indices, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Max_pool1dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self, result=result)
        grad_fn._kernel_size = kernel_size
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._dilation = dilation
        grad_fn._ceil_mode = ceil_mode
        grad_fn._return_indices = return_indices
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def max_pool1d_autograd_post(result, self, kernel_size, stride, padding, dilation, ceil_mode=False, return_indices=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Max_pool1dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self, result=result)
        grad_fn._kernel_size = kernel_size
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._dilation = dilation
        grad_fn._ceil_mode = ceil_mode
        grad_fn._return_indices = return_indices
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def max_pool3d_autograd(self, kernel_size, stride, padding=None, dilation=None, ceil_mode=False, return_indices=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("max_pool3d", raw_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, return_indices, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Max_pool3dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self, result=result)
        grad_fn._kernel_size = kernel_size
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._dilation = dilation
        grad_fn._ceil_mode = ceil_mode
        grad_fn._return_indices = return_indices
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def max_pool3d_autograd_post(result, self, kernel_size, stride, padding=None, dilation=None, ceil_mode=False, return_indices=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Max_pool3dBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self, result=result)
        grad_fn._kernel_size = kernel_size
        grad_fn._stride = stride
        grad_fn._padding = padding
        grad_fn._dilation = dilation
        grad_fn._ceil_mode = ceil_mode
        grad_fn._return_indices = return_indices
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def moveaxis_autograd(input, source, destination, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("moveaxis", raw_keyset, input, source, destination, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.MoveaxisBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._source = source
        grad_fn._destination = destination
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def moveaxis_autograd_post(result, input, source, destination, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.MoveaxisBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._source = source
        grad_fn._destination = destination
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def movedim_autograd(input, source, destination, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("movedim", raw_keyset, input, source, destination, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.MovedimBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._source = source
        grad_fn._destination = destination
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def movedim_autograd_post(result, input, source, destination, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.MovedimBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._source = source
        grad_fn._destination = destination
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def nanmean_autograd(input, dim=None, keepdim=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("nanmean", raw_keyset, input, dim, keepdim, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.NanmeanBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dim = dim
        grad_fn._keepdim = keepdim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def nanmean_autograd_post(result, input, dim=None, keepdim=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.NanmeanBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dim = dim
        grad_fn._keepdim = keepdim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def nanquantile_autograd(input, q, dim=None, keepdim=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("nanquantile", raw_keyset, input, q, dim, keepdim, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.NanquantileBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._q = q
        grad_fn._dim = dim
        grad_fn._keepdim = keepdim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def nanquantile_autograd_post(result, input, q, dim=None, keepdim=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.NanquantileBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._q = q
        grad_fn._dim = dim
        grad_fn._keepdim = keepdim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def narrow_autograd(input, dim, start, length, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("narrow", raw_keyset, input, dim, start, length, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.NarrowBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dim = dim
        grad_fn._start = start
        grad_fn._length = length
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def narrow_autograd_post(result, input, dim, start, length, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.NarrowBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dim = dim
        grad_fn._start = start
        grad_fn._length = length
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def normalize_autograd(input, p=2.0, dim=1, eps=1e-12, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("normalize", raw_keyset, input, p, dim, eps, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.NormalizeBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._p = p
        grad_fn._dim = dim
        grad_fn._eps = eps
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def normalize_autograd_post(result, input, p=2.0, dim=1, eps=1e-12, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.NormalizeBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._p = p
        grad_fn._dim = dim
        grad_fn._eps = eps
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def outer_autograd(self, other, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("outer", raw_keyset, self, other, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.OuterBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def outer_autograd_post(result, self, other, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.OuterBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def pad_autograd(input, pad, mode="constant", value=0, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("pad", raw_keyset, input, pad, mode, value, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.PadBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._pad = pad
        grad_fn._mode = mode
        grad_fn._value = value
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def pad_autograd_post(result, input, pad, mode="constant", value=0, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.PadBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._pad = pad
        grad_fn._mode = mode
        grad_fn._value = value
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def pad_sequence_autograd(sequences, batch_first=False, padding_value=0.0, padding_side="right", **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("pad_sequence", raw_keyset, sequences, batch_first, padding_value, padding_side, **_kwargs)
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in sequences)):
        grad_fn = _FL.Pad_sequenceBackward0((*sequences,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._batch_first = batch_first
        grad_fn._padding_value = padding_value
        grad_fn._padding_side = padding_side
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def pad_sequence_autograd_post(result, sequences, batch_first=False, padding_value=0.0, padding_side="right", *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in sequences)):
        grad_fn = _FL.Pad_sequenceBackward0((*sequences,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._batch_first = batch_first
        grad_fn._padding_value = padding_value
        grad_fn._padding_side = padding_side
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def prelu_autograd(self, weight, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("prelu", raw_keyset, self, weight, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(weight, 'requires_grad', False)):
        grad_fn = _FL.PreluBackward0((self, weight,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self, weight=weight)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def prelu_autograd_post(result, self, weight, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(weight, 'requires_grad', False)):
        grad_fn = _FL.PreluBackward0((self, weight,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self, weight=weight)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def quantile_autograd(input, q, dim=None, keepdim=False, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("quantile", raw_keyset, input, q, dim, keepdim, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.QuantileBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._q = q
        grad_fn._dim = dim
        grad_fn._keepdim = keepdim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def quantile_autograd_post(result, input, q, dim=None, keepdim=False, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.QuantileBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._q = q
        grad_fn._dim = dim
        grad_fn._keepdim = keepdim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def relu6_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("relu6", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Relu6Backward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def relu6_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Relu6Backward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def repeat_interleave_autograd(input, repeats, dim=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("repeat_interleave", raw_keyset, input, repeats, dim, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Repeat_interleaveBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._repeats = repeats
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def repeat_interleave_autograd_post(result, input, repeats, dim=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Repeat_interleaveBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._repeats = repeats
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def rms_norm_autograd(input, normalized_shape, weight, eps=1e-6, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("rms_norm", raw_keyset, input, normalized_shape, weight, eps, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Rms_normBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input, weight=weight)
        grad_fn._normalized_shape = normalized_shape
        grad_fn._eps = eps
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def rms_norm_autograd_post(result, input, normalized_shape, weight, eps=1e-6, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Rms_normBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input, weight=weight)
        grad_fn._normalized_shape = normalized_shape
        grad_fn._eps = eps
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def row_stack_autograd(tensors, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("row_stack", raw_keyset, tensors, **_kwargs)
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.Row_stackBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def row_stack_autograd_post(result, tensors, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.Row_stackBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def selu_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("selu", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.SeluBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def selu_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.SeluBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def signbit_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("signbit", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.SignbitBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def signbit_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.SignbitBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def softmax_autograd(self, dim, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("softmax", raw_keyset, self, dim, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _F.SoftmaxBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def softmax_autograd_post(result, self, dim, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _F.SoftmaxBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def softsign_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("softsign", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.SoftsignBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def softsign_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.SoftsignBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_digamma_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("special_digamma", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_digammaBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_digamma_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_digammaBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_erfinv_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("special_erfinv", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_erfinvBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_erfinv_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_erfinvBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_gammainc_autograd(self, other, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("special_gammainc", raw_keyset, self, other, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.Special_gammaincBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_gammainc_autograd_post(result, self, other, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.Special_gammaincBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_gammaincc_autograd(self, other, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("special_gammaincc", raw_keyset, self, other, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.Special_gammainccBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_gammaincc_autograd_post(result, self, other, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.Special_gammainccBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_gammaln_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("special_gammaln", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_gammalnBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_gammaln_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_gammalnBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_i0_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("special_i0", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_i0Backward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_i0_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_i0Backward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_logit_autograd(input, eps=None, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("special_logit", raw_keyset, input, eps, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Special_logitBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._eps = eps
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_logit_autograd_post(result, input, eps=None, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Special_logitBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._eps = eps
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_multigammaln_autograd(self, p, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("special_multigammaln", raw_keyset, self, p, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_multigammalnBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._p = p
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_multigammaln_autograd_post(result, self, p, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_multigammalnBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._p = p
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_ndtr_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("special_ndtr", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_ndtrBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_ndtr_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_ndtrBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_polygamma_autograd(n, self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("special_polygamma", raw_keyset, n, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_polygammaBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._n = n
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_polygamma_autograd_post(result, n, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_polygammaBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        grad_fn._n = n
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_sinc_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("special_sinc", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_sincBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_sinc_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.Special_sincBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_xlogy_autograd(self, other, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("special_xlogy", raw_keyset, self, other, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.Special_xlogyBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def special_xlogy_autograd_post(result, self, other, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.Special_xlogyBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def square_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("square", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.SquareBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def square_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _FL.SquareBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def take_along_dim_autograd(input, indices, dim, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("take_along_dim", raw_keyset, input, indices, dim, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Take_along_dimBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(indices=indices, input_=input)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def take_along_dim_autograd_post(result, input, indices, dim, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.Take_along_dimBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(indices=indices, input_=input)
        grad_fn._dim = dim
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def tensordot_autograd(self, other, dims=2, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("tensordot", raw_keyset, self, other, dims, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.TensordotBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        grad_fn._dims = dims
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def tensordot_autograd_post(result, self, other, dims=2, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.TensordotBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        grad_fn._dims = dims
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def tile_autograd(input, dims, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("tile", raw_keyset, input, dims, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.TileBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dims = dims
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def tile_autograd_post(result, input, dims, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.TileBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dims = dims
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def true_divide_autograd(self, other, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("true_divide", raw_keyset, self, other, **_kwargs)
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.True_divideBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def true_divide_autograd_post(result, self, other, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (getattr(self, 'requires_grad', False) or getattr(other, 'requires_grad', False)):
        grad_fn = _FL.True_divideBackward0((self, other,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(other=other, self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def unflatten_autograd(input, dim, sizes, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("unflatten", raw_keyset, input, dim, sizes, **_kwargs)
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.UnflattenBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dim = dim
        grad_fn._sizes = sizes
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def unflatten_autograd_post(result, input, dim, sizes, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (input.requires_grad):
        grad_fn = _FL.UnflattenBackward0((input,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(input_=input)
        grad_fn._dim = dim
        grad_fn._sizes = sizes
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def uniform_autograd(self, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("uniform", raw_keyset, self, **_kwargs)
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _F.UniformBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def uniform_autograd_post(result, self, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (self.requires_grad):
        grad_fn = _F.UniformBackward0((self,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        grad_fn._save(self_=self)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def vstack_autograd(tensors, **_kwargs):
    active_keyset = current_dispatch_keyset()
    raw_keyset = _strip_autograd_keys(active_keyset)
    result = redispatch("vstack", raw_keyset, tensors, **_kwargs)
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.VstackBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result


def vstack_autograd_post(result, tensors, *, raw_keyset, active_keyset, **_kwargs):
    if GradMode.enabled and (any(getattr(t, 'requires_grad', False) for t in tensors)):
        grad_fn = _FL.VstackBackward0((*tensors,), raw_keyset=raw_keyset, active_keyset=active_keyset)
        annotate_node_creation(grad_fn)
        result.grad_fn = grad_fn
        result.requires_grad = True
    return result

def register_legacy_autograd_kernels():
    """Register all legacy (non-generated) autograd wrappers."""
    from .._dispatch.registration import register_autograd_kernels, register_autograd_post_kernels

    register_autograd_post_kernels('sum_to_size', sum_to_size_autograd_post)
    register_autograd_kernels('contiguous', default=contiguous_autograd, cpu=contiguous_autograd, cuda=contiguous_autograd, npu=contiguous_autograd, meta=contiguous_autograd)
    register_autograd_post_kernels('contiguous', contiguous_autograd_post)
    register_autograd_kernels('relu6', default=relu6_autograd, cpu=relu6_autograd, cuda=relu6_autograd, npu=relu6_autograd, meta=relu6_autograd)
    register_autograd_post_kernels('relu6', relu6_autograd_post)
    register_autograd_kernels('selu', default=selu_autograd, cpu=selu_autograd, cuda=selu_autograd, npu=selu_autograd, meta=selu_autograd)
    register_autograd_post_kernels('selu', selu_autograd_post)
    register_autograd_kernels('softmax', default=softmax_autograd, cpu=softmax_autograd, cuda=softmax_autograd, npu=softmax_autograd, meta=softmax_autograd)
    register_autograd_post_kernels('softmax', softmax_autograd_post)
    register_autograd_kernels('log_softmax', default=log_softmax_autograd, cpu=log_softmax_autograd, cuda=log_softmax_autograd, npu=log_softmax_autograd, meta=log_softmax_autograd)
    register_autograd_post_kernels('log_softmax', log_softmax_autograd_post)
    register_autograd_kernels('prelu', default=prelu_autograd, cpu=prelu_autograd, cuda=prelu_autograd, npu=prelu_autograd, meta=prelu_autograd)
    register_autograd_post_kernels('prelu', prelu_autograd_post)
    register_autograd_kernels('layer_norm', default=layer_norm_autograd, cpu=layer_norm_autograd, cuda=layer_norm_autograd, npu=layer_norm_autograd, meta=layer_norm_autograd)
    register_autograd_post_kernels('layer_norm', layer_norm_autograd_post)
    register_autograd_kernels('batch_norm', default=batch_norm_autograd, cpu=batch_norm_autograd, cuda=batch_norm_autograd, npu=batch_norm_autograd, meta=batch_norm_autograd)
    register_autograd_post_kernels('batch_norm', batch_norm_autograd_post)
    register_autograd_kernels('group_norm', default=group_norm_autograd, cpu=group_norm_autograd, cuda=group_norm_autograd, npu=group_norm_autograd, meta=group_norm_autograd)
    register_autograd_post_kernels('group_norm', group_norm_autograd_post)
    register_autograd_kernels('rms_norm', default=rms_norm_autograd, cpu=rms_norm_autograd, cuda=rms_norm_autograd, npu=rms_norm_autograd, meta=rms_norm_autograd)
    register_autograd_post_kernels('rms_norm', rms_norm_autograd_post)
    register_autograd_kernels('conv1d', default=conv1d_autograd, cpu=conv1d_autograd, cuda=conv1d_autograd, npu=conv1d_autograd, meta=conv1d_autograd)
    register_autograd_post_kernels('conv1d', conv1d_autograd_post)
    register_autograd_kernels('conv2d', default=conv2d_autograd, cpu=conv2d_autograd, cuda=conv2d_autograd, npu=conv2d_autograd, meta=conv2d_autograd)
    register_autograd_post_kernels('conv2d', conv2d_autograd_post)
    register_autograd_kernels('conv3d', default=conv3d_autograd, cpu=conv3d_autograd, cuda=conv3d_autograd, npu=conv3d_autograd, meta=conv3d_autograd)
    register_autograd_post_kernels('conv3d', conv3d_autograd_post)
    register_autograd_kernels('conv_transpose1d', default=conv_transpose1d_autograd, cpu=conv_transpose1d_autograd, cuda=conv_transpose1d_autograd, npu=conv_transpose1d_autograd, meta=conv_transpose1d_autograd)
    register_autograd_post_kernels('conv_transpose1d', conv_transpose1d_autograd_post)
    register_autograd_kernels('conv_transpose2d', default=conv_transpose2d_autograd, cpu=conv_transpose2d_autograd, cuda=conv_transpose2d_autograd, npu=conv_transpose2d_autograd, meta=conv_transpose2d_autograd)
    register_autograd_post_kernels('conv_transpose2d', conv_transpose2d_autograd_post)
    register_autograd_kernels('conv_transpose3d', default=conv_transpose3d_autograd, cpu=conv_transpose3d_autograd, cuda=conv_transpose3d_autograd, npu=conv_transpose3d_autograd, meta=conv_transpose3d_autograd)
    register_autograd_post_kernels('conv_transpose3d', conv_transpose3d_autograd_post)
    register_autograd_kernels('max_pool1d', default=max_pool1d_autograd, cpu=max_pool1d_autograd, cuda=max_pool1d_autograd, npu=max_pool1d_autograd, meta=max_pool1d_autograd)
    register_autograd_post_kernels('max_pool1d', max_pool1d_autograd_post)
    register_autograd_kernels('max_pool3d', default=max_pool3d_autograd, cpu=max_pool3d_autograd, cuda=max_pool3d_autograd, npu=max_pool3d_autograd, meta=max_pool3d_autograd)
    register_autograd_post_kernels('max_pool3d', max_pool3d_autograd_post)
    register_autograd_kernels('avg_pool1d', default=avg_pool1d_autograd, cpu=avg_pool1d_autograd, cuda=avg_pool1d_autograd, npu=avg_pool1d_autograd, meta=avg_pool1d_autograd)
    register_autograd_post_kernels('avg_pool1d', avg_pool1d_autograd_post)
    register_autograd_kernels('adaptive_avg_pool1d', default=adaptive_avg_pool1d_autograd, cpu=adaptive_avg_pool1d_autograd, cuda=adaptive_avg_pool1d_autograd, npu=adaptive_avg_pool1d_autograd, meta=adaptive_avg_pool1d_autograd)
    register_autograd_post_kernels('adaptive_avg_pool1d', adaptive_avg_pool1d_autograd_post)
    register_autograd_kernels('adaptive_avg_pool2d', default=adaptive_avg_pool2d_autograd, cpu=adaptive_avg_pool2d_autograd, cuda=adaptive_avg_pool2d_autograd, npu=adaptive_avg_pool2d_autograd, meta=adaptive_avg_pool2d_autograd)
    register_autograd_post_kernels('adaptive_avg_pool2d', adaptive_avg_pool2d_autograd_post)
    register_autograd_kernels('adaptive_avg_pool3d', default=adaptive_avg_pool3d_autograd, cpu=adaptive_avg_pool3d_autograd, cuda=adaptive_avg_pool3d_autograd, npu=adaptive_avg_pool3d_autograd, meta=adaptive_avg_pool3d_autograd)
    register_autograd_post_kernels('adaptive_avg_pool3d', adaptive_avg_pool3d_autograd_post)
    register_autograd_kernels('adaptive_max_pool1d', default=adaptive_max_pool1d_autograd, cpu=adaptive_max_pool1d_autograd, cuda=adaptive_max_pool1d_autograd, npu=adaptive_max_pool1d_autograd, meta=adaptive_max_pool1d_autograd)
    register_autograd_post_kernels('adaptive_max_pool1d', adaptive_max_pool1d_autograd_post)
    register_autograd_kernels('softsign', default=softsign_autograd, cpu=softsign_autograd, cuda=softsign_autograd, npu=softsign_autograd, meta=softsign_autograd)
    register_autograd_post_kernels('softsign', softsign_autograd_post)
    register_autograd_kernels('square', default=square_autograd, cpu=square_autograd, cuda=square_autograd, npu=square_autograd, meta=square_autograd)
    register_autograd_post_kernels('square', square_autograd_post)
    register_autograd_kernels('signbit', default=signbit_autograd, cpu=signbit_autograd, cuda=signbit_autograd, npu=signbit_autograd, meta=signbit_autograd)
    register_autograd_post_kernels('signbit', signbit_autograd_post)
    register_autograd_kernels('true_divide', default=true_divide_autograd, cpu=true_divide_autograd, cuda=true_divide_autograd, npu=true_divide_autograd, meta=true_divide_autograd)
    register_autograd_post_kernels('true_divide', true_divide_autograd_post)
    register_autograd_kernels('outer', default=outer_autograd, cpu=outer_autograd, cuda=outer_autograd, npu=outer_autograd, meta=outer_autograd)
    register_autograd_post_kernels('outer', outer_autograd_post)
    register_autograd_kernels('floor_divide', default=floor_divide_autograd, cpu=floor_divide_autograd, cuda=floor_divide_autograd, npu=floor_divide_autograd, meta=floor_divide_autograd)
    register_autograd_post_kernels('floor_divide', floor_divide_autograd_post)
    register_autograd_kernels('inner', default=inner_autograd, cpu=inner_autograd, cuda=inner_autograd, npu=inner_autograd, meta=inner_autograd)
    register_autograd_post_kernels('inner', inner_autograd_post)
    register_autograd_kernels('heaviside', default=heaviside_autograd, cpu=heaviside_autograd, cuda=heaviside_autograd, npu=heaviside_autograd, meta=heaviside_autograd)
    register_autograd_post_kernels('heaviside', heaviside_autograd_post)
    register_autograd_kernels('pad', default=pad_autograd, cpu=pad_autograd, cuda=pad_autograd, npu=pad_autograd, meta=pad_autograd)
    register_autograd_post_kernels('pad', pad_autograd_post)
    register_autograd_kernels('tile', default=tile_autograd, cpu=tile_autograd, cuda=tile_autograd, npu=tile_autograd, meta=tile_autograd)
    register_autograd_post_kernels('tile', tile_autograd_post)
    register_autograd_kernels('flatten', default=flatten_autograd, cpu=flatten_autograd, cuda=flatten_autograd, npu=flatten_autograd, meta=flatten_autograd)
    register_autograd_post_kernels('flatten', flatten_autograd_post)
    register_autograd_kernels('unflatten', default=unflatten_autograd, cpu=unflatten_autograd, cuda=unflatten_autograd, npu=unflatten_autograd, meta=unflatten_autograd)
    register_autograd_post_kernels('unflatten', unflatten_autograd_post)
    register_autograd_kernels('movedim', default=movedim_autograd, cpu=movedim_autograd, cuda=movedim_autograd, npu=movedim_autograd, meta=movedim_autograd)
    register_autograd_post_kernels('movedim', movedim_autograd_post)
    register_autograd_kernels('moveaxis', default=moveaxis_autograd, cpu=moveaxis_autograd, cuda=moveaxis_autograd, npu=moveaxis_autograd, meta=moveaxis_autograd)
    register_autograd_post_kernels('moveaxis', moveaxis_autograd_post)
    register_autograd_kernels('instance_norm', default=instance_norm_autograd, cpu=instance_norm_autograd, cuda=instance_norm_autograd, npu=instance_norm_autograd, meta=instance_norm_autograd)
    register_autograd_post_kernels('instance_norm', instance_norm_autograd_post)
    register_autograd_kernels('normalize', default=normalize_autograd, cpu=normalize_autograd, cuda=normalize_autograd, npu=normalize_autograd, meta=normalize_autograd)
    register_autograd_post_kernels('normalize', normalize_autograd_post)
    register_autograd_kernels('repeat_interleave', default=repeat_interleave_autograd, cpu=repeat_interleave_autograd, cuda=repeat_interleave_autograd, npu=repeat_interleave_autograd, meta=repeat_interleave_autograd)
    register_autograd_post_kernels('repeat_interleave', repeat_interleave_autograd_post)
    register_autograd_kernels('take_along_dim', default=take_along_dim_autograd, cpu=take_along_dim_autograd, cuda=take_along_dim_autograd, npu=take_along_dim_autograd, meta=take_along_dim_autograd)
    register_autograd_post_kernels('take_along_dim', take_along_dim_autograd_post)
    register_autograd_kernels('narrow', default=narrow_autograd, cpu=narrow_autograd, cuda=narrow_autograd, npu=narrow_autograd, meta=narrow_autograd)
    register_autograd_post_kernels('narrow', narrow_autograd_post)
    register_autograd_kernels('broadcast_to', default=broadcast_to_autograd, cpu=broadcast_to_autograd, cuda=broadcast_to_autograd, npu=broadcast_to_autograd, meta=broadcast_to_autograd)
    register_autograd_post_kernels('broadcast_to', broadcast_to_autograd_post)
    register_autograd_kernels('diff', default=diff_autograd, cpu=diff_autograd, cuda=diff_autograd, npu=diff_autograd, meta=diff_autograd)
    register_autograd_post_kernels('diff', diff_autograd_post)
    register_autograd_kernels('det', default=det_autograd, cpu=det_autograd, cuda=det_autograd, npu=det_autograd, meta=det_autograd)
    register_autograd_post_kernels('det', det_autograd_post)
    register_autograd_kernels('diag', default=diag_autograd, cpu=diag_autograd, cuda=diag_autograd, npu=diag_autograd, meta=diag_autograd)
    register_autograd_post_kernels('diag', diag_autograd_post)
    register_autograd_kernels('nanmean', default=nanmean_autograd, cpu=nanmean_autograd, cuda=nanmean_autograd, npu=nanmean_autograd, meta=nanmean_autograd)
    register_autograd_post_kernels('nanmean', nanmean_autograd_post)
    register_autograd_kernels('matrix_power', default=matrix_power_autograd, cpu=matrix_power_autograd, cuda=matrix_power_autograd, npu=matrix_power_autograd, meta=matrix_power_autograd)
    register_autograd_post_kernels('matrix_power', matrix_power_autograd_post)
    register_autograd_kernels('linalg_matrix_power', default=linalg_matrix_power_autograd, cpu=linalg_matrix_power_autograd, cuda=linalg_matrix_power_autograd, npu=linalg_matrix_power_autograd, meta=linalg_matrix_power_autograd)
    register_autograd_post_kernels('linalg_matrix_power', linalg_matrix_power_autograd_post)
    register_autograd_kernels('special_logit', default=special_logit_autograd, cpu=special_logit_autograd, cuda=special_logit_autograd, npu=special_logit_autograd, meta=special_logit_autograd)
    register_autograd_post_kernels('special_logit', special_logit_autograd_post)
    register_autograd_kernels('linalg_norm', default=linalg_norm_autograd, cpu=linalg_norm_autograd, cuda=linalg_norm_autograd, npu=linalg_norm_autograd, meta=linalg_norm_autograd)
    register_autograd_post_kernels('linalg_norm', linalg_norm_autograd_post)
    register_autograd_kernels('linalg_matrix_norm', default=linalg_matrix_norm_autograd, cpu=linalg_matrix_norm_autograd, cuda=linalg_matrix_norm_autograd, npu=linalg_matrix_norm_autograd, meta=linalg_matrix_norm_autograd)
    register_autograd_post_kernels('linalg_matrix_norm', linalg_matrix_norm_autograd_post)
    register_autograd_kernels('linalg_cond', default=linalg_cond_autograd, cpu=linalg_cond_autograd, cuda=linalg_cond_autograd, npu=linalg_cond_autograd, meta=linalg_cond_autograd)
    register_autograd_post_kernels('linalg_cond', linalg_cond_autograd_post)
    register_autograd_kernels('linalg_matrix_rank', default=linalg_matrix_rank_autograd, cpu=linalg_matrix_rank_autograd, cuda=linalg_matrix_rank_autograd, npu=linalg_matrix_rank_autograd, meta=linalg_matrix_rank_autograd)
    register_autograd_post_kernels('linalg_matrix_rank', linalg_matrix_rank_autograd_post)
    register_autograd_kernels('linalg_tensorinv', default=linalg_tensorinv_autograd, cpu=linalg_tensorinv_autograd, cuda=linalg_tensorinv_autograd, npu=linalg_tensorinv_autograd, meta=linalg_tensorinv_autograd)
    register_autograd_post_kernels('linalg_tensorinv', linalg_tensorinv_autograd_post)
    register_autograd_kernels('linalg_tensorsolve', default=linalg_tensorsolve_autograd, cpu=linalg_tensorsolve_autograd, cuda=linalg_tensorsolve_autograd, npu=linalg_tensorsolve_autograd, meta=linalg_tensorsolve_autograd)
    register_autograd_post_kernels('linalg_tensorsolve', linalg_tensorsolve_autograd_post)
    register_autograd_kernels('linalg_vander', default=linalg_vander_autograd, cpu=linalg_vander_autograd, cuda=linalg_vander_autograd, npu=linalg_vander_autograd, meta=linalg_vander_autograd)
    register_autograd_post_kernels('linalg_vander', linalg_vander_autograd_post)
    register_autograd_kernels('linalg_eigvalsh', default=linalg_eigvalsh_autograd, cpu=linalg_eigvalsh_autograd, cuda=linalg_eigvalsh_autograd, npu=linalg_eigvalsh_autograd, meta=linalg_eigvalsh_autograd)
    register_autograd_post_kernels('linalg_eigvalsh', linalg_eigvalsh_autograd_post)
    register_autograd_kernels('getitem', default=getitem_autograd, cpu=getitem_autograd, cuda=getitem_autograd, npu=getitem_autograd, meta=getitem_autograd)
    register_autograd_post_kernels('getitem', getitem_autograd_post)
    register_autograd_kernels('quantile', default=quantile_autograd, cpu=quantile_autograd, cuda=quantile_autograd, npu=quantile_autograd, meta=quantile_autograd)
    register_autograd_post_kernels('quantile', quantile_autograd_post)
    register_autograd_kernels('nanquantile', default=nanquantile_autograd, cpu=nanquantile_autograd, cuda=nanquantile_autograd, npu=nanquantile_autograd, meta=nanquantile_autograd)
    register_autograd_post_kernels('nanquantile', nanquantile_autograd_post)
    register_autograd_kernels('special_digamma', default=special_digamma_autograd, cpu=special_digamma_autograd, cuda=special_digamma_autograd, npu=special_digamma_autograd, meta=special_digamma_autograd)
    register_autograd_post_kernels('special_digamma', special_digamma_autograd_post)
    register_autograd_kernels('special_gammaln', default=special_gammaln_autograd, cpu=special_gammaln_autograd, cuda=special_gammaln_autograd, npu=special_gammaln_autograd, meta=special_gammaln_autograd)
    register_autograd_post_kernels('special_gammaln', special_gammaln_autograd_post)
    register_autograd_kernels('special_erfinv', default=special_erfinv_autograd, cpu=special_erfinv_autograd, cuda=special_erfinv_autograd, npu=special_erfinv_autograd, meta=special_erfinv_autograd)
    register_autograd_post_kernels('special_erfinv', special_erfinv_autograd_post)
    register_autograd_kernels('special_ndtr', default=special_ndtr_autograd, cpu=special_ndtr_autograd, cuda=special_ndtr_autograd, npu=special_ndtr_autograd, meta=special_ndtr_autograd)
    register_autograd_post_kernels('special_ndtr', special_ndtr_autograd_post)
    register_autograd_kernels('special_sinc', default=special_sinc_autograd, cpu=special_sinc_autograd, cuda=special_sinc_autograd, npu=special_sinc_autograd, meta=special_sinc_autograd)
    register_autograd_post_kernels('special_sinc', special_sinc_autograd_post)
    register_autograd_kernels('special_i0', default=special_i0_autograd, cpu=special_i0_autograd, cuda=special_i0_autograd, npu=special_i0_autograd, meta=special_i0_autograd)
    register_autograd_post_kernels('special_i0', special_i0_autograd_post)
    register_autograd_kernels('uniform', default=uniform_autograd, cpu=uniform_autograd, cuda=uniform_autograd, npu=uniform_autograd, meta=uniform_autograd)
    register_autograd_post_kernels('uniform', uniform_autograd_post)
    register_autograd_kernels('special_xlogy', default=special_xlogy_autograd, cpu=special_xlogy_autograd, cuda=special_xlogy_autograd, npu=special_xlogy_autograd, meta=special_xlogy_autograd)
    register_autograd_post_kernels('special_xlogy', special_xlogy_autograd_post)
    register_autograd_kernels('special_gammainc', default=special_gammainc_autograd, cpu=special_gammainc_autograd, cuda=special_gammainc_autograd, npu=special_gammainc_autograd, meta=special_gammainc_autograd)
    register_autograd_post_kernels('special_gammainc', special_gammainc_autograd_post)
    register_autograd_kernels('special_gammaincc', default=special_gammaincc_autograd, cpu=special_gammaincc_autograd, cuda=special_gammaincc_autograd, npu=special_gammaincc_autograd, meta=special_gammaincc_autograd)
    register_autograd_post_kernels('special_gammaincc', special_gammaincc_autograd_post)
    register_autograd_kernels('linalg_inv', default=linalg_inv_autograd, cpu=linalg_inv_autograd, cuda=linalg_inv_autograd, npu=linalg_inv_autograd, meta=linalg_inv_autograd)
    register_autograd_post_kernels('linalg_inv', linalg_inv_autograd_post)
    register_autograd_kernels('linalg_eigvals', default=linalg_eigvals_autograd, cpu=linalg_eigvals_autograd, cuda=linalg_eigvals_autograd, npu=linalg_eigvals_autograd, meta=linalg_eigvals_autograd)
    register_autograd_post_kernels('linalg_eigvals', linalg_eigvals_autograd_post)
    register_autograd_kernels('linalg_svdvals', default=linalg_svdvals_autograd, cpu=linalg_svdvals_autograd, cuda=linalg_svdvals_autograd, npu=linalg_svdvals_autograd, meta=linalg_svdvals_autograd)
    register_autograd_post_kernels('linalg_svdvals', linalg_svdvals_autograd_post)
    register_autograd_kernels('fft_fft', default=fft_fft_autograd, cpu=fft_fft_autograd, cuda=fft_fft_autograd, npu=fft_fft_autograd, meta=fft_fft_autograd)
    register_autograd_post_kernels('fft_fft', fft_fft_autograd_post)
    register_autograd_kernels('fft_ifft', default=fft_ifft_autograd, cpu=fft_ifft_autograd, cuda=fft_ifft_autograd, npu=fft_ifft_autograd, meta=fft_ifft_autograd)
    register_autograd_post_kernels('fft_ifft', fft_ifft_autograd_post)
    register_autograd_kernels('fft_fft2', default=fft_fft2_autograd, cpu=fft_fft2_autograd, cuda=fft_fft2_autograd, npu=fft_fft2_autograd, meta=fft_fft2_autograd)
    register_autograd_post_kernels('fft_fft2', fft_fft2_autograd_post)
    register_autograd_kernels('fft_ifft2', default=fft_ifft2_autograd, cpu=fft_ifft2_autograd, cuda=fft_ifft2_autograd, npu=fft_ifft2_autograd, meta=fft_ifft2_autograd)
    register_autograd_post_kernels('fft_ifft2', fft_ifft2_autograd_post)
    register_autograd_kernels('fft_fftn', default=fft_fftn_autograd, cpu=fft_fftn_autograd, cuda=fft_fftn_autograd, npu=fft_fftn_autograd, meta=fft_fftn_autograd)
    register_autograd_post_kernels('fft_fftn', fft_fftn_autograd_post)
    register_autograd_kernels('fft_ifftn', default=fft_ifftn_autograd, cpu=fft_ifftn_autograd, cuda=fft_ifftn_autograd, npu=fft_ifftn_autograd, meta=fft_ifftn_autograd)
    register_autograd_post_kernels('fft_ifftn', fft_ifftn_autograd_post)
    register_autograd_kernels('fft_rfft', default=fft_rfft_autograd, cpu=fft_rfft_autograd, cuda=fft_rfft_autograd, npu=fft_rfft_autograd, meta=fft_rfft_autograd)
    register_autograd_post_kernels('fft_rfft', fft_rfft_autograd_post)
    register_autograd_kernels('fft_irfft', default=fft_irfft_autograd, cpu=fft_irfft_autograd, cuda=fft_irfft_autograd, npu=fft_irfft_autograd, meta=fft_irfft_autograd)
    register_autograd_post_kernels('fft_irfft', fft_irfft_autograd_post)
    register_autograd_kernels('fft_rfft2', default=fft_rfft2_autograd, cpu=fft_rfft2_autograd, cuda=fft_rfft2_autograd, npu=fft_rfft2_autograd, meta=fft_rfft2_autograd)
    register_autograd_post_kernels('fft_rfft2', fft_rfft2_autograd_post)
    register_autograd_kernels('fft_irfft2', default=fft_irfft2_autograd, cpu=fft_irfft2_autograd, cuda=fft_irfft2_autograd, npu=fft_irfft2_autograd, meta=fft_irfft2_autograd)
    register_autograd_post_kernels('fft_irfft2', fft_irfft2_autograd_post)
    register_autograd_kernels('fft_rfftn', default=fft_rfftn_autograd, cpu=fft_rfftn_autograd, cuda=fft_rfftn_autograd, npu=fft_rfftn_autograd, meta=fft_rfftn_autograd)
    register_autograd_post_kernels('fft_rfftn', fft_rfftn_autograd_post)
    register_autograd_kernels('fft_irfftn', default=fft_irfftn_autograd, cpu=fft_irfftn_autograd, cuda=fft_irfftn_autograd, npu=fft_irfftn_autograd, meta=fft_irfftn_autograd)
    register_autograd_post_kernels('fft_irfftn', fft_irfftn_autograd_post)
    register_autograd_kernels('fft_hfft', default=fft_hfft_autograd, cpu=fft_hfft_autograd, cuda=fft_hfft_autograd, npu=fft_hfft_autograd, meta=fft_hfft_autograd)
    register_autograd_post_kernels('fft_hfft', fft_hfft_autograd_post)
    register_autograd_kernels('fft_ihfft', default=fft_ihfft_autograd, cpu=fft_ihfft_autograd, cuda=fft_ihfft_autograd, npu=fft_ihfft_autograd, meta=fft_ihfft_autograd)
    register_autograd_post_kernels('fft_ihfft', fft_ihfft_autograd_post)
    register_autograd_kernels('fft_fftshift', default=fft_fftshift_autograd, cpu=fft_fftshift_autograd, cuda=fft_fftshift_autograd, npu=fft_fftshift_autograd, meta=fft_fftshift_autograd)
    register_autograd_post_kernels('fft_fftshift', fft_fftshift_autograd_post)
    register_autograd_kernels('fft_ifftshift', default=fft_ifftshift_autograd, cpu=fft_ifftshift_autograd, cuda=fft_ifftshift_autograd, npu=fft_ifftshift_autograd, meta=fft_ifftshift_autograd)
    register_autograd_post_kernels('fft_ifftshift', fft_ifftshift_autograd_post)
    register_autograd_kernels('cross', default=cross_autograd, cpu=cross_autograd, cuda=cross_autograd, npu=cross_autograd, meta=cross_autograd)
    register_autograd_post_kernels('cross', cross_autograd_post)
    register_autograd_kernels('tensordot', default=tensordot_autograd, cpu=tensordot_autograd, cuda=tensordot_autograd, npu=tensordot_autograd, meta=tensordot_autograd)
    register_autograd_post_kernels('tensordot', tensordot_autograd_post)
    register_autograd_kernels('grid_sample', default=grid_sample_autograd, cpu=grid_sample_autograd, cuda=grid_sample_autograd, npu=grid_sample_autograd, meta=grid_sample_autograd)
    register_autograd_post_kernels('grid_sample', grid_sample_autograd_post)
    register_autograd_kernels('affine_grid', default=affine_grid_autograd, cpu=affine_grid_autograd, cuda=affine_grid_autograd, npu=affine_grid_autograd, meta=affine_grid_autograd)
    register_autograd_post_kernels('affine_grid', affine_grid_autograd_post)
    register_autograd_kernels('ctc_loss', default=ctc_loss_autograd, cpu=ctc_loss_autograd, cuda=ctc_loss_autograd, npu=ctc_loss_autograd, meta=ctc_loss_autograd)
    register_autograd_post_kernels('ctc_loss', ctc_loss_autograd_post)
    register_autograd_kernels('cdist', default=cdist_autograd, cpu=cdist_autograd, cuda=cdist_autograd, npu=cdist_autograd, meta=cdist_autograd)
    register_autograd_post_kernels('cdist', cdist_autograd_post)
    register_autograd_kernels('special_polygamma', default=special_polygamma_autograd, cpu=special_polygamma_autograd, cuda=special_polygamma_autograd, npu=special_polygamma_autograd, meta=special_polygamma_autograd)
    register_autograd_post_kernels('special_polygamma', special_polygamma_autograd_post)
    register_autograd_kernels('special_multigammaln', default=special_multigammaln_autograd, cpu=special_multigammaln_autograd, cuda=special_multigammaln_autograd, npu=special_multigammaln_autograd, meta=special_multigammaln_autograd)
    register_autograd_post_kernels('special_multigammaln', special_multigammaln_autograd_post)
    register_autograd_kernels('linalg_det', default=linalg_det_autograd, cpu=linalg_det_autograd, cuda=linalg_det_autograd, npu=linalg_det_autograd, meta=linalg_det_autograd)
    register_autograd_post_kernels('linalg_det', linalg_det_autograd_post)
    register_autograd_kernels('linalg_slogdet', default=linalg_slogdet_autograd, cpu=linalg_slogdet_autograd, cuda=linalg_slogdet_autograd, npu=linalg_slogdet_autograd, meta=linalg_slogdet_autograd)
    register_autograd_post_kernels('linalg_slogdet', linalg_slogdet_autograd_post)
    register_autograd_kernels('linalg_cholesky', default=linalg_cholesky_autograd, cpu=linalg_cholesky_autograd, cuda=linalg_cholesky_autograd, npu=linalg_cholesky_autograd, meta=linalg_cholesky_autograd)
    register_autograd_post_kernels('linalg_cholesky', linalg_cholesky_autograd_post)
    register_autograd_kernels('linalg_solve', default=linalg_solve_autograd, cpu=linalg_solve_autograd, cuda=linalg_solve_autograd, npu=linalg_solve_autograd, meta=linalg_solve_autograd)
    register_autograd_post_kernels('linalg_solve', linalg_solve_autograd_post)
    register_autograd_kernels('hstack', default=hstack_autograd, cpu=hstack_autograd, cuda=hstack_autograd, npu=hstack_autograd, meta=hstack_autograd)
    register_autograd_post_kernels('hstack', hstack_autograd_post)
    register_autograd_kernels('vstack', default=vstack_autograd, cpu=vstack_autograd, cuda=vstack_autograd, npu=vstack_autograd, meta=vstack_autograd)
    register_autograd_post_kernels('vstack', vstack_autograd_post)
    register_autograd_kernels('row_stack', default=row_stack_autograd, cpu=row_stack_autograd, cuda=row_stack_autograd, npu=row_stack_autograd, meta=row_stack_autograd)
    register_autograd_post_kernels('row_stack', row_stack_autograd_post)
    register_autograd_kernels('dstack', default=dstack_autograd, cpu=dstack_autograd, cuda=dstack_autograd, npu=dstack_autograd, meta=dstack_autograd)
    register_autograd_post_kernels('dstack', dstack_autograd_post)
    register_autograd_kernels('column_stack', default=column_stack_autograd, cpu=column_stack_autograd, cuda=column_stack_autograd, npu=column_stack_autograd, meta=column_stack_autograd)
    register_autograd_post_kernels('column_stack', column_stack_autograd_post)
    register_autograd_kernels('concat', default=concat_autograd, cpu=concat_autograd, cuda=concat_autograd, npu=concat_autograd, meta=concat_autograd)
    register_autograd_post_kernels('concat', concat_autograd_post)
    register_autograd_kernels('concatenate', default=concatenate_autograd, cpu=concatenate_autograd, cuda=concatenate_autograd, npu=concatenate_autograd, meta=concatenate_autograd)
    register_autograd_post_kernels('concatenate', concatenate_autograd_post)
    register_autograd_kernels('pad_sequence', default=pad_sequence_autograd, cpu=pad_sequence_autograd, cuda=pad_sequence_autograd, npu=pad_sequence_autograd, meta=pad_sequence_autograd)
    register_autograd_post_kernels('pad_sequence', pad_sequence_autograd_post)

