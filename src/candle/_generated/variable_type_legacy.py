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

