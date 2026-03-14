import math

from ..module import Module
from ..parameter import Parameter
from ..._creation import empty
from .. import functional as F
from .. import init as init


def _single(x):
    return (x,) if isinstance(x, int) else tuple(x)


def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def _triple(x):
    return (x, x, x) if isinstance(x, int) else tuple(x)


class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, transposed, output_padding, groups, bias, padding_mode='zeros',
                 device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            weight_shape = [in_channels, out_channels // groups] + list(kernel_size)
        else:
            weight_shape = [out_channels, in_channels // groups] + list(kernel_size)
        self.weight = Parameter(empty(*weight_shape, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(empty(out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}'
             f', stride={self.stride}')
        if self.padding != (0,) * len(self.padding):
            s += f', padding={self.padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self._parameters.get('bias') is None:
            s += ', bias=False'
        return s


class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding) if not isinstance(padding, str) else padding
        dilation = _single(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, False, (0,), groups, bias, padding_mode, device, dtype)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding) if not isinstance(padding, str) else padding
        dilation = _pair(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, False, (0, 0), groups, bias, padding_mode, device, dtype)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvTranspose1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros',
                 device=None, dtype=None):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, True, output_padding, groups, bias, padding_mode, device, dtype)

    def forward(self, input, output_size=None):
        return F.conv_transpose1d(input, self.weight, self.bias, self.stride,
                                  self.padding, self.output_padding, self.groups, self.dilation)


class ConvTranspose2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros',
                 device=None, dtype=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, True, output_padding, groups, bias, padding_mode, device, dtype)

    def forward(self, input, output_size=None):
        return F.conv_transpose2d(input, self.weight, self.bias, self.stride,
                                  self.padding, self.output_padding, self.groups, self.dilation)


class Conv3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding) if not isinstance(padding, str) else padding
        dilation = _triple(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, False, (0, 0, 0), groups, bias, padding_mode, device, dtype)

    def forward(self, input):
        return F.conv3d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvTranspose3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros',
                 device=None, dtype=None):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, True, output_padding, groups, bias, padding_mode, device, dtype)

    def forward(self, input, output_size=None):
        return F.conv_transpose3d(input, self.weight, self.bias, self.stride,
                                  self.padding, self.output_padding, self.groups, self.dilation)
