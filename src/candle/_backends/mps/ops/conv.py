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

def conv1d(input, weight, bias=None, stride=(1,), padding=(0,), dilation=(1,), groups=1):
    """Conv1d via conv2d: unsqueeze spatial dim."""
    from ...._tensor import Tensor
    # Unsqueeze to 4D and delegate to conv2d (which has GPU path)
    inp_shape = input.shape
    w_shape = weight.shape
    # Build 4D tensors staying on device
    inp4d = input.unsqueeze(2)  # (N, C, 1, L)
    w4d = weight.unsqueeze(2)   # (O, C/g, 1, kL)
    out = conv2d(inp4d, w4d, bias, stride=(1, stride[0]),
                 padding=(0, padding[0]), dilation=(1, dilation[0]), groups=groups)
    # Squeeze H=1 dim
    return out.squeeze(2)

def _conv2d_gpu(input, weight, bias, sH, sW, pH, pW, dH, dW,
                N, C_in, H_in, W_in, C_out, kH, kW, H_out, W_out):
    """GPU conv2d via Metal compute shader."""
    d = _get_dispatcher()
    sfx = _kernel_suffix(input.dtype)
    numel = N * C_out * H_out * W_out

    input_buf = _metal_buf(input)
    weight_buf = _metal_buf(weight)
    out_buf = _alloc_output_buf(numel, input.dtype)

    # Bias: need a valid buffer even when None (Metal requires all bindings)
    has_bias = 0
    if bias is not None and _can_use_gpu(bias):
        bias_buf = _metal_buf(bias)
        has_bias = 1
    else:
        from ..runtime import get_runtime
        bias_buf = get_runtime().create_buffer(4)  # dummy 4 bytes

    d.dispatch_conv2d(
        f"conv2d_{sfx}", input_buf, weight_buf, bias_buf, out_buf,
        N, C_in, H_in, W_in, C_out, kH, kW,
        H_out, W_out, sH, sW, pH, pW, dH, dW,
        has_bias, numel)

    out_shape = (N, C_out, H_out, W_out)
    out_stride = tuple()
    s = 1
    for dim in reversed(out_shape):
        out_stride = (s,) + out_stride
        s *= dim
    return _from_metal_buffer(out_buf, out_shape, out_stride, input.dtype, input.device)

def conv2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """Conv2d forward. Input: (N,C,H,W), Weight: (O,C/g,kH,kW).

    GPU path via MPSCNNConvolution for float32/float16, contiguous, groups=1.
    Falls back to numpy otherwise.
    """
    # GPU path: float32/float16, contiguous, groups=1
    if (groups == 1 and _can_use_gpu(input) and _can_use_gpu(weight) and
        input.dtype in (float32_dtype, float16_dtype) and
        input.is_contiguous() and weight.is_contiguous()):
        N, C_in, H_in, W_in = input.shape
        C_out, C_in_w, kH, kW = weight.shape
        if C_in == C_in_w:
            sH, sW = stride if isinstance(stride, tuple) else (stride, stride)
            pH, pW = padding if isinstance(padding, tuple) else (padding, padding)
            dH, dW = dilation if isinstance(dilation, tuple) else (dilation, dilation)

            # Check output dimensions are valid
            H_out = (H_in + 2 * pH - (kH - 1) * dH - 1) // sH + 1
            W_out = (W_in + 2 * pW - (kW - 1) * dW - 1) // sW + 1
            if H_out > 0 and W_out > 0:
                return _conv2d_gpu(input, weight, bias,
                                   sH, sW, pH, pW, dH, dW,
                                   N, C_in, H_in, W_in, C_out, kH, kW, H_out, W_out)

    # NumPy fallback
    inp = _to_numpy(input)
    w = _to_numpy(weight)
    N, C_in, H, W = inp.shape
    C_out, C_in_g, kH, kW = w.shape
    sH, sW = stride if isinstance(stride, tuple) else (stride, stride)
    pH, pW = padding if isinstance(padding, tuple) else (padding, padding)
    dH, dW = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    # Effective kernel size with dilation
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1

    H_out = (H + 2 * pH - ekH) // sH + 1
    W_out = (W + 2 * pW - ekW) // sW + 1

    # Pad input
    if pH > 0 or pW > 0:
        inp = np.pad(inp, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode='constant')

    out = np.zeros((N, C_out, H_out, W_out), dtype=inp.dtype)

    for g in range(groups):
        c_out_per_g = C_out // groups
        c_out_s = g * c_out_per_g
        c_in_s = g * C_in_g
        for co_local in range(c_out_per_g):
            co = c_out_s + co_local
            for ci_local in range(C_in_g):
                ci = c_in_s + ci_local
                kernel = w[co, ci_local]
                # For dilated kernels, use the dilated positions
                for oh in range(H_out):
                    for ow in range(W_out):
                        val = 0.0
                        for kh in range(kH):
                            for kw in range(kW):
                                ih = oh * sH + kh * dH
                                iw = ow * sW + kw * dW
                                # Broadcasting over batch dimension
                                out[:, co, oh, ow] += inp[:, ci, ih, iw] * kernel[kh, kw]

    if bias is not None:
        b = _to_numpy(bias)
        out += b[np.newaxis, :, np.newaxis, np.newaxis]

    return _from_numpy(np.ascontiguousarray(out.astype(inp.dtype)), input.dtype, input.device)

def conv3d(input, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1):
    """Conv3d forward using numpy. Input: (N,C,D,H,W), Weight: (O,C/g,kD,kH,kW)."""
    inp = _to_numpy(input)
    w = _to_numpy(weight)
    N, C_in, D, H, W = inp.shape
    C_out, C_in_g, kD, kH, kW = w.shape
    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    ekD = (kD - 1) * dD + 1
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1

    D_out = (D + 2 * pD - ekD) // sD + 1
    H_out = (H + 2 * pH - ekH) // sH + 1
    W_out = (W + 2 * pW - ekW) // sW + 1

    if pD > 0 or pH > 0 or pW > 0:
        inp = np.pad(inp, ((0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)), mode='constant')

    out = np.zeros((N, C_out, D_out, H_out, W_out), dtype=inp.dtype)

    for g in range(groups):
        c_out_per_g = C_out // groups
        c_out_s = g * c_out_per_g
        c_in_s = g * C_in_g
        for co_local in range(c_out_per_g):
            co = c_out_s + co_local
            for ci_local in range(C_in_g):
                ci = c_in_s + ci_local
                kernel = w[co, ci_local]
                for od in range(D_out):
                    for oh in range(H_out):
                        for ow in range(W_out):
                            for kd in range(kD):
                                for kh in range(kH):
                                    for kw in range(kW):
                                        id_ = od * sD + kd * dD
                                        ih = oh * sH + kh * dH
                                        iw = ow * sW + kw * dW
                                        out[:, co, od, oh, ow] += inp[:, ci, id_, ih, iw] * kernel[kd, kh, kw]

    if bias is not None:
        b = _to_numpy(bias)
        out += b.reshape(1, C_out, 1, 1, 1)

    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)

def conv_transpose1d(input, weight, bias=None, stride=(1,), padding=(0,),
                     output_padding=(0,), groups=1, dilation=(1,)):
    """Conv_transpose1d via conv_transpose2d: unsqueeze spatial dim."""
    inp = _to_numpy(input)   # (N, C, L)
    w = _to_numpy(weight)    # (C_in, C_out/g, kL)
    inp4d = inp[:, :, np.newaxis, :]
    w4d = w[:, :, np.newaxis, :]
    inp_t = _from_numpy(np.ascontiguousarray(inp4d), input.dtype, input.device)
    w_t = _from_numpy(np.ascontiguousarray(w4d), weight.dtype, weight.device)
    out = conv_transpose2d(inp_t, w_t, bias, stride=(1, stride[0]),
                           padding=(0, padding[0]), output_padding=(0, output_padding[0]),
                           groups=groups, dilation=(1, dilation[0]))
    out_np = _to_numpy(out)[:, :, 0, :]
    return _from_numpy(np.ascontiguousarray(out_np), input.dtype, input.device)

def conv_transpose2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0),
                     output_padding=(0, 0), groups=1, dilation=(1, 1)):
    """Transposed conv2d using numpy."""
    inp = _to_numpy(input)
    w = _to_numpy(weight)   # (C_in, C_out/g, kH, kW) — note: transposed weight layout
    N, C_in, H_in, W_in = inp.shape
    C_in_w, C_out_g, kH, kW = w.shape
    sH, sW = stride
    pH, pW = padding
    opH, opW = output_padding
    dH, dW = dilation

    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1

    H_out = (H_in - 1) * sH - 2 * pH + ekH + opH
    W_out = (W_in - 1) * sW - 2 * pW + ekW + opW
    C_out = C_out_g * groups

    # We compute in a padded buffer and then slice
    H_buf = (H_in - 1) * sH + ekH
    W_buf = (W_in - 1) * sW + ekW
    buf = np.zeros((N, C_out, H_buf, W_buf), dtype=inp.dtype)

    for g in range(groups):
        c_in_per_g = C_in // groups
        c_in_s = g * c_in_per_g
        c_out_s = g * C_out_g
        for ci_local in range(c_in_per_g):
            ci = c_in_s + ci_local
            for co_local in range(C_out_g):
                co = c_out_s + co_local
                kernel = w[ci_local, co_local]
                for ih in range(H_in):
                    for iw in range(W_in):
                        oh_start = ih * sH
                        ow_start = iw * sW
                        for kh in range(kH):
                            for kw in range(kW):
                                oh = oh_start + kh * dH
                                ow = ow_start + kw * dW
                                buf[:, co, oh, ow] += inp[:, ci, ih, iw] * kernel[kh, kw]

    # Slice to remove padding and apply output_padding
    out = buf[:, :, pH:pH + H_out, pW:pW + W_out]

    if bias is not None:
        b = _to_numpy(bias)
        out = out + b[np.newaxis, :, np.newaxis, np.newaxis]

    return _from_numpy(np.ascontiguousarray(out.astype(inp.dtype)), input.dtype, input.device)

def _conv_transpose3d_scatter(out, a, w, n, c_in, c_out_start, C_out_per_g,
                               D_in, H_in, W_in, D_out, H_out, W_out,
                               kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW):
    """Scatter one input channel's contribution for conv_transpose3d."""
    for d in range(D_in):
        for h in range(H_in):
            for wi in range(W_in):
                val = a[n, c_in, d, h, wi]
                if val == 0:
                    continue
                for c_out_local in range(C_out_per_g):
                    c_out = c_out_start + c_out_local
                    for kd in range(kD):
                        od = d * sD - pD + kd * dD
                        if od < 0 or od >= D_out:
                            continue
                        for kh in range(kH):
                            oh = h * sH - pH + kh * dH
                            if oh < 0 or oh >= H_out:
                                continue
                            for kw in range(kW):
                                ow = wi * sW - pW + kw * dW
                                if ow < 0 or ow >= W_out:
                                    continue
                                out[n, c_out, od, oh, ow] += val * w[c_in, c_out_local, kd, kh, kw]

def conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation):
    """Transposed convolution 3D via numpy."""
    a = _to_numpy(input)
    w = _to_numpy(weight)
    sD, sH, sW = stride
    pD, pH, pW = padding
    opD, opH, opW = output_padding
    dD, dH, dW = dilation

    N, C_in, D_in, H_in, W_in = a.shape
    C_in_w, C_out_per_g, kD, kH, kW = w.shape
    C_out = C_out_per_g * groups

    D_out = (D_in - 1) * sD - 2 * pD + dD * (kD - 1) + opD + 1
    H_out = (H_in - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1
    W_out = (W_in - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1

    out = np.zeros((N, C_out, D_out, H_out, W_out), dtype=a.dtype)
    c_in_per_g = C_in // groups

    for g in range(groups):
        for n in range(N):
            for c_in_local in range(c_in_per_g):
                c_in = g * c_in_per_g + c_in_local
                _conv_transpose3d_scatter(out, a, w, n, c_in, g * C_out_per_g, C_out_per_g,
                                          D_in, H_in, W_in, D_out, H_out, W_out,
                                          kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW)

    if bias is not None:
        b = _to_numpy(bias)
        out += b.reshape(1, C_out, 1, 1, 1)

    return _from_numpy(out, input.dtype, input.device)

def max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode=False, return_indices=False):
    """Max pooling 1D — GPU composite via lift to max_pool2d."""
    # GPU path: lift (N,C,W) → (N,C,1,W), use 2D shader, squeeze back
    if (_can_use_gpu(input) and input.is_contiguous()
            and input.dtype in (float32_dtype, float16_dtype)
            and not return_indices and input.ndim == 3):
        from ...common.view import reshape
        N, C, W = input.shape
        inp4d = reshape(input, (N, C, 1, W))
        out4d = max_pool2d(inp4d, (1, kernel_size[0]), (1, stride[0]),
                           (0, padding[0]), (1, dilation[0]), ceil_mode)
        return reshape(out4d, (N, C, out4d.shape[3]))
    a = _to_numpy(input)
    kW = kernel_size[0]
    sW = stride[0]
    pW = padding[0]
    dW = dilation[0]

    if input.ndim == 3:
        N, C, W = a.shape
    else:
        raise ValueError(f"Expected 3D input, got {input.ndim}D")

    # Pad input
    if pW > 0:
        a = np.pad(a, ((0, 0), (0, 0), (pW, pW)), mode='constant', constant_values=-np.inf)
    W_padded = a.shape[2]

    if ceil_mode:
        oW = int(np.ceil((W_padded - dW * (kW - 1) - 1) / sW + 1))
    else:
        oW = int(np.floor((W_padded - dW * (kW - 1) - 1) / sW + 1))

    out = np.empty((N, C, oW), dtype=a.dtype)
    for ow in range(oW):
        w_start = ow * sW
        vals = [a[:, :, w_start + k * dW] for k in range(kW) if w_start + k * dW < W_padded]
        out[:, :, ow] = np.maximum.reduce(vals)

    return _from_numpy(out, input.dtype, input.device)

def max_pool2d(input, kernel_size, stride, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """MaxPool2d with GPU path for float32/float16."""
    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    sH, sW = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pH, pW = (padding, padding) if isinstance(padding, int) else tuple(padding)
    dH, dW = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)

    if len(input.shape) == 4:
        N, C, H, W = input.shape
    else:
        N, C, H, W = 1, input.shape[0], input.shape[1], input.shape[2]

    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1
    if ceil_mode:
        H_out = math.ceil((H + 2 * pH - ekH) / sH) + 1
        W_out = math.ceil((W + 2 * pW - ekW) / sW) + 1
    else:
        H_out = (H + 2 * pH - ekH) // sH + 1
        W_out = (W + 2 * pW - ekW) // sW + 1

    # GPU path
    if (_can_use_gpu(input) and input.is_contiguous()
            and input.dtype in (float32_dtype, float16_dtype)
            and not return_indices and len(input.shape) == 4
            and H_out > 0 and W_out > 0):
        import struct
        total = N * C * H_out * W_out
        sfx = _kernel_suffix(input.dtype)
        d = _get_dispatcher()
        out_buf = _alloc_output_buf(total, input.dtype)
        # params: N, C, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW, dH, dW, total
        params = struct.pack("15I", N, C, H, W, H_out, W_out, kH, kW,
                             sH, sW, pH, pW, dH, dW, total)
        d.dispatch_pool2d(f"max_pool2d_{sfx}", _metal_buf(input), out_buf,
                          params, total)
        out_shape = (N, C, H_out, W_out)
        s = 1
        out_stride = ()
        for dim in reversed(out_shape):
            out_stride = (s,) + out_stride
            s *= dim
        return _from_metal_buffer(out_buf, out_shape, out_stride, input.dtype, input.device)

    # NumPy fallback
    inp = _to_numpy(input)
    if len(inp.shape) == 4:
        N, C, H, W = inp.shape
    else:
        N, C, H, W = 1, inp.shape[0], inp.shape[1], inp.shape[2]

    if pH > 0 or pW > 0:
        inp = np.pad(inp, ((0, 0), (0, 0), (pH, pH), (pW, pW)),
                     mode='constant', constant_values=-np.inf)

    out = np.full((N, C, H_out, W_out), -np.inf, dtype=inp.dtype)
    for oh in range(H_out):
        for ow in range(W_out):
            for kh in range(kH):
                for kw in range(kW):
                    ih = oh * sH + kh * dH
                    iw = ow * sW + kw * dW
                    if ih < inp.shape[2] and iw < inp.shape[3]:
                        out[:, :, oh, ow] = np.maximum(out[:, :, oh, ow], inp[:, :, ih, iw])

    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)

def max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode=False, return_indices=False):
    """Max pooling 3D via numpy sliding window."""
    a = _to_numpy(input)
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    N, C, D, H, W = a.shape

    if pD > 0 or pH > 0 or pW > 0:
        a = np.pad(a, ((0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)),
                   mode='constant', constant_values=-np.inf)

    D_pad, H_pad, W_pad = a.shape[2], a.shape[3], a.shape[4]

    if ceil_mode:
        oD = int(np.ceil((D_pad - dD * (kD - 1) - 1) / sD + 1))
        oH = int(np.ceil((H_pad - dH * (kH - 1) - 1) / sH + 1))
        oW = int(np.ceil((W_pad - dW * (kW - 1) - 1) / sW + 1))
    else:
        oD = int(np.floor((D_pad - dD * (kD - 1) - 1) / sD + 1))
        oH = int(np.floor((H_pad - dH * (kH - 1) - 1) / sH + 1))
        oW = int(np.floor((W_pad - dW * (kW - 1) - 1) / sW + 1))

    out = np.empty((N, C, oD, oH, oW), dtype=a.dtype)
    for od in range(oD):
        for oh in range(oH):
            for ow in range(oW):
                d_start = od * sD
                h_start = oh * sH
                w_start = ow * sW
                window_vals = []
                for kd in range(kD):
                    di = d_start + kd * dD
                    if di >= D_pad:
                        continue
                    for kh in range(kH):
                        hi = h_start + kh * dH
                        if hi >= H_pad:
                            continue
                        for kw in range(kW):
                            wi = w_start + kw * dW
                            if wi >= W_pad:
                                continue
                            window_vals.append(a[:, :, di, hi, wi])
                if window_vals:
                    out[:, :, od, oh, ow] = np.maximum.reduce(window_vals)

    return _from_numpy(out, input.dtype, input.device)

def avg_pool1d(input, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True):
    """Avg pooling 1D — GPU composite via lift to avg_pool2d."""
    # GPU path: lift (N,C,W) → (N,C,1,W), use 2D shader, squeeze back
    if (_can_use_gpu(input) and input.is_contiguous()
            and input.dtype in (float32_dtype, float16_dtype)
            and count_include_pad and input.ndim == 3):
        from ...common.view import reshape
        N, C, W = input.shape
        inp4d = reshape(input, (N, C, 1, W))
        out4d = avg_pool2d(inp4d, (1, kernel_size[0]), (1, stride[0]),
                           (0, padding[0]), ceil_mode)
        return reshape(out4d, (N, C, out4d.shape[3]))
    a = _to_numpy(input)
    kW = kernel_size[0]
    sW = stride[0]
    pW = padding[0]

    N, C, W = a.shape

    if pW > 0:
        a = np.pad(a, ((0, 0), (0, 0), (pW, pW)), mode='constant', constant_values=0)
    W_padded = a.shape[2]

    if ceil_mode:
        oW = int(np.ceil((W_padded - kW) / sW + 1))
    else:
        oW = int(np.floor((W_padded - kW) / sW + 1))

    out = np.empty((N, C, oW), dtype=a.dtype)
    for ow in range(oW):
        w_start = ow * sW
        w_end = min(w_start + kW, W_padded)
        window = a[:, :, w_start:w_end]
        if count_include_pad:
            out[:, :, ow] = window.sum(axis=2) / kW
        else:
            real_start = max(w_start - pW, 0)
            real_end = min(w_end - pW, W)
            count = max(real_end - real_start, 1)
            out[:, :, ow] = window.sum(axis=2) / count

    return _from_numpy(out, input.dtype, input.device)

def avg_pool2d(input, kernel_size, stride, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None):
    """AvgPool2d with GPU path for float32/float16."""
    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    sH, sW = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pH, pW = (padding, padding) if isinstance(padding, int) else tuple(padding)

    if len(input.shape) == 4:
        N, C, H, W = input.shape
    else:
        N, C, H, W = 1, input.shape[0], input.shape[1], input.shape[2]

    if ceil_mode:
        H_out = math.ceil((H + 2 * pH - kH) / sH) + 1
        W_out = math.ceil((W + 2 * pW - kW) / sW) + 1
    else:
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

    # GPU path: count_include_pad only, no divisor_override
    if (_can_use_gpu(input) and input.is_contiguous()
            and input.dtype in (float32_dtype, float16_dtype)
            and len(input.shape) == 4 and count_include_pad
            and divisor_override is None
            and H_out > 0 and W_out > 0):
        import struct
        total = N * C * H_out * W_out
        sfx = _kernel_suffix(input.dtype)
        d = _get_dispatcher()
        out_buf = _alloc_output_buf(total, input.dtype)
        # params: N, C, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW, count_include_pad, total
        params = struct.pack("14I", N, C, H, W, H_out, W_out, kH, kW,
                             sH, sW, pH, pW, 1, total)
        d.dispatch_pool2d(f"avg_pool2d_{sfx}", _metal_buf(input), out_buf,
                          params, total)
        out_shape = (N, C, H_out, W_out)
        s = 1
        out_stride = ()
        for dim in reversed(out_shape):
            out_stride = (s,) + out_stride
            s *= dim
        return _from_metal_buffer(out_buf, out_shape, out_stride, input.dtype, input.device)

    # NumPy fallback
    inp = _to_numpy(input)
    if len(inp.shape) == 4:
        N, C, H, W = inp.shape
    else:
        N, C, H, W = 1, inp.shape[0], inp.shape[1], inp.shape[2]

    if pH > 0 or pW > 0:
        inp = np.pad(inp, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode='constant')

    out = np.zeros((N, C, H_out, W_out), dtype=inp.dtype)
    for oh in range(H_out):
        for ow in range(W_out):
            h_start = oh * sH
            w_start = ow * sW
            h_end = min(h_start + kH, inp.shape[2])
            w_end = min(w_start + kW, inp.shape[3])
            window = inp[:, :, h_start:h_end, w_start:w_end]
            if divisor_override is not None:
                out[:, :, oh, ow] = window.sum(axis=(-2, -1)) / divisor_override
            elif count_include_pad:
                out[:, :, oh, ow] = window.sum(axis=(-2, -1)) / (kH * kW)
            else:
                actual_h = min(h_end, H + pH) - max(h_start, pH)
                actual_w = min(w_end, W + pW) - max(w_start, pW)
                count = max(actual_h * actual_w, 1)
                out[:, :, oh, ow] = window.sum(axis=(-2, -1)) / count

    return _from_numpy(np.ascontiguousarray(out.astype(inp.dtype)), input.dtype, input.device)

def avg_pool3d(input, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True):
    """Avg pooling 3D via numpy sliding window."""
    a = _to_numpy(input)
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding

    N, C, D, H, W = a.shape

    if pD > 0 or pH > 0 or pW > 0:
        a = np.pad(a, ((0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)),
                   mode='constant', constant_values=0)

    D_pad, H_pad, W_pad = a.shape[2], a.shape[3], a.shape[4]

    if ceil_mode:
        oD = int(np.ceil((D_pad - kD) / sD + 1))
        oH = int(np.ceil((H_pad - kH) / sH + 1))
        oW = int(np.ceil((W_pad - kW) / sW + 1))
    else:
        oD = int(np.floor((D_pad - kD) / sD + 1))
        oH = int(np.floor((H_pad - kH) / sH + 1))
        oW = int(np.floor((W_pad - kW) / sW + 1))

    out = np.empty((N, C, oD, oH, oW), dtype=a.dtype)
    pool_size = kD * kH * kW
    for od in range(oD):
        for oh in range(oH):
            for ow in range(oW):
                d_start = od * sD
                h_start = oh * sH
                w_start = ow * sW
                d_end = min(d_start + kD, D_pad)
                h_end = min(h_start + kH, H_pad)
                w_end = min(w_start + kW, W_pad)
                window = a[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                if count_include_pad:
                    out[:, :, od, oh, ow] = window.sum(axis=(2, 3, 4)) / pool_size
                else:
                    count = window.shape[2] * window.shape[3] * window.shape[4]
                    out[:, :, od, oh, ow] = window.sum(axis=(2, 3, 4)) / max(count, 1)

    return _from_numpy(out, input.dtype, input.device)

def adaptive_avg_pool1d(input, output_size):
    """Adaptive avg pool 1D — GPU composite via lift to adaptive_avg_pool2d."""
    # GPU path: lift (N,C,W) → (N,C,1,W), use 2D shader, squeeze back
    oW = output_size[0]
    if (_can_use_gpu(input) and input.is_contiguous()
            and input.dtype in (float32_dtype, float16_dtype)
            and input.ndim == 3):
        from ...common.view import reshape
        N, C, W = input.shape
        inp4d = reshape(input, (N, C, 1, W))
        out4d = adaptive_avg_pool2d(inp4d, (1, oW))
        return reshape(out4d, (N, C, oW))
    a = _to_numpy(input)
    N, C, W = a.shape
    oW = output_size[0]

    out = np.empty((N, C, oW), dtype=a.dtype)
    for ow in range(oW):
        w_start = int(np.floor(ow * W / oW))
        w_end = int(np.ceil((ow + 1) * W / oW))
        out[:, :, ow] = a[:, :, w_start:w_end].mean(axis=2)

    return _from_numpy(out, input.dtype, input.device)

def adaptive_avg_pool2d(input, output_size):
    """AdaptiveAvgPool2d with GPU path."""
    if isinstance(output_size, int):
        oH = oW = output_size
    else:
        oH, oW = output_size
    # GPU path
    if (_can_use_gpu(input) and input.is_contiguous()
            and input.dtype in (float32_dtype, float16_dtype)
            and len(input.shape) == 4):
        N, C, H, W = input.shape
        total = N * C * oH * oW
        sfx = _kernel_suffix(input.dtype)
        d = _get_dispatcher()
        out_buf = _alloc_output_buf(total, input.dtype)
        d.dispatch_adaptive_avg_pool2d(
            f"adaptive_avg_pool2d_{sfx}", _metal_buf(input), out_buf,
            N, C, H, W, oH, oW, total)
        out_shape = (N, C, oH, oW)
        s = 1
        out_stride = ()
        for dim in reversed(out_shape):
            out_stride = (s,) + out_stride
            s *= dim
        return _from_metal_buffer(out_buf, out_shape, out_stride, input.dtype, input.device)
    inp = _to_numpy(input)  # (N, C, H, W)
    N, C, H, W = inp.shape
    out = np.zeros((N, C, oH, oW), dtype=inp.dtype)
    for oh in range(oH):
        h_start = oh * H // oH
        h_end = (oh + 1) * H // oH
        for ow in range(oW):
            w_start = ow * W // oW
            w_end = (ow + 1) * W // oW
            out[:, :, oh, ow] = inp[:, :, h_start:h_end, w_start:w_end].mean(axis=(-2, -1))
    return _from_numpy(np.ascontiguousarray(out.astype(inp.dtype)), input.dtype, input.device)


# ---------------------------------------------------------------------------
# Group 1: Math ops
# ---------------------------------------------------------------------------

def adaptive_avg_pool3d(input, output_size):
    """Adaptive avg pool 3D: compute regions from output_size."""
    a = _to_numpy(input)
    N, C, D, H, W = a.shape
    oD, oH, oW = output_size

    out = np.empty((N, C, oD, oH, oW), dtype=a.dtype)
    for od in range(oD):
        d_start = int(np.floor(od * D / oD))
        d_end = int(np.ceil((od + 1) * D / oD))
        for oh in range(oH):
            h_start = int(np.floor(oh * H / oH))
            h_end = int(np.ceil((oh + 1) * H / oH))
            for ow in range(oW):
                w_start = int(np.floor(ow * W / oW))
                w_end = int(np.ceil((ow + 1) * W / oW))
                out[:, :, od, oh, ow] = a[:, :, d_start:d_end, h_start:h_end, w_start:w_end].mean(axis=(2, 3, 4))

    return _from_numpy(out, input.dtype, input.device)

def adaptive_max_pool1d(input, output_size, return_indices=False):
    """Adaptive max pooling 1D via numpy."""
    a = _to_numpy(input)
    N, C, L = a.shape
    oL = output_size if isinstance(output_size, int) else output_size[0]

    out = np.empty((N, C, oL), dtype=a.dtype)
    if return_indices:
        out_indices = np.empty((N, C, oL), dtype=np.int64)
        for ol in range(oL):
            l_start = ol * L // oL
            l_end = (ol + 1) * L // oL
            region = a[:, :, l_start:l_end]
            out[:, :, ol] = region.max(axis=2)
            out_indices[:, :, ol] = region.argmax(axis=2) + l_start
        result = _from_numpy(out, input.dtype, input.device)
        return result, _from_numpy(out_indices, int64_dtype, input.device)

    for ol in range(oL):
        l_start = ol * L // oL
        l_end = (ol + 1) * L // oL
        region = a[:, :, l_start:l_end]
        out[:, :, ol] = region.max(axis=2)

    return _from_numpy(out, input.dtype, input.device)

def adaptive_max_pool2d(input, output_size, return_indices=False):
    """Adaptive max pooling 2D via numpy."""
    a = _to_numpy(input)
    N, C, H, W = a.shape
    if isinstance(output_size, int):
        oH = oW = output_size
    else:
        oH, oW = output_size

    out = np.empty((N, C, oH, oW), dtype=a.dtype)
    if return_indices:
        out_indices = np.empty((N, C, oH, oW), dtype=np.int64)
        for oh in range(oH):
            h_start = oh * H // oH
            h_end = (oh + 1) * H // oH
            for ow in range(oW):
                w_start = ow * W // oW
                w_end = (ow + 1) * W // oW
                region = a[:, :, h_start:h_end, w_start:w_end]
                region_flat = region.reshape(N, C, -1)
                out[:, :, oh, ow] = region_flat.max(axis=2)
                local_idx = region_flat.argmax(axis=2)
                rW = w_end - w_start
                local_h = local_idx // rW + h_start
                local_w = local_idx % rW + w_start
                out_indices[:, :, oh, ow] = local_h * W + local_w
        result = _from_numpy(out, input.dtype, input.device)
        return result, _from_numpy(out_indices, int64_dtype, input.device)

    for oh in range(oH):
        h_start = oh * H // oH
        h_end = (oh + 1) * H // oH
        for ow in range(oW):
            w_start = ow * W // oW
            w_end = (ow + 1) * W // oW
            region = a[:, :, h_start:h_end, w_start:w_end]
            region_flat = region.reshape(N, C, -1)
            out[:, :, oh, ow] = region_flat.max(axis=2)

    return _from_numpy(out, input.dtype, input.device)

def upsample_nearest1d(a, output_size):
    """Nearest-neighbor 1D upsampling — GPU composite via lift to upsample_nearest2d."""
    W_out = output_size[0]
    # GPU path: lift (N,C,W) → (N,C,1,W), use 2D shader, squeeze back
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)
            and len(a.shape) == 3):
        from ...common.view import reshape
        N, C, W_in = a.shape
        inp4d = reshape(a, (N, C, 1, W_in))
        out4d = upsample_nearest2d(inp4d, (1, W_out))
        return reshape(out4d, (N, C, W_out))
    arr = _to_numpy(a)
    W_out = output_size[0]
    W_in = arr.shape[2]
    indices = (np.arange(W_out, dtype=np.float64) * W_in / W_out).astype(np.intp)
    np.clip(indices, 0, W_in - 1, out=indices)
    out = arr[:, :, indices]
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def upsample_nearest2d(a, output_size):
    """Nearest-neighbor 2D upsampling with GPU path."""
    H_out, W_out = output_size
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)
            and len(a.shape) == 4):
        N, C, H_in, W_in = a.shape
        total = N * C * H_out * W_out
        sfx = _kernel_suffix(a.dtype)
        d = _get_dispatcher()
        out_buf = _alloc_output_buf(total, a.dtype)
        d.dispatch_upsample(f"upsample_nearest2d_{sfx}", _metal_buf(a), out_buf,
                            N, C, H_in, W_in, H_out, W_out, 0, total)
        out_shape = (N, C, H_out, W_out)
        s = 1
        out_stride = ()
        for dim in reversed(out_shape):
            out_stride = (s,) + out_stride
            s *= dim
        return _from_metal_buffer(out_buf, out_shape, out_stride, a.dtype, a.device)
    arr = _to_numpy(a)
    H_in, W_in = arr.shape[2], arr.shape[3]
    h_idx = (np.arange(H_out, dtype=np.float64) * H_in / H_out).astype(np.intp)
    w_idx = (np.arange(W_out, dtype=np.float64) * W_in / W_out).astype(np.intp)
    np.clip(h_idx, 0, H_in - 1, out=h_idx)
    np.clip(w_idx, 0, W_in - 1, out=w_idx)
    out = arr[:, :, h_idx[:, None], w_idx[None, :]]
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def upsample_linear1d(a, output_size, align_corners=False, scales=None):
    """Linear interpolation 1D upsampling. Input: (N, C, W_in) -> (N, C, W_out)."""
    arr = _to_numpy(a).astype(np.float64)
    W_out = output_size[0]
    W_in = arr.shape[2]

    if W_out == 1:
        out = arr[:, :, :1]
        return _from_numpy(np.ascontiguousarray(out.astype(_to_numpy(a).dtype)), a.dtype, a.device)

    if align_corners and W_in > 1 and W_out > 1:
        x = np.linspace(0, W_in - 1, W_out)
    else:
        x = (np.arange(W_out, dtype=np.float64) + 0.5) * W_in / W_out - 0.5

    x = np.clip(x, 0, W_in - 1)
    x0 = np.floor(x).astype(np.intp)
    x1 = np.minimum(x0 + 1, W_in - 1)
    wx = (x - x0).reshape(1, 1, -1)

    out = arr[:, :, x0] * (1.0 - wx) + arr[:, :, x1] * wx
    return _from_numpy(np.ascontiguousarray(out.astype(_to_numpy(a).dtype)), a.dtype, a.device)

def upsample_bilinear2d(a, output_size, align_corners=False, scales_h=None, scales_w=None):
    """Bilinear 2D upsampling with GPU path."""
    H_out, W_out = output_size
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)
            and len(a.shape) == 4):
        N, C, H_in, W_in = a.shape
        total = N * C * H_out * W_out
        sfx = _kernel_suffix(a.dtype)
        d = _get_dispatcher()
        out_buf = _alloc_output_buf(total, a.dtype)
        ac = 1 if align_corners else 0
        d.dispatch_upsample(f"upsample_bilinear2d_{sfx}", _metal_buf(a), out_buf,
                            N, C, H_in, W_in, H_out, W_out, ac, total)
        out_shape = (N, C, H_out, W_out)
        s = 1
        out_stride = ()
        for dim in reversed(out_shape):
            out_stride = (s,) + out_stride
            s *= dim
        return _from_metal_buffer(out_buf, out_shape, out_stride, a.dtype, a.device)
    arr = _to_numpy(a).astype(np.float64)
    H_out, W_out = output_size
    H_in, W_in = arr.shape[2], arr.shape[3]

    if align_corners and H_in > 1 and H_out > 1:
        h = np.linspace(0, H_in - 1, H_out)
    else:
        h = (np.arange(H_out, dtype=np.float64) + 0.5) * H_in / H_out - 0.5
    if align_corners and W_in > 1 and W_out > 1:
        w = np.linspace(0, W_in - 1, W_out)
    else:
        w = (np.arange(W_out, dtype=np.float64) + 0.5) * W_in / W_out - 0.5

    h = np.clip(h, 0, H_in - 1)
    w = np.clip(w, 0, W_in - 1)

    h0 = np.floor(h).astype(np.intp)
    h1 = np.minimum(h0 + 1, H_in - 1)
    w0 = np.floor(w).astype(np.intp)
    w1 = np.minimum(w0 + 1, W_in - 1)

    wh = (h - h0).reshape(1, 1, -1, 1)
    ww = (w - w0).reshape(1, 1, 1, -1)

    out = (arr[:, :, h0[:, None], w0[None, :]] * (1 - wh) * (1 - ww) +
           arr[:, :, h0[:, None], w1[None, :]] * (1 - wh) * ww +
           arr[:, :, h1[:, None], w0[None, :]] * wh * (1 - ww) +
           arr[:, :, h1[:, None], w1[None, :]] * wh * ww)
    return _from_numpy(np.ascontiguousarray(out.astype(_to_numpy(a).dtype)), a.dtype, a.device)

def upsample_bicubic2d(a, output_size, align_corners=False, scales_h=None, scales_w=None):
    """Bicubic 2D upsampling. Input: (N, C, H_in, W_in) -> (N, C, H_out, W_out)."""
    arr = _to_numpy(a).astype(np.float64)
    H_out, W_out = output_size
    H_in, W_in = arr.shape[2], arr.shape[3]

    def _cubic_weight(t):
        """Keys cubic kernel (a=-0.75)."""
        at = np.abs(t)
        return np.where(at <= 1, (1.5 * at - 2.5) * at * at + 1,
               np.where(at < 2, ((-0.5 * at + 2.5) * at - 4) * at + 2, 0.0))

    if align_corners and H_in > 1 and H_out > 1:
        h = np.linspace(0, H_in - 1, H_out)
    else:
        h = (np.arange(H_out, dtype=np.float64) + 0.5) * H_in / H_out - 0.5
    if align_corners and W_in > 1 and W_out > 1:
        w = np.linspace(0, W_in - 1, W_out)
    else:
        w = (np.arange(W_out, dtype=np.float64) + 0.5) * W_in / W_out - 0.5

    N, C = arr.shape[:2]
    out = np.zeros((N, C, H_out, W_out), dtype=np.float64)
    for j in range(H_out):
        for i in range(W_out):
            hy, wx = h[j], w[i]
            hy0 = int(np.floor(hy)) - 1
            wx0 = int(np.floor(wx)) - 1
            for dh in range(4):
                for dw in range(4):
                    hh = min(max(hy0 + dh, 0), H_in - 1)
                    ww = min(max(wx0 + dw, 0), W_in - 1)
                    weight = _cubic_weight(hy - (hy0 + dh)) * _cubic_weight(wx - (wx0 + dw))
                    out[:, :, j, i] += arr[:, :, hh, ww] * weight
    return _from_numpy(np.ascontiguousarray(out.astype(_to_numpy(a).dtype)), a.dtype, a.device)

def pad(a, pad_widths, mode='constant', value=0):
    ndim = len(a.shape)

    if len(pad_widths) % 2 != 0:
        raise ValueError("Padding length must be divisible by 2")

    n_pairs = len(pad_widths) // 2
    if n_pairs > ndim:
        raise ValueError("Padding length too large for input dimensions")

    pads = [(0, 0)] * ndim
    for i in range(n_pairs):
        dim = ndim - 1 - i
        pads[dim] = (int(pad_widths[2 * i]), int(pad_widths[2 * i + 1]))

    # GPU path: constant mode, no negative padding, contiguous float/half
    has_negative = any(l < 0 or r < 0 for l, r in pads)
    if (mode == 'constant' and not has_negative
            and _can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)
            and ndim <= 8):
        import struct
        out_shape = []
        pad_before = []
        for d in range(ndim):
            left, right = pads[d]
            out_shape.append(a.shape[d] + left + right)
            pad_before.append(left)
        total = 1
        for s in out_shape:
            total *= s
        if total > 0:
            sfx = _kernel_suffix(a.dtype)
            d = _get_dispatcher()
            out_buf = _alloc_output_buf(total, a.dtype)
            in_shape_packed = struct.pack(f"{ndim}I", *a.shape)
            pad_before_packed = struct.pack(f"{ndim}i", *pad_before)
            out_shape_packed = struct.pack(f"{ndim}I", *out_shape)
            if a.dtype == float32_dtype:
                fill_bytes = struct.pack("f", float(value))
                fill_size = 4
            else:
                import numpy as np
                fill_bytes = np.float16(value).tobytes()
                fill_size = 2
            d.dispatch_pad_constant(
                f"pad_constant_{sfx}", _metal_buf(a), out_buf,
                in_shape_packed, pad_before_packed, out_shape_packed,
                fill_bytes, fill_size, ndim, total)
            out_shape_t = tuple(out_shape)
            s = 1
            out_stride = ()
            for dim in reversed(out_shape_t):
                out_stride = (s,) + out_stride
                s *= dim
            return _from_metal_buffer(out_buf, out_shape_t, out_stride, a.dtype, a.device)

    arr = _to_numpy(a)

    # Negative padding crops first, then positive padding extends.
    slices = [slice(None)] * ndim
    for dim, (left, right) in enumerate(pads):
        start = max(-left, 0)
        end = arr.shape[dim] - max(-right, 0)
        length = end - start
        if length < 0:
            raise RuntimeError("narrow(): length must be non-negative.")
        slices[dim] = slice(start, end)
    result = arr[tuple(slices)]

    np_pad = [(max(left, 0), max(right, 0)) for left, right in pads]
    if any(left or right for left, right in np_pad):
        if mode == 'constant':
            result = np.pad(result, np_pad, mode='constant', constant_values=value)
        elif mode == 'reflect':
            result = np.pad(result, np_pad, mode='reflect')
        elif mode == 'replicate':
            result = np.pad(result, np_pad, mode='edge')
        elif mode == 'circular':
            result = np.pad(result, np_pad, mode='wrap')
        else:
            raise ValueError(f"Unsupported pad mode: {mode}")
    else:
        result = np.ascontiguousarray(result)

    return _from_numpy(result, a.dtype, a.device)

def im2col(a, kernel_size, dilation, padding, stride):
    """F.unfold: Extract sliding local blocks from 4D input (N, C, H, W).

    Returns tensor of shape (N, C*kH*kW, L) where L = number of valid blocks.
    kernel_size, dilation, padding, stride are all 2-tuples.
    """
    arr = _to_numpy(a)
    N, C, H, W = arr.shape
    kH, kW = kernel_size
    dH, dW = dilation
    pH, pW = padding
    sH, sW = stride

    # Effective kernel size with dilation
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1

    # Output spatial dimensions
    H_out = (H + 2 * pH - ekH) // sH + 1
    W_out = (W + 2 * pW - ekW) // sW + 1
    L = H_out * W_out

    # Pad input if needed
    if pH > 0 or pW > 0:
        arr = np.pad(arr, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode='constant')

    # Ensure contiguous for stride tricks
    arr = np.ascontiguousarray(arr)

    # Use stride tricks to extract patches efficiently
    # Shape: (N, C, kH, kW, H_out, W_out)
    shape = (N, C, kH, kW, H_out, W_out)
    strides = (
        arr.strides[0],          # batch
        arr.strides[1],          # channel
        arr.strides[2] * dH,     # kernel height (dilated)
        arr.strides[3] * dW,     # kernel width (dilated)
        arr.strides[2] * sH,     # output height (stride)
        arr.strides[3] * sW,     # output width (stride)
    )

    patches = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    # Reshape to (N, C*kH*kW, H_out*W_out)
    out = patches.reshape(N, C * kH * kW, L)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def col2im(a, output_size, kernel_size, dilation, padding, stride):
    """F.fold: Combine array of sliding local blocks into a 4D tensor.

    Input shape: (N, C*kH*kW, L)
    Returns tensor of shape (N, C, output_size[0], output_size[1]).
    Overlapping regions are summed (matching PyTorch behavior).
    """
    arr = _to_numpy(a)
    N, C_kk, L = arr.shape
    kH, kW = kernel_size
    dH, dW = dilation
    pH, pW = padding
    sH, sW = stride
    H_out, W_out = output_size

    # Effective kernel size with dilation
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1

    # Number of output positions
    H_col = (H_out + 2 * pH - ekH) // sH + 1
    W_col = (W_out + 2 * pW - ekW) // sW + 1

    if H_col * W_col != L:
        raise ValueError(
            f"Expected L={H_col * W_col} (H_col={H_col}, W_col={W_col}) but got L={L}"
        )

    C = C_kk // (kH * kW)
    if C * kH * kW != C_kk:
        raise ValueError(
            f"C*kH*kW ({C}*{kH}*{kW}={C * kH * kW}) does not match input dim 1 ({C_kk})"
        )

    # Padded output dimensions
    H_pad = H_out + 2 * pH
    W_pad = W_out + 2 * pW

    # Create output with padding
    out = np.zeros((N, C, H_pad, W_pad), dtype=arr.dtype)

    # Reshape input columns to (N, C, kH, kW, H_col, W_col)
    cols = arr.reshape(N, C, kH, kW, H_col, W_col)

    # Scatter-add patches back into output
    for i in range(kH):
        for j in range(kW):
            h_start = i * dH
            w_start = j * dW
            for h_idx in range(H_col):
                for w_idx in range(W_col):
                    out[:, :, h_start + h_idx * sH, w_start + w_idx * sW] += cols[:, :, i, j, h_idx, w_idx]

    # Remove padding
    if pH > 0 or pW > 0:
        out = out[:, :, pH:pH + H_out, pW:pW + W_out]

    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


# ---------------------------------------------------------------------------
# uniform (out-of-place) — needed by F.gumbel_softmax
# ---------------------------------------------------------------------------

def affine_grid(theta, size, align_corners=None):
    """Generate 2D or 3D affine sampling grid from transformation matrix.

    Args:
        theta: Tensor of shape (N, 2, 3) for 4-D or (N, 3, 4) for 5-D.
        size: output spatial size, e.g. torch.Size([N, C, H, W]).
        align_corners: if True, [-1, 1] maps to pixel centres at corners;
                       if False, [-1, 1] maps to the edges of the corner pixels.
    Returns:
        grid: (N, H, W, 2) for 4-D or (N, D, H, W, 3) for 5-D.
    """
    if align_corners is None:
        align_corners = False

    theta_np = _to_numpy(theta).astype(np.float32)  # (N, 2, 3) or (N, 3, 4)
    N = theta_np.shape[0]

    if len(size) == 4:
        # 2-D spatial
        _, _, H, W = size
        # Build base grid of (x, y, 1) coordinates
        if align_corners:
            # linspace from -1 to 1 with H / W points
            ys = np.linspace(-1.0, 1.0, H, dtype=np.float32) if H > 1 else np.array([0.0], dtype=np.float32)
            xs = np.linspace(-1.0, 1.0, W, dtype=np.float32) if W > 1 else np.array([0.0], dtype=np.float32)
        else:
            # coordinates centred in each cell, ranging from -1+1/W to 1-1/W
            ys = (np.arange(H, dtype=np.float32) * 2.0 + 1.0) / H - 1.0 if H > 0 else np.zeros(0, dtype=np.float32)
            xs = (np.arange(W, dtype=np.float32) * 2.0 + 1.0) / W - 1.0 if W > 0 else np.zeros(0, dtype=np.float32)

        # meshgrid: grid_x (H, W), grid_y (H, W)
        grid_x, grid_y = np.meshgrid(xs, ys)  # both (H, W)
        ones = np.ones_like(grid_x)

        # base_grid: (H*W, 3) — columns are x, y, 1
        base_grid = np.stack([grid_x.ravel(), grid_y.ravel(), ones.ravel()], axis=1)  # (H*W, 3)

        # Apply theta: out = base_grid @ theta^T  =>  (N, H*W, 2)
        # theta: (N, 2, 3), base_grid: (H*W, 3)
        # result[n] = base_grid @ theta[n].T  ->  (H*W, 2)
        out = np.einsum('ij,nkj->nik', base_grid, theta_np)  # (N, H*W, 2)
        out = out.reshape(N, H, W, 2)

    elif len(size) == 5:
        # 3-D spatial
        _, _, D, H, W = size
        if align_corners:
            zs = np.linspace(-1.0, 1.0, D, dtype=np.float32) if D > 1 else np.array([0.0], dtype=np.float32)
            ys = np.linspace(-1.0, 1.0, H, dtype=np.float32) if H > 1 else np.array([0.0], dtype=np.float32)
            xs = np.linspace(-1.0, 1.0, W, dtype=np.float32) if W > 1 else np.array([0.0], dtype=np.float32)
        else:
            zs = (np.arange(D, dtype=np.float32) * 2.0 + 1.0) / D - 1.0 if D > 0 else np.zeros(0, dtype=np.float32)
            ys = (np.arange(H, dtype=np.float32) * 2.0 + 1.0) / H - 1.0 if H > 0 else np.zeros(0, dtype=np.float32)
            xs = (np.arange(W, dtype=np.float32) * 2.0 + 1.0) / W - 1.0 if W > 0 else np.zeros(0, dtype=np.float32)

        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='xy')
        # grid_x, grid_y, grid_z are (H, W, D) due to meshgrid 'xy' indexing
        # Transpose to (D, H, W)
        grid_x = np.transpose(grid_x, (2, 0, 1))
        grid_y = np.transpose(grid_y, (2, 0, 1))
        grid_z = np.transpose(grid_z, (2, 0, 1))
        ones = np.ones_like(grid_x)

        base_grid = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel(), ones.ravel()], axis=1)  # (D*H*W, 4)
        out = np.einsum('ij,nkj->nik', base_grid, theta_np)  # (N, D*H*W, 3)
        out = out.reshape(N, D, H, W, 3)
    else:
        raise ValueError("affine_grid only supports 4-D (2-D spatial) and 5-D (3-D spatial) inputs")

    return _from_numpy(np.ascontiguousarray(out), theta.dtype, theta.device)

def _grid_sample_denormalize(coord, length, align_corners):
    """Convert normalised coordinate [-1, 1] to pixel coordinate."""
    if align_corners:
        # -1 -> 0, 1 -> length-1
        return ((coord + 1.0) / 2.0) * (length - 1)
    else:
        # -1 -> -0.5, 1 -> length - 0.5
        return ((coord + 1.0) * length - 1.0) / 2.0

def _grid_sample_compute_source(coord, length, padding_mode):
    """Apply padding mode to source pixel coordinate.

    Returns (index_array, in_bounds_mask).
    For 'zeros': index is clamped but mask marks OOB.
    For 'border': index is clamped.
    For 'reflection': index is reflected.
    """
    if padding_mode == 'border':
        coord = np.clip(coord, 0, length - 1)
        return coord, None
    elif padding_mode == 'reflection':
        # Reflect: the range [0, length-1] is mirrored.
        # Double the period and fold back.
        if length > 1:
            # Period = 2 * (length - 1)
            period = 2.0 * (length - 1)
            coord = np.abs(np.mod(coord, period))
            coord = np.where(coord > length - 1, period - coord, coord)
        else:
            coord = np.zeros_like(coord)
        coord = np.clip(coord, 0, length - 1)
        return coord, None
    else:
        # zeros padding: mark out-of-bounds, clamp index
        mask = (coord >= 0) & (coord < length)
        coord = np.clip(coord, 0, length - 1)
        return coord, mask

def _cubic_interp_weight(t):
    """Compute cubic convolution weights for distance t (|t| values).

    Uses the Keys cubic with a = -0.75 (same as PyTorch).
    Returns array of same shape as t.
    """
    t = np.abs(t)
    w = np.where(
        t <= 1.0,
        (1.5 * t - 2.5) * t * t + 1.0,
        np.where(
            t < 2.0,
            ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0,
            0.0,
        ),
    )
    return w

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
    """Sample input tensor at grid coordinates.

    Args:
        input: (N, C, H_in, W_in) tensor.
        grid:  (N, H_out, W_out, 2) tensor with values in [-1, 1].
               The last dimension is (x, y) where x indexes W and y indexes H.
        mode: 'bilinear', 'nearest', or 'bicubic'.
        padding_mode: 'zeros', 'border', or 'reflection'.
        align_corners: bool.
    Returns:
        (N, C, H_out, W_out) tensor.
    """
    if align_corners is None:
        align_corners = False

    inp = _to_numpy(input).astype(np.float64)  # (N, C, H_in, W_in)
    g = _to_numpy(grid).astype(np.float64)      # (N, H_out, W_out, 2)

    N, C, H_in, W_in = inp.shape
    _, H_out, W_out, _ = g.shape

    # De-normalise grid coordinates
    gx = _grid_sample_denormalize(g[..., 0], W_in, align_corners)  # (N, H_out, W_out)
    gy = _grid_sample_denormalize(g[..., 1], H_in, align_corners)  # (N, H_out, W_out)

    if mode == 'nearest':
        ix = np.round(gx).astype(np.int64)
        iy = np.round(gy).astype(np.int64)

        ix_src, mask_x = _grid_sample_compute_source(ix.astype(np.float64), W_in, padding_mode)
        iy_src, mask_y = _grid_sample_compute_source(iy.astype(np.float64), H_in, padding_mode)
        ix_src = ix_src.astype(np.int64)
        iy_src = iy_src.astype(np.int64)

        # Gather: out[n, c, h, w] = inp[n, c, iy_src[n,h,w], ix_src[n,h,w]]
        n_idx = np.arange(N)[:, None, None]  # (N, 1, 1)
        out = inp[n_idx, :, iy_src[:, :, :], ix_src[:, :, :]]  # (N, H_out, W_out, C)
        out = out.transpose(0, 3, 1, 2)  # (N, C, H_out, W_out)

        if padding_mode == 'zeros':
            mask = mask_x & mask_y  # (N, H_out, W_out)
            out = out * mask[:, np.newaxis, :, :]

    elif mode == 'bilinear':
        ix0 = np.floor(gx).astype(np.int64)
        iy0 = np.floor(gy).astype(np.int64)

        tx = gx - np.floor(gx)  # fractional part
        ty = gy - np.floor(gy)

        out = np.zeros((N, C, H_out, W_out), dtype=np.float64)

        for dy in range(2):
            for dx in range(2):
                cy = iy0 + dy
                cx = ix0 + dx

                cx_src, mask_x = _grid_sample_compute_source(cx.astype(np.float64), W_in, padding_mode)
                cy_src, mask_y = _grid_sample_compute_source(cy.astype(np.float64), H_in, padding_mode)
                cx_src = cx_src.astype(np.int64)
                cy_src = cy_src.astype(np.int64)

                wx = (1.0 - tx) if dx == 0 else tx
                wy = (1.0 - ty) if dy == 0 else ty
                w = wx * wy  # (N, H_out, W_out)

                n_idx = np.arange(N)[:, None, None]
                val = inp[n_idx, :, cy_src, cx_src]  # (N, H_out, W_out, C)
                val = val.transpose(0, 3, 1, 2)       # (N, C, H_out, W_out)

                if padding_mode == 'zeros':
                    mask = mask_x & mask_y
                    val = val * mask[:, np.newaxis, :, :]

                out += val * w[:, np.newaxis, :, :]

    elif mode == 'bicubic':
        ix0 = np.floor(gx).astype(np.int64)
        iy0 = np.floor(gy).astype(np.int64)

        tx = gx - np.floor(gx)
        ty = gy - np.floor(gy)

        out = np.zeros((N, C, H_out, W_out), dtype=np.float64)

        for dy in range(-1, 3):
            for dx in range(-1, 3):
                cy = iy0 + dy
                cx = ix0 + dx

                cx_src, mask_x = _grid_sample_compute_source(cx.astype(np.float64), W_in, padding_mode)
                cy_src, mask_y = _grid_sample_compute_source(cy.astype(np.float64), H_in, padding_mode)
                cx_src = cx_src.astype(np.int64)
                cy_src = cy_src.astype(np.int64)

                wx = _cubic_interp_weight(tx - dx)
                wy = _cubic_interp_weight(ty - dy)
                w = wx * wy

                n_idx = np.arange(N)[:, None, None]
                val = inp[n_idx, :, cy_src, cx_src]
                val = val.transpose(0, 3, 1, 2)

                if padding_mode == 'zeros':
                    mask = mask_x & mask_y
                    val = val * mask[:, np.newaxis, :, :]

                out += val * w[:, np.newaxis, :, :]

    else:
        raise ValueError(f"grid_sample mode must be 'bilinear', 'nearest', or 'bicubic', got '{mode}'")

    out = out.astype(to_numpy_dtype(input.dtype))
    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)


# ---------------------------------------------------------------------------
# F.unfold (im2col) / F.fold (col2im)
# ---------------------------------------------------------------------------

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
    """CTC Loss forward via numpy alpha (forward variable) algorithm.

    Args:
        log_probs: (T, N, C) log probabilities
        targets: (N, S) or (sum(target_lengths),) target sequences
        input_lengths: (N,) lengths of inputs
        target_lengths: (N,) lengths of targets
        blank: blank label index
        reduction: 'none', 'mean', or 'sum'
        zero_infinity: if True, zero out infinite losses
    """
    lp = _to_numpy(log_probs).astype(np.float64)
    T, N, C = lp.shape

    if isinstance(targets, Tensor):
        tgt = _to_numpy(targets)
    else:
        tgt = np.array(targets)

    if isinstance(input_lengths, Tensor):
        inp_lens = _to_numpy(input_lengths).astype(np.int64)
    else:
        inp_lens = np.array(input_lengths, dtype=np.int64)

    if isinstance(target_lengths, Tensor):
        tgt_lens = _to_numpy(target_lengths).astype(np.int64)
    else:
        tgt_lens = np.array(target_lengths, dtype=np.int64)

    NEG_INF = -1e30
    losses = np.zeros(N, dtype=np.float64)

    # Determine if targets is 1D (concatenated) or 2D
    is_1d = (tgt.ndim == 1)
    offset = 0

    for b in range(N):
        T_b = int(inp_lens[b])
        S_b = int(tgt_lens[b])

        if is_1d:
            labels_b = tgt[offset:offset + S_b]
            offset += S_b
        else:
            labels_b = tgt[b, :S_b]

        # Build extended labels with blanks: [blank, l0, blank, l1, blank, ...]
        L = 2 * S_b + 1
        ext = np.full(L, blank, dtype=np.int64)
        for s in range(S_b):
            ext[2 * s + 1] = labels_b[s]

        # Alpha (forward variables): alpha[t, s] = log-prob of emitting ext[:s+1] up to time t
        alpha = np.full((T_b, L), NEG_INF, dtype=np.float64)
        alpha[0, 0] = lp[0, b, ext[0]]
        if L > 1:
            alpha[0, 1] = lp[0, b, ext[1]]

        for t in range(1, T_b):
            for s in range(L):
                a = alpha[t - 1, s]
                if s > 0:
                    a = np.logaddexp(a, alpha[t - 1, s - 1])
                if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                    a = np.logaddexp(a, alpha[t - 1, s - 2])
                alpha[t, s] = a + lp[t, b, ext[s]]

        # Loss = -log(alpha[T-1, L-1] + alpha[T-1, L-2])
        log_likelihood = alpha[T_b - 1, L - 1]
        if L > 1:
            log_likelihood = np.logaddexp(log_likelihood, alpha[T_b - 1, L - 2])
        loss = -log_likelihood

        if zero_infinity and np.isinf(loss):
            loss = 0.0
        losses[b] = loss

    if reduction == 'none':
        result = losses
    elif reduction == 'sum':
        result = np.array(losses.sum(), dtype=np.float64)
    else:  # 'mean'
        tgt_lens_f = tgt_lens.astype(np.float64)
        tgt_lens_f = np.maximum(tgt_lens_f, 1.0)
        result = np.array((losses / tgt_lens_f).mean(), dtype=np.float64)

    out_dtype = log_probs.dtype
    return _from_numpy(result.astype(to_numpy_dtype(out_dtype)), out_dtype, log_probs.device)

