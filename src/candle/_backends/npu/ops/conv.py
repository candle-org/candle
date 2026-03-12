"""Convolution, pooling, upsampling, and padding operations for NPU."""
from ._helpers import (
    _unwrap_storage, _wrap_tensor, _unary_op, _binary_op,
    _broadcast_shape, _broadcast_shape_checked,
    _numel, _dtype_itemsize, _use_soc_fallback,
    _scalar_to_npu_tensor, _scalar_to_npu_tensor_no_add,
    _npu_broadcast_to, _npu_arange_1d, _npu_linear_index,
    _npu_add_scalar_, npu_index_put_impl,
    _normalize_reduction_dims, _reduce_out_shape,
    _cast_tensor_dtype, _normalize_tensor_sequence_args,
    _matmul_out_shape,
    _iter_indices, _broadcast_index, _batch_offset,
    bool_dtype, int32_dtype, int64_dtype, float_dtype,
    npu_typed_storage_from_ptr, reshape,
    aclnn, npu_runtime, npu_state, ops_soc,
)
from .elementwise import logaddexp, where
from .linalg import matmul
from .math import add, ceil, div, floor, isinf
from .reduce import maximum, mean, sum_
from .shape import contiguous, expand, gather, index_select, repeat, split, stack, tile


def conv2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """Conv2d forward using aclnnConvolution."""
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    N, C_in, H, W = input.shape
    C_out, C_in_g, kH, kW = weight.shape
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1
    H_out = (H + 2 * pH - ekH) // sH + 1
    W_out = (W + 2 * pW - ekW) // sW + 1
    out_shape = (N, C_out, H_out, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    bias_ptr = None
    bias_shape = None
    bias_stride = None
    if bias is not None:
        bias_ptr = _unwrap_storage(bias).data_ptr()
        bias_shape = bias.shape
        bias_stride = bias.stride

    aclnn.convolution(
        _unwrap_storage(input).data_ptr(),
        _unwrap_storage(weight).data_ptr(),
        bias_ptr,
        input.shape, input.stride,
        weight.shape, weight.stride,
        bias_shape, bias_stride,
        input.dtype,
        stride, padding, dilation,
        False,  # transposed
        (0, 0),  # output_padding
        groups,
        out_ptr, out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def conv1d(input, weight, bias=None, stride=(1,), padding=(0,), dilation=(1,), groups=1):
    """Conv1d forward via conv2d with unsqueezed spatial dim."""
    from ...common import view as view_backend
    # Unsqueeze: (N, C, L) -> (N, C, 1, L)
    input_4d = view_backend.unsqueeze(input, 2)
    weight_4d = view_backend.unsqueeze(weight, 2)
    out_4d = conv2d(input_4d, weight_4d, bias,
                    stride=(1, stride[0]),
                    padding=(0, padding[0]),
                    dilation=(1, dilation[0]),
                    groups=groups)
    # Squeeze: (N, C_out, 1, L_out) -> (N, C_out, L_out)
    return view_backend.squeeze(out_4d, 2)


def conv_transpose2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0),
                     output_padding=(0, 0), groups=1, dilation=(1, 1)):
    """ConvTranspose2d forward using aclnnConvolution with transposed=True."""
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    N, C_in, H_in, W_in = input.shape
    C_in_w, C_out_g, kH, kW = weight.shape
    sH, sW = stride
    pH, pW = padding
    opH, opW = output_padding
    dH, dW = dilation
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1
    H_out = (H_in - 1) * sH - 2 * pH + ekH + opH
    W_out = (W_in - 1) * sW - 2 * pW + ekW + opW
    C_out = C_out_g * groups
    out_shape = (N, C_out, H_out, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    bias_ptr = None
    bias_shape = None
    bias_stride = None
    if bias is not None:
        bias_ptr = _unwrap_storage(bias).data_ptr()
        bias_shape = bias.shape
        bias_stride = bias.stride

    aclnn.convolution(
        _unwrap_storage(input).data_ptr(),
        _unwrap_storage(weight).data_ptr(),
        bias_ptr,
        input.shape, input.stride,
        weight.shape, weight.stride,
        bias_shape, bias_stride,
        input.dtype,
        stride, padding, dilation,
        True,  # transposed
        output_padding,
        groups,
        out_ptr, out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def conv_transpose1d(input, weight, bias=None, stride=(1,), padding=(0,),
                     output_padding=(0,), groups=1, dilation=(1,)):
    """ConvTranspose1d forward via conv_transpose2d with unsqueezed spatial dim."""
    from ...common import view as view_backend
    input_4d = view_backend.unsqueeze(input, 2)
    weight_4d = view_backend.unsqueeze(weight, 2)
    out_4d = conv_transpose2d(input_4d, weight_4d, bias,
                              stride=(1, stride[0]),
                              padding=(0, padding[0]),
                              output_padding=(0, output_padding[0]),
                              groups=groups,
                              dilation=(1, dilation[0]))
    return view_backend.squeeze(out_4d, 2)


def conv3d_op(input, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0),
              dilation=(1, 1, 1), groups=1):
    """Conv3d forward via vol2col + mm pattern (like im2col_op but for 5D).

    Reshapes 3D convolution into 2D matrix multiplication:
    - Extract sliding 3D blocks (vol2col) using gather indices
    - Reshape weight to 2D
    - Compute output via matmul
    """
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    import numpy as _np

    N, C_in, D, H, W = input.shape
    C_out, C_in_g, kD, kH, kW = weight.shape
    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    ekD = (kD - 1) * dD + 1
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1

    D_out = (D + 2 * pD - ekD) // sD + 1
    H_out = (H + 2 * pH - ekH) // sH + 1
    W_out = (W + 2 * pW - ekW) // sW + 1

    # Pad input if needed
    a = input
    if pD > 0 or pH > 0 or pW > 0:
        a = dispatch("pad", "npu", a, (pW, pW, pH, pH, pD, pD))
    a = contiguous(a)

    _, _, D_pad, H_pad, W_pad = a.shape

    # Build vol2col gather indices on CPU then copy to NPU
    # For each output position and kernel position, compute flat index
    n_cols = D_out * H_out * W_out
    n_rows = C_in_g * kD * kH * kW

    indices = _np.zeros((n_rows, n_cols), dtype=_np.int64)
    for kd in range(kD):
        for kh in range(kH):
            for kw in range(kW):
                row = (kd * kH + kh) * kW + kw
                for od in range(D_out):
                    for oh in range(H_out):
                        for ow in range(W_out):
                            col = (od * H_out + oh) * W_out + ow
                            id_ = od * sD + kd * dD
                            ih = oh * sH + kh * dH
                            iw = ow * sW + kw * dW
                            indices[row, col] = (id_ * H_pad + ih) * W_pad + iw

    runtime = npu_runtime.get_runtime((input.device.index or 0))
    idx_ptr, _ = npu_runtime._copy_cpu_to_npu(indices.ravel(), runtime=runtime)
    idx_storage = npu_typed_storage_from_ptr(idx_ptr, n_rows * n_cols, int64_dtype, device=input.device)
    idx_tensor = _wrap_tensor(idx_storage, (n_rows, n_cols), npu_runtime._contiguous_stride((n_rows, n_cols)))

    # Flatten spatial dims of input per channel
    spatial_size = D_pad * H_pad * W_pad

    # Process each group
    outs = []
    c_out_per_g = C_out // groups
    for g in range(groups):
        c_in_start = g * C_in_g
        c_out_start = g * c_out_per_g
        # For each batch element
        batch_outs = []
        for n in range(N):
            # Extract input channels for this group: (C_in_g, D*H*W)
            from ...._creation import arange as _arange
            cin_idx = _arange(c_in_start, c_in_start + C_in_g, dtype=int64_dtype, device=input.device)
            a_group = index_select(contiguous(a), 1, cin_idx)  # (1, C_in_g, D_pad, H_pad, W_pad) -> need single batch
            # Get single batch element
            n_idx = view_backend.reshape(
                _cast_tensor_dtype(_scalar_to_npu_tensor(n, _arange(0, 1, dtype=int64_dtype, device=input.device)), int64_dtype),
                (1,))
            a_n = index_select(contiguous(a_group), 0, n_idx)  # (1, C_in_g, D_pad, H_pad, W_pad)
            a_flat = view_backend.reshape(contiguous(a_n), (C_in_g, spatial_size))  # (C_in_g, D*H*W)

            # Gather columns: for each channel, gather using spatial indices
            # We need to expand indices for all input channels
            cols_parts = []
            for ci in range(C_in_g):
                ci_idx = view_backend.reshape(
                    _cast_tensor_dtype(_scalar_to_npu_tensor(ci, _arange(0, 1, dtype=int64_dtype, device=input.device)), int64_dtype),
                    (1,))
                a_ci = index_select(contiguous(a_flat), 0, ci_idx)  # (1, spatial_size)
                a_ci_flat = view_backend.reshape(contiguous(a_ci), (spatial_size,))
                # Gather: pick indices for all kernel positions of this channel
                ki_start = ci * kD * kH * kW
                ki_end = (ci + 1) * kD * kH * kW
                ki_idx = _arange(ki_start, ki_end, dtype=int64_dtype, device=input.device)
                ci_indices = index_select(contiguous(idx_tensor), 0, ki_idx)  # (kD*kH*kW, n_cols)
                ci_indices_flat = view_backend.reshape(contiguous(ci_indices), (kD * kH * kW * n_cols,))
                gathered = index_select(a_ci_flat, 0, ci_indices_flat)
                cols_parts.append(view_backend.reshape(contiguous(gathered), (kD * kH * kW, n_cols)))

            col_matrix = dispatch("cat", "npu", cols_parts, dim=0)  # (C_in_g * kD*kH*kW, n_cols)

            # Weight for this group: (c_out_per_g, C_in_g * kD * kH * kW)
            cout_idx = _arange(c_out_start, c_out_start + c_out_per_g, dtype=int64_dtype, device=input.device)
            w_group = index_select(contiguous(weight), 0, cout_idx)
            w_2d = view_backend.reshape(contiguous(w_group), (c_out_per_g, C_in_g * kD * kH * kW))

            # Output: w_2d @ col_matrix = (c_out_per_g, n_cols)
            out_n = matmul(contiguous(w_2d), contiguous(col_matrix))
            batch_outs.append(view_backend.reshape(contiguous(out_n), (1, c_out_per_g, D_out, H_out, W_out)))

        group_out = dispatch("cat", "npu", batch_outs, dim=0)  # (N, c_out_per_g, D_out, H_out, W_out)
        outs.append(group_out)

    if groups > 1:
        result = dispatch("cat", "npu", outs, dim=1)
    else:
        result = outs[0]

    if bias is not None:
        bias_5d = view_backend.reshape(contiguous(bias), (1, C_out, 1, 1, 1))
        bias_broad = _npu_broadcast_to(bias_5d, result.shape)
        result = add(result, bias_broad)

    return result


def conv_transpose3d_op(input, weight, bias, stride, padding, output_padding, groups, dilation):
    """Transposed 3D convolution via col2vol scatter + mm pattern.

    For each input position (d,h,w), the weight kernel is scattered to
    the output at positions determined by stride/dilation. This is the
    adjoint of the forward convolution (vol2col + mm).
    """
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    import numpy as _np

    sD, sH, sW = stride
    pD, pH, pW = padding
    opD, opH, opW = output_padding
    dD, dH, dW = dilation

    N, C_in, D_in, H_in, W_in = input.shape
    C_in_w, C_out_per_g, kD, kH, kW = weight.shape
    C_out = C_out_per_g * groups
    c_in_per_g = C_in // groups

    D_out = (D_in - 1) * sD - 2 * pD + dD * (kD - 1) + opD + 1
    H_out = (H_in - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1
    W_out = (W_in - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1

    # Build col2vol scatter indices on CPU
    # For each input position and kernel position, compute the output flat index
    n_in = D_in * H_in * W_in
    spatial_out = D_out * H_out * W_out

    # For each (kd, kh, kw, id, ih, iw), the output position is:
    # od = id * sD + kd * dD - pD, oh = ih * sH + kh * dH - pH, ow = iw * sW + kw * dW - pW
    # Build scatter: we accumulate w_t @ x into output via col2vol
    # Use addmm-like approach: compute w^T @ x_flat to get (C_out_per_g * kD*kH*kW, n_in)
    # then scatter each kernel element to the correct output position

    # Simpler approach for correctness: compute output via element-wise accumulation
    # For each group, output[n, cout, od, oh, ow] += sum over cin, kd, kh, kw of
    #   input[n, cin, id, ih, iw] * weight[cin, cout, kd, kh, kw]
    # where id = (od + pD - kd * dD) / sD (if divisible)

    # Build scatter index mapping on CPU then use scatter_add on NPU
    # For efficiency, use matmul-based approach:
    # col = W^T @ x_flat for each group, then col2vol via index scatter

    result = dispatch("zeros", "npu", (N, C_out, D_out, H_out, W_out),
                      dtype=input.dtype, device=input.device)
    result_flat = view_backend.reshape(contiguous(result), (N, C_out, spatial_out))

    for g in range(groups):
        from ...._creation import arange as _arange
        cin_idx = _arange(g * c_in_per_g, (g + 1) * c_in_per_g, dtype=int64_dtype, device=input.device)
        w_g = index_select(contiguous(weight), 0, cin_idx)  # (c_in_per_g, C_out_per_g, kD, kH, kW)
        # Transpose to (C_out_per_g, c_in_per_g, kD, kH, kW)
        w_t = view_backend.permute(contiguous(w_g), [1, 0, 2, 3, 4])
        w_2d = view_backend.reshape(contiguous(w_t), (C_out_per_g, c_in_per_g * kD * kH * kW))

        # Build col2vol indices: for each kernel position and input position,
        # compute output flat index
        col_indices = _np.full((kD * kH * kW, n_in), -1, dtype=_np.int64)
        for kd in range(kD):
            for kh in range(kH):
                for kw in range(kW):
                    ki = (kd * kH + kh) * kW + kw
                    for id_ in range(D_in):
                        for ih in range(H_in):
                            for iw in range(W_in):
                                ii = (id_ * H_in + ih) * W_in + iw
                                od = id_ * sD + kd * dD - pD
                                oh = ih * sH + kh * dH - pH
                                ow = iw * sW + kw * dW - pW
                                if 0 <= od < D_out and 0 <= oh < H_out and 0 <= ow < W_out:
                                    col_indices[ki, ii] = (od * H_out + oh) * W_out + ow

        # For each batch element and kernel position
        for n in range(N):
            n_idx = view_backend.reshape(
                _cast_tensor_dtype(_scalar_to_npu_tensor(n, _arange(0, 1, dtype=int64_dtype, device=input.device)), int64_dtype),
                (1,))
            x_idx = _arange(g * c_in_per_g, (g + 1) * c_in_per_g, dtype=int64_dtype, device=input.device)
            x_n = index_select(index_select(contiguous(input), 0, n_idx), 1, x_idx)
            x_flat = view_backend.reshape(contiguous(x_n), (c_in_per_g, n_in))

            # For each kernel position, compute contribution and scatter
            for ki in range(kD * kH * kW):
                # Extract weight slice for this kernel position
                # w_slice: (C_out_per_g, c_in_per_g) from w_2d columns [ki*c_in_per_g : (ki+1)*c_in_per_g]
                # Actually w_2d shape is (C_out_per_g, c_in_per_g * kD*kH*kW)
                ki_cin_start = ki * c_in_per_g  # Incorrect — weight layout is (cout, cin, kD, kH, kW) flattened
                # Actually after reshape: w_2d[cout, cin * kD*kH*kW + ki] — no, it's cin*kD*kH*kW
                # The flatten is over (c_in_per_g, kD, kH, kW), so index = cin * (kD*kH*kW) + ki
                # We need w_slice[cout, cin] = w_2d[cout, cin * kD*kH*kW + ki]
                w_col_indices = _np.array([cin * kD * kH * kW + ki for cin in range(c_in_per_g)], dtype=_np.int64)
                runtime = npu_runtime.get_runtime((input.device.index or 0))
                wci_ptr, _ = npu_runtime._copy_cpu_to_npu(w_col_indices, runtime=runtime)
                wci_storage = npu_typed_storage_from_ptr(wci_ptr, c_in_per_g, int64_dtype, device=input.device)
                wci_t = _wrap_tensor(wci_storage, (c_in_per_g,), (1,))
                w_slice = index_select(contiguous(w_2d), 1, wci_t)  # (C_out_per_g, c_in_per_g)

                # Contribution: w_slice @ x_flat = (C_out_per_g, n_in)
                contrib = matmul(contiguous(w_slice), contiguous(x_flat))

                # Now scatter contrib to output positions using col_indices[ki]
                valid_mask = col_indices[ki] >= 0
                valid_in_indices = _np.where(valid_mask)[0]
                if len(valid_in_indices) == 0:
                    continue
                valid_out_indices = col_indices[ki][valid_in_indices]

                # Gather valid contributions
                vi_ptr, _ = npu_runtime._copy_cpu_to_npu(
                    _np.array(valid_in_indices, dtype=_np.int64), runtime=runtime)
                vi_storage = npu_typed_storage_from_ptr(vi_ptr, len(valid_in_indices), int64_dtype, device=input.device)
                vi_t = _wrap_tensor(vi_storage, (len(valid_in_indices),), (1,))
                valid_contrib = index_select(contiguous(contrib), 1, vi_t)  # (C_out_per_g, n_valid)

                # Scatter-add to output at valid_out_indices
                # Use index_put with accumulate=True
                vo_ptr, _ = npu_runtime._copy_cpu_to_npu(
                    _np.array(valid_out_indices, dtype=_np.int64), runtime=runtime)
                vo_storage = npu_typed_storage_from_ptr(vo_ptr, len(valid_out_indices), int64_dtype, device=input.device)
                vo_t = _wrap_tensor(vo_storage, (len(valid_out_indices),), (1,))

                # Add contributions to result_flat[n, g*C_out_per_g:(g+1)*C_out_per_g, valid_out_indices]
                cout_start = g * C_out_per_g
                for co in range(C_out_per_g):
                    co_global = cout_start + co
                    co_idx = view_backend.reshape(
                        _cast_tensor_dtype(_scalar_to_npu_tensor(co, _arange(0, 1, dtype=int64_dtype, device=input.device)), int64_dtype),
                        (1,))
                    contrib_co = index_select(contiguous(valid_contrib), 0, co_idx)
                    contrib_co = view_backend.reshape(contiguous(contrib_co), (len(valid_out_indices),))

                    # Get current output slice
                    out_co_idx = view_backend.reshape(
                        _cast_tensor_dtype(_scalar_to_npu_tensor(co_global, _arange(0, 1, dtype=int64_dtype, device=input.device)), int64_dtype),
                        (1,))
                    out_row = index_select(index_select(contiguous(result_flat), 0, n_idx), 1, out_co_idx)
                    out_row = view_backend.reshape(contiguous(out_row), (spatial_out,))

                    # Scatter add
                    gathered_existing = index_select(out_row, 0, vo_t)
                    updated = add(gathered_existing, contrib_co)

                    # Write back via building full row
                    # This is inefficient but correct — use index_put if available
                    npu_index_put_impl(
                        view_backend.reshape(contiguous(out_row), (spatial_out,)),
                        vo_t,
                        updated,
                        accumulate=False,
                    )

    result = view_backend.reshape(contiguous(result_flat), (N, C_out, D_out, H_out, W_out))

    if bias is not None:
        bias_5d = view_backend.reshape(contiguous(bias), (1, C_out, 1, 1, 1))
        bias_broad = _npu_broadcast_to(bias_5d, result.shape)
        result = add(result, bias_broad)

    return result


def max_pool2d(input, kernel_size, stride, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """MaxPool2d forward using aclnnMaxPool2dWithMask (supports fp32/fp16 on Ascend910B)."""
    import math as _math
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    sH, sW = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pH, pW = (padding, padding) if isinstance(padding, int) else tuple(padding)
    dH, dW = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)

    N, C, H, W = input.shape
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1
    if ceil_mode:
        H_out = _math.ceil((H + 2 * pH - ekH) / sH) + 1
        W_out = _math.ceil((W + 2 * pW - ekW) / sW) + 1
    else:
        H_out = (H + 2 * pH - ekH) // sH + 1
        W_out = (W + 2 * pW - ekW) // sW + 1

    out_shape = (N, C, H_out, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    # aclnnMaxPool2dWithMask returns a mask tensor (int8) used for backward.
    # mask shape: (N, C, kH*kW, (ceil(outH*outW/16)+1)*32)
    BLOCKSIZE = 16
    mask_H = kH * kW
    mask_W = (_math.ceil(H_out * W_out / BLOCKSIZE) + 1) * 32
    mask_shape = (N, C, mask_H, mask_W)
    mask_stride = npu_runtime._contiguous_stride(mask_shape)
    mask_numel = _numel(mask_shape)
    mask_ptr = npu_runtime._alloc_device(max(mask_numel, 1), runtime=runtime)  # int8 = 1 byte each

    aclnn.max_pool2d_with_mask(
        _unwrap_storage(input).data_ptr(), out_ptr, mask_ptr,
        input.shape, input.stride, input.dtype,
        [kH, kW], [sH, sW], [pH, pW], [dH, dW], ceil_mode,
        out_shape, out_stride, mask_shape, mask_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    out = _wrap_tensor(out_storage, out_shape, out_stride)
    out._backward_data = {
        "mask_ptr": mask_ptr, "mask_shape": mask_shape, "mask_stride": mask_stride,
        "kernel_size": (kH, kW), "strides": (sH, sW), "padding": (pH, pW),
        "dilation": (dH, dW), "ceil_mode": ceil_mode,
    }
    return out


def max_pool3d(input, kernel_size, stride, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """MaxPool3d forward using aclnnMaxPool3dWithArgmax (supports fp32/fp16 on Ascend)."""
    import math as _math
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    kD, kH, kW = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
    sD, sH, sW = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
    pD, pH, pW = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
    dD, dH, dW = (dilation,) * 3 if isinstance(dilation, int) else tuple(dilation)

    N, C, D, H, W = input.shape
    ekD = (kD - 1) * dD + 1
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1
    if ceil_mode:
        D_out = _math.ceil((D + 2 * pD - ekD) / sD) + 1
        H_out = _math.ceil((H + 2 * pH - ekH) / sH) + 1
        W_out = _math.ceil((W + 2 * pW - ekW) / sW) + 1
    else:
        D_out = (D + 2 * pD - ekD) // sD + 1
        H_out = (H + 2 * pH - ekH) // sH + 1
        W_out = (W + 2 * pW - ekW) // sW + 1

    out_shape = (N, C, D_out, H_out, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    # aclnnMaxPool3dWithArgmax returns argmax indices as int32 with same shape as output
    indices_shape = out_shape
    indices_stride = out_stride
    indices_numel = out_numel
    indices_ptr = npu_runtime._alloc_device(max(indices_numel, 1) * 4, runtime=runtime)  # int32 = 4 bytes

    aclnn.max_pool3d_with_argmax(
        _unwrap_storage(input).data_ptr(), out_ptr, indices_ptr,
        input.shape, input.stride, input.dtype,
        [kD, kH, kW], [sD, sH, sW], [pD, pH, pW], [dD, dH, dW], ceil_mode,
        out_shape, out_stride, indices_shape, indices_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    out = _wrap_tensor(out_storage, out_shape, out_stride)
    out._backward_data = {
        "indices_ptr": indices_ptr, "indices_shape": indices_shape,
        "indices_stride": indices_stride,
        "kernel_size": (kD, kH, kW), "strides": (sD, sH, sW),
        "padding": (pD, pH, pW), "dilation": (dD, dH, dW),
        "ceil_mode": ceil_mode,
    }
    return out


def avg_pool2d(input, kernel_size, stride, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None):
    """AvgPool2d forward using aclnnAvgPool2d."""
    import math as _math
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    sH, sW = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pH, pW = (padding, padding) if isinstance(padding, int) else tuple(padding)

    N, C, H, W = input.shape
    if ceil_mode:
        H_out = _math.ceil((H + 2 * pH - kH) / sH) + 1
        W_out = _math.ceil((W + 2 * pW - kW) / sW) + 1
    else:
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

    out_shape = (N, C, H_out, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.avg_pool2d(
        _unwrap_storage(input).data_ptr(), out_ptr,
        input.shape, input.stride, input.dtype,
        [kH, kW], [sH, sW], [pH, pW],
        ceil_mode, count_include_pad, divisor_override,
        out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def adaptive_avg_pool2d(input, output_size):
    """AdaptiveAvgPool2d forward — composite implementation via avg_pool2d.

    Uses avg_pool2d with computed kernel_size/stride/padding to avoid
    aclnnAdaptiveAvgPool2d cross-op contamination issues on Ascend910B
    (CANN 8.3.RC2 bug where cubeMathType=1 ops corrupt AdaptiveAvgPool2d state).
    """
    import math as _math

    if isinstance(output_size, int):
        oH = oW = output_size
    else:
        oH, oW = output_size

    N, C, H, W = input.shape

    # Compute avg_pool2d parameters that produce the desired adaptive output.
    # PyTorch's adaptive pooling algorithm (from ATen/native/AdaptiveAveragePooling.cpp):
    #   start_index(i) = floor(i * input_size / output_size)
    #   end_index(i)   = ceil((i+1) * input_size / output_size)
    # When input_size is evenly divisible by output_size, this simplifies to
    # a regular avg_pool2d with stride = input_size // output_size and
    # kernel_size = input_size - (output_size - 1) * stride.

    def _can_use_regular_pool(in_sz, out_sz):
        """Check if adaptive pool can be expressed as regular avg_pool2d."""
        if out_sz == 0:
            return False
        if in_sz % out_sz == 0:
            return True
        # Also works when all windows have the same size
        stride = in_sz // out_sz
        kernel = in_sz - (out_sz - 1) * stride
        # Verify all windows produce valid output
        return stride > 0 and kernel > 0 and (out_sz - 1) * stride + kernel == in_sz

    if _can_use_regular_pool(H, oH) and _can_use_regular_pool(W, oW):
        sH = H // oH
        sW = W // oW
        kH = H - (oH - 1) * sH
        kW = W - (oW - 1) * sW
        return avg_pool2d(input, kernel_size=(kH, kW), stride=(sH, sW),
                          padding=0, ceil_mode=False,
                          count_include_pad=True, divisor_override=None)

    # Fallback: try the native ACLNN kernel for non-uniform window sizes
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    out_shape = (N, C, oH, oW)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.adaptive_avg_pool2d(
        _unwrap_storage(input).data_ptr(), out_ptr,
        input.shape, input.stride, input.dtype,
        [oH, oW], out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def adaptive_max_pool2d(input, output_size, return_indices=False):
    """AdaptiveMaxPool2d forward using aclnnAdaptiveMaxPool2d (supports fp32/fp16 on Ascend)."""
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    if isinstance(output_size, int):
        oH = oW = output_size
    else:
        oH, oW = output_size

    # Handle both 3D (C, H, W) and 4D (N, C, H, W) input
    unsqueezed = False
    if len(input.shape) == 3:
        unsqueezed = True
        C, H, W = input.shape
        input = input.unsqueeze(0)  # (1, C, H, W)
        N = 1
    else:
        N, C, H, W = input.shape

    out_shape = (N, C, oH, oW)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    # indices are int64 (8 bytes each)
    indices_shape = out_shape
    indices_stride = out_stride
    indices_numel = out_numel
    indices_ptr = npu_runtime._alloc_device(max(indices_numel, 1) * 8, runtime=runtime)

    aclnn.adaptive_max_pool2d(
        _unwrap_storage(input).data_ptr(), out_ptr, indices_ptr,
        input.shape, input.stride, input.dtype,
        [oH, oW],
        out_shape, out_stride,
        indices_shape, indices_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    out = _wrap_tensor(out_storage, out_shape, out_stride)
    out._backward_data = {
        "indices_ptr": indices_ptr, "indices_shape": indices_shape,
        "indices_stride": indices_stride,
    }

    if unsqueezed:
        out = out.squeeze(0)

    if return_indices:
        indices_storage = npu_typed_storage_from_ptr(indices_ptr, max(indices_numel, 1), int64_dtype, device=input.device)
        indices_tensor = _wrap_tensor(indices_storage, indices_shape, indices_stride)
        if unsqueezed:
            indices_tensor = indices_tensor.squeeze(0)
        return out, indices_tensor

    return out


# ---------------------------------------------------------------
# P1 ops: std, reciprocal, addmm, einsum, upsample_nearest2d,
#          upsample_bilinear2d, one_hot
# ---------------------------------------------------------------


def adaptive_avg_pool3d_op(input, output_size):
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))
    s = _unwrap_storage(input)

    if len(input.shape) == 4:
        N, C, D, H, W = 1, *input.shape
        in_5d = True
    else:
        N, C, D, H, W = input.shape
        in_5d = False

    oD, oH, oW = output_size
    out_shape_5d = (N, C, oD, oH, oW)
    out_stride_5d = npu_runtime._contiguous_stride(out_shape_5d)
    out_nbytes = _numel(out_shape_5d) * _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_nbytes, 4), runtime=runtime)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape_5d), input.dtype, device=input.device)

    in_shape = input.shape if not in_5d else (N, C, D, H, W)
    in_stride = input.stride if not in_5d else npu_runtime._contiguous_stride(in_shape)

    aclnn.adaptive_avg_pool3d(
        s.data_ptr(), out_ptr,
        in_shape, in_stride, out_shape_5d, out_stride_5d,
        input.dtype, output_size,
        runtime=runtime, stream=stream.stream,
    )
    result = _wrap_tensor(out_storage, out_shape_5d, out_stride_5d)
    if in_5d:
        from ...common import view as view_backend
        result = view_backend.reshape(result, (C, oD, oH, oW))
    return result


def avg_pool3d_op(input, kernel_size, stride, padding, ceil_mode=False,
                  count_include_pad=True):
    """Avg pool 3D via slice + mean over pooling windows on NPU."""
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    import math as _math
    import numpy as _np

    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding

    N, C, D, H, W = input.shape

    # Pad if needed
    a = input
    if pD > 0 or pH > 0 or pW > 0:
        a = dispatch("pad", "npu", a, (pW, pW, pH, pH, pD, pD))
    a = contiguous(a)

    _, _, D_pad, H_pad, W_pad = a.shape

    if ceil_mode:
        oD = _math.ceil((D_pad - kD) / sD) + 1
        oH = _math.ceil((H_pad - kH) / sH) + 1
        oW = _math.ceil((W_pad - kW) / sW) + 1
    else:
        oD = (D_pad - kD) // sD + 1
        oH = (H_pad - kH) // sH + 1
        oW = (W_pad - kW) // sW + 1

    # Build gather indices for all output positions and pool windows
    pool_size = kD * kH * kW
    n_out = oD * oH * oW

    # For each output position, gather kD*kH*kW values from flattened spatial dims
    indices = _np.zeros((pool_size, n_out), dtype=_np.int64)
    for kd in range(kD):
        for kh in range(kH):
            for kw in range(kW):
                row = (kd * kH + kh) * kW + kw
                for od in range(oD):
                    for oh in range(oH):
                        for ow in range(oW):
                            col = (od * oH + oh) * oW + ow
                            id_ = od * sD + kd
                            ih = oh * sH + kh
                            iw = ow * sW + kw
                            indices[row, col] = (id_ * H_pad + ih) * W_pad + iw

    runtime = npu_runtime.get_runtime((input.device.index or 0))
    spatial = D_pad * H_pad * W_pad

    # Flatten spatial dims: (N, C, D*H*W)
    a_flat = view_backend.reshape(contiguous(a), (N * C, spatial))

    # Copy indices to NPU
    idx_flat = indices.ravel()
    idx_ptr, _ = npu_runtime._copy_cpu_to_npu(idx_flat, runtime=runtime)
    idx_storage = npu_typed_storage_from_ptr(idx_ptr, len(idx_flat), int64_dtype, device=input.device)
    idx_t = _wrap_tensor(idx_storage, (pool_size * n_out,), (1,))

    # Gather for all N*C at once
    gathered = index_select(contiguous(a_flat), 1, idx_t)  # (N*C, pool_size * n_out)
    gathered = view_backend.reshape(contiguous(gathered), (N * C, pool_size, n_out))

    # Mean over pool dimension
    pooled = sum_(gathered, dim=1)  # (N*C, n_out)
    if count_include_pad:
        divisor = _scalar_to_npu_tensor(float(pool_size), pooled)
    else:
        divisor = _scalar_to_npu_tensor(float(pool_size), pooled)
    pooled = div(pooled, divisor)

    return view_backend.reshape(contiguous(pooled), (N, C, oD, oH, oW))


def adaptive_avg_pool1d_op(input, output_size):
    """Adaptive average pooling 1D via lifting to 2D."""
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    # (N, C, W) → (N, C, 1, W) → adaptive_avg_pool2d → (N, C, 1, oW) → (N, C, oW)
    N, C, W = input.shape
    oW = output_size[0] if isinstance(output_size, (list, tuple)) else output_size
    input_4d = view_backend.reshape(input, (N, C, 1, W))
    out_4d = dispatch("adaptive_avg_pool2d", "npu", input_4d, [1, oW])
    return view_backend.reshape(out_4d, (N, C, oW))


def avg_pool1d_op(input, kernel_size, stride, padding=0, ceil_mode=False,
                  count_include_pad=True):
    """Average pooling 1D via lifting to 2D."""
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    kW = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
    sW = stride[0] if isinstance(stride, (list, tuple)) else stride
    pW = padding[0] if isinstance(padding, (list, tuple)) else padding
    N, C, W = input.shape
    input_4d = view_backend.reshape(input, (N, C, 1, W))
    out_4d = dispatch("avg_pool2d", "npu", input_4d, [1, kW], [1, sW], [0, pW],
                      ceil_mode, count_include_pad)
    oW = out_4d.shape[3]
    return view_backend.reshape(out_4d, (N, C, oW))


def max_pool1d_op(input, kernel_size, stride, padding=0, dilation=1,
                  ceil_mode=False, return_indices=False):
    """Max pooling 1D via lifting to 2D."""
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    kW = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
    sW = stride[0] if isinstance(stride, (list, tuple)) else stride
    pW = padding[0] if isinstance(padding, (list, tuple)) else padding
    dW = dilation[0] if isinstance(dilation, (list, tuple)) else dilation
    N, C, W = input.shape
    input_4d = view_backend.reshape(input, (N, C, 1, W))
    result = dispatch("max_pool2d", "npu", input_4d, [1, kW], [1, sW], [0, pW],
                      [1, dW], ceil_mode, return_indices)
    if return_indices:
        out_4d, idx_4d = result
        oW = out_4d.shape[3]
        return view_backend.reshape(out_4d, (N, C, oW)), view_backend.reshape(idx_4d, (N, C, oW))
    oW = result.shape[3]
    return view_backend.reshape(result, (N, C, oW))


def adaptive_max_pool1d_op(input, output_size, return_indices=False):
    """Adaptive max pooling 1D via computed kernel/stride + max_pool1d."""
    from ...._dispatch.dispatcher import dispatch
    N, C, W = input.shape
    oW = output_size[0] if isinstance(output_size, (list, tuple)) else output_size
    # Compute equivalent kernel/stride for adaptive pooling
    kW = (W + oW - 1) // oW
    sW = W // oW
    pW = 0
    return max_pool1d_op(input, [kW], [sW], [pW], [1], False, return_indices)


# ===========================================================================
# Phase 4: Optimizer composites
# ===========================================================================


def upsample_nearest2d(input, output_size):
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    N, C = input.shape[0], input.shape[1]
    oH, oW = output_size
    out_shape = (N, C, oH, oW)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.upsample_nearest2d(
        _unwrap_storage(input).data_ptr(), out_ptr,
        input.shape, input.stride, input.dtype,
        output_size, out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def upsample_bilinear2d(input, output_size, align_corners, scales_h, scales_w):
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    N, C = input.shape[0], input.shape[1]
    oH, oW = output_size
    out_shape = (N, C, oH, oW)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.upsample_bilinear2d(
        _unwrap_storage(input).data_ptr(), out_ptr,
        input.shape, input.stride, input.dtype,
        output_size, align_corners, scales_h, scales_w,
        out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def upsample_bicubic2d_op(a, output_size, align_corners=False, scales_h=None, scales_w=None):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    s = _unwrap_storage(a)

    N, C, H_in, W_in = a.shape
    H_out, W_out = output_size
    out_shape = (N, C, H_out, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_nbytes = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_nbytes, 4), runtime=runtime)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)

    aclnn.upsample_bicubic2d(
        s.data_ptr(), out_ptr,
        a.shape, a.stride, out_shape, out_stride,
        a.dtype, output_size, align_corners, scales_h, scales_w,
        runtime=runtime, stream=stream.stream,
    )
    return _wrap_tensor(out_storage, out_shape, out_stride)


def upsample_linear1d_op(a, output_size, align_corners=False, scales=None):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    s = _unwrap_storage(a)

    N, C, W_in = a.shape
    W_out = output_size[0]
    out_shape = (N, C, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_nbytes = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_nbytes, 4), runtime=runtime)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)

    aclnn.upsample_linear1d(
        s.data_ptr(), out_ptr,
        a.shape, a.stride, out_shape, out_stride,
        a.dtype, output_size, align_corners, scales,
        runtime=runtime, stream=stream.stream,
    )
    return _wrap_tensor(out_storage, out_shape, out_stride)


def upsample_nearest1d_op(a, output_size, scales=None):
    """Upsample nearest 1D via 2D upsample (ACLNN broken on 910B)."""
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    N, C, W = a.shape
    oW = output_size[0] if isinstance(output_size, (list, tuple)) else output_size
    a_4d = view_backend.reshape(a, (N, C, 1, W))
    out_4d = dispatch("upsample_nearest2d", "npu", a_4d, [1, oW], None, scales)
    return view_backend.reshape(out_4d, (N, C, oW))


def im2col_op(a, kernel_size, dilation, padding, stride):
    """F.unfold: extract sliding local blocks.

    Composite implementation: aclnnIm2col returns 561103 on CANN 8.3.RC2.
    Uses pad + flatten + gather with existing NPU ops.
    """
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend

    N, C, H, W = a.shape
    kH, kW = kernel_size
    dH, dW = dilation
    pH, pW = padding
    sH, sW = stride
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1
    out_H = (H + 2 * pH - ekH) // sH + 1
    out_W = (W + 2 * pW - ekW) // sW + 1
    L = out_H * out_W

    if pH > 0 or pW > 0:
        a = dispatch("pad", "npu", a, (pW, pW, pH, pH))
    a = contiguous(a)

    import numpy as _np
    _, _, H_pad, W_pad = a.shape

    # Build gather indices: for each kernel position, compute flat index into H_pad*W_pad plane
    patches = []
    for kh in range(kH):
        for kw in range(kW):
            row_indices = []
            for oh in range(out_H):
                for ow in range(out_W):
                    r = oh * sH + kh * dH
                    c = ow * sW + kw * dW
                    row_indices.append(r * W_pad + c)
            patches.append(row_indices)

    # Stack into (kH*kW, L), tile to (C*kH*kW, L) with per-channel offsets
    idx_2d = _np.stack([_np.array(p, dtype=_np.int64) for p in patches], axis=0)
    idx_full = _np.tile(idx_2d, (C, 1))

    offsets = _np.arange(C, dtype=_np.int64).reshape(C, 1) * (H_pad * W_pad)
    offsets_tiled = _np.repeat(offsets, kH * kW, axis=0)
    idx_with_offset = idx_full + offsets_tiled

    # Broadcast to (N, C*kH*kW, L), then flatten last two dims for gather
    idx_with_offset_batch = _np.broadcast_to(
        idx_with_offset[None], (N, C * kH * kW, L)
    ).copy()
    idx_flat = idx_with_offset_batch.reshape(N, C * kH * kW * L)

    # Flatten input to (N, C*H_pad*W_pad)
    a_fully_flat = view_backend.reshape(a, (N, C * H_pad * W_pad))

    # Copy index to NPU and gather
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    idx_ptr, _ = npu_runtime._copy_cpu_to_npu(idx_flat, runtime=runtime)
    idx_shape = (N, C * kH * kW * L)
    idx_stride = npu_runtime._contiguous_stride(idx_shape)
    idx_storage = npu_typed_storage_from_ptr(
        idx_ptr, _numel(idx_shape), int64_dtype, device=a.device
    )
    idx_tensor = _wrap_tensor(idx_storage, idx_shape, idx_stride)

    result = gather(a_fully_flat, -1, idx_tensor)
    out_shape = (N, C * kH * kW, L)
    result = view_backend.reshape(result, out_shape)
    return result


def col2im_op(a, output_size, kernel_size, dilation, padding, stride):
    """F.fold: combine sliding local blocks into a 4D tensor.

    Uses the same composite approach as im2col but in reverse via scatter_add.
    """
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    import numpy as _np

    N, C_kk, L = a.shape
    kH, kW = kernel_size
    dH, dW = dilation
    pH, pW = padding
    sH, sW = stride
    H_out, W_out = output_size
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1
    H_col = (H_out + 2 * pH - ekH) // sH + 1
    W_col = (W_out + 2 * pW - ekW) // sW + 1
    C = C_kk // (kH * kW)
    H_pad = H_out + 2 * pH
    W_pad = W_out + 2 * pW

    # Build gather indices (same approach as im2col but reversed)
    flat_indices = []
    for ki in range(kH):
        for kj in range(kW):
            for hi in range(H_col):
                for wi in range(W_col):
                    h = ki * dH + hi * sH
                    w = kj * dW + wi * sW
                    flat_indices.append(h * W_pad + w)
    idx_np = _np.array(flat_indices, dtype=_np.int64)
    # Shape: (kH*kW * H_col*W_col,) -> expand for (N, C, ...)
    idx_np = _np.tile(idx_np, 1)

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    # Create output: (N, C, H_pad * W_pad) filled with zeros
    out = dispatch("zeros", "npu", (N, C, H_pad * W_pad), dtype=a.dtype, device=a.device)
    # Reshape input: (N, C, kH*kW, H_col*W_col) -> (N, C, kH*kW*H_col*W_col)
    a_reshaped = view_backend.reshape(a, (N, C, kH * kW * L))

    # Upload indices
    idx_ptr, _ = npu_runtime._copy_cpu_to_npu(idx_np, runtime=runtime)
    idx_shape = (kH * kW * H_col * W_col,)
    idx_stride = npu_runtime._contiguous_stride(idx_shape)
    idx_storage = npu_typed_storage_from_ptr(idx_ptr, _numel(idx_shape), int64_dtype, device=a.device)
    idx_tensor_1d = _wrap_tensor(idx_storage, idx_shape, idx_stride)
    # Expand to (N, C, kH*kW*L) — use tile instead of expand (expand view bug)
    idx_reshaped = view_backend.reshape(idx_tensor_1d, (1, 1, kH * kW * H_col * W_col))
    idx_expanded = dispatch("tile", "npu", idx_reshaped, (N, C, 1))

    from ...._functional import scatter_add_ as _scatter_add
    _scatter_add(out, 2, idx_expanded, a_reshaped)

    out = view_backend.reshape(out, (N, C, H_pad, W_pad))
    # Remove padding
    if pH > 0 or pW > 0:
        out = dispatch("narrow", "npu", out, 2, pH, H_out)
        out = dispatch("narrow", "npu", out, 3, pW, W_out)
        out = contiguous(out)
    return out


# ---- ACLNN large-kernel ops (Phase 1, confirmed working on 910B) ----


def grid_sample_op(input, grid, mode='bilinear', padding_mode='zeros',
                   align_corners=None):
    """F.grid_sample via aclnnGridSampler2D."""
    if align_corners is None:
        align_corners = False
    mode_map = {'bilinear': 0, 'nearest': 1, 'bicubic': 2}
    pad_map = {'zeros': 0, 'border': 1, 'reflection': 2}
    interp = mode_map.get(mode, 0)
    pad = pad_map.get(padding_mode, 0)

    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    N, C = input.shape[0], input.shape[1]
    H_out, W_out = grid.shape[1], grid.shape[2]
    out_shape = (N, C, H_out, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(
        _numel(out_shape) * _dtype_itemsize(input.dtype), runtime=runtime
    )
    aclnn.sgrid_sampler2d(
        _unwrap_storage(input).data_ptr(), _unwrap_storage(grid).data_ptr(),
        out_ptr,
        input.shape, input.stride, grid.shape, grid.stride,
        out_shape, out_stride, input.dtype,
        interp, pad, align_corners,
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(
        out_ptr, _numel(out_shape), input.dtype, device=input.device
    )
    return _wrap_tensor(out_storage, out_shape, out_stride)


def affine_grid_op(theta, size, align_corners=None):
    """F.affine_grid via aclnnAffineGrid."""
    if align_corners is None:
        align_corners = False

    runtime = npu_runtime.get_runtime((theta.device.index or 0))
    stream = npu_state.current_stream((theta.device.index or 0))

    N = size[0]
    if len(size) == 4:
        H, W = size[2], size[3]
        out_shape = (N, H, W, 2)
    else:
        D, H, W = size[2], size[3], size[4]
        out_shape = (N, D, H, W, 3)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(
        _numel(out_shape) * _dtype_itemsize(theta.dtype), runtime=runtime
    )
    aclnn.saffine_grid(
        _unwrap_storage(theta).data_ptr(), out_ptr,
        theta.shape, theta.stride, theta.dtype,
        list(size), align_corners,
        out_shape, out_stride,
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(
        out_ptr, _numel(out_shape), theta.dtype, device=theta.device
    )
    return _wrap_tensor(out_storage, out_shape, out_stride)


# ---------- P1: View / reshape composite ops ----------


def pad(input, pad, mode='constant', value=0):
    if input.device.type != "npu":
        raise ValueError("NPU pad expects NPU tensors")
    if mode != "constant":
        raise NotImplementedError("NPU pad currently supports constant mode only")
    if not isinstance(pad, (tuple, list)):
        raise TypeError("pad must be a tuple/list of ints")
    if len(pad) % 2 != 0:
        raise ValueError("pad length must be even")
    if len(pad) > 2 * input.dim():
        raise ValueError("padding length too large")
    pad_vals = tuple(int(v) for v in pad)

    out_shape = list(input.shape)
    n_pairs = len(pad_vals) // 2
    for i in range(n_pairs):
        dim = input.dim() - 1 - i
        left = pad_vals[2 * i]
        right = pad_vals[2 * i + 1]
        out_shape[dim] = out_shape[dim] + left + right
        if out_shape[dim] < 0:
            raise RuntimeError("negative output size is not supported")
    out_shape = tuple(out_shape)

    out_stride = npu_runtime._contiguous_stride(out_shape)
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))
    out_ptr = npu_runtime._alloc_device(max(_numel(out_shape), 1) * _dtype_itemsize(input.dtype), runtime=runtime)

    if not aclnn.constant_pad_nd_symbols_ok():
        raise RuntimeError("aclnnConstantPadNd symbols not available")

    aclnn.constant_pad_nd(
        _unwrap_storage(input).data_ptr(),
        out_ptr,
        input.shape,
        input.stride,
        input.dtype,
        pad_vals,
        value,
        out_shape,
        out_stride,
        input.dtype,
        runtime,
        stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def ctc_loss_op(log_probs, targets, input_lengths, target_lengths,
                blank=0, reduction='mean', zero_infinity=False):
    """CTC Loss forward via alpha (forward variable) algorithm on NPU.

    Uses element-wise NPU ops for the forward pass computation.
    """
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    from ...._creation import arange as _arange
    import numpy as _np

    T, N, C = log_probs.shape

    # Sync input_lengths and target_lengths to CPU for loop control
    runtime = npu_runtime.get_runtime((log_probs.device.index or 0))
    runtime.synchronize()

    def _sync_to_cpu_int(tensor):
        if not hasattr(tensor, 'data_ptr'):
            return list(tensor) if hasattr(tensor, '__iter__') else [int(tensor)]
        nbytes = _numel(tensor.shape) * _dtype_itemsize(tensor.dtype)
        if nbytes == 0:
            return []
        from .. import acl_loader
        acl = acl_loader.ensure_acl()
        host_ptr, ret = acl.rt.malloc_host(int(nbytes))
        if ret != 0:
            raise RuntimeError(f"malloc_host failed: {ret}")
        npu_runtime.memcpy_d2h(
            host_ptr,
            int(nbytes),
            _unwrap_storage(tensor).data_ptr(),
            runtime=runtime,
        )
        data = _np.empty(int(nbytes), dtype=_np.uint8)
        import ctypes
        ctypes.memmove(data.ctypes.data, host_ptr, int(nbytes))
        acl.rt.free_host(host_ptr)
        dtype_name = str(tensor.dtype).split(".")[-1]
        np_dtype = {'int32': _np.int32, 'int64': _np.int64, 'float32': _np.float32}.get(dtype_name, _np.int64)
        return _np.frombuffer(data.tobytes(), dtype=np_dtype).tolist()

    inp_lens = _sync_to_cpu_int(input_lengths)
    tgt_lens = _sync_to_cpu_int(target_lengths)

    # Sync targets to CPU for label indexing
    tgt_cpu = _sync_to_cpu_int(targets)
    tgt_np = _np.array(tgt_cpu, dtype=_np.int64)
    if hasattr(targets, 'shape') and len(targets.shape) == 2:
        tgt_np = tgt_np.reshape(targets.shape)

    NEG_INF = -1e30
    losses_np = _np.zeros(N, dtype=_np.float32)
    is_1d = (tgt_np.ndim == 1)
    offset = 0

    # Run the alpha algorithm per batch element
    # This uses CPU numpy for the dynamic programming loop (data-dependent control flow)
    # but the actual log_probs indexing uses NPU gather ops
    # For simplicity and correctness, sync log_probs to CPU
    lp_nbytes = _numel(log_probs.shape) * _dtype_itemsize(log_probs.dtype)
    from .. import acl_loader
    acl = acl_loader.ensure_acl()
    host_ptr2, ret = acl.rt.malloc_host(int(lp_nbytes))
    if ret != 0:
        raise RuntimeError(f"malloc_host failed: {ret}")
    npu_runtime.memcpy_d2h(
        host_ptr2,
        int(lp_nbytes),
        _unwrap_storage(log_probs).data_ptr(),
        runtime=runtime,
    )
    lp_data = _np.empty(int(lp_nbytes), dtype=_np.uint8)
    import ctypes
    ctypes.memmove(lp_data.ctypes.data, host_ptr2, int(lp_nbytes))
    acl.rt.free_host(host_ptr2)
    dtype_name = str(log_probs.dtype).split(".")[-1]
    np_dtype = {'float16': _np.float16, 'float32': _np.float32, 'float64': _np.float64}.get(dtype_name, _np.float32)
    lp = _np.frombuffer(lp_data.tobytes(), dtype=np_dtype).reshape(T, N, C).astype(_np.float64)

    for b in range(N):
        T_b = int(inp_lens[b])
        S_b = int(tgt_lens[b])

        if is_1d:
            labels_b = tgt_np[offset:offset + S_b]
            offset += S_b
        else:
            labels_b = tgt_np[b, :S_b]

        L = 2 * S_b + 1
        ext = _np.full(L, blank, dtype=_np.int64)
        for s in range(S_b):
            ext[2 * s + 1] = labels_b[s]

        alpha = _np.full((T_b, L), NEG_INF, dtype=_np.float64)
        alpha[0, 0] = lp[0, b, ext[0]]
        if L > 1:
            alpha[0, 1] = lp[0, b, ext[1]]

        for t in range(1, T_b):
            for s in range(L):
                a_val = alpha[t - 1, s]
                if s > 0:
                    a_val = _np.logaddexp(a_val, alpha[t - 1, s - 1])
                if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                    a_val = _np.logaddexp(a_val, alpha[t - 1, s - 2])
                alpha[t, s] = a_val + lp[t, b, ext[s]]

        log_likelihood = alpha[T_b - 1, L - 1]
        if L > 1:
            log_likelihood = _np.logaddexp(log_likelihood, alpha[T_b - 1, L - 2])
        loss = -log_likelihood

        if zero_infinity and _np.isinf(loss):
            loss = 0.0
        losses_np[b] = loss

    if reduction == 'none':
        result_np = losses_np
    elif reduction == 'sum':
        result_np = _np.array([losses_np.sum()], dtype=_np.float32)
    else:  # mean
        tgt_lens_f = _np.maximum(_np.array(tgt_lens, dtype=_np.float32), 1.0)
        result_np = _np.array([(losses_np / tgt_lens_f).mean()], dtype=_np.float32)

    result_np = result_np.astype(np_dtype)
    result_ptr, _ = npu_runtime._copy_cpu_to_npu(result_np, runtime=runtime)
    result_shape = tuple(result_np.shape)
    result_stride = npu_runtime._contiguous_stride(result_shape)
    result_storage = npu_typed_storage_from_ptr(result_ptr, max(1, _numel(result_shape)),
                                                 log_probs.dtype, device=log_probs.device)
    return _wrap_tensor(result_storage, result_shape, result_stride)

# ---------- Other missing ops ----------
