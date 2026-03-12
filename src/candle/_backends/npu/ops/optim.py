"""Optimizer step operations for NPU."""
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
from .comparison import gt, lt
from .elementwise import where
from .math import abs, add, div, mul, neg, sign, sqrt, sub
from .random import copy_
from .reduce import maximum, minimum


def _adam_step_op(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq,
                  step, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize):
    runtime = npu_runtime.get_runtime((param.device.index or 0))
    stream = npu_state.current_stream((param.device.index or 0))

    p_s = _unwrap_storage(param)
    g_s = _unwrap_storage(grad)
    ea_s = _unwrap_storage(exp_avg)
    eas_s = _unwrap_storage(exp_avg_sq)
    # Create step tensor on device
    import numpy as _np
    step_np = _np.array([float(step)], dtype=_np.float32)
    step_ptr, _ = npu_runtime._copy_cpu_to_npu(step_np, runtime=runtime)
    step_shape = (1,)
    step_stride = (1,)

    max_v_ptr = None
    if amsgrad and max_exp_avg_sq is not None:
        max_v_ptr = _unwrap_storage(max_exp_avg_sq).data_ptr()

    aclnn.apply_adam_w_v2(
        p_s.data_ptr(), ea_s.data_ptr(), eas_s.data_ptr(),
        max_v_ptr, g_s.data_ptr(), step_ptr,
        param.shape, param.stride, step_shape, step_stride,
        param.dtype,
        float(lr), float(beta1), float(beta2),
        float(weight_decay), float(eps),
        bool(amsgrad), bool(maximize),
        runtime=runtime, stream=stream.stream,
    )
    return param


def _adamw_step_op(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq,
                   step, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize):
    return _adam_step_op(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq,
                         step, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize)


# ===========================================================================
# Phase 2: Activation function composites
# ===========================================================================


def _sgd_step_op(param, grad, buf, lr, momentum, dampening, weight_decay,
                 nesterov, maximize):
    """SGD step as composite of NPU arithmetic ops."""
    g = neg(grad) if maximize else grad
    if weight_decay != 0:
        wd_t = _scalar_to_npu_tensor(weight_decay, param)
        g = add(g, mul(wd_t, param))
    if momentum != 0:
        mom_t = _scalar_to_npu_tensor(momentum, buf)
        damp_t = _scalar_to_npu_tensor(1.0 - dampening, buf)
        # buf = momentum * buf + (1-dampening) * g
        new_buf = add(mul(mom_t, buf), mul(damp_t, g))
        copy_(buf, new_buf)
        if nesterov:
            g = add(g, mul(mom_t, buf))
        else:
            g = buf
    lr_t = _scalar_to_npu_tensor(lr, param)
    new_param = sub(param, mul(lr_t, g))
    copy_(param, new_param)
    return param


def _adagrad_step_op(param, grad, state_sum, step, lr, lr_decay,
                     weight_decay, eps, maximize):
    """Adagrad step."""
    g = neg(grad) if maximize else grad
    if weight_decay != 0:
        g = add(g, mul(_scalar_to_npu_tensor(weight_decay, param), param))
    # state_sum += g * g
    copy_(state_sum, add(state_sum, mul(g, g)))
    # clr = lr / (1 + (step-1) * lr_decay)
    clr = lr / (1.0 + (step - 1) * lr_decay)
    clr_t = _scalar_to_npu_tensor(clr, param)
    eps_t = _scalar_to_npu_tensor(eps, param)
    # param -= clr * g / (sqrt(state_sum) + eps)
    new_param = sub(param, mul(clr_t, div(g, add(sqrt(state_sum), eps_t))))
    copy_(param, new_param)
    return param


def _rmsprop_step_op(param, grad, square_avg, grad_avg, buf,
                     step, lr, alpha, eps, weight_decay, momentum,
                     centered, maximize):
    """RMSprop step."""
    g = neg(grad) if maximize else grad
    if weight_decay != 0:
        g = add(g, mul(_scalar_to_npu_tensor(weight_decay, param), param))
    alpha_t = _scalar_to_npu_tensor(alpha, square_avg)
    one_minus_alpha_t = _scalar_to_npu_tensor(1.0 - alpha, square_avg)
    # square_avg = alpha * square_avg + (1-alpha) * g * g
    copy_(square_avg, add(mul(alpha_t, square_avg), mul(one_minus_alpha_t, mul(g, g))))
    eps_t = _scalar_to_npu_tensor(eps, param)
    if centered:
        # grad_avg = alpha * grad_avg + (1-alpha) * g
        copy_(grad_avg, add(mul(alpha_t, grad_avg), mul(one_minus_alpha_t, g)))
        avg = sub(square_avg, mul(grad_avg, grad_avg))
        denom = add(sqrt(avg), eps_t)
    else:
        denom = add(sqrt(square_avg), eps_t)
    lr_t = _scalar_to_npu_tensor(lr, param)
    if momentum > 0:
        mom_t = _scalar_to_npu_tensor(momentum, buf)
        copy_(buf, add(mul(mom_t, buf), div(g, denom)))
        copy_(param, sub(param, mul(lr_t, buf)))
    else:
        copy_(param, sub(param, mul(lr_t, div(g, denom))))
    return param


def _adadelta_step_op(param, grad, square_avg, acc_delta, lr, rho, eps,
                      weight_decay, maximize):
    """Adadelta step."""
    g = neg(grad) if maximize else grad
    if weight_decay != 0:
        g = add(g, mul(_scalar_to_npu_tensor(weight_decay, param), param))
    rho_t = _scalar_to_npu_tensor(rho, square_avg)
    one_rho_t = _scalar_to_npu_tensor(1.0 - rho, square_avg)
    eps_t = _scalar_to_npu_tensor(eps, param)
    # square_avg = rho * square_avg + (1-rho) * g^2
    copy_(square_avg, add(mul(rho_t, square_avg), mul(one_rho_t, mul(g, g))))
    # delta = sqrt(acc_delta + eps) / sqrt(square_avg + eps) * g
    std = sqrt(add(acc_delta, eps_t))
    delta = mul(div(std, sqrt(add(square_avg, eps_t))), g)
    # acc_delta = rho * acc_delta + (1-rho) * delta^2
    copy_(acc_delta, add(mul(rho_t, acc_delta), mul(one_rho_t, mul(delta, delta))))
    lr_t = _scalar_to_npu_tensor(lr, param)
    copy_(param, sub(param, mul(lr_t, delta)))
    return param


def _adamax_step_op(param, grad, exp_avg, exp_inf, step, lr, beta1, beta2,
                    eps, weight_decay, maximize):
    """Adamax step."""
    g = neg(grad) if maximize else grad
    if weight_decay != 0:
        g = add(g, mul(_scalar_to_npu_tensor(weight_decay, param), param))
    b1_t = _scalar_to_npu_tensor(beta1, exp_avg)
    one_b1_t = _scalar_to_npu_tensor(1.0 - beta1, exp_avg)
    b2_t = _scalar_to_npu_tensor(beta2, exp_inf)
    eps_t = _scalar_to_npu_tensor(eps, param)
    # exp_avg = beta1 * exp_avg + (1-beta1) * g
    copy_(exp_avg, add(mul(b1_t, exp_avg), mul(one_b1_t, g)))
    # exp_inf = max(beta2 * exp_inf, abs(g) + eps)
    copy_(exp_inf, maximum(mul(b2_t, exp_inf), add(abs(g), eps_t)))
    # bias correction
    bc1 = 1.0 - beta1 ** step
    step_size = lr / bc1
    step_t = _scalar_to_npu_tensor(step_size, param)
    copy_(param, sub(param, mul(step_t, div(exp_avg, exp_inf))))
    return param


def _asgd_step_op(param, grad, ax, step, lr, lambd, alpha, t0,
                  weight_decay, maximize):
    """Averaged SGD step."""
    import math
    g = neg(grad) if maximize else grad
    if weight_decay != 0:
        g = add(g, mul(_scalar_to_npu_tensor(weight_decay, param), param))
    eta = lr / ((1.0 + lambd * lr * step) ** alpha)
    eta_t = _scalar_to_npu_tensor(eta, param)
    new_param = sub(param, mul(eta_t, g))
    copy_(param, new_param)
    if step >= t0:
        mu_t_val = 1.0 / max(1, step - t0 + 1)
        mu_t = _scalar_to_npu_tensor(mu_t_val, ax)
        # ax = ax + mu * (param - ax)
        copy_(ax, add(ax, mul(mu_t, sub(param, ax))))
    else:
        copy_(ax, param)
    return param


def _nadam_step_op(param, grad, exp_avg, exp_avg_sq, step,
                   lr, beta1, beta2, eps, weight_decay,
                   mu, mu_next, mu_product, mu_product_next, maximize):
    """NAdam step."""
    g = neg(grad) if maximize else grad
    if weight_decay != 0:
        g = add(g, mul(_scalar_to_npu_tensor(weight_decay, param), param))
    b1_t = _scalar_to_npu_tensor(beta1, exp_avg)
    one_b1_t = _scalar_to_npu_tensor(1.0 - beta1, exp_avg)
    b2_t = _scalar_to_npu_tensor(beta2, exp_avg_sq)
    one_b2_t = _scalar_to_npu_tensor(1.0 - beta2, exp_avg_sq)
    eps_t = _scalar_to_npu_tensor(eps, param)
    # Update moments
    copy_(exp_avg, add(mul(b1_t, exp_avg), mul(one_b1_t, g)))
    copy_(exp_avg_sq, add(mul(b2_t, exp_avg_sq), mul(one_b2_t, mul(g, g))))
    # Bias correction for v
    bc2 = 1.0 - beta2 ** step
    # Nesterov-corrected first moment
    c1 = mu_next / (1.0 - mu_product_next)
    c2 = mu / (1.0 - mu_product)
    ea_hat = add(mul(_scalar_to_npu_tensor(c1, exp_avg), exp_avg),
                 mul(_scalar_to_npu_tensor(c2, g), g))
    eas_hat_t = _scalar_to_npu_tensor(1.0 / bc2, exp_avg_sq)
    eas_hat = mul(exp_avg_sq, eas_hat_t)
    lr_t = _scalar_to_npu_tensor(lr, param)
    copy_(param, sub(param, mul(lr_t, div(ea_hat, add(sqrt(eas_hat), eps_t)))))
    return param


def _radam_step_op(param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2,
                   eps, weight_decay, maximize):
    """RAdam step."""
    import math
    g = neg(grad) if maximize else grad
    if weight_decay != 0:
        g = add(g, mul(_scalar_to_npu_tensor(weight_decay, param), param))
    b1_t = _scalar_to_npu_tensor(beta1, exp_avg)
    one_b1_t = _scalar_to_npu_tensor(1.0 - beta1, exp_avg)
    b2_t = _scalar_to_npu_tensor(beta2, exp_avg_sq)
    one_b2_t = _scalar_to_npu_tensor(1.0 - beta2, exp_avg_sq)
    eps_t = _scalar_to_npu_tensor(eps, param)
    # Update moments
    copy_(exp_avg, add(mul(b1_t, exp_avg), mul(one_b1_t, g)))
    copy_(exp_avg_sq, add(mul(b2_t, exp_avg_sq), mul(one_b2_t, mul(g, g))))
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    ea_corrected_t = _scalar_to_npu_tensor(1.0 / bc1, exp_avg)
    ea_corrected = mul(exp_avg, ea_corrected_t)
    rho_inf = 2.0 / (1.0 - beta2) - 1.0
    rho_t = rho_inf - 2.0 * step * (beta2 ** step) / bc2
    lr_t = _scalar_to_npu_tensor(lr, param)
    if rho_t > 5:
        eas_corrected_t = _scalar_to_npu_tensor(1.0 / bc2, exp_avg_sq)
        eas_corrected = mul(exp_avg_sq, eas_corrected_t)
        rect = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf /
                         ((rho_inf - 4) * (rho_inf - 2) * rho_t))
        rect_t = _scalar_to_npu_tensor(rect, param)
        copy_(param, sub(param, mul(lr_t, mul(rect_t, div(ea_corrected,
                                                          add(sqrt(eas_corrected), eps_t))))))
    else:
        copy_(param, sub(param, mul(lr_t, ea_corrected)))
    return param


def _rprop_step_op(param, grad, prev, step_sizes, lr, etaminus, etaplus,
                   step_size_min, step_size_max, maximize):
    """Rprop step."""
    g = neg(grad) if maximize else grad
    # sign = g * prev
    sign_prod = mul(g, prev)
    zero = _scalar_to_npu_tensor(0.0, param)
    pos_mask = gt(sign_prod, zero)
    neg_mask = lt(sign_prod, zero)
    etaplus_t = _scalar_to_npu_tensor(etaplus, step_sizes)
    etaminus_t = _scalar_to_npu_tensor(etaminus, step_sizes)
    max_t = _scalar_to_npu_tensor(step_size_max, step_sizes)
    min_t = _scalar_to_npu_tensor(step_size_min, step_sizes)
    # Adapt step sizes
    new_steps = where(pos_mask, minimum(mul(step_sizes, etaplus_t), max_t),
                      where(neg_mask, maximum(mul(step_sizes, etaminus_t), min_t),
                            step_sizes))
    copy_(step_sizes, new_steps)
    # Update params: param -= sign(g) * step_sizes
    g_sign = sign(g)
    update = mul(g_sign, step_sizes)
    # Zero out gradient where sign was negative (for prev update)
    g_for_prev = where(neg_mask, zero, g)
    copy_(param, sub(param, update))
    copy_(prev, g_for_prev)
    return param


def _sparse_adam_step_op(param, grad, exp_avg, exp_avg_sq, step, lr, beta1,
                         beta2, eps):
    """Sparse Adam step (simplified: updates all elements)."""
    b1_t = _scalar_to_npu_tensor(beta1, exp_avg)
    one_b1_t = _scalar_to_npu_tensor(1.0 - beta1, exp_avg)
    b2_t = _scalar_to_npu_tensor(beta2, exp_avg_sq)
    one_b2_t = _scalar_to_npu_tensor(1.0 - beta2, exp_avg_sq)
    eps_t = _scalar_to_npu_tensor(eps, param)
    # Update moments
    copy_(exp_avg, add(mul(b1_t, exp_avg), mul(one_b1_t, grad)))
    copy_(exp_avg_sq, add(mul(b2_t, exp_avg_sq), mul(one_b2_t, mul(grad, grad))))
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat_t = _scalar_to_npu_tensor(1.0 / bc1, exp_avg)
    v_hat_t = _scalar_to_npu_tensor(1.0 / bc2, exp_avg_sq)
    m_hat = mul(exp_avg, m_hat_t)
    v_hat = mul(exp_avg_sq, v_hat_t)
    lr_t = _scalar_to_npu_tensor(lr, param)
    copy_(param, sub(param, mul(lr_t, div(m_hat, add(sqrt(v_hat), eps_t)))))
    return param


# ===========================================================================
# Phase 5: Special function composites
# ===========================================================================
