import candle as torch
from candle.nn import functional as F
import numpy as np
import pytest


def test_silu_cpu():
    x = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], device='cpu')
    result = F.silu(x)
    # silu(x) = x * sigmoid(x)
    expected = x * torch.sigmoid(x)
    assert torch.allclose(result, expected, atol=1e-6)


def test_leaky_relu_cpu():
    x = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], device='cpu')
    result = F.leaky_relu(x)
    # Default negative_slope=0.01
    expected = torch.tensor([[-0.01, 0.0, 1.0, 2.0]], device='cpu')
    assert torch.allclose(result, expected, atol=1e-6)


def test_leaky_relu_custom_slope():
    x = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], device='cpu')
    result = F.leaky_relu(x, negative_slope=0.2)
    expected = torch.tensor([[-0.2, 0.0, 1.0, 2.0]], device='cpu')
    assert torch.allclose(result, expected, atol=1e-6)


def test_elu_cpu():
    x = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], device='cpu')
    result = F.elu(x)
    # Default alpha=1.0, elu(x) = x if x > 0 else alpha * (exp(x) - 1)
    expected_vals = [-0.6321205588, 0.0, 1.0, 2.0]
    expected = torch.tensor([expected_vals], device='cpu')
    assert torch.allclose(result, expected, atol=1e-6)


def test_elu_custom_alpha():
    x = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], device='cpu')
    result = F.elu(x, alpha=2.0)
    # elu(x) = x if x > 0 else alpha * (exp(x) - 1)
    expected_vals = [-1.264241117, 0.0, 1.0, 2.0]
    expected = torch.tensor([expected_vals], device='cpu')
    assert torch.allclose(result, expected, atol=1e-6)


def test_mish_cpu():
    x = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], device='cpu')
    result = F.mish(x)
    # mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    # Compute expected using numpy
    x_np = np.array([[-1.0, 0.0, 1.0, 2.0]])
    expected_np = x_np * np.tanh(np.log1p(np.exp(x_np)))
    expected = torch.tensor(expected_np.tolist(), device='cpu')
    assert torch.allclose(result, expected, atol=1e-6)


def test_prelu_cpu():
    x = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], device='cpu')
    weight = torch.tensor([0.25], device='cpu')
    result = F.prelu(x, weight)
    # prelu(x) = x if x > 0 else weight * x
    expected = torch.tensor([[-0.25, 0.0, 1.0, 2.0]], device='cpu')
    assert torch.allclose(result, expected, atol=1e-6)


# --- Activation Autograd Tests ---

def test_silu_backward():
    x = torch.tensor([-1.0, 0.0, 1.0], device='cpu')
    x.requires_grad = True
    y = F.silu(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_leaky_relu_backward():
    x = torch.tensor([-1.0, 0.0, 1.0], device='cpu')
    x.requires_grad = True
    y = F.leaky_relu(x, negative_slope=0.01)
    y.sum().backward()
    assert x.grad is not None
    # For x > 0, grad = 1; for x <= 0, grad = negative_slope
    expected = np.array([0.01, 0.01, 1.0])
    np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5)


def test_elu_backward():
    x = torch.tensor([-1.0, 0.0, 1.0], device='cpu')
    x.requires_grad = True
    y = F.elu(x, alpha=1.0)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_mish_backward():
    x = torch.tensor([-1.0, 0.0, 1.0], device='cpu')
    x.requires_grad = True
    y = F.mish(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_prelu_backward():
    x = torch.tensor([-1.0, 0.0, 1.0], device='cpu')
    x.requires_grad = True
    weight = torch.tensor([0.25], device='cpu')
    y = F.prelu(x, weight)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# --- Loss Function Tests ---

def test_l1_loss_mean():
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cpu')
    target = torch.tensor([[1.5, 1.5], [3.5, 3.5]], device='cpu')
    result = F.l1_loss(input, target, reduction='mean')
    # |1.0-1.5| + |2.0-1.5| + |3.0-3.5| + |4.0-3.5| = 0.5 + 0.5 + 0.5 + 0.5 = 2.0
    # mean = 2.0 / 4 = 0.5
    expected = torch.tensor(0.5, device='cpu')
    assert torch.allclose(result, expected, atol=1e-6)


def test_l1_loss_sum():
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cpu')
    target = torch.tensor([[1.5, 1.5], [3.5, 3.5]], device='cpu')
    result = F.l1_loss(input, target, reduction='sum')
    # sum = 2.0
    expected = torch.tensor(2.0, device='cpu')
    assert torch.allclose(result, expected, atol=1e-6)


def test_l1_loss_none():
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cpu')
    target = torch.tensor([[1.5, 1.5], [3.5, 3.5]], device='cpu')
    result = F.l1_loss(input, target, reduction='none')
    expected = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cpu')
    assert torch.allclose(result, expected, atol=1e-6)


def test_mse_loss_mean():
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cpu')
    target = torch.tensor([[1.5, 1.5], [3.5, 3.5]], device='cpu')
    result = F.mse_loss(input, target, reduction='mean')
    # (1.0-1.5)^2 + (2.0-1.5)^2 + (3.0-3.5)^2 + (4.0-3.5)^2 = 0.25 + 0.25 + 0.25 + 0.25 = 1.0
    # mean = 1.0 / 4 = 0.25
    expected = torch.tensor(0.25, device='cpu')
    assert torch.allclose(result, expected, atol=1e-6)


def test_mse_loss_sum():
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cpu')
    target = torch.tensor([[1.5, 1.5], [3.5, 3.5]], device='cpu')
    result = F.mse_loss(input, target, reduction='sum')
    # sum = 1.0
    expected = torch.tensor(1.0, device='cpu')
    assert torch.allclose(result, expected, atol=1e-6)


def test_smooth_l1_loss_mean():
    input = torch.tensor([[0.5, 2.0], [0.0, 1.5]], device='cpu')
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device='cpu')
    result = F.smooth_l1_loss(input, target, reduction='mean', beta=1.0)
    # For beta=1.0:
    # |0.5-0.0| = 0.5 < 1.0 -> 0.5 * 0.5^2 / 1.0 = 0.125
    # |2.0-0.0| = 2.0 >= 1.0 -> 2.0 - 0.5 * 1.0 = 1.5
    # |0.0-0.0| = 0.0 < 1.0 -> 0.5 * 0.0^2 / 1.0 = 0.0
    # |1.5-0.0| = 1.5 >= 1.0 -> 1.5 - 0.5 * 1.0 = 1.0
    # mean = (0.125 + 1.5 + 0.0 + 1.0) / 4 = 2.625 / 4 = 0.65625
    expected = torch.tensor(0.65625, device='cpu')
    assert torch.allclose(result, expected, atol=1e-6)


def test_l1_loss_backward():
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cpu')
    input.requires_grad = True
    target = torch.tensor([[1.5, 1.5], [3.5, 3.5]], device='cpu')
    loss = F.l1_loss(input, target, reduction='mean')
    loss.backward()
    assert input.grad is not None
    assert input.grad.shape == input.shape


def test_mse_loss_backward():
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cpu')
    input.requires_grad = True
    target = torch.tensor([[1.5, 1.5], [3.5, 3.5]], device='cpu')
    loss = F.mse_loss(input, target, reduction='mean')
    loss.backward()
    assert input.grad is not None
    assert input.grad.shape == input.shape


# --- Normalization Tests ---

def test_batch_norm_eval():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cpu')  # (N=2, C=2)
    running_mean = torch.tensor([0.0, 0.0], device='cpu')
    running_var = torch.tensor([1.0, 1.0], device='cpu')
    weight = torch.tensor([1.0, 1.0], device='cpu')
    bias = torch.tensor([0.0, 0.0], device='cpu')
    out = F.batch_norm(x, running_mean, running_var, weight, bias, training=False)
    # With mean=0, var=1, weight=1, bias=0: output = input
    assert torch.allclose(out, x, atol=1e-5)


def test_batch_norm_eval_with_stats():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cpu')  # (N=2, C=2)
    running_mean = torch.tensor([2.0, 3.0], device='cpu')
    running_var = torch.tensor([1.0, 1.0], device='cpu')
    out = F.batch_norm(x, running_mean, running_var, training=False)
    # (x - mean) / sqrt(var + eps)
    expected_vals = [[-1.0, -1.0], [1.0, 1.0]]
    expected = torch.tensor(expected_vals, device='cpu')
    assert torch.allclose(out, expected, atol=1e-4)


def test_group_norm_cpu():
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device='cpu')  # (N=1, C=2, L=2)
    out = F.group_norm(x, num_groups=2)
    # Each group (channel) is normalized independently
    # Channel 0: [1,2] -> mean=1.5, var=0.25 -> (x-1.5)/sqrt(0.25+1e-5)
    # Channel 1: [3,4] -> mean=3.5, var=0.25 -> (x-3.5)/sqrt(0.25+1e-5)
    xn = x.numpy()
    result = np.zeros_like(xn)
    for c in range(2):
        ch = xn[0, c]
        mean = ch.mean()
        var = ch.var()
        result[0, c] = (ch - mean) / np.sqrt(var + 1e-5)
    expected = torch.tensor(result.tolist(), device='cpu')
    assert torch.allclose(out, expected, atol=1e-4)


# --- Dropout Tests ---

def test_dropout_eval():
    x = torch.tensor([1.0, 2.0, 3.0], device='cpu')
    out = F.dropout(x, p=0.5, training=False)
    assert torch.allclose(out, x, atol=1e-5)


def test_dropout_training():
    x = torch.tensor([1.0] * 1000, device='cpu')
    out = F.dropout(x, p=0.5, training=True)
    # ~50% should be zeroed, rest scaled by 2
    zeros = (out.numpy() == 0).sum()
    assert 300 < zeros < 700  # rough check


# --- Pad Tests ---

def test_pad_constant():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cpu')
    out = F.pad(x, (1, 1, 0, 0), mode='constant', value=0)
    expected = torch.tensor([[0.0, 1.0, 2.0, 0.0], [0.0, 3.0, 4.0, 0.0]], device='cpu')
    assert torch.allclose(out, expected, atol=1e-5)


def test_pad_reflect():
    x = torch.tensor([[1.0, 2.0, 3.0]], device='cpu')
    out = F.pad(x, (1, 1), mode='reflect')
    expected = torch.tensor([[2.0, 1.0, 2.0, 3.0, 2.0]], device='cpu')
    assert torch.allclose(out, expected, atol=1e-5)


# --- Loss Function Tests (NLL, Cross Entropy, BCE, KL Div) ---

def test_nll_loss_mean():
    # log_softmax output shape: (N=2, C=3)
    log_probs = torch.tensor([[-0.5, -1.0, -2.0], [-0.3, -0.8, -1.5]], device='cpu')
    target = torch.tensor([0, 2], device='cpu')
    loss = F.nll_loss(log_probs, target, reduction='mean')
    # nll = (-log_probs[0, 0] + -log_probs[1, 2]) / 2 = (0.5 + 1.5) / 2 = 1.0
    expected = torch.tensor(1.0, device='cpu')
    assert torch.allclose(loss, expected, atol=1e-5)


def test_cross_entropy_mean():
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], device='cpu')
    target = torch.tensor([0, 1], device='cpu')
    loss = F.cross_entropy(logits, target, reduction='mean')
    # Manual: log_softmax + nll_loss
    xn = logits.numpy()
    log_probs = xn - np.log(np.exp(xn).sum(axis=1, keepdims=True))
    expected_val = -(log_probs[0, 0] + log_probs[1, 1]) / 2
    expected = torch.tensor(expected_val, device='cpu')
    assert torch.allclose(loss, expected, atol=1e-4)


def test_binary_cross_entropy_mean():
    pred = torch.tensor([0.8, 0.4, 0.6], device='cpu')
    target = torch.tensor([1.0, 0.0, 1.0], device='cpu')
    loss = F.binary_cross_entropy(pred, target, reduction='mean')
    pn = pred.numpy()
    tn = target.numpy()
    eps = 1e-12
    expected_val = -np.mean(tn * np.log(pn + eps) + (1 - tn) * np.log(1 - pn + eps))
    expected = torch.tensor(expected_val, device='cpu')
    assert torch.allclose(loss, expected, atol=1e-4)


def test_bce_with_logits_mean():
    logits = torch.tensor([1.0, -1.0, 0.5], device='cpu')
    target = torch.tensor([1.0, 0.0, 1.0], device='cpu')
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
    # sigmoid(logits) then BCE
    ln = logits.numpy()
    tn = target.numpy()
    sig = 1.0 / (1.0 + np.exp(-ln))
    eps = 1e-12
    expected_val = -np.mean(tn * np.log(sig + eps) + (1 - tn) * np.log(1 - sig + eps))
    expected = torch.tensor(expected_val, device='cpu')
    assert torch.allclose(loss, expected, atol=1e-4)


def test_kl_div_mean():
    log_pred = torch.tensor([-0.5, -1.0, -1.5], device='cpu')
    target = torch.tensor([0.4, 0.3, 0.3], device='cpu')
    loss = F.kl_div(log_pred, target, reduction='mean')
    # KL = target * (log(target) - log_pred)
    pn = log_pred.numpy()
    tn = target.numpy()
    expected_val = np.mean(tn * (np.log(tn + 1e-12) - pn))
    expected = torch.tensor(expected_val, device='cpu')
    assert torch.allclose(loss, expected, atol=1e-4)


# --- CPU Parity P1 Tests ---


def _assert_inplace_wrapper(fn, *args, expected):
    x = torch.tensor([-1.0, 0.0, 2.0], device='cpu')
    out = fn(x, *args)
    assert out is x
    assert torch.allclose(x, expected, atol=1e-6)


def test_relu_inplace_wrapper_exists_and_mutates():
    _assert_inplace_wrapper(F.relu_, expected=torch.tensor([0.0, 0.0, 2.0], device='cpu'))


def test_elu_inplace_wrapper_exists_and_mutates():
    expected = F.elu(torch.tensor([-1.0, 0.0, 2.0], device='cpu'))
    _assert_inplace_wrapper(F.elu_, expected=expected)


def test_hardtanh_inplace_wrapper_exists_and_mutates():
    expected = torch.tensor([-0.5, 0.0, 0.5], device='cpu')
    _assert_inplace_wrapper(F.hardtanh_, -0.5, 0.5, expected=expected)


def test_rrelu_inplace_wrapper_exists_and_mutates_training_false():
    x = torch.tensor([-1.0, 0.0, 2.0], device='cpu')
    out = F.rrelu_(x, training=False)
    assert out is x
    # training=False should be deterministic slope=(lower+upper)/2
    expected = F.rrelu(torch.tensor([-1.0, 0.0, 2.0], device='cpu'), training=False)
    assert torch.allclose(x, expected, atol=1e-6)


def test_threshold_inplace_wrapper_exists_and_mutates():
    expected = torch.tensor([3.0, 3.0, 2.0], device='cpu')
    _assert_inplace_wrapper(F.threshold_, 0.5, 3.0, expected=expected)


def test_selu_inplace_wrapper_exists_and_mutates():
    expected = F.selu(torch.tensor([-1.0, 0.0, 2.0], device='cpu'))
    _assert_inplace_wrapper(F.selu_, expected=expected)


def test_celu_inplace_wrapper_exists_and_mutates():
    expected = F.celu(torch.tensor([-1.0, 0.0, 2.0], device='cpu'), alpha=0.75)
    _assert_inplace_wrapper(F.celu_, 0.75, expected=expected)


def test_max_pool1d_with_indices_returns_tuple():
    x = torch.tensor([[[1.0, 3.0, 2.0, 4.0]]], device='cpu')
    out, idx = F.max_pool1d_with_indices(x, kernel_size=2, stride=2)
    assert out.shape == (1, 1, 2)
    assert idx.shape == (1, 1, 2)


def test_max_pool2d_with_indices_returns_tuple():
    x = torch.tensor([[[[1.0, 3.0], [2.0, 4.0]]]], device='cpu')
    out, idx = F.max_pool2d_with_indices(x, kernel_size=2)
    assert out.shape == (1, 1, 1, 1)
    assert idx.shape == (1, 1, 1, 1)


def test_adaptive_max_pool1d_wrapper_exists():
    x = torch.tensor([[[1.0, 3.0, 2.0, 4.0]]], device='cpu')
    out = F.adaptive_max_pool1d(x, output_size=2)
    assert out.shape == (1, 1, 2)


def test_adaptive_max_pool2d_return_indices_returns_tuple():
    x = torch.tensor([[[[1.0, 3.0], [2.0, 4.0]]]], device='cpu')
    out, idx = F.adaptive_max_pool2d(x, output_size=(1, 1), return_indices=True)
    assert out.shape == (1, 1, 1, 1)
    assert idx.shape == (1, 1, 1, 1)


def test_max_unpool1d_scatter_matches_indices():
    x = torch.tensor([[[1.0, 3.0, 2.0, 4.0]]], device='cpu')
    pooled, indices = F.max_pool1d_with_indices(x, kernel_size=2, stride=2)
    out = F.max_unpool1d(pooled, indices, kernel_size=2, stride=2, output_size=(1, 1, 4))
    expected = torch.tensor([[[0.0, 3.0, 0.0, 4.0]]], device='cpu')
    assert torch.allclose(out, expected, atol=1e-6)


def test_max_unpool2d_scatter_matches_indices():
    x = torch.tensor([[[[1.0, 3.0], [2.0, 4.0]]]], device='cpu')
    pooled, indices = F.max_pool2d_with_indices(x, kernel_size=2, stride=2)
    out = F.max_unpool2d(pooled, indices, kernel_size=2, stride=2, output_size=(1, 1, 2, 2))
    expected = torch.tensor([[[[0.0, 0.0], [0.0, 4.0]]]], device='cpu')
    assert torch.allclose(out, expected, atol=1e-6)


def test_max_unpool1d_invalid_index_raises():
    x = torch.tensor([[[2.0, 1.0]]], device='cpu')
    pooled = torch.tensor([[[2.0]]], device='cpu')
    bad_indices = torch.tensor([[[5]]], device='cpu', dtype=torch.int64)
    with pytest.raises((RuntimeError, ValueError, IndexError)):
        F.max_unpool1d(pooled, bad_indices, kernel_size=2, stride=2, output_size=(1, 1, 2))
