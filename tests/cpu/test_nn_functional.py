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


def test_hardswish_matches_torch():
    import torch as torch_ref

    x = torch.tensor([-4.0, -3.0, -1.0, 0.0, 2.0, 4.0], dtype=torch.float32)
    out = F.hardswish(x)

    x_ref = torch_ref.tensor([-4.0, -3.0, -1.0, 0.0, 2.0, 4.0], dtype=torch_ref.float32)
    ref = torch_ref.nn.functional.hardswish(x_ref)

    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_hardsigmoid_matches_torch():
    import torch as torch_ref

    x = torch.tensor([-4.0, -3.0, -1.0, 0.0, 2.0, 4.0], dtype=torch.float32)
    out = F.hardsigmoid(x)

    x_ref = torch_ref.tensor([-4.0, -3.0, -1.0, 0.0, 2.0, 4.0], dtype=torch_ref.float32)
    ref = torch_ref.nn.functional.hardsigmoid(x_ref)

    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_softsign_matches_torch():
    import torch as torch_ref

    x = torch.tensor([-4.0, -1.0, 0.0, 2.0, 4.0], dtype=torch.float32)
    out = F.softsign(x)

    x_ref = torch_ref.tensor([-4.0, -1.0, 0.0, 2.0, 4.0], dtype=torch_ref.float32)
    ref = torch_ref.nn.functional.softsign(x_ref)

    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_hardswish_routes_through_dispatch(monkeypatch):
    import candle._dispatch as dispatch_mod

    calls = []
    expected = torch.tensor([42.0], dtype=torch.float32)

    def fake_dispatch(name, dispatch_device, input):
        calls.append((name, dispatch_device, tuple(input.shape)))
        return expected

    monkeypatch.setattr(dispatch_mod, "dispatch", fake_dispatch)
    x = torch.tensor([1.0], dtype=torch.float32)

    out = F.hardswish(x)

    assert out is expected
    assert calls == [("hardswish", "cpu", (1,))]


def test_hardsigmoid_routes_through_dispatch(monkeypatch):
    import candle._dispatch as dispatch_mod

    calls = []
    expected = torch.tensor([42.0], dtype=torch.float32)

    def fake_dispatch(name, dispatch_device, input):
        calls.append((name, dispatch_device, tuple(input.shape)))
        return expected

    monkeypatch.setattr(dispatch_mod, "dispatch", fake_dispatch)
    x = torch.tensor([1.0], dtype=torch.float32)

    out = F.hardsigmoid(x)

    assert out is expected
    assert calls == [("hardsigmoid", "cpu", (1,))]


def test_softsign_routes_through_dispatch(monkeypatch):
    import candle._dispatch as dispatch_mod

    calls = []
    expected = torch.tensor([42.0], dtype=torch.float32)

    def fake_dispatch(name, dispatch_device, input):
        calls.append((name, dispatch_device, tuple(input.shape)))
        return expected

    monkeypatch.setattr(dispatch_mod, "dispatch", fake_dispatch)
    x = torch.tensor([1.0], dtype=torch.float32)

    out = F.softsign(x)

    assert out is expected
    assert calls == [("softsign", "cpu", (1,))]


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


def test_normalize_matches_torch_l2_dim1():
    import torch as torch_ref
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float32)
    out = F.normalize(x, p=2.0, dim=1, eps=1e-12)

    x_ref = torch_ref.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch_ref.float32)
    ref = torch_ref.nn.functional.normalize(x_ref, p=2.0, dim=1, eps=1e-12)

    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_normalize_matches_torch_l1_dim0():
    import torch as torch_ref
    x = torch.tensor([[1.0, -1.0], [3.0, 1.0]], dtype=torch.float32)
    out = F.normalize(x, p=1.0, dim=0, eps=1e-12)

    x_ref = torch_ref.tensor([[1.0, -1.0], [3.0, 1.0]], dtype=torch_ref.float32)
    ref = torch_ref.nn.functional.normalize(x_ref, p=1.0, dim=0, eps=1e-12)

    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_normalize_zero_norm_respects_eps_matches_torch():
    import torch as torch_ref
    x = torch.tensor([[0.0, 0.0], [1e-20, -1e-20]], dtype=torch.float32)
    out = F.normalize(x, p=2.0, dim=1, eps=1e-12)

    x_ref = torch_ref.tensor([[0.0, 0.0], [1e-20, -1e-20]], dtype=torch_ref.float32)
    ref = torch_ref.nn.functional.normalize(x_ref, p=2.0, dim=1, eps=1e-12)

    np.testing.assert_allclose(out.numpy(), ref.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_normalize_routes_through_dispatch(monkeypatch):
    calls = []
    expected = torch.tensor([[42.0]], dtype=torch.float32)

    def fake_dispatch(name, dispatch_device, input, p, dim, eps):
        calls.append((name, dispatch_device, input.shape, p, dim, eps))
        return expected

    import candle._dispatch as dispatch_mod
    monkeypatch.setattr(dispatch_mod, "dispatch", fake_dispatch)
    x = torch.tensor([[3.0]], dtype=torch.float32)

    out = F.normalize(x, p=2.0, dim=1, eps=1e-12)

    assert out is expected
    assert calls == [("normalize", "cpu", (1, 1), 2.0, 1, 1e-12)]


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


def test_max_pool2d_with_indices_flattening_matches_torch():
    import torch as torch_ref

    x = torch.tensor([[[[1.0, 3.0], [2.0, 4.0]]]], device='cpu')
    x_ref = torch_ref.tensor(x.numpy())
    pooled, indices = F.max_pool2d_with_indices(x, kernel_size=2, stride=2)
    ref_pooled, ref_indices = torch_ref.nn.functional.max_pool2d(
        x_ref,
        kernel_size=2,
        stride=2,
        return_indices=True,
    )
    ref_pooled_c = torch.tensor(ref_pooled.numpy())
    ref_indices_c = torch.tensor(ref_indices.numpy(), dtype=torch.int64)
    assert torch.allclose(pooled, ref_pooled_c, atol=1e-6)
    assert torch.equal(indices, ref_indices_c)



def test_max_unpool1d_infers_output_size_when_none_matches_torch():
    import torch as torch_ref

    x = torch.tensor([[[1.0, 3.0, 2.0, 4.0]]], device='cpu')
    x_ref = torch_ref.tensor(x.numpy())
    pooled, indices = F.max_pool1d_with_indices(x, kernel_size=2, stride=2)
    out = F.max_unpool1d(pooled, indices, kernel_size=2, stride=2)
    ref_pooled, ref_indices = torch_ref.nn.functional.max_pool1d(
        x_ref,
        kernel_size=2,
        stride=2,
        return_indices=True,
    )
    ref_out = torch_ref.nn.functional.max_unpool1d(
        ref_pooled,
        ref_indices,
        kernel_size=2,
        stride=2,
    )
    ref_out_c = torch.tensor(ref_out.numpy())
    assert out.shape == ref_out_c.shape
    assert torch.allclose(out, ref_out_c, atol=1e-6)



def test_max_pool1d_padding_too_large_matches_torch():
    import torch as torch_ref

    x = torch.tensor([[[1.0, 2.0, 3.0]]], device='cpu')
    x_ref = torch_ref.tensor(x.numpy())
    with pytest.raises(RuntimeError):
        torch_ref.nn.functional.max_pool1d(x_ref, kernel_size=3, stride=1, padding=2)
    with pytest.raises(RuntimeError):
        F.max_pool1d(x, kernel_size=3, stride=1, padding=2)



def test_max_pool2d_padding_too_large_matches_torch():
    import torch as torch_ref

    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device='cpu')
    x_ref = torch_ref.tensor(x.numpy())
    with pytest.raises(RuntimeError):
        torch_ref.nn.functional.max_pool2d(x_ref, kernel_size=2, stride=1, padding=2)
    with pytest.raises(RuntimeError):
        F.max_pool2d(x, kernel_size=2, stride=1, padding=2)


def test_max_pool3d_with_indices_returns_tuple():
    x = torch.tensor([[[[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]]]], device='cpu')
    out, idx = F.max_pool3d_with_indices(x, kernel_size=2, stride=2)
    assert out.shape == (1, 1, 1, 1, 1)
    assert idx.shape == (1, 1, 1, 1, 1)


def test_adaptive_max_pool2d_with_indices_wrapper_exists():
    x = torch.tensor([[[[1.0, 3.0], [2.0, 4.0]]]], device='cpu')
    out, idx = F.adaptive_max_pool2d_with_indices(x, output_size=(1, 1))
    assert out.shape == (1, 1, 1, 1)
    assert idx.shape == (1, 1, 1, 1)


def test_adaptive_max_pool3d_wrapper_exists():
    x = torch.tensor([[[[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]]]], device='cpu')
    out = F.adaptive_max_pool3d(x, output_size=(1, 1, 1))
    assert out.shape == (1, 1, 1, 1, 1)


def test_adaptive_max_pool3d_with_indices_wrapper_exists():
    x = torch.tensor([[[[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]]]], device='cpu')
    out, idx = F.adaptive_max_pool3d_with_indices(x, output_size=(1, 1, 1))
    assert out.shape == (1, 1, 1, 1, 1)
    assert idx.shape == (1, 1, 1, 1, 1)


def test_max_unpool3d_scatter_matches_indices():
    x = torch.tensor([[[[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]]]], device='cpu')
    pooled, indices = F.max_pool3d_with_indices(x, kernel_size=2, stride=2)
    out = F.max_unpool3d(pooled, indices, kernel_size=2, stride=2, output_size=(1, 1, 2, 2, 2))
    expected = torch.tensor([[[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 8.0]]]]], device='cpu')
    assert torch.allclose(out, expected, atol=1e-6)


def test_max_unpool3d_invalid_index_raises():
    pooled = torch.tensor([[[[[2.0]]]]], device='cpu')
    bad_indices = torch.tensor([[[[[99]]]]], device='cpu', dtype=torch.int64)
    with pytest.raises((RuntimeError, ValueError, IndexError)):
        F.max_unpool3d(pooled, bad_indices, kernel_size=2, stride=2, output_size=(1, 1, 2, 2, 2))


def test_adaptive_max_pool1d_with_indices_wrapper_exists():
    x = torch.tensor([[[1.0, 3.0, 2.0, 4.0]]], device='cpu')
    out, idx = F.adaptive_max_pool1d_with_indices(x, output_size=2)
    assert out.shape == (1, 1, 2)
    assert idx.shape == (1, 1, 2)


def test_leaky_relu_inplace_wrapper_exists_and_mutates():
    x = torch.tensor([-2.0, 0.0, 2.0], device='cpu')
    out = F.leaky_relu_(x, negative_slope=0.1)
    assert out is x
    expected = torch.tensor([-0.2, 0.0, 2.0], device='cpu')
    assert torch.allclose(x, expected, atol=1e-6)



def _assert_inplace_leaf_requires_grad_raises(fn, *args, **kwargs):
    x = torch.tensor([-1.0, 0.0, 2.0], device='cpu')
    x.requires_grad = True
    with pytest.raises(RuntimeError):
        fn(x, *args, **kwargs)


def test_inplace_activation_non_leaf_requires_grad_allowed():
    base = torch.tensor([-1.0, 0.0, 2.0], device='cpu')
    base.requires_grad = True
    x = base * 1.0
    out = F.relu_(x)
    assert out is x
    assert torch.allclose(x, torch.tensor([0.0, 0.0, 2.0], device='cpu'), atol=1e-6)


def test_inplace_activation_leaf_requires_grad_errors():
    _assert_inplace_leaf_requires_grad_raises(F.relu_)
    _assert_inplace_leaf_requires_grad_raises(F.elu_)
    _assert_inplace_leaf_requires_grad_raises(F.hardtanh_, -0.5, 0.5)
    _assert_inplace_leaf_requires_grad_raises(F.rrelu_, training=False)
    _assert_inplace_leaf_requires_grad_raises(F.threshold_, 0.5, 3.0)
    _assert_inplace_leaf_requires_grad_raises(F.selu_)
    _assert_inplace_leaf_requires_grad_raises(F.celu_, 0.75)
    _assert_inplace_leaf_requires_grad_raises(F.leaky_relu_, negative_slope=0.1)


def test_rrelu_training_true_backward_uses_forward_slope():
    # When training=True, backward should use the same random slope as forward.
    torch.manual_seed(123)
    x = torch.tensor([-2.0, -1.0, 0.0, 2.0], device='cpu')
    x.requires_grad = True
    out = F.rrelu(x, lower=0.1, upper=0.2, training=True)
    out.sum().backward()
    # For negative inputs, grad should match out / input (slope from forward).
    neg_mask = x < 0
    slopes = out[neg_mask].detach() / x[neg_mask].detach()
    assert torch.allclose(x.grad[neg_mask], slopes, atol=1e-6)


def test_upsample_alias_matches_interpolate_nearest_2d():
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device='cpu')
    out = F.upsample(x, size=(4, 4), mode='nearest')
    ref = F.interpolate(x, size=(4, 4), mode='nearest')
    assert out.shape == (1, 1, 4, 4)
    assert torch.allclose(out, ref, atol=1e-6)


def test_upsample_nearest_alias_exists():
    x = torch.tensor([[[1.0, 2.0, 3.0]]], device='cpu')
    out = F.upsample_nearest(x, size=6)
    ref = F.interpolate(x, size=6, mode='nearest')
    assert out.shape == (1, 1, 6)
    assert torch.allclose(out, ref, atol=1e-6)


def test_upsample_bilinear_alias_exists():
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device='cpu')
    out = F.upsample_bilinear(x, size=(4, 4), align_corners=False)
    ref = F.interpolate(x, size=(4, 4), mode='bilinear', align_corners=False)
    assert out.shape == (1, 1, 4, 4)
    assert torch.allclose(out, ref, atol=1e-6)


def test_lp_pool1d_wrapper_exists_matches_torch():
    x = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]], device='cpu')
    out = F.lp_pool1d(x, norm_type=2.0, kernel_size=2, stride=2)
    expected = torch.tensor([[[np.sqrt(5.0), 5.0]]], device='cpu')
    assert out.shape == (1, 1, 2)
    assert torch.allclose(out, expected, atol=1e-6)


def test_lp_pool2d_wrapper_exists_matches_torch():
    x = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]], device='cpu')
    out = F.lp_pool2d(x, norm_type=2.0, kernel_size=2, stride=2)
    expected = torch.tensor([[[[np.sqrt(30.0)]]]], device='cpu')
    assert out.shape == (1, 1, 1, 1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_lp_pool3d_wrapper_exists_matches_expected():
    x = torch.tensor([[[[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]]]], device='cpu')
    out = F.lp_pool3d(x, norm_type=2.0, kernel_size=2, stride=2)
    expected = torch.tensor([[[[[np.sqrt(204.0)]]]]], device='cpu')
    assert out.shape == (1, 1, 1, 1, 1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_multilabel_margin_loss_wrapper_exists_basic():
    # input: (N=1, C=4), target uses class indices and -1 padding
    input = torch.tensor([[0.9, 0.2, 0.8, -0.1]], device='cpu')
    target = torch.tensor([[0, 2, -1, -1]], device='cpu', dtype=torch.int64)
    loss = F.multilabel_margin_loss(input, target, reduction='mean')
    # For y={0,2}, non-targets={1,3}:
    # max(0, 1 - (x0-x1))=0.3, max(0,1-(x0-x3))=0.0
    # max(0, 1 - (x2-x1))=0.4, max(0,1-(x2-x3))=0.1
    # sum=0.8, divide by C=4 -> 0.2
    expected = torch.tensor(0.2, device='cpu')
    assert torch.allclose(loss, expected, atol=1e-6)


def test_native_channel_shuffle_wrapper_exists_matches_channel_shuffle():
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], device='cpu')  # (N=1,C=4,L=1)
    out = F.native_channel_shuffle(x, groups=2)

    # Compare against torch reference on CPU using detached numpy values.
    import torch as torch_ref
    x_ref = torch_ref.tensor(x.numpy())
    ref = torch_ref.nn.functional.native_channel_shuffle(x_ref, groups=2)

    assert out.shape == x.shape
    np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=1e-6, atol=1e-6)


def test_triplet_margin_with_distance_loss_wrapper_exists():
    anchor = torch.tensor([[0.0, 0.0]], device='cpu')
    positive = torch.tensor([[1.0, 0.0]], device='cpu')
    negative = torch.tensor([[3.0, 0.0]], device='cpu')
    loss = F.triplet_margin_with_distance_loss(anchor, positive, negative, margin=1.0, reduction='mean')
    # d(ap)=1, d(an)=3 => max(1 - 3 + 1, 0)=0
    expected = torch.tensor(0.0, device='cpu')
    assert torch.allclose(loss, expected, atol=1e-6)


def test_conv_tbc_wrapper_exists():
    # T=3, B=1, C_in=2
    x = torch.tensor([
        [[1.0, 2.0]],
        [[3.0, 4.0]],
        [[5.0, 6.0]],
    ], device='cpu')
    # weight shape: (K=2, C_in=2, C_out=1)
    w = torch.tensor([
        [[1.0], [1.0]],
        [[1.0], [1.0]],
    ], device='cpu')
    b = torch.tensor([0.0], device='cpu')
    out = F.conv_tbc(x, w, b, pad=0)
    # out[t] = sum over k,c of x[t+k,c]*w[k,c]
    expected = torch.tensor([
        [[10.0]],
        [[18.0]],
    ], device='cpu')
    assert out.shape == (2, 1, 1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_multi_head_attention_forward_wrapper_exists_basic_need_weights_true():
    # Basic parity check against torch reference on CPU.
    import torch as torch_ref
    import torch.nn.functional as F_ref

    torch_ref.manual_seed(0)
    L, S, N, E, H = 2, 2, 1, 4, 2
    query = torch_ref.randn(L, N, E)
    key = torch_ref.randn(S, N, E)
    value = torch_ref.randn(S, N, E)

    in_proj_weight = torch_ref.randn(3 * E, E)
    in_proj_bias = torch_ref.randn(3 * E)
    out_proj_weight = torch_ref.randn(E, E)
    out_proj_bias = torch_ref.randn(E)

    out_ref, w_ref = F_ref.multi_head_attention_forward(
        query,
        key,
        value,
        embed_dim_to_check=E,
        num_heads=H,
        in_proj_weight=in_proj_weight,
        in_proj_bias=in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=out_proj_weight,
        out_proj_bias=out_proj_bias,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )

    # Run candle version with identical inputs.
    q = torch.tensor(query.numpy())
    k = torch.tensor(key.numpy())
    v = torch.tensor(value.numpy())
    w = torch.tensor(in_proj_weight.numpy())
    b = torch.tensor(in_proj_bias.numpy())
    ow = torch.tensor(out_proj_weight.numpy())
    ob = torch.tensor(out_proj_bias.numpy())

    out, attn_w = F.multi_head_attention_forward(
        q,
        k,
        v,
        embed_dim_to_check=E,
        num_heads=H,
        in_proj_weight=w,
        in_proj_bias=b,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=ow,
        out_proj_bias=ob,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )

    assert out.shape == (L, N, E)
    assert attn_w is not None
    assert attn_w.shape == (N, L, S)
    np.testing.assert_allclose(out.numpy(), out_ref.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(attn_w.numpy(), w_ref.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_multi_head_attention_forward_attn_mask_bool_and_key_padding_mask_bool():
    # Verify boolean mask semantics match torch: True means masked out.
    import torch as torch_ref
    import torch.nn.functional as F_ref

    torch_ref.manual_seed(0)
    L, S, N, E, H = 2, 2, 1, 4, 2
    query = torch_ref.randn(L, N, E)
    key = torch_ref.randn(S, N, E)
    value = torch_ref.randn(S, N, E)
    in_proj_weight = torch_ref.randn(3 * E, E)
    in_proj_bias = torch_ref.randn(3 * E)
    out_proj_weight = torch_ref.randn(E, E)
    out_proj_bias = torch_ref.randn(E)

    attn_mask = torch_ref.tensor([[False, True], [False, False]])
    key_padding_mask = torch_ref.tensor([[False, True]])

    out_ref, w_ref = F_ref.multi_head_attention_forward(
        query,
        key,
        value,
        embed_dim_to_check=E,
        num_heads=H,
        in_proj_weight=in_proj_weight,
        in_proj_bias=in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=out_proj_weight,
        out_proj_bias=out_proj_bias,
        training=False,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        attn_mask=attn_mask,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )

    q = torch.tensor(query.numpy())
    k = torch.tensor(key.numpy())
    v = torch.tensor(value.numpy())
    w = torch.tensor(in_proj_weight.numpy())
    b = torch.tensor(in_proj_bias.numpy())
    ow = torch.tensor(out_proj_weight.numpy())
    ob = torch.tensor(out_proj_bias.numpy())
    am = torch.tensor(attn_mask.numpy(), dtype=torch.bool)
    kpm = torch.tensor(key_padding_mask.numpy(), dtype=torch.bool)

    out, attn_w = F.multi_head_attention_forward(
        q,
        k,
        v,
        embed_dim_to_check=E,
        num_heads=H,
        in_proj_weight=w,
        in_proj_bias=b,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=ow,
        out_proj_bias=ob,
        training=False,
        key_padding_mask=kpm,
        need_weights=True,
        attn_mask=am,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )

    np.testing.assert_allclose(out.numpy(), out_ref.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(attn_w.numpy(), w_ref.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_multi_head_attention_forward_wrapper_need_weights_false_returns_none_weights():
    import torch as torch_ref
    import torch.nn.functional as F_ref

    torch_ref.manual_seed(0)
    L, S, N, E, H = 2, 2, 1, 4, 2
    query = torch_ref.randn(L, N, E)
    key = torch_ref.randn(S, N, E)
    value = torch_ref.randn(S, N, E)
    in_proj_weight = torch_ref.randn(3 * E, E)
    in_proj_bias = torch_ref.randn(3 * E)
    out_proj_weight = torch_ref.randn(E, E)
    out_proj_bias = torch_ref.randn(E)

    out_ref, w_ref = F_ref.multi_head_attention_forward(
        query,
        key,
        value,
        embed_dim_to_check=E,
        num_heads=H,
        in_proj_weight=in_proj_weight,
        in_proj_bias=in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=out_proj_weight,
        out_proj_bias=out_proj_bias,
        training=False,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )
    assert w_ref is None

    q = torch.tensor(query.numpy())
    k = torch.tensor(key.numpy())
    v = torch.tensor(value.numpy())
    w = torch.tensor(in_proj_weight.numpy())
    b = torch.tensor(in_proj_bias.numpy())
    ow = torch.tensor(out_proj_weight.numpy())
    ob = torch.tensor(out_proj_bias.numpy())

    out, attn_w = F.multi_head_attention_forward(
        q,
        k,
        v,
        embed_dim_to_check=E,
        num_heads=H,
        in_proj_weight=w,
        in_proj_bias=b,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=ow,
        out_proj_bias=ob,
        training=False,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )

    assert attn_w is None
    np.testing.assert_allclose(out.numpy(), out_ref.detach().numpy(), rtol=1e-5, atol=1e-5)
