"""Backward tests for composite ops used in transformer training."""
import numpy as np
import candle as torch
import candle.nn.functional as F


def test_cross_entropy_backward_basic():
    """cross_entropy = log_softmax + nll_loss, both composites."""
    torch.manual_seed(42)
    x = torch.randn(4, 10, device='cpu')
    x.requires_grad = True
    target = torch.tensor([3, 7, 1, 0], device='cpu')
    loss = F.cross_entropy(x, target)
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    # Gradient should be finite
    assert np.all(np.isfinite(x.grad.numpy()))


def test_cross_entropy_backward_vs_torch():
    """Numerical parity with PyTorch."""
    import torch as real_torch
    np.random.seed(42)
    data = np.random.randn(8, 5).astype(np.float32)
    tgt = np.array([0, 2, 4, 1, 3, 0, 2, 1], dtype=np.int64)

    # candle
    x_c = torch.tensor(data, device='cpu')
    x_c.requires_grad = True
    loss_c = F.cross_entropy(x_c, torch.tensor(tgt, device='cpu'))
    loss_c.backward()

    # torch
    x_t = real_torch.tensor(data, requires_grad=True)
    loss_t = real_torch.nn.functional.cross_entropy(x_t, real_torch.tensor(tgt))
    loss_t.backward()

    np.testing.assert_allclose(loss_c.detach().numpy(), loss_t.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(x_c.grad.numpy(), x_t.grad.numpy(), atol=1e-5)


def test_nll_loss_backward_basic():
    """nll_loss uses gather, neg, mul, div, sum — all have backward."""
    torch.manual_seed(42)
    x = torch.randn(4, 10, device='cpu')
    x.requires_grad = True
    target = torch.tensor([3, 7, 1, 0], device='cpu')
    loss = F.nll_loss(F.log_softmax(x, dim=1), target)
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_nll_loss_backward_reduction_sum():
    """nll_loss with reduction='sum'."""
    torch.manual_seed(42)
    x = torch.randn(4, 10, device='cpu')
    x.requires_grad = True
    target = torch.tensor([3, 7, 1, 0], device='cpu')
    loss = F.nll_loss(F.log_softmax(x, dim=1), target, reduction='sum')
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_nll_loss_backward_reduction_none():
    """nll_loss with reduction='none' returns per-sample losses."""
    torch.manual_seed(42)
    x = torch.randn(4, 10, device='cpu')
    x.requires_grad = True
    target = torch.tensor([3, 7, 1, 0], device='cpu')
    losses = F.nll_loss(F.log_softmax(x, dim=1), target, reduction='none')
    losses.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_nll_loss_backward_ignore_index():
    """nll_loss with ignore_index should zero out ignored samples."""
    import torch as real_torch
    np.random.seed(42)
    data = np.random.randn(4, 5).astype(np.float32)
    tgt = np.array([0, -100, 2, 1], dtype=np.int64)  # sample 1 ignored

    x_c = torch.tensor(data, device='cpu')
    x_c.requires_grad = True
    loss_c = F.cross_entropy(x_c, torch.tensor(tgt, device='cpu'), ignore_index=-100)
    loss_c.backward()

    x_t = real_torch.tensor(data, requires_grad=True)
    loss_t = real_torch.nn.functional.cross_entropy(x_t, real_torch.tensor(tgt), ignore_index=-100)
    loss_t.backward()

    np.testing.assert_allclose(x_c.grad.numpy(), x_t.grad.numpy(), atol=1e-5)


# ---- SDPA backward ----


def test_sdpa_backward_basic():
    """SDPA = matmul + mul + softmax + matmul, all have backward."""
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 8, 16
    q = torch.randn(B, H, L, D, device='cpu')
    k = torch.randn(B, H, L, D, device='cpu')
    v = torch.randn(B, H, L, D, device='cpu')
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    out = F.scaled_dot_product_attention(q, k, v)
    out.sum().backward()

    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert v.grad is not None and v.grad.shape == v.shape
    assert np.all(np.isfinite(q.grad.numpy()))
    assert np.all(np.isfinite(k.grad.numpy()))
    assert np.all(np.isfinite(v.grad.numpy()))


def test_sdpa_backward_vs_torch():
    """Numerical parity with PyTorch SDPA backward."""
    import torch as real_torch
    np.random.seed(42)
    B, H, L, D = 2, 2, 4, 8
    q_np = np.random.randn(B, H, L, D).astype(np.float32)
    k_np = np.random.randn(B, H, L, D).astype(np.float32)
    v_np = np.random.randn(B, H, L, D).astype(np.float32)

    q_c = torch.tensor(q_np, device='cpu'); q_c.requires_grad = True
    k_c = torch.tensor(k_np, device='cpu'); k_c.requires_grad = True
    v_c = torch.tensor(v_np, device='cpu'); v_c.requires_grad = True
    out_c = F.scaled_dot_product_attention(q_c, k_c, v_c)
    out_c.sum().backward()

    with real_torch.nn.attention.sdpa_kernel(real_torch.nn.attention.SDPBackend.MATH):
        q_t = real_torch.tensor(q_np, requires_grad=True)
        k_t = real_torch.tensor(k_np, requires_grad=True)
        v_t = real_torch.tensor(v_np, requires_grad=True)
        out_t = real_torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
        out_t.sum().backward()

    np.testing.assert_allclose(out_c.detach().numpy(), out_t.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(q_c.grad.numpy(), q_t.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(k_c.grad.numpy(), k_t.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(v_c.grad.numpy(), v_t.grad.numpy(), atol=1e-4)


def test_sdpa_backward_with_causal_mask():
    """SDPA with is_causal=True should still propagate gradients."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 6, 8
    q = torch.randn(B, H, L, D, device='cpu'); q.requires_grad = True
    k = torch.randn(B, H, L, D, device='cpu'); k.requires_grad = True
    v = torch.randn(B, H, L, D, device='cpu'); v.requires_grad = True

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out.sum().backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_sdpa_backward_with_attn_mask():
    """SDPA with float attn_mask should propagate gradients."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 6, 8
    q = torch.randn(B, H, L, D, device='cpu'); q.requires_grad = True
    k = torch.randn(B, H, L, D, device='cpu'); k.requires_grad = True
    v = torch.randn(B, H, L, D, device='cpu'); v.requires_grad = True
    mask = torch.randn(L, L, device='cpu')

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out.sum().backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


# ---- Dropout backward ----


def test_dropout_backward():
    """dropout has a dedicated backward in autograd.py."""
    torch.manual_seed(42)
    x = torch.randn(4, 16, device='cpu')
    x.requires_grad = True
    out = F.dropout(x, p=0.5, training=True)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    unique_vals = np.unique(np.round(x.grad.numpy(), 5))
    for v in unique_vals:
        assert v == 0.0 or abs(v - 2.0) < 1e-4, f"Unexpected grad value: {v}"


def test_dropout_backward_p_zero():
    """dropout with p=0 is identity, gradient should be all ones."""
    x = torch.randn(4, 8, device='cpu')
    x.requires_grad = True
    out = F.dropout(x, p=0.0, training=True)
    out.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.ones_like(x.grad.numpy()))
