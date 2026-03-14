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


# ---- MultiheadAttention backward ----


def test_mha_self_attention_backward():
    """MHA self-attention: all projection weights should get gradients."""
    torch.manual_seed(42)
    embed_dim, num_heads, seq_len, batch = 8, 2, 4, 2
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    x = torch.randn(batch, seq_len, embed_dim, device='cpu')
    x.requires_grad = True
    out, _ = mha(x, x, x)
    out.sum().backward()

    assert x.grad is not None
    assert mha.in_proj_weight.grad is not None
    assert mha.in_proj_bias.grad is not None
    assert mha.out_proj.weight.grad is not None
    assert mha.out_proj.bias.grad is not None
    # Weight grads should be non-zero
    assert np.abs(mha.in_proj_weight.grad.numpy()).sum() > 0
    assert np.abs(mha.out_proj.weight.grad.numpy()).sum() > 0


def test_mha_cross_attention_backward():
    """MHA cross-attention: query and key/value from different sources."""
    torch.manual_seed(42)
    embed_dim, num_heads = 8, 2
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    q = torch.randn(2, 4, embed_dim, device='cpu'); q.requires_grad = True
    kv = torch.randn(2, 6, embed_dim, device='cpu'); kv.requires_grad = True
    out, _ = mha(q, kv, kv)
    out.sum().backward()

    assert q.grad is not None
    assert kv.grad is not None
    assert mha.in_proj_weight.grad is not None


def test_mha_need_weights_true():
    """MHA with need_weights=True returns attention weights and still backprops."""
    torch.manual_seed(42)
    embed_dim, num_heads = 8, 2
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    x = torch.randn(2, 4, embed_dim, device='cpu')
    x.requires_grad = True
    out, attn_weights = mha(x, x, x, need_weights=True)
    assert attn_weights is not None
    assert attn_weights.shape == (2, 4, 4)  # (N, L, S) when averaged over heads
    out.sum().backward()
    assert x.grad is not None


# ---- End-to-end transformer training step ----


def test_transformer_training_step():
    """Full forward + backward + optimizer step on a small transformer."""
    torch.manual_seed(42)
    batch, seq_len, embed_dim, num_heads = 2, 4, 8, 2

    # Build a minimal transformer encoder layer
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    ln1 = torch.nn.LayerNorm(embed_dim)
    linear1 = torch.nn.Linear(embed_dim, 16)
    linear2 = torch.nn.Linear(16, embed_dim)
    ln2 = torch.nn.LayerNorm(embed_dim)

    params = (
        list(mha.parameters()) +
        list(ln1.parameters()) +
        list(ln2.parameters()) +
        list(linear1.parameters()) +
        list(linear2.parameters())
    )
    optimizer = torch.optim.SGD(params, lr=0.01)

    # Save initial weights
    init_weights = {id(p): p.detach().numpy().copy() for p in params}

    # Forward pass
    x = torch.randn(batch, seq_len, embed_dim, device='cpu')
    # Self-attention + residual
    attn_out, _ = mha(x, x, x)
    x2 = ln1(x + attn_out)
    # FFN + residual
    ffn_out = linear2(F.relu(linear1(x2)))
    x3 = ln2(x2 + ffn_out)

    # Loss
    target = torch.randn(batch, seq_len, embed_dim, device='cpu')
    loss = ((x3 - target) ** 2).mean()

    # Backward
    loss.backward()

    # Check gradients exist for all parameters
    for p in params:
        assert p.grad is not None, f"Parameter with shape {p.shape} has no gradient"

    # Optimizer step
    optimizer.step()

    # Verify weights changed
    changed_count = 0
    for p in params:
        old = init_weights[id(p)]
        new = p.detach().numpy()
        if not np.allclose(old, new, atol=1e-10):
            changed_count += 1

    assert changed_count == len(params), (
        f"Only {changed_count}/{len(params)} parameters changed after optimizer step"
    )
