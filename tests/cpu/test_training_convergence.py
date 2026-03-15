"""Multi-step training convergence tests.

Verify that loss actually decreases over multiple optimizer steps on toy
problems — something a single forward+backward step cannot prove.
"""
import numpy as np
import candle as torch
import candle.nn as nn
import candle.nn.functional as F


def test_mlp_xor_convergence():
    """A 2-layer MLP should learn XOR in < 500 steps."""
    torch.manual_seed(42)
    # XOR dataset
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for _ in range(500):
        optimizer.zero_grad()
        out = model(X)
        loss = F.mse_loss(out, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.1, \
        f"Loss should decrease significantly: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert losses[-1] < 0.05, f"Final loss {losses[-1]:.4f} should be < 0.05"


def test_linear_regression_convergence():
    """Linear regression on y = 2x + 1 should converge with SGD."""
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    Y = 2.0 * X + 1.0

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    initial_loss = None
    final_loss = None
    for step in range(200):
        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, Y)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.01, \
        f"Loss should drop 100x: {initial_loss:.4f} -> {final_loss:.4f}"
    # Check learned parameters
    w = model.weight.detach().numpy().item()
    b = model.bias.detach().numpy().item()
    assert abs(w - 2.0) < 0.1, f"Weight should be ~2.0, got {w:.3f}"
    assert abs(b - 1.0) < 0.1, f"Bias should be ~1.0, got {b:.3f}"


def test_transformer_encoder_convergence():
    """A small transformer encoder should reduce loss on a toy sequence task."""
    torch.manual_seed(42)
    batch, seq_len, d_model, nhead = 4, 8, 16, 2

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=32, batch_first=True
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    head = nn.Linear(d_model, d_model)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        src = torch.randn(batch, seq_len, d_model)
        target = torch.randn(batch, seq_len, d_model)
        out = head(encoder(src))
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # With random targets, loss should still decrease (model fits noise)
    avg_first_5 = np.mean(losses[:5])
    avg_last_5 = np.mean(losses[-5:])
    assert avg_last_5 < avg_first_5, \
        f"Loss should decrease: first5={avg_first_5:.4f}, last5={avg_last_5:.4f}"


def test_gradient_accumulation():
    """Gradient accumulation should produce same result as large batch."""
    torch.manual_seed(42)
    X = torch.randn(8, 4)
    Y = torch.randn(8, 2)

    # Single large batch
    model_a = nn.Linear(4, 2)
    model_a_w = model_a.weight.detach().numpy().copy()
    model_a_b = model_a.bias.detach().numpy().copy()
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    optimizer_a.zero_grad()
    loss_a = F.mse_loss(model_a(X), Y)
    loss_a.backward()
    optimizer_a.step()

    # Gradient accumulation over 4 micro-batches of size 2
    model_b = nn.Linear(4, 2)
    # Copy same initial weights
    model_b.weight = nn.Parameter(torch.tensor(model_a_w))
    model_b.bias = nn.Parameter(torch.tensor(model_a_b))
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.1)

    accum_steps = 4
    optimizer_b.zero_grad()
    for i in range(accum_steps):
        micro_x = X[i * 2:(i + 1) * 2]
        micro_y = Y[i * 2:(i + 1) * 2]
        micro_loss = F.mse_loss(model_b(micro_x), micro_y)
        (micro_loss / accum_steps).backward()
    optimizer_b.step()

    # Weights should be very close
    np.testing.assert_allclose(
        model_a.weight.detach().numpy(),
        model_b.weight.detach().numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        model_a.bias.detach().numpy(),
        model_b.bias.detach().numpy(),
        atol=1e-5,
    )


def test_adamw_weight_decay():
    """AdamW decoupled weight decay should shrink weights toward zero."""
    torch.manual_seed(42)
    model = nn.Linear(4, 2, bias=False)
    initial_norm = np.linalg.norm(model.weight.detach().numpy())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)

    for _ in range(100):
        optimizer.zero_grad()
        x = torch.randn(8, 4)
        # Zero target — loss drives outputs to zero, decay drives weights to zero
        loss = (model(x) ** 2).mean()
        loss.backward()
        optimizer.step()

    final_norm = np.linalg.norm(model.weight.detach().numpy())
    assert final_norm < initial_norm * 0.5, \
        f"Weight decay should shrink weights: {initial_norm:.4f} -> {final_norm:.4f}"


def test_grad_clipping_during_training():
    """Gradient clipping should cap gradient norms."""
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    x = torch.randn(2, 4) * 100  # Large input to create large gradients
    y = torch.randn(2, 2)

    optimizer.zero_grad()
    loss = F.mse_loss(model(x), y)
    loss.backward()

    # Clip gradients
    max_norm = 1.0
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    assert total_norm > max_norm, "Pre-clip norm should exceed max_norm"

    # After clipping, each parameter's grad norm should be scaled down
    clipped_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            clipped_norm += (p.grad.detach().numpy() ** 2).sum()
    clipped_norm = np.sqrt(clipped_norm)
    assert abs(clipped_norm - max_norm) < 0.01, \
        f"Clipped norm should be ~{max_norm}, got {clipped_norm:.4f}"
