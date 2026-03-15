"""Tests for model save/load round-trip correctness.

Verify that saving a candle model and loading it back produces
identical forward outputs and preserves all state.
"""
import os
import tempfile
import numpy as np
import candle as torch
import candle.nn as nn


def _assert_state_dicts_equal(sd1, sd2, msg=""):
    """Assert two state dicts have identical keys and values."""
    assert set(sd1.keys()) == set(sd2.keys()), \
        f"{msg} Key mismatch: {set(sd1.keys())} vs {set(sd2.keys())}"
    for key in sd1:
        np.testing.assert_allclose(
            sd1[key].numpy(), sd2[key].numpy(), atol=1e-6,
            err_msg=f"{msg} Value mismatch for key '{key}'"
        )


def test_linear_save_load_roundtrip():
    """Linear model: save -> load -> forward should be identical."""
    torch.manual_seed(42)
    model = nn.Linear(8, 4)
    x = torch.randn(2, 8)
    out_before = model(x).detach().numpy().copy()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(model.state_dict(), path)
        model2 = nn.Linear(8, 4)
        model2.load_state_dict(torch.load(path))
        out_after = model2(x).detach().numpy()
        np.testing.assert_allclose(out_before, out_after, atol=1e-6)
    finally:
        os.unlink(path)


def test_sequential_save_load_roundtrip():
    """Sequential model with multiple layers: round-trip preserves output."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    x = torch.randn(2, 8)
    out_before = model(x).detach().numpy().copy()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(model.state_dict(), path)
        model2 = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        model2.load_state_dict(torch.load(path))
        out_after = model2(x).detach().numpy()
        np.testing.assert_allclose(out_before, out_after, atol=1e-6)
    finally:
        os.unlink(path)


def test_layernorm_save_load_roundtrip():
    """LayerNorm state (weight + bias) should survive round-trip."""
    torch.manual_seed(42)
    ln = nn.LayerNorm(8)
    x = torch.randn(2, 8)
    out_before = ln(x).detach().numpy().copy()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(ln.state_dict(), path)
        ln2 = nn.LayerNorm(8)
        ln2.load_state_dict(torch.load(path))
        out_after = ln2(x).detach().numpy()
        np.testing.assert_allclose(out_before, out_after, atol=1e-6)
    finally:
        os.unlink(path)


def test_embedding_save_load_roundtrip():
    """Embedding weights should survive round-trip."""
    torch.manual_seed(42)
    emb = nn.Embedding(100, 32)
    idx = torch.tensor([0, 5, 99])
    out_before = emb(idx).detach().numpy().copy()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(emb.state_dict(), path)
        emb2 = nn.Embedding(100, 32)
        emb2.load_state_dict(torch.load(path))
        out_after = emb2(idx).detach().numpy()
        np.testing.assert_allclose(out_before, out_after, atol=1e-6)
    finally:
        os.unlink(path)


def test_mha_save_load_roundtrip():
    """MultiheadAttention state should survive round-trip."""
    torch.manual_seed(42)
    mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
    x = torch.randn(2, 4, 8)
    out_before, _ = mha(x, x, x)
    out_before = out_before.detach().numpy().copy()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(mha.state_dict(), path)
        mha2 = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        mha2.load_state_dict(torch.load(path))
        out_after, _ = mha2(x, x, x)
        out_after = out_after.detach().numpy()
        np.testing.assert_allclose(out_before, out_after, atol=1e-5)
    finally:
        os.unlink(path)


def test_state_dict_keys_match():
    """state_dict keys should be identical before save and after load."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.LayerNorm(8),
        nn.Linear(8, 2),
    )
    sd_original = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(sd_original, path)
        sd_loaded = torch.load(path)
        assert set(sd_original.keys()) == set(sd_loaded.keys())
    finally:
        os.unlink(path)


def test_optimizer_state_save_load():
    """Optimizer state should survive save/load round-trip."""
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Run a few steps to populate optimizer state
    for _ in range(5):
        optimizer.zero_grad()
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        optimizer.step()

    opt_state = optimizer.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(opt_state, path)
        loaded_state = torch.load(path)
        assert loaded_state["param_groups"][0]["lr"] == 0.01
        assert len(loaded_state["state"]) == len(opt_state["state"])
    finally:
        os.unlink(path)


def test_full_training_checkpoint_resume():
    """Save model + optimizer mid-training, load, and resume — loss continues dropping."""
    torch.manual_seed(42)
    X = torch.randn(32, 4)
    Y = torch.randn(32, 2)

    # Phase 1: train for 50 steps
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(50):
        optimizer.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        optimizer.step()
    loss_at_checkpoint = loss.item()

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    try:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, ckpt_path)

        # Phase 2: load and resume training
        model2 = nn.Linear(4, 2)
        ckpt = torch.load(ckpt_path)
        model2.load_state_dict(ckpt["model"])
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        optimizer2.load_state_dict(ckpt["optimizer"])

        for _ in range(50):
            optimizer2.zero_grad()
            loss = ((model2(X) - Y) ** 2).mean()
            loss.backward()
            optimizer2.step()
        loss_after_resume = loss.item()

        assert loss_after_resume < loss_at_checkpoint, \
            f"Loss should continue decreasing after resume: {loss_at_checkpoint:.4f} -> {loss_after_resume:.4f}"
    finally:
        os.unlink(ckpt_path)
