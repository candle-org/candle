"""Tests for nn.Module weight initialization parity with PyTorch."""
import numpy as np
import candle as torch


def test_embedding_init_not_zeros():
    """Embedding weights should be initialized with normal_(0, 1), not zeros."""
    emb = torch.nn.Embedding(100, 32)
    w = emb.weight.detach().numpy()
    assert not np.allclose(w, 0.0), "Embedding weights should not be all zeros"
    assert np.abs(w).sum() > 0, "Embedding weights should have non-zero values"


def test_embedding_init_distribution():
    """Embedding weights should follow roughly N(0, 1)."""
    torch.manual_seed(42)
    emb = torch.nn.Embedding(10000, 64)
    w = emb.weight.detach().numpy()
    assert abs(w.mean()) < 0.05, f"Mean should be near 0, got {w.mean()}"
    assert abs(w.std() - 1.0) < 0.1, f"Std should be near 1.0, got {w.std()}"


def test_embedding_padding_idx_zero():
    """Embedding with padding_idx should have zeros at that index."""
    torch.manual_seed(42)
    emb = torch.nn.Embedding(100, 32, padding_idx=0)
    w = emb.weight.detach().numpy()
    np.testing.assert_allclose(w[0], 0.0)
    assert not np.allclose(w[1], 0.0), "Non-padding rows should be non-zero"


def test_embedding_from_pretrained_skips_init():
    """from_pretrained should use provided weights, not re-initialize."""
    data = torch.randn(10, 8)
    emb = torch.nn.Embedding.from_pretrained(data)
    np.testing.assert_allclose(emb.weight.detach().numpy(), data.detach().numpy())


def test_embedding_bag_init_not_zeros():
    """EmbeddingBag weights should be initialized with normal_(0, 1)."""
    emb = torch.nn.EmbeddingBag(100, 32)
    w = emb.weight.detach().numpy()
    assert not np.allclose(w, 0.0), "EmbeddingBag weights should not be all zeros"


def test_embedding_init_vs_torch():
    """Verify Embedding init distribution matches PyTorch's."""
    import torch as real_torch
    torch.manual_seed(42)
    real_torch.manual_seed(42)
    emb_c = torch.nn.Embedding(1000, 64)
    emb_t = real_torch.nn.Embedding(1000, 64)
    w_c = emb_c.weight.detach().numpy()
    w_t = emb_t.weight.detach().numpy()
    # Distribution should be similar (both N(0,1))
    assert abs(w_c.mean() - w_t.mean()) < 0.15
    assert abs(w_c.std() - w_t.std()) < 0.15


def test_lstm_init_not_zeros():
    """LSTM weights should be initialized with uniform_(-stdv, stdv)."""
    lstm = torch.nn.LSTM(input_size=32, hidden_size=64, num_layers=1)
    w_ih = lstm.weight_ih_l0.detach().numpy()
    w_hh = lstm.weight_hh_l0.detach().numpy()
    assert not np.allclose(w_ih, 0.0), "LSTM weight_ih should not be all zeros"
    assert not np.allclose(w_hh, 0.0), "LSTM weight_hh should not be all zeros"


def test_lstm_init_distribution():
    """LSTM weights should follow uniform(-1/sqrt(H), 1/sqrt(H))."""
    torch.manual_seed(42)
    hidden_size = 64
    lstm = torch.nn.LSTM(input_size=32, hidden_size=hidden_size)
    stdv = 1.0 / np.sqrt(hidden_size)
    w = lstm.weight_ih_l0.detach().numpy()
    assert w.min() >= -stdv - 0.01, f"Min {w.min()} should be >= {-stdv}"
    assert w.max() <= stdv + 0.01, f"Max {w.max()} should be <= {stdv}"


def test_gru_init_not_zeros():
    """GRU weights should be initialized, not zeros."""
    gru = torch.nn.GRU(input_size=16, hidden_size=32)
    w = gru.weight_ih_l0.detach().numpy()
    assert not np.allclose(w, 0.0), "GRU weights should not be all zeros"


def test_rnn_init_not_zeros():
    """RNN weights should be initialized, not zeros."""
    rnn = torch.nn.RNN(input_size=16, hidden_size=32)
    w = rnn.weight_ih_l0.detach().numpy()
    assert not np.allclose(w, 0.0), "RNN weights should not be all zeros"


def test_rnn_init_vs_torch():
    """RNNBase init distribution should match PyTorch's uniform range."""
    import torch as real_torch
    hidden_size = 64
    torch.manual_seed(42)
    real_torch.manual_seed(42)
    lstm_c = torch.nn.LSTM(32, hidden_size)
    lstm_t = real_torch.nn.LSTM(32, hidden_size)
    stdv = 1.0 / np.sqrt(hidden_size)
    w_c = lstm_c.weight_ih_l0.detach().numpy()
    w_t = lstm_t.weight_ih_l0.detach().numpy()
    # Both should be in [-stdv, stdv]
    assert w_c.min() >= -stdv - 0.01
    assert w_c.max() <= stdv + 0.01
    assert w_t.min() >= -stdv - 0.01
    assert w_t.max() <= stdv + 0.01


def test_rnn_cell_init_not_zeros():
    """RNNCell/LSTMCell/GRUCell should also be properly initialized."""
    rnn_cell = torch.nn.RNNCell(16, 32)
    lstm_cell = torch.nn.LSTMCell(16, 32)
    gru_cell = torch.nn.GRUCell(16, 32)
    assert not np.allclose(rnn_cell.weight_ih.detach().numpy(), 0.0)
    assert not np.allclose(lstm_cell.weight_ih.detach().numpy(), 0.0)
    assert not np.allclose(gru_cell.weight_ih.detach().numpy(), 0.0)


def test_randint_overload():
    """randint(low, (size,)) should work like randint(low, size=(size,))."""
    torch.manual_seed(42)
    a = torch.randint(10, (3, 4))
    assert a.shape == (3, 4)
    assert a.dtype == torch.int64
    assert (a.numpy() >= 0).all()
    assert (a.numpy() < 10).all()


def test_randint_overload_with_high():
    """randint(low, high, size) should still work."""
    a = torch.randint(5, 10, (2, 3))
    assert a.shape == (2, 3)
    assert (a.numpy() >= 5).all()
    assert (a.numpy() < 10).all()
