import candle as torch
import torch as real_torch


def test_var_mean_returns_tuple():
    x = torch.randn(2, 3)
    v, m = torch.var_mean(x)
    assert v.shape == m.shape


def test_var_mean_dim_keepdim():
    x = torch.randn(2, 3)
    v, m = torch.var_mean(x, dim=1, keepdim=True)
    assert v.shape == (2, 1)
    assert m.shape == (2, 1)


def test_var_mean_matches_torch_contract():
    x = torch.randn(2, 3)
    v, m = torch.var_mean(x, dim=1, keepdim=True)
    x_ref = real_torch.randn(2, 3)
    v_ref, m_ref = real_torch.var_mean(x_ref, dim=1, keepdim=True)
    assert tuple(v.shape) == tuple(v_ref.shape)
    assert tuple(m.shape) == tuple(m_ref.shape)
