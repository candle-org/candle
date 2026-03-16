import candle as torch
import pytest
import torch as real_torch


def test_view_as_real_complex_shape_dtype():
    x = torch.randn(2, 3, dtype=torch.complex64)
    y = torch.view_as_real(x)
    assert y.shape == (2, 3, 2)
    assert "float" in str(y.dtype)


def test_view_as_complex_roundtrip():
    x = torch.randn(2, 3, dtype=torch.complex64)
    y = torch.view_as_real(x)
    z = torch.view_as_complex(y)
    assert z.shape == x.shape
    assert str(z.dtype) == str(x.dtype)


def test_view_as_complex_invalid_last_dim():
    x = torch.randn(2, 3, 4)
    with pytest.raises(RuntimeError):
        torch.view_as_complex(x)


def test_view_as_real_stride_matches_torch_contract():
    x = torch.randn(2, 3, dtype=torch.complex64)
    y = torch.view_as_real(x)
    x_ref = real_torch.randn(2, 3, dtype=real_torch.complex64)
    y_ref = real_torch.view_as_real(x_ref)
    assert tuple(y.stride) == tuple(y_ref.stride())


def test_view_as_real_matches_torch_contract():
    x = torch.randn(2, 3, dtype=torch.complex64)
    y = torch.view_as_real(x)
    x_ref = real_torch.randn(2, 3, dtype=real_torch.complex64)
    y_ref = real_torch.view_as_real(x_ref)
    assert tuple(y.shape) == tuple(y_ref.shape)


def test_view_as_complex_matches_torch_contract():
    x = torch.view_as_real(torch.randn(2, 3, dtype=torch.complex64))
    z = torch.view_as_complex(x)
    x_ref = real_torch.view_as_real(real_torch.randn(2, 3, dtype=real_torch.complex64))
    z_ref = real_torch.view_as_complex(x_ref)
    assert tuple(z.shape) == tuple(z_ref.shape)
