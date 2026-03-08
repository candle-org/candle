import numpy as np
import candle as torch


def test_zeros_mps():
    x = torch.zeros((2, 3), device="mps")
    assert x.shape == (2, 3)
    assert x.device.type == "mps"
    np.testing.assert_array_equal(x.cpu().numpy(), np.zeros((2, 3)))


def test_ones_mps():
    x = torch.ones((2, 3), device="mps")
    assert x.shape == (2, 3)
    assert x.device.type == "mps"
    np.testing.assert_array_equal(x.cpu().numpy(), np.ones((2, 3)))


def test_empty_mps():
    x = torch.empty((2, 3), device="mps")
    assert x.shape == (2, 3)
    assert x.device.type == "mps"


def test_full_mps():
    x = torch.full((2, 3), 1.5, device="mps")
    assert x.shape == (2, 3)
    assert x.device.type == "mps"
    expected = np.full((2, 3), 1.5)
    np.testing.assert_allclose(x.cpu().numpy(), expected)


def test_arange_mps():
    x = torch.arange(0, 5, device="mps")
    assert x.shape == (5,)
    assert x.device.type == "mps"
    np.testing.assert_array_equal(x.cpu().numpy(), np.arange(0, 5))


def test_linspace_mps():
    x = torch.linspace(0.0, 1.0, 5, device="mps")
    assert x.shape == (5,)
    assert x.device.type == "mps"
    np.testing.assert_allclose(x.cpu().numpy(), np.linspace(0.0, 1.0, 5))


def test_rand_mps():
    x = torch.rand((2, 3), device="mps")
    assert x.shape == (2, 3)
    assert x.device.type == "mps"
    vals = x.cpu().numpy()
    assert np.all(vals >= 0.0) and np.all(vals < 1.0)


def test_randn_mps():
    x = torch.randn((2, 3), device="mps")
    assert x.shape == (2, 3)
    assert x.device.type == "mps"


def test_eye_mps():
    x = torch.eye(3, 2, device="mps")
    assert x.shape == (3, 2)
    assert x.device.type == "mps"
    expected = np.eye(3, 2)
    np.testing.assert_array_equal(x.cpu().numpy(), expected)


def test_zeros_dtype_mps():
    x = torch.zeros((2, 3), dtype=torch.float32, device="mps")
    assert x.dtype == torch.float32
    assert x.device.type == "mps"


def test_ones_int_mps():
    x = torch.ones((2, 3), dtype=torch.int64, device="mps")
    assert x.dtype == torch.int64
    assert x.device.type == "mps"
    np.testing.assert_array_equal(x.cpu().numpy(), np.ones((2, 3), dtype=np.int64))
