import numpy as np
import candle as torch


def test_cpu_to_mps():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = x.to("mps")
    assert y.device.type == "mps"
    np.testing.assert_allclose(y.cpu().numpy(), x.numpy())


def test_to_mps_method():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = x.mps()
    assert y.device.type == "mps"
    np.testing.assert_allclose(y.cpu().numpy(), x.numpy())


def test_mps_to_cpu():
    x = torch.tensor([1.0, 2.0, 3.0], device="mps")
    y = x.to("cpu")
    assert y.device.type == "cpu"
    np.testing.assert_allclose(y.numpy(), np.array([1.0, 2.0, 3.0]))


def test_cpu_method():
    x = torch.tensor([1.0, 2.0, 3.0], device="mps")
    y = x.cpu()
    assert y.device.type == "cpu"
    np.testing.assert_allclose(y.numpy(), np.array([1.0, 2.0, 3.0]))


def test_roundtrip_data_integrity():
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x = torch.tensor(x_np)
    y = x.to("mps")
    z = y.to("cpu")
    np.testing.assert_allclose(z.numpy(), x_np)


def test_roundtrip_int_data():
    x_np = np.array([10, 20, 30], dtype=np.int64)
    x = torch.tensor(x_np)
    y = x.to("mps")
    z = y.to("cpu")
    np.testing.assert_array_equal(z.numpy(), x_np)


def test_roundtrip_multidim():
    x_np = np.random.randn(3, 4, 5).astype(np.float32)
    x = torch.tensor(x_np)
    y = x.to("mps").to("cpu")
    np.testing.assert_allclose(y.numpy(), x_np, rtol=1e-5)
