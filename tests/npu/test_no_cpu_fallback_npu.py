import pytest

import candle as torch
from candle._tensor import Tensor


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_mul_scalar_stays_on_npu():
    torch.manual_seed(1234)
    x = torch.randn(128, device="npu")
    scale = torch.tensor(3.0, device="npu")
    y = torch.mul(x, scale)

    assert y.device.type == "npu"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_golden_ops_stay_on_npu():
    x = torch.randn((32, 8), device="npu")
    w = torch.randn((8, 4), device="npu")
    out = torch.matmul(x, w)
    diff = torch.sub(out, torch.zeros_like(out))
    loss = torch.mean(torch.pow(diff, 2.0))

    assert out.device.type == "npu"
    assert diff.device.type == "npu"
    assert loss.device.type == "npu"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_tril_indices_respects_npu_device():
    out = torch.tril_indices(3, 4, device="npu")
    assert out.device.type == "npu"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_triu_indices_respects_npu_device():
    out = torch.triu_indices(3, 4, device="npu")
    assert out.device.type == "npu"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_cartesian_prod_no_cpu_roundtrip(monkeypatch):
    original_to = Tensor.to

    def guard_to(self, *args, **kwargs):
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]
        if getattr(self, "device", None) is not None and self.device.type == "npu":
            if device == "cpu" or getattr(device, "type", None) == "cpu":
                raise AssertionError("cartesian_prod should not move NPU tensors to CPU")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "to", guard_to)
    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")

    out = torch.cartesian_prod(a, b)
    assert out.device.type == "npu"
    assert out.shape == (4, 2)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_block_diag_no_cpu_roundtrip(monkeypatch):
    original_to = Tensor.to

    def guard_to(self, *args, **kwargs):
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]
        if getattr(self, "device", None) is not None and self.device.type == "npu":
            if device == "cpu" or getattr(device, "type", None) == "cpu":
                raise AssertionError("block_diag should not move NPU tensors to CPU")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "to", guard_to)
    a = torch.tensor([[1.0, 2.0]], device="npu")
    b = torch.tensor([[3.0], [4.0]], device="npu")

    out = torch.block_diag(a, b)
    assert out.device.type == "npu"
    assert out.shape == (3, 3)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_repeat_interleave_tensor_repeats_stays_on_device():
    x = torch.tensor([[1, 2], [3, 4]], device="npu", dtype=torch.float32)
    repeats = torch.tensor([1, 2], device="npu", dtype=torch.int64)
    out = torch.repeat_interleave(x, repeats=repeats, dim=1)
    assert out.device.type == "npu"
    assert out.shape == (2, 3)
    import numpy as np
    expected = np.array([[1, 2, 2], [3, 4, 4]], dtype=np.float32)
    np.testing.assert_allclose(out.to("cpu").numpy(), expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_baddbmm_tensor_alpha_beta_stays_on_device():
    B, N, M, P = 2, 3, 5, 4
    self_tensor = torch.ones(B, N, P, dtype=torch.float32, device="npu")
    batch1 = torch.ones(B, N, M, dtype=torch.float32, device="npu")
    batch2 = torch.ones(B, M, P, dtype=torch.float32, device="npu")
    alpha = torch.tensor(0.5, device="npu")
    beta = torch.tensor(2.0, device="npu")
    out = torch.baddbmm(self_tensor, batch1, batch2, beta=beta, alpha=alpha)
    assert out.device.type == "npu"
    # 2.0 * 1 + 0.5 * (ones @ ones) = 2 + 0.5*5 = 4.5
    import numpy as np
    np.testing.assert_allclose(out.to("cpu").numpy(), np.full((B, N, P), 4.5))
