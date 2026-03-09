import pytest

import candle as torch


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_mul_scalar_after_randn_on_npu():
    torch.manual_seed(1234)

    x = torch.randn(128, device="npu")
    scale = torch.tensor(3.0, device="npu")

    y = torch.mul(x, scale)
    assert y.device.type == "npu"
    assert y.shape == x.shape
