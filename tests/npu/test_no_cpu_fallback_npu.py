import pytest

import candle as torch


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
