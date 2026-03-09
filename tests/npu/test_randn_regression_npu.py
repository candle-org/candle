import pytest

import candle as torch


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_randn_npu_basic_shape_is_available():
    x = torch.randn((32, 8), device="npu")
    assert x.device.type == "npu"
    assert x.shape == (32, 8)

    y = torch.randn((32, 8), device="npu")
    assert y.device.type == "npu"
    assert y.shape == (32, 8)
