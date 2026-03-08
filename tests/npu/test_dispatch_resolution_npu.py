import candle as torch


def test_dispatch_prefers_npu_over_cpu():
    a = torch.ones((2,), device="npu")
    b = torch.ones((2,), device="npu")
    c = torch.add(a, b)
    assert c.device.type == "npu"
