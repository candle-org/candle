import pytest


@pytest.fixture
def npu_device():
    import candle as torch
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    return torch.device("npu:0")
