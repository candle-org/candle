import candle as torch


def test_to_device_roundtrip():
    x = torch.tensor([1.0, 2.0])
    y = x.to("cpu")
    assert y.device.type == "cpu"


def test_tensor_to_cuda_and_back_roundtrip_with_fake_runtime(monkeypatch):
    import candle._backends.cuda.runtime as cuda_runtime

    monkeypatch.setattr(cuda_runtime, "is_available", lambda: True)
    monkeypatch.setattr(cuda_runtime, "device_count", lambda: 1)

    x = torch.tensor([1.0, 2.0])
    y = x.to("cuda")
    z = y.to("cpu")

    assert y.device.type == "cuda"
    assert z.device.type == "cpu"
    assert z.tolist() == [1.0, 2.0]
