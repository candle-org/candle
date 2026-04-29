import pytest

import candle as torch
from candle._C._dispatcher_core import cy_prepare_dispatch_inputs
from candle._dispatch.dispatcher import dispatch_with_keyset
from candle._dispatch.keys import DispatchKey, DispatchKeySet


def test_dispatch_prefers_meta_when_input_meta():
    a = torch.ones((2,), device="meta")
    b = torch.ones((2,), device="meta")
    c = torch.add(a, b)
    assert c.device.type == "meta"


def test_dispatch_rejects_cross_npu_device_index():
    class _FakeTensor:
        def __init__(self, dev):
            self.device = torch.Device(dev)
            self.dtype = torch.float32

    a = _FakeTensor("npu:0")
    b = _FakeTensor("npu:1")
    keyset = DispatchKeySet(int(DispatchKey.NPU))

    with pytest.raises(RuntimeError, match=r"npu:1.*npu:0"):
        dispatch_with_keyset("add", keyset, None, a, b)


def test_dispatch_prefers_cuda_over_cpu(monkeypatch):
    import candle._backends.cuda.runtime as cuda_runtime

    monkeypatch.setattr(cuda_runtime, "is_available", lambda: True)
    monkeypatch.setattr(cuda_runtime, "device_count", lambda: 1)

    a = torch.ones((2,), device="cuda")
    b = torch.ones((2,), device="cuda")
    c = torch.add(a, b)
    assert c.device.type == "cuda"


def test_dispatch_core_preserves_kernel_resolution():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([3.0, 4.0], dtype=torch.float32)
    out = x.add(y)
    assert out.tolist() == [4.0, 6.0]


def test_dispatch_core_preserves_optional_device_handling():
    x = torch.ones((2, 2), dtype=torch.float32)
    y = x.to(device="cpu")
    assert y.device.type == "cpu"
    assert y.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_dispatch_core_injects_device_kwarg_for_accepting_kernel():
    device = torch.Device("cpu")

    def _kernel(*, device=None):
        return device

    args, kwargs = cy_prepare_dispatch_inputs(_kernel, (), {}, device)

    assert args == ()
    assert kwargs["device"] == device


def test_dispatch_core_strips_explicit_device_for_nonaccepting_kernel():
    device = torch.Device("cpu")

    def _kernel(*, alpha=None):
        return alpha

    args, kwargs = cy_prepare_dispatch_inputs(_kernel, (), {"device": device, "alpha": 2}, device)

    assert args == ()
    assert kwargs == {"alpha": 2}


def test_dispatch_core_preserves_explicit_device_for_accepting_kernel():
    dispatch_device = torch.Device("cpu")
    explicit_device = torch.Device("meta")

    def _kernel(*, device=None):
        return device

    args, kwargs = cy_prepare_dispatch_inputs(
        _kernel,
        (),
        {"device": explicit_device},
        dispatch_device,
    )

    assert args == ()
    assert kwargs["device"] == explicit_device
