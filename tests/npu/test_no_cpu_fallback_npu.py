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
def test_npu_cartesian_prod_rejects_cpu_roundtrip_path(monkeypatch):
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

    with pytest.raises(RuntimeError, match="NPU cartesian_prod is not implemented without CPU fallback"):
        torch.cartesian_prod(a, b)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_block_diag_rejects_cpu_roundtrip_path(monkeypatch):
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

    with pytest.raises(RuntimeError, match="NPU block_diag is not implemented without CPU fallback"):
        torch.block_diag(a, b)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_repeat_interleave_tensor_repeats_rejects_cpu_roundtrip(monkeypatch):
    original_to = Tensor.to

    def guard_to(self, *args, **kwargs):
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]
        if getattr(self, "device", None) is not None and self.device.type == "npu":
            if device == "cpu" or getattr(device, "type", None) == "cpu":
                raise AssertionError("repeat_interleave should not move NPU tensors to CPU")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "to", guard_to)
    x = torch.tensor([[1, 2], [3, 4]], device="npu")
    repeats = torch.tensor([1, 2], device="npu", dtype=torch.int64)

    with pytest.raises(RuntimeError, match="NPU repeat_interleave with tensor repeats is not implemented without CPU fallback"):
        torch.repeat_interleave(x, repeats=repeats, dim=1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_baddbmm_tensor_alpha_beta_rejects_cpu_readback(monkeypatch):
    self_tensor = torch.randn((2, 3, 4), device="npu")
    batch1 = torch.randn((2, 3, 5), device="npu")
    batch2 = torch.randn((2, 5, 4), device="npu")
    alpha = torch.tensor(0.5, device="npu")
    beta = torch.tensor(2.0, device="npu")

    storage_cls = type(alpha.storage())

    def fail_to_numpy(self):
        raise AssertionError("baddbmm should not read NPU tensor scalars on CPU")

    monkeypatch.setattr(storage_cls, "to_numpy", fail_to_numpy, raising=False)

    with pytest.raises(RuntimeError, match="NPU baddbmm does not support tensor alpha/beta without CPU fallback"):
        torch.baddbmm(self_tensor, batch1, batch2, beta=beta, alpha=alpha)
