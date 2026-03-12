import pytest

from candle import npu as npu_api
from candle import tensor
from candle._backends.npu import runtime as npu_runtime


def _block_cpu_roundtrip(monkeypatch):
    def _fail(*_args, **_kwargs):
        raise AssertionError("CPU round-trip is not allowed in NPU indexing path")

    monkeypatch.setattr(npu_runtime, "memcpy_d2h", _fail)
    monkeypatch.setattr(npu_runtime, "_copy_cpu_to_npu", _fail)


@pytest.mark.skipif(not npu_api.is_available(), reason="NPU not available")
@pytest.mark.parametrize("case", ["basic", "tensor", "bool"])
def test_npu_indexing_avoids_cpu_roundtrip(monkeypatch, case):
    x = tensor([[1, 2, 3], [4, 5, 6]], device="npu")

    if case == "tensor":
        idx = tensor([0, 2], device="npu")
    elif case == "bool":
        mask = tensor([[True, False, True], [False, True, False]], device="npu")

    _block_cpu_roundtrip(monkeypatch)

    if case == "basic":
        y = x[:, 1]
        assert y is not None
    elif case == "tensor":
        y = x[0, idx]
        assert y is not None
    else:
        with pytest.raises(RuntimeError, match="NPU boolean mask indexing is not supported"):
            _ = x[mask]
