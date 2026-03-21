import pytest

import candle as torch


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_fast_binary_op_fallback_avoids_recursive_binary_op(monkeypatch):
    from candle._backends.npu.ops import _helpers
    from candle._cython import _npu_ops_fallback

    monkeypatch.setattr(_helpers, "_fast_binary_op", _npu_ops_fallback.fast_binary_op)
    monkeypatch.setattr(_helpers, "_HAS_FAST_OPS", True)

    a = torch.randn((2, 2), device="npu", dtype=torch.float16)
    b = torch.randn((2, 2), device="npu", dtype=torch.float16)

    out = torch.add(a, b)
    torch.npu.synchronize()

    expected = a.to("cpu").numpy() + b.to("cpu").numpy()
    actual = out.to("cpu").numpy()
    assert actual.shape == expected.shape
    assert (abs(actual - expected) < 1e-3).all()
