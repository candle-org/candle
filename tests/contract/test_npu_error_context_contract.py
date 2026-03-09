import pytest

import candle as torch
from candle._dispatch.dispatcher import dispatch


def test_npu_normalize_device_error_mentions_npu_not_cuda():
    with pytest.raises(ValueError, match="Expected a npu device"):
        torch.npu._normalize_npu_device("cpu")


def test_dispatch_error_contains_op_and_device_context():
    x = torch.ones((2, 2))
    y = torch.ones((2, 2))

    with pytest.raises(RuntimeError) as exc:
        dispatch("_definitely_missing_op_", "npu", x, y)

    message = str(exc.value)
    assert "op=_definitely_missing_op_" in message
    assert "device=npu" in message
