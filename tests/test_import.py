import candle as torch


def test_import_has_tensor():
    assert hasattr(torch, "tensor")
