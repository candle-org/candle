import candle as torch


def test_import_has_tensor():
    assert hasattr(torch, "tensor")


def test_import_has_nn_module():
    assert hasattr(torch, "nn")
    assert hasattr(torch.nn, "Linear")
