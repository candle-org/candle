import candle as torch
import torch as pt


def _assert_shape_dtype(out, ref):
    assert out.shape == ref.shape
    assert str(out.dtype) == str(ref.dtype)


def test_randint_high_size_tuple():
    out = torch.randint(10, (3,))
    ref = pt.randint(10, (3,))
    _assert_shape_dtype(out, ref)


def test_randint_high_size_list():
    out = torch.randint(2, [3, 2])
    ref = pt.randint(2, [3, 2])
    _assert_shape_dtype(out, ref)
