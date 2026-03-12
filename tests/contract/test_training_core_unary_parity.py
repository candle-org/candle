
import candle as torch
import pytest

from .helpers import run_training_core_parity_case


UNARY_OPS = [
    "abs", "neg", "exp", "log", "sqrt", "sin", "cos", "tan", "tanh", "sigmoid",
    "floor", "ceil", "round", "trunc", "frac", "log2", "log10", "exp2",
    "rsqrt", "reciprocal", "sign", "signbit", "isnan", "isinf", "isfinite",
    "sinh", "cosh", "asinh", "acosh", "atanh", "erf", "erfc", "softplus",
    "relu6", "gelu", "silu", "mish", "square",
]


def _torch_unary(op_name, x):
    import torch as real_torch
    if hasattr(real_torch, op_name):
        return getattr(real_torch, op_name)(x)
    return getattr(real_torch.nn.functional, op_name)(x)


@pytest.mark.parametrize("op_name", UNARY_OPS)
def test_unary_forward_parity(op_name):
    import torch as real_torch


@pytest.mark.parametrize("op_name", ["frac", "gelu", "silu", "mish"])
def test_unary_int_error_parity(op_name):
    import torch as real_torch

    def candle_call():
        fn = getattr(torch, op_name) if hasattr(torch, op_name) else getattr(torch.nn.functional, op_name)
        fn(torch.tensor([1, -2, 3], dtype=torch.int64))

    def torch_call():
        fn = getattr(real_torch, op_name) if hasattr(real_torch, op_name) else getattr(real_torch.nn.functional, op_name)
        fn(real_torch.tensor([1, -2, 3], dtype=real_torch.int64))

    try:
        torch_call()
    except Exception as torch_exc:
        torch_type = type(torch_exc)
        torch_msg = str(torch_exc)
    else:
        torch_type = None
        torch_msg = None

    try:
        candle_call()
    except Exception as candle_exc:
        candle_type = type(candle_exc)
        candle_msg = str(candle_exc)
    else:
        candle_type = None
        candle_msg = None

    assert candle_type is torch_type
    assert candle_msg == torch_msg
