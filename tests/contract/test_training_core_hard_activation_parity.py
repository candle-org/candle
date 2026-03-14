import candle as torch

from .helpers import run_training_core_parity_case


def test_hardswish_forward_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="hardswish",
        candle_fn=lambda x: torch.nn.functional.hardswish(x),
        torch_fn=lambda x: real_torch.nn.functional.hardswish(x),
        candle_inputs=lambda: (
            torch.tensor([-4.0, -3.0, -1.0, 0.0, 2.0, 4.0], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([-4.0, -3.0, -1.0, 0.0, 2.0, 4.0], dtype=real_torch.float32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_hardsigmoid_forward_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="hardsigmoid",
        candle_fn=lambda x: torch.nn.functional.hardsigmoid(x),
        torch_fn=lambda x: real_torch.nn.functional.hardsigmoid(x),
        candle_inputs=lambda: (
            torch.tensor([-4.0, -3.0, -1.0, 0.0, 2.0, 4.0], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([-4.0, -3.0, -1.0, 0.0, 2.0, 4.0], dtype=real_torch.float32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_softsign_forward_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="softsign",
        candle_fn=lambda x: torch.nn.functional.softsign(x),
        torch_fn=lambda x: real_torch.nn.functional.softsign(x),
        candle_inputs=lambda: (
            torch.tensor([-4.0, -1.0, 0.0, 2.0, 4.0], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([-4.0, -1.0, 0.0, 2.0, 4.0], dtype=real_torch.float32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["shape_match"] is True
    assert result["value_match"] is True
