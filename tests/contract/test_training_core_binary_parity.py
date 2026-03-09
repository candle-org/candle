import candle as torch
import pytest

from .helpers import run_training_core_parity_case


def test_add_dtype_promotion_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="add",
        candle_fn=lambda a, b: torch.add(a, b),
        torch_fn=lambda a, b: real_torch.add(a, b),
        candle_inputs=lambda: (
            torch.tensor([1, 2, 3], dtype=torch.int64),
            torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([1, 2, 3], dtype=real_torch.int64),
            real_torch.tensor([0.5, 0.5, 0.5], dtype=real_torch.float32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_mul_dtype_promotion_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="mul",
        candle_fn=lambda a, b: torch.mul(a, b),
        torch_fn=lambda a, b: real_torch.mul(a, b),
        candle_inputs=lambda: (
            torch.tensor([1, 2, 3], dtype=torch.int64),
            torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([1, 2, 3], dtype=real_torch.int64),
            real_torch.tensor([0.5, 0.5, 0.5], dtype=real_torch.float32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_div_dtype_promotion_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="div",
        candle_fn=lambda a, b: torch.div(a, b),
        torch_fn=lambda a, b: real_torch.div(a, b),
        candle_inputs=lambda: (
            torch.tensor([2, 4, 6], dtype=torch.int64),
            torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([2, 4, 6], dtype=real_torch.int64),
            real_torch.tensor([2.0, 2.0, 2.0], dtype=real_torch.float32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_true_divide_dtype_promotion_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="true_divide",
        candle_fn=lambda a, b: torch.true_divide(a, b),
        torch_fn=lambda a, b: real_torch.true_divide(a, b),
        candle_inputs=lambda: (
            torch.tensor([2, 4, 6], dtype=torch.int64),
            torch.tensor([2, 2, 2], dtype=torch.int32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([2, 4, 6], dtype=real_torch.int64),
            real_torch.tensor([2, 2, 2], dtype=real_torch.int32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_add_bool_int64_promotion_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="add",
        candle_fn=lambda a, b: torch.add(a, b),
        torch_fn=lambda a, b: real_torch.add(a, b),
        candle_inputs=lambda: (
            torch.tensor([True, False, True], dtype=torch.bool),
            torch.tensor([1, 2, 3], dtype=torch.int64),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([True, False, True], dtype=real_torch.bool),
            real_torch.tensor([1, 2, 3], dtype=real_torch.int64),
        ),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_add_broadcast_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="add",
        candle_fn=lambda a, b: torch.add(a, b),
        torch_fn=lambda a, b: real_torch.add(a, b),
        candle_inputs=lambda: (
            torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([[1.0], [2.0]], dtype=real_torch.float32),
            real_torch.tensor([10.0, 20.0, 30.0], dtype=real_torch.float32),
        ),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_add_broadcast_error_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="add",
        candle_fn=lambda a, b: torch.add(a, b),
        torch_fn=lambda a, b: real_torch.add(a, b),
        candle_inputs=lambda: (
            torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([[1.0, 2.0]], dtype=real_torch.float32),
            real_torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=real_torch.float32),
        ),
        expect_error=True,
    )

    assert result["error_type_match"] is True


def test_add_inplace_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="add_",
        candle_fn=lambda a, b: a.add_(b),
        torch_fn=lambda a, b: a.add_(b),
        candle_inputs=lambda: (
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([1.0, 2.0, 3.0], dtype=real_torch.float32),
            real_torch.tensor([0.5, 0.5, 0.5], dtype=real_torch.float32),
        ),
    )

    assert result["value_match"] is True
