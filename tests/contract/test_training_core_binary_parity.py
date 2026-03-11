import pytest
import candle as torch

from .helpers import run_training_core_parity_case


def test_add_dtype_promotion_matches_torch_contract():
    import torch as real_torch

    a = torch.tensor([1, 2, 3], dtype=torch.int64)
    b = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
    result = run_training_core_parity_case(
        op_name="add",
        candle_fn=lambda x, y: torch.add(x, y),
        torch_fn=lambda x, y: real_torch.add(x, y),
        candle_inputs=lambda: (a, b),
        torch_inputs=lambda: (
            real_torch.tensor([1, 2, 3], dtype=real_torch.int64),
            real_torch.tensor([1.5, 2.5, 3.5], dtype=real_torch.float32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_mul_dtype_promotion_matches_torch_contract():
    import torch as real_torch

    a = torch.tensor([1, 2, 3], dtype=torch.int64)
    b = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
    result = run_training_core_parity_case(
        op_name="mul",
        candle_fn=lambda x, y: torch.mul(x, y),
        torch_fn=lambda x, y: real_torch.mul(x, y),
        candle_inputs=lambda: (a, b),
        torch_inputs=lambda: (
            real_torch.tensor([1, 2, 3], dtype=real_torch.int64),
            real_torch.tensor([1.5, 2.5, 3.5], dtype=real_torch.float32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_div_dtype_promotion_matches_torch_contract():
    import torch as real_torch

    a = torch.tensor([1, 2, 3], dtype=torch.int64)
    b = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32)
    result = run_training_core_parity_case(
        op_name="div",
        candle_fn=lambda x, y: torch.div(x, y),
        torch_fn=lambda x, y: real_torch.div(x, y),
        candle_inputs=lambda: (a, b),
        torch_inputs=lambda: (
            real_torch.tensor([1, 2, 3], dtype=real_torch.int64),
            real_torch.tensor([2.0, 4.0, 6.0], dtype=real_torch.float32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_true_divide_dtype_promotion_matches_torch_contract():
    import torch as real_torch

    a = torch.tensor([1, 2, 3], dtype=torch.int64)
    b = torch.tensor([2, 4, 6], dtype=torch.int32)
    result = run_training_core_parity_case(
        op_name="true_divide",
        candle_fn=lambda x, y: torch.true_divide(x, y),
        torch_fn=lambda x, y: real_torch.true_divide(x, y),
        candle_inputs=lambda: (a, b),
        torch_inputs=lambda: (
            real_torch.tensor([1, 2, 3], dtype=real_torch.int64),
            real_torch.tensor([2, 4, 6], dtype=real_torch.int32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_add_bool_int64_promotion_matches_torch_contract():
    import torch as real_torch

    a = torch.tensor([True, False, True], dtype=torch.bool)
    b = torch.tensor([1, 2, 3], dtype=torch.int64)
    result = run_training_core_parity_case(
        op_name="add",
        candle_fn=lambda x, y: torch.add(x, y),
        torch_fn=lambda x, y: real_torch.add(x, y),
        candle_inputs=lambda: (a, b),
        torch_inputs=lambda: (
            real_torch.tensor([True, False, True], dtype=real_torch.bool),
            real_torch.tensor([1, 2, 3], dtype=real_torch.int64),
        ),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_add_broadcast_matches_torch_contract():
    import torch as real_torch

    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    b = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32)
    result = run_training_core_parity_case(
        op_name="add",
        candle_fn=lambda x, y: torch.add(x, y),
        torch_fn=lambda x, y: real_torch.add(x, y),
        candle_inputs=lambda: (a, b),
        torch_inputs=lambda: (
            real_torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=real_torch.float32),
            real_torch.tensor([0.5, -1.0, 2.0], dtype=real_torch.float32),
        ),
    )

    assert result["shape_match"] is True
    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_add_broadcast_error_matches_torch_contract():
    import torch as real_torch

    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    b = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    result = run_training_core_parity_case(
        op_name="add",
        candle_fn=lambda x, y: torch.add(x, y),
        torch_fn=lambda x, y: real_torch.add(x, y),
        candle_inputs=lambda: (a, b),
        torch_inputs=lambda: (
            real_torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=real_torch.float32),
            real_torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=real_torch.float32),
        ),
        expect_error=True,
    )

    assert result["error_type_match"] is True


def test_add_inplace_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="add_",
        candle_fn=lambda x, y: torch.add_(x, y),
        torch_fn=lambda x, y: x.add_(y),
        candle_inputs=lambda: (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
            torch.tensor([[0.5, -1.0, 2.0], [1.0, 1.5, -0.5]], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=real_torch.float32),
            real_torch.tensor([[0.5, -1.0, 2.0], [1.0, 1.5, -0.5]], dtype=real_torch.float32),
        ),
    )

    assert result["value_match"] is True
