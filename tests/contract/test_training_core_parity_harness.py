import candle as torch
import pytest

from .helpers import run_training_core_parity_case


def test_parity_harness_compares_forward_dtype_and_value():
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
    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_parity_harness_compares_exception_type():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="mean",
        candle_fn=lambda a: torch.mean(a),
        torch_fn=lambda a: real_torch.mean(a),
        candle_inputs=lambda: (torch.tensor([1, 2, 3], dtype=torch.int64),),
        torch_inputs=lambda: (real_torch.tensor([1, 2, 3], dtype=real_torch.int64),),
        expect_error=True,
    )

    assert result["error_type_match"] is True


def test_parity_harness_compares_gradients():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="sum_relu_mul",
        candle_fn=lambda a, b: torch.sum(torch.relu(torch.mul(a, b))),
        torch_fn=lambda a, b: real_torch.sum(real_torch.relu(real_torch.mul(a, b))),
        candle_inputs=lambda: (
            torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32, requires_grad=True),
            torch.tensor([0.5, 4.0, -1.0], dtype=torch.float32, requires_grad=True),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([1.0, -2.0, 3.0], dtype=real_torch.float32, requires_grad=True),
            real_torch.tensor([0.5, 4.0, -1.0], dtype=real_torch.float32, requires_grad=True),
        ),
        check_backward=True,
    )

    assert result["grad_count_match"] is True
    assert result["grad_value_match"] is True


def test_parity_harness_compares_error_message_fragment_for_view_inplace():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="view_add_",
        candle_fn=lambda x: x.view((4,)).add_(1.0),
        torch_fn=lambda x: x.view((4,)).add_(1.0),
        candle_inputs=lambda: (torch.ones((2, 2), dtype=torch.float32).requires_grad_(),),
        torch_inputs=lambda: (real_torch.ones((2, 2), dtype=real_torch.float32, requires_grad=True),),
        expect_error=True,
        check_error_message=True,
        error_message_fragment="view of a leaf Variable that requires grad",
    )

    assert result["error_type_match"] is True
    assert result["error_message_match"] is True
