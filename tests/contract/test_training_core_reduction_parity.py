import candle as torch

from .helpers import run_training_core_parity_case


def test_mean_int_without_dtype_matches_torch_contract():
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


def test_mean_int_with_dtype_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="mean",
        candle_fn=lambda a: torch.mean(a, dtype=torch.float32),
        torch_fn=lambda a: real_torch.mean(a, dtype=real_torch.float32),
        candle_inputs=lambda: (torch.tensor([1, 2, 3], dtype=torch.int64),),
        torch_inputs=lambda: (real_torch.tensor([1, 2, 3], dtype=real_torch.int64),),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_sum_dtype_accumulates_in_target_dtype_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="sum",
        candle_fn=lambda a: torch.sum(a, dtype=torch.int64),
        torch_fn=lambda a: real_torch.sum(a, dtype=real_torch.int64),
        candle_inputs=lambda: (torch.tensor([120, 120], dtype=torch.int8),),
        torch_inputs=lambda: (real_torch.tensor([120, 120], dtype=real_torch.int8),),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_sum_negative_dim_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="sum",
        candle_fn=lambda a: torch.sum(a, dim=-1),
        torch_fn=lambda a: real_torch.sum(a, dim=-1),
        candle_inputs=lambda: (torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=real_torch.float32),),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_mean_tuple_dim_keepdim_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="mean",
        candle_fn=lambda a: torch.mean(a, dim=(0, 2), keepdim=True),
        torch_fn=lambda a: real_torch.mean(a, dim=(0, 2), keepdim=True),
        candle_inputs=lambda: (torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=real_torch.float32),),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_sum_backward_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="sum",
        candle_fn=lambda a: torch.sum(a),
        torch_fn=lambda a: real_torch.sum(a),
        candle_inputs=lambda: (torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True),),
        torch_inputs=lambda: (real_torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=real_torch.float32, requires_grad=True),),
        check_backward=True,
    )

    assert result["grad_count_match"] is True
    assert result["grad_value_match"] is True


def test_mean_backward_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="mean",
        candle_fn=lambda a: torch.mean(a),
        torch_fn=lambda a: real_torch.mean(a),
        candle_inputs=lambda: (torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True),),
        torch_inputs=lambda: (real_torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=real_torch.float32, requires_grad=True),),
        check_backward=True,
    )

    assert result["grad_count_match"] is True
    assert result["grad_value_match"] is True
