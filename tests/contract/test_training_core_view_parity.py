import candle as torch

from .helpers import run_training_core_parity_case


def test_view_contiguous_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="view",
        candle_fn=lambda x: x.view((3, 2)),
        torch_fn=lambda x: x.view((3, 2)),
        candle_inputs=lambda: (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=real_torch.float32),),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_view_non_contiguous_error_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="view",
        candle_fn=lambda x: x.transpose(0, 1).view((6,)),
        torch_fn=lambda x: x.transpose(0, 1).view((6,)),
        candle_inputs=lambda: (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=real_torch.float32),),
        expect_error=True,
    )

    assert result["error_type_match"] is True


def test_view_inplace_error_matches_torch_contract():
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


def test_view_backward_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="transpose_sum",
        candle_fn=lambda x: x.transpose(0, 1).sum(),
        torch_fn=lambda x: x.transpose(0, 1).sum(),
        candle_inputs=lambda: (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=real_torch.float32, requires_grad=True),
        ),
        check_backward=True,
    )

    assert result["grad_count_match"] is True
    assert result["grad_value_match"] is True
