import candle as torch

from .helpers import run_training_core_parity_case


def test_slice_forward_matches_torch_contract():
    import torch as real_torch

    candle_out = torch._dispatch.dispatch("slice", "cpu", torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32), 1, 0, 2, 1)
    torch_out = real_torch.ops.aten.slice.Tensor(
        real_torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=real_torch.float32),
        1,
        0,
        2,
        1,
    )

    assert candle_out.shape == torch_out.shape
    assert candle_out.numpy().tolist() == torch_out.numpy().tolist()


def test_slice_copy_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="slice_copy",
        candle_fn=lambda x: torch.slice_copy(x, 1, 0, 2, 1),
        torch_fn=lambda x: real_torch.slice_copy(x, 1, 0, 2, 1),
        candle_inputs=lambda: (torch.arange(6, dtype=torch.float32).reshape(2, 3),),
        torch_inputs=lambda: (real_torch.arange(6, dtype=real_torch.float32).reshape(2, 3),),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_slice_scatter_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="slice_scatter",
        candle_fn=lambda x, src: torch.slice_scatter(x, src, 1, 0, 2, 1),
        torch_fn=lambda x, src: real_torch.slice_scatter(x, src, 1, 0, 2, 1),
        candle_inputs=lambda: (
            torch.zeros((2, 3), dtype=torch.float32),
            torch.tensor([[9.0, 8.0], [7.0, 6.0]], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.zeros((2, 3), dtype=real_torch.float32),
            real_torch.tensor([[9.0, 8.0], [7.0, 6.0]], dtype=real_torch.float32),
        ),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_slice_step_must_be_positive_matches_torch_contract():
    import torch as real_torch

    candle_input = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch_input = real_torch.arange(6, dtype=real_torch.float32).reshape(2, 3)

    try:
        real_torch.ops.aten.slice.Tensor(torch_input, 1, 0, 2, -1)
    except Exception as torch_exc:
        torch_type = type(torch_exc)
    else:
        torch_type = None

    try:
        torch._dispatch.dispatch("slice", "cpu", candle_input, 1, 0, 2, -1)
    except Exception as candle_exc:
        candle_type = type(candle_exc)
    else:
        candle_type = None

    assert candle_type is torch_type


def test_slice_backward_matches_torch_contract():
    import torch as real_torch

    candle_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
    torch_input = real_torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=real_torch.float32, requires_grad=True)

    candle_out = torch._dispatch.dispatch("slice", "cpu", candle_input, 1, 0, 2, 1).sum()
    torch_out = real_torch.ops.aten.slice.Tensor(torch_input, 1, 0, 2, 1).sum()

    candle_out.backward()
    torch_out.backward()

    assert candle_input.grad is not None
    assert torch_input.grad is not None
    assert candle_input.grad.numpy().tolist() == torch_input.grad.numpy().tolist()


def test_expand_copy_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="expand_copy",
        candle_fn=lambda x: torch.expand_copy(x, (2, 3)),
        torch_fn=lambda x: real_torch.expand_copy(x, (2, 3)),
        candle_inputs=lambda: (torch.tensor([[1.0], [2.0]], dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.tensor([[1.0], [2.0]], dtype=real_torch.float32),),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_as_strided_copy_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="as_strided_copy",
        candle_fn=lambda x: torch.as_strided_copy(x, (2, 2), (2, 1), 0),
        torch_fn=lambda x: real_torch.as_strided_copy(x, (2, 2), (2, 1), 0),
        candle_inputs=lambda: (torch.arange(6, dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.arange(6, dtype=real_torch.float32),),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_as_strided_scatter_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="as_strided_scatter",
        candle_fn=lambda x, src: torch.as_strided_scatter(x, src, (2, 2), (2, 1), 0),
        torch_fn=lambda x, src: real_torch.as_strided_scatter(x, src, (2, 2), (2, 1), 0),
        candle_inputs=lambda: (
            torch.zeros((6,), dtype=torch.float32),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.zeros((6,), dtype=real_torch.float32),
            real_torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=real_torch.float32),
        ),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_as_strided_scatter_backward_matches_torch_contract():
    import torch as real_torch

    candle_input = torch.zeros((6,), dtype=torch.float32).requires_grad_()
    candle_src = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).requires_grad_()
    torch_input = real_torch.zeros((6,), dtype=real_torch.float32, requires_grad=True)
    torch_src = real_torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=real_torch.float32, requires_grad=True)

    candle_out = torch._dispatch.dispatch("as_strided_scatter", "cpu", candle_input, candle_src, (2, 2), (2, 1), 0).sum()
    torch_out = real_torch.as_strided_scatter(torch_input, torch_src, (2, 2), (2, 1), 0).sum()

    candle_out.backward()
    torch_out.backward()

    assert candle_input.grad is not None
    assert torch_input.grad is not None
    assert torch_src.grad is not None
    assert candle_input.grad.numpy().tolist() == torch_input.grad.numpy().tolist()
    assert candle_src.grad is None


def test_as_strided_offset_out_of_bounds_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="as_strided_offset_oob",
        candle_fn=lambda x: torch.as_strided_copy(x, (2, 2), (2, 1), 10),
        torch_fn=lambda x: real_torch.as_strided_copy(x, (2, 2), (2, 1), 10),
        candle_inputs=lambda: (torch.arange(6, dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.arange(6, dtype=real_torch.float32),),
        expect_error=True,
    )

    assert result["error_type_match"] is True


def test_as_strided_negative_stride_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="as_strided_negative_stride",
        candle_fn=lambda x: torch.as_strided_copy(x, (2, 2), (-1, 1), 4),
        torch_fn=lambda x: real_torch.as_strided_copy(x, (2, 2), (-1, 1), 4),
        candle_inputs=lambda: (torch.arange(6, dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.arange(6, dtype=real_torch.float32),),
        expect_error=True,
    )

    assert result["error_type_match"] is True
