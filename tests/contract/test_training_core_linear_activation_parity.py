import candle as torch

from .helpers import run_training_core_parity_case


def test_linear_forward_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="linear",
        candle_fn=lambda x, w, b: torch.nn.functional.linear(x, w, b),
        torch_fn=lambda x, w, b: real_torch.nn.functional.linear(x, w, b),
        candle_inputs=lambda: (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32),
            torch.tensor([0.5, -0.5], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=real_torch.float32),
            real_torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=real_torch.float32),
            real_torch.tensor([0.5, -0.5], dtype=real_torch.float32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["value_match"] is True


def test_relu_forward_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="relu",
        candle_fn=lambda x: torch.nn.functional.relu(x),
        torch_fn=lambda x: real_torch.nn.functional.relu(x),
        candle_inputs=lambda: (torch.tensor([[-1.0, 0.0, 2.0]], dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.tensor([[-1.0, 0.0, 2.0]], dtype=real_torch.float32),),
    )

    assert result["value_match"] is True


def test_gelu_forward_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="gelu",
        candle_fn=lambda x: torch.nn.functional.gelu(x),
        torch_fn=lambda x: real_torch.nn.functional.gelu(x),
        candle_inputs=lambda: (torch.tensor([[-1.0, 0.0, 1.0, 2.0]], dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.tensor([[-1.0, 0.0, 1.0, 2.0]], dtype=real_torch.float32),),
        atol=1e-6,
        rtol=1e-5,
    )

    assert result["value_match"] is True


def test_silu_forward_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="silu",
        candle_fn=lambda x: torch.nn.functional.silu(x),
        torch_fn=lambda x: real_torch.nn.functional.silu(x),
        candle_inputs=lambda: (torch.tensor([[-1.0, 0.0, 1.0, 2.0]], dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.tensor([[-1.0, 0.0, 1.0, 2.0]], dtype=real_torch.float32),),
        atol=1e-6,
        rtol=1e-5,
    )

    assert result["value_match"] is True


def test_linear_relu_backward_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="linear_relu",
        candle_fn=lambda x, w, b: torch.nn.functional.relu(torch.nn.functional.linear(x, w, b)).sum(),
        torch_fn=lambda x, w, b: real_torch.nn.functional.relu(real_torch.nn.functional.linear(x, w, b)).sum(),
        candle_inputs=lambda: (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32, requires_grad=True),
            torch.tensor([0.5, -0.5], dtype=torch.float32, requires_grad=True),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=real_torch.float32, requires_grad=True),
            real_torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=real_torch.float32, requires_grad=True),
            real_torch.tensor([0.5, -0.5], dtype=real_torch.float32, requires_grad=True),
        ),
        check_backward=True,
        atol=1e-6,
        rtol=1e-5,
    )

    assert result["grad_count_match"] is True
    assert result["grad_value_match"] is True
