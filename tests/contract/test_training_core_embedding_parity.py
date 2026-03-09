import candle as torch

from .helpers import run_training_core_parity_case


def test_embedding_backward_matches_torch_padding_idx_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="embedding_padding_idx",
        candle_fn=lambda indices, weight: torch.nn.functional.embedding(indices, weight, padding_idx=0).sum(),
        torch_fn=lambda indices, weight: real_torch.nn.functional.embedding(indices, weight, padding_idx=0).sum(),
        candle_inputs=lambda: (
            torch.tensor([0, 2, 0], dtype=torch.int64),
            torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                dtype=torch.float32,
                requires_grad=True,
            ),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([0, 2, 0], dtype=real_torch.int64),
            real_torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                dtype=real_torch.float32,
                requires_grad=True,
            ),
        ),
        check_backward=True,
        atol=1e-6,
        rtol=1e-5,
    )

    assert result["grad_count_match"] is True
    assert result["grad_value_match"] is True


def test_embedding_backward_matches_torch_scale_grad_by_freq_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="embedding_scale_grad_by_freq",
        candle_fn=lambda indices, weight: torch.nn.functional.embedding(indices, weight, scale_grad_by_freq=True).sum(),
        torch_fn=lambda indices, weight: real_torch.nn.functional.embedding(indices, weight, scale_grad_by_freq=True).sum(),
        candle_inputs=lambda: (
            torch.tensor([0, 2, 0], dtype=torch.int64),
            torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                dtype=torch.float32,
                requires_grad=True,
            ),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([0, 2, 0], dtype=real_torch.int64),
            real_torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                dtype=real_torch.float32,
                requires_grad=True,
            ),
        ),
        check_backward=True,
        atol=1e-6,
        rtol=1e-5,
    )

    assert result["grad_count_match"] is True
    assert result["grad_value_match"] is True
