import candle as torch

from .helpers import run_training_core_parity_case


def test_index_select_negative_index_matches_torch_error_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="index_select_negative_index",
        candle_fn=lambda x, index: torch.index_select(x, dim=1, index=index),
        torch_fn=lambda x, index: real_torch.index_select(x, dim=1, index=index),
        candle_inputs=lambda: (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
            torch.tensor([-1, 0], dtype=torch.int64),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=real_torch.float32),
            real_torch.tensor([-1, 0], dtype=real_torch.int64),
        ),
        expect_error=True,
    )

    assert result["error_type_match"] is True
