import candle as torch


def test_sum_to_size_forward_matches_torch_contract():
    import torch as real_torch

    candle_out = torch.sum_to_size(torch.arange(6, dtype=torch.float32).reshape(2, 3), 1, 3)
    torch_out = real_torch.arange(6, dtype=real_torch.float32).reshape(2, 3).sum_to_size(1, 3)

    assert candle_out.shape == torch_out.shape
    assert candle_out.numpy().tolist() == torch_out.numpy().tolist()


def test_sum_to_size_scalar_shape_matches_torch_contract():
    import torch as real_torch

    candle_out = torch.sum_to_size(torch.arange(6, dtype=torch.float32).reshape(2, 3), ())
    torch_out = real_torch.arange(6, dtype=real_torch.float32).reshape(2, 3).sum_to_size(())

    assert candle_out.shape == torch_out.shape
    assert candle_out.numpy().tolist() == torch_out.numpy().tolist()


def test_sum_to_size_invalid_shape_message_matches_torch_contract():
    import torch as real_torch

    candle_in = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch_in = real_torch.arange(6, dtype=real_torch.float32).reshape(2, 3)

    try:
        torch_in.sum_to_size(2, 2)
    except Exception as torch_exc:
        torch_type = type(torch_exc)
        torch_msg = str(torch_exc)
    else:
        torch_type = None
        torch_msg = None

    try:
        torch.sum_to_size(candle_in, 2, 2)
    except Exception as candle_exc:
        candle_type = type(candle_exc)
        candle_msg = str(candle_exc)
    else:
        candle_type = None
        candle_msg = None

    assert candle_type is torch_type
    assert candle_msg == torch_msg


def test_sum_to_size_type_error_matches_torch_contract():
    import torch as real_torch

    candle_in = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch_in = real_torch.arange(6, dtype=real_torch.float32).reshape(2, 3)

    try:
        torch_in.sum_to_size(True)
    except Exception as torch_exc:
        torch_type = type(torch_exc)
        torch_msg = str(torch_exc)
    else:
        torch_type = None
        torch_msg = None

    try:
        torch.sum_to_size(candle_in, True)
    except Exception as candle_exc:
        candle_type = type(candle_exc)
        candle_msg = str(candle_exc)
    else:
        candle_type = None
        candle_msg = None

    assert candle_type is torch_type
    assert candle_msg == torch_msg


def test_sum_to_size_type_error_matrix_matches_torch_contract():
    import torch as real_torch

    candle_in = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch_in = real_torch.arange(6, dtype=real_torch.float32).reshape(2, 3)

    cases = [
        ("top_bool", True),
        ("top_float", 1.5),
        ("top_str", "1"),
        ("top_none", None),
        ("first_str", ("1", 3)),
        ("first_float", (1.5, 3)),
        ("first_bool", (True, 3)),
        ("first_none", (None, 3)),
        ("first_list", ([1], 3)),
        ("late_str", (1, "1")),
        ("late_float", (1, 1.5)),
        ("late_none", (1, None)),
        ("late_list", (1, [3])),
        ("late_bool_then_str", (1, True, "a")),
        ("late_bool_then_none", (1, True, None)),
        ("first_bool_list", [True, 3]),
        ("mismatch_zero", [0, 3]),
        ("mismatch_neg", [-1, 3]),
        ("mismatch_long", [1, 2, 3]),
        ("mismatch_shape", [2, 2]),
        ("mismatch_false", [1, False]),
    ]

    for name, size in cases:
        try:
            torch_in.sum_to_size(size)
        except Exception as torch_exc:
            torch_type = type(torch_exc)
            torch_msg = str(torch_exc)
        else:
            torch_type = None
            torch_msg = None

        try:
            torch.sum_to_size(candle_in, size)
        except Exception as candle_exc:
            candle_type = type(candle_exc)
            candle_msg = str(candle_exc)
        else:
            candle_type = None
            candle_msg = None

        assert candle_type is torch_type, f"{name} type mismatch"
        assert candle_msg == torch_msg, f"{name} message mismatch"


def test_sum_to_size_valid_sizes_match_torch_contract():
    import torch as real_torch

    candle_in = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch_in = real_torch.arange(6, dtype=real_torch.float32).reshape(2, 3)

    for size in ([], [1, True], (1, True)):
        candle_out = torch.sum_to_size(candle_in, size)
        torch_out = torch_in.sum_to_size(size)

        assert candle_out.shape == torch_out.shape
        assert candle_out.numpy().tolist() == torch_out.numpy().tolist()


def test_sum_to_size_returns_view_when_size_matches():
    candle_in = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    candle_out = torch.sum_to_size(candle_in, 2, 3)

    assert candle_out.shape == candle_in.shape
    assert candle_out.storage() is candle_in.storage()


def test_sum_to_size_backward_matches_torch_contract():
    import torch as real_torch

    candle_leaf = torch.arange(6, dtype=torch.float32).requires_grad_()
    candle_in = candle_leaf.reshape(2, 3)
    torch_leaf = real_torch.arange(6, dtype=real_torch.float32, requires_grad=True)
    torch_in = torch_leaf.reshape(2, 3)

    candle_out = torch.sum_to_size(candle_in, 1, 3).sum()
    torch_out = torch_in.sum_to_size(1, 3).sum()

    candle_out.backward()
    torch_out.backward()

    assert candle_leaf.grad is not None
    assert torch_leaf.grad is not None
    assert candle_leaf.grad.numpy().tolist() == torch_leaf.grad.numpy().tolist()
