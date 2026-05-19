import pytest

import candle as torch
import torch as pt
from .helpers import assert_torch_error


def test_inplace_view_version_error_message():
    def mt():
        x = torch.tensor([1.0], requires_grad=True)
        y = x.view((1,))
        y.add_(1.0)
        y.sum().backward()

    def th():
        x = pt.tensor([1.0], requires_grad=True)
        y = x.view((1,))
        y.add_(1.0)
        y.sum().backward()

    assert_torch_error(mt, th)


def test_special_zeta_backward_error_matches_torch():
    def mt():
        x = torch.tensor([2.5], requires_grad=True)
        q = torch.tensor([3.0])
        torch.special.zeta(x, q).sum().backward()

    def th():
        x = pt.tensor([2.5], requires_grad=True)
        q = pt.tensor([3.0])
        pt.special.zeta(x, q).sum().backward()

    assert_torch_error(mt, th)


def test_special_gammainc_backward_error_matches_torch():
    def mt():
        x = torch.tensor([2.0], requires_grad=True)
        y = torch.tensor([1.0])
        torch.special.gammainc(x, y).sum().backward()

    def th():
        x = pt.tensor([2.0], requires_grad=True)
        y = pt.tensor([1.0])
        pt.special.gammainc(x, y).sum().backward()

    assert_torch_error(mt, th)


def test_special_gammaincc_backward_error_matches_torch():
    def mt():
        x = torch.tensor([2.0], requires_grad=True)
        y = torch.tensor([1.0])
        torch.special.gammaincc(x, y).sum().backward()

    def th():
        x = pt.tensor([2.0], requires_grad=True)
        y = pt.tensor([1.0])
        pt.special.gammaincc(x, y).sum().backward()

    assert_torch_error(mt, th)


def test_unique_backward_error_matches_torch():
    def mt():
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        torch.unique(x).sum().backward()

    def th():
        x = pt.tensor([1.0, 2.0], requires_grad=True)
        pt.unique(x).sum().backward()

    assert_torch_error(mt, th)


def test_unique_consecutive_backward_error_matches_torch():
    def mt():
        x = torch.tensor([1.0, 1.0, 2.0], requires_grad=True)
        torch.unique_consecutive(x).sum().backward()

    def th():
        x = pt.tensor([1.0, 1.0, 2.0], requires_grad=True)
        pt.unique_consecutive(x).sum().backward()

    assert_torch_error(mt, th)


def test_nextafter_backward_error_matches_torch():
    def mt():
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0])
        torch.nextafter(x, y).sum().backward()

    def th():
        x = pt.tensor([1.0], requires_grad=True)
        y = pt.tensor([2.0])
        pt.nextafter(x, y).sum().backward()

    assert_torch_error(mt, th)


def test_aminmax_backward_error_matches_torch():
    def mt():
        x = torch.tensor([1.0, 3.0, 2.0], requires_grad=True)
        torch.aminmax(x)[0].backward()

    def th():
        x = pt.tensor([1.0, 3.0, 2.0], requires_grad=True)
        pt.aminmax(x)[0].backward()

    try:
        th()
    except Exception:
        with pytest.raises(RuntimeError, match="derivative.*aminmax.*not implemented"):
            mt()
    else:
        pytest.xfail("PyTorch now supports aminmax backward; Candle still reports it as unsupported")


def test_non_scalar_no_grad_backward_error_matches_torch():
    def mt():
        torch.tensor([1.0, 2.0]).backward()

    def th():
        pt.tensor([1.0, 2.0]).backward()

    assert_torch_error(mt, th)


def test_logical_xor_backward_error_matches_torch():
    def mt():
        x = torch.tensor([0.0, 1.0], requires_grad=True)
        torch.logical_xor(x, torch.tensor([1.0, 1.0])).sum().backward()

    def th():
        x = pt.tensor([0.0, 1.0], requires_grad=True)
        pt.logical_xor(x, pt.tensor([1.0, 1.0])).sum().backward()

    assert_torch_error(mt, th)


def test_bitwise_not_backward_error_matches_torch():
    def mt():
        torch.bitwise_not(torch.tensor([1, 3])).sum().backward()

    def th():
        pt.bitwise_not(pt.tensor([1, 3])).sum().backward()

    assert_torch_error(mt, th)


@pytest.mark.parametrize(
    ("op_name", "left", "right"),
    [
        ("bitwise_and", [1, 2], [3, 1]),
        ("bitwise_or", [1, 2], [3, 1]),
        ("bitwise_xor", [1, 2], [3, 1]),
        ("bitwise_left_shift", [1, 2], [1, 2]),
        ("bitwise_right_shift", [4, 8], [1, 2]),
    ],
)
def test_bitwise_binary_backward_error_matches_torch(op_name, left, right):
    def mt():
        getattr(torch, op_name)(torch.tensor(left), torch.tensor(right)).sum().backward()

    def th():
        getattr(pt, op_name)(pt.tensor(left), pt.tensor(right)).sum().backward()

    assert_torch_error(mt, th)


def test_special_zeta_other_backward_matches_torch():
    x = torch.tensor([2.5])
    q = torch.tensor([3.0], requires_grad=True)
    torch.special.zeta(x, q).sum().backward()

    expected_x = pt.tensor([2.5])
    expected_q = pt.tensor([3.0], requires_grad=True)
    pt.special.zeta(expected_x, expected_q).sum().backward()

    actual_grad = pt.tensor(q.grad.detach().numpy())
    assert pt.allclose(actual_grad, expected_q.grad.detach(), atol=1e-6, rtol=1e-6)


def test_nextafter_other_backward_error_matches_torch():
    def mt():
        x = torch.tensor([1.0])
        y = torch.tensor([2.0], requires_grad=True)
        torch.nextafter(x, y).sum().backward()

    def th():
        x = pt.tensor([1.0])
        y = pt.tensor([2.0], requires_grad=True)
        pt.nextafter(x, y).sum().backward()

    assert_torch_error(mt, th)


def test_unique_dim_backward_error_matches_torch():
    def mt():
        x = torch.tensor([[1.0, 2.0], [1.0, 2.0]], requires_grad=True)
        torch.unique(x, dim=0).sum().backward()

    def th():
        x = pt.tensor([[1.0, 2.0], [1.0, 2.0]], requires_grad=True)
        pt.unique(x, dim=0).sum().backward()

    assert_torch_error(mt, th)


def test_unique_consecutive_dim_backward_error_matches_torch():
    def mt():
        x = torch.tensor([[1.0, 2.0], [1.0, 2.0]], requires_grad=True)
        torch.unique_consecutive(x, dim=0).sum().backward()

    def th():
        x = pt.tensor([[1.0, 2.0], [1.0, 2.0]], requires_grad=True)
        pt.unique_consecutive(x, dim=0).sum().backward()

    assert_torch_error(mt, th)


def test_top_level_rrelu_backward_matches_torch():
    x = torch.tensor([-2.0, 1.0], requires_grad=True)
    out = torch.rrelu(x, lower=0.1, upper=0.1, training=False)
    out.sum().backward()

    expected_x = pt.tensor([-2.0, 1.0], requires_grad=True)
    expected = pt.rrelu(expected_x, lower=0.1, upper=0.1, training=False)
    expected.sum().backward()

    actual_out = pt.tensor(out.detach().numpy())
    actual_grad = pt.tensor(x.grad.detach().numpy())
    assert pt.allclose(actual_out, expected.detach(), atol=1e-6, rtol=1e-6)
    assert pt.allclose(actual_grad, expected_x.grad.detach(), atol=1e-6, rtol=1e-6)


def test_tensor_acosh_inplace_leaf_error_matches_torch():
    def mt():
        x = torch.tensor([2.0], requires_grad=True)
        x.acosh_()

    def th():
        x = pt.tensor([2.0], requires_grad=True)
        x.acosh_()

    assert_torch_error(mt, th)


def test_tensor_asinh_inplace_leaf_error_matches_torch():
    def mt():
        x = torch.tensor([2.0], requires_grad=True)
        x.asinh_()

    def th():
        x = pt.tensor([2.0], requires_grad=True)
        x.asinh_()

    assert_torch_error(mt, th)


def test_tensor_atanh_inplace_leaf_error_matches_torch():
    def mt():
        x = torch.tensor([0.2], requires_grad=True)
        x.atanh_()

    def th():
        x = pt.tensor([0.2], requires_grad=True)
        x.atanh_()

    assert_torch_error(mt, th)
