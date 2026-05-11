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


def test_unique_backward_error_matches_torch():
    def mt():
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        torch.unique(x).sum().backward()

    def th():
        x = pt.tensor([1.0, 2.0], requires_grad=True)
        pt.unique(x).sum().backward()

    assert_torch_error(mt, th)
