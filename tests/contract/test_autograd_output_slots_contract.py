import pytest

import candle as torch
from candle.autograd import Function
from candle.autograd.engine import backward


class _MarkDirtyIdentity(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.mark_dirty(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output,)


class _Duplicate(Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone(), x.clone()

    @staticmethod
    def backward(ctx, ga, gb):
        return (ga + gb,)


class _SaveThenRead(Function):
    @staticmethod
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return t.clone()

    @staticmethod
    def backward(ctx, grad_output):
        ctx.saved_tensors
        return (grad_output,)


class _MarkDirtyTwice(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.mark_dirty(x)
        ctx.mark_dirty(y)
        return x, y

    @staticmethod
    def backward(ctx, grad_x, grad_y):
        return grad_x, grad_y


class _MarkDirtyKeyword(Function):
    @staticmethod
    def forward(ctx, x, *, y):
        ctx.mark_dirty(y)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def test_mark_dirty_bumps_saved_tensor_version():
    x = torch.tensor([1.0], requires_grad=True)
    y = _SaveThenRead.apply(x)
    _MarkDirtyIdentity.apply(x)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        backward(y.sum())



def test_mark_dirty_multiple_calls_accumulate_dirty_inputs():
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([2.0], requires_grad=True)
    sx = _SaveThenRead.apply(x)
    sy = _SaveThenRead.apply(y)

    _MarkDirtyTwice.apply(x, y)

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        backward(sx.sum())
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        backward(sy.sum())



def test_mark_dirty_keyword_tensor_bumps_saved_version():
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([2.0], requires_grad=True)
    sy = _SaveThenRead.apply(y)

    _MarkDirtyKeyword.apply(x, y=y)

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        backward(sy.sum())



def test_duplicate_outputs_have_distinct_output_slots():
    x = torch.tensor([3.0], requires_grad=True)
    a, b = _Duplicate.apply(x)
    assert a.output_nr != b.output_nr



def test_duplicate_output_slot_zero_one_order():
    x = torch.tensor([3.0], requires_grad=True)
    a, b = _Duplicate.apply(x)
    assert (a.output_nr, b.output_nr) == (0, 1)




def test_duplicate_outputs_propagate_output_slots_to_downstream_next_functions():
    x = torch.tensor([3.0], requires_grad=True)
    a, b = _Duplicate.apply(x)
    z = a + b
    nf = z.grad_fn.next_functions
    assert [idx for _, idx in nf] == [0, 1]



def test_duplicate_outputs_route_gradients_to_same_input():
    x = torch.tensor([3.0], requires_grad=True)
    a, b = _Duplicate.apply(x)
    backward((a + b).sum())

def test_duplicate_second_output_backward_uses_second_slot_only():
    x = torch.tensor([3.0], requires_grad=True)
    a, b = _Duplicate.apply(x)
    backward(b.sum())
    assert x.grad.item() == 1.0
