import pytest
import candle as torch


def test_saved_tensor_hooks_basic_roundtrip():
    packed = []

    def pack(tensor):
        packed.append(tensor)
        return tensor.numpy().copy()

    def unpack(data):
        return torch.tensor(data)

    x = torch.tensor([1.0, 2.0])
    x.requires_grad = True
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = torch.mul(x, x)
        y.sum().backward()
    assert len(packed) >= 1
    assert x.grad is not None


def test_saved_tensor_hooks_disallow_no_grad():
    def pack(tensor):
        return tensor

    def unpack(tensor):
        return tensor

    x = torch.tensor([1.0, 2.0])
    x.requires_grad = True
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = torch.mul(x, x)
        y.sum().backward()
    assert x.grad is not None


def test_saved_tensor_hooks_pack_raises():
    def pack(_tensor):
        raise RuntimeError("pack failed")

    def unpack(tensor):
        return tensor

    x = torch.tensor([1.0, 2.0])
    x.requires_grad = True
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        with pytest.raises(RuntimeError):
            y = torch.mul(x, x)
            y.sum().backward()


def test_saved_tensor_register_hooks_requires_callables():
    x = torch.tensor([1.0], requires_grad=True)
    y = x * x
    raw = y.grad_fn._raw_saved_self
    with pytest.raises(TypeError):
        raw.register_hooks(lambda x: x)
    with pytest.raises(TypeError):
        raw.register_hooks(1, 1)


def test_saved_tensor_none_and_release_errors():
    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(None)
            return x

        @staticmethod
        def backward(ctx, g):
            return g

    x = torch.tensor([1.0], requires_grad=True)
    y = Fn.apply(x)
    raw = y.grad_fn._raw_saved_tensors[0]
    with pytest.raises(RuntimeError, match="None is forbidden"):
        raw.register_hooks(lambda x: x, lambda x: x)
    y.sum().backward()
    with pytest.raises(RuntimeError, match="after they have already been freed"):
        _ = y.grad_fn.saved_tensors()


def test_saved_tensor_pack_hook_inplace_modification_raises():
    def pack(x):
        x += 1
        return x

    def unpack(x):
        return x

    x = torch.tensor([1.0], requires_grad=True)
    y = x * x
    raw = y.grad_fn._raw_saved_self
    with pytest.raises(RuntimeError, match="pack hook is modifying|in-place operation"):
        raw.register_hooks(pack, unpack)
