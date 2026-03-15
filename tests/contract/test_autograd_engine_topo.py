import candle as torch


def test_autograd_engine_accumulates_shared_subgraph():
    a = torch.ones((2, 2)).requires_grad_()
    b = a * a
    c = b + b
    c.sum().backward()
    assert a.grad is not None
    # b = a * a, c = b + b => dc/da = 4a
    assert (a.grad.numpy() == 4).all()


def test_autograd_engine_accumulates_reused_leaf():
    x = torch.randn(5, 5, requires_grad=True)
    y = (torch.rand(5, 5) + 0.1).requires_grad_(True)
    z = torch.randn(5, 5, requires_grad=True)
    grad_output = torch.randn(5, 5)

    term3 = 4 * z**2 * x / y
    (x + term3).backward(grad_output)

    expected = (4 * z.pow(2) / y + 1) * grad_output
    assert x.grad is not None
    torch.testing.assert_close(x.grad, expected)


def test_autograd_engine_reentrant_backward():
    # Backward inside backward hook should be supported.
    a = torch.ones((2, 2)).requires_grad_()
    b = a * a
    c = b.sum()
    d = (a * a).sum()
    called = {"ok": False}

    def hook(_grad):
        d.backward()
        called["ok"] = True

    c.register_hook(hook)
    c.backward()
    assert called["ok"]


def test_saved_tensors_hooks_unpacked_once_per_node():
    a = torch.ones((2, 2)).requires_grad_()
    counters = {"unpack": 0}

    def pack(t):
        return t

    def unpack(t):
        counters["unpack"] += 1
        return t

    from candle.autograd.graph import saved_tensors_hooks

    with saved_tensors_hooks(pack, unpack):
        b = a * a
        c = b + b
    c.sum().backward()
    # mul saves two tensors; add does not save inputs.
    assert counters["unpack"] == 2
