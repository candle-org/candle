import numpy as np
import candle as torch


def test_autograd_add_mul():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    x.requires_grad = True
    y.requires_grad = True
    z = torch.mul(torch.add(x, y), x)
    z.sum().backward()
    assert x.grad is not None
    assert y.grad is not None


def test_autograd_mul_tensor_scalar_backward():
    x = torch.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True
    y = x * 0.5
    y.sum().backward()
    assert x.grad is not None


def test_autograd_getitem_backward_and_retain_grad():
    x = torch.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True

    y = x * 2.0
    y.retain_grad()
    z = y[0]
    z.backward()

    assert y.grad is not None
    assert y.grad.tolist() == [1.0, 0.0, 0.0]
    assert x.grad is not None
    assert x.grad.tolist() == [2.0, 0.0, 0.0]


def test_autograd_flatten_propagates_grad_to_base_tensor():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x.requires_grad = True

    y = x.flatten()

    # After 1B-B, flatten on a contiguous tensor is a pure view: it carries
    # _view_func / _rev_view_func so the engine rebases gradient onto the
    # base directly, with no FlattenBackward0 grad_fn. This mirrors PyTorch.
    assert y._base is x
    assert callable(y._view_func)
    assert callable(y._rev_view_func)

    y[0].backward()

    assert x.grad is not None
    assert x.grad.shape == (2, 3)
    assert x.grad.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_autograd_core_nn_ops_keep_graph():
    import candle.nn.functional as F

    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]])
    x.requires_grad = True

    y = F.layer_norm(x, (x.shape[-1],))
    y = F.gelu(y)
    y = F.softmax(y, dim=-1)
    y = F.dropout(y, p=0.1, training=True)

    y.retain_grad()
    y.flatten()[0].backward()

    assert y.grad is not None
    assert x.grad is not None


def test_autograd_batched_matmul_backward_shape_safe():
    a = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[0.5, -1.0, 2.0], [3.0, 1.5, -2.0]],
        ]
    )
    b = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
            [[2.0, 1.0], [1.0, 0.0], [0.0, 3.0]],
        ]
    )
    a.requires_grad = True
    b.requires_grad = True

    out = torch.matmul(a, b)
    out.flatten()[0].backward()

    assert a.grad is not None
    assert b.grad is not None


def test_autograd_matmul_vector_matrix_backward_shape_and_values():
    x = torch.tensor([1.0, -2.0, 3.0])
    w = torch.tensor(
        [
            [0.5, 1.0],
            [-1.5, 2.0],
            [3.0, -0.5],
        ]
    )
    x.requires_grad = True
    w.requires_grad = True

    y = torch.matmul(x, w)
    y.sum().backward()

    assert x.grad is not None
    assert w.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([1.5, 0.5, 2.5], dtype=np.float32))
    np.testing.assert_allclose(
        w.grad.numpy(),
        np.array(
            [
                [1.0, 1.0],
                [-2.0, -2.0],
                [3.0, 3.0],
            ],
            dtype=np.float32,
        ),
    )


def test_autograd_matmul_matrix_vector_backward_shape_and_values():
    a = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.0, -3.0, 4.0],
        ]
    )
    x = torch.tensor([2.0, -1.0, 0.5])
    a.requires_grad = True
    x.requires_grad = True

    y = torch.matmul(a, x)
    y.sum().backward()

    assert a.grad is not None
    assert x.grad is not None
    np.testing.assert_allclose(
        a.grad.numpy(),
        np.array(
            [
                [2.0, -1.0, 0.5],
                [2.0, -1.0, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(x.grad.numpy(), np.array([1.0, -1.0, 3.0], dtype=np.float32))
