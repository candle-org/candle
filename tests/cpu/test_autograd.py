import numpy as np
import candle as torch
from candle.nn import functional as F


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


def test_autograd_broadcast_to_reduces_grad_to_input_shape():
    from candle._dispatch.dispatcher import dispatch

    x = torch.tensor([[1.0, 2.0, 3.0]])
    x.requires_grad = True

    y = dispatch("broadcast_to", x.device.type, x, (2, 3))
    assert type(y.grad_fn).__name__ == "BroadcastToBackward0"
    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (1, 3)
    assert x.grad.tolist() == [[2.0, 2.0, 2.0]]


def test_autograd_moveaxis_rebases_grad_to_input_axes():
    from candle._dispatch.dispatcher import dispatch

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x.requires_grad = True

    y = dispatch("moveaxis", x.device.type, x, 0, 1)
    assert type(y.grad_fn).__name__ == "MoveaxisBackward0"
    mask = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    (y * mask).sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (2, 3)
    assert x.grad.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def test_autograd_tile_reduces_grad_to_input_shape():
    from candle._dispatch.dispatcher import dispatch

    x = torch.tensor([[1.0, 2.0, 3.0]])
    x.requires_grad = True

    y = dispatch("tile", x.device.type, x, (2, 1))
    assert type(y.grad_fn).__name__ == "TileBackward0"
    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (1, 3)
    assert x.grad.tolist() == [[2.0, 2.0, 2.0]]


def test_autograd_repeat_interleave_reduces_grad_to_input_shape():
    from candle._dispatch.dispatcher import dispatch

    x = torch.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True

    y = dispatch("repeat_interleave", x.device.type, x, 2, 0)
    assert type(y.grad_fn).__name__ == "RepeatInterleaveBackward0"
    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (3,)
    assert x.grad.tolist() == [2.0, 2.0, 2.0]


def test_autograd_take_along_dim_scatters_grad_to_input_positions():
    from candle._dispatch.dispatcher import dispatch

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    indices = torch.tensor([[2, 0], [1, 1]])
    x.requires_grad = True
    indices.requires_grad = True

    y = dispatch("take_along_dim", x.device.type, x, indices, 1)
    assert type(y.grad_fn).__name__ == "TakeAlongDimBackward0"
    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (2, 3)
    assert x.grad.tolist() == [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]]
    assert indices.grad is None


def test_autograd_index_select_scatters_grad_to_input_rows():
    from candle._dispatch.dispatcher import dispatch

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    index = torch.tensor([0, 2, 0])
    x.requires_grad = True
    index.requires_grad = True

    y = dispatch("index_select", x.device.type, x, 0, index)
    assert type(y.grad_fn).__name__ == "IndexSelectBackward0"
    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (3, 3)
    assert x.grad.tolist() == [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert index.grad is None


def test_autograd_gather_scatters_grad_to_input_positions():
    from candle._dispatch.dispatcher import dispatch

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    index = torch.tensor([[2, 0, 2], [1, 1, 0]])
    x.requires_grad = True
    index.requires_grad = True

    y = dispatch("gather", x.device.type, x, 1, index)
    assert type(y.grad_fn).__name__ == "GatherBackward0"
    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (2, 3)
    assert x.grad.tolist() == [[1.0, 0.0, 2.0], [1.0, 2.0, 0.0]]
    assert index.grad is None


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


def test_autograd_cumsum_propagates_grad_to_input():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x.requires_grad = True

    y = torch.cumsum(x, 0)
    assert type(y.grad_fn).__name__ == "CumsumBackward0"
    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (4,)
    assert x.grad.tolist() == [4.0, 3.0, 2.0, 1.0]


def test_autograd_cummax_propagates_grad_to_selected_inputs():
    x = torch.tensor([1.0, 3.0, 2.0, 5.0])
    x.requires_grad = True

    values, indices = torch.cummax(x, 0)
    assert type(values.grad_fn).__name__ == "CummaxBackward0"
    assert indices.requires_grad is False
    values.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (4,)
    assert x.grad.tolist() == [1.0, 2.0, 0.0, 1.0]


def test_autograd_max_pool2d_propagates_grad_to_max_positions():
    x = torch.tensor([[[[1.0, 4.0], [3.0, 2.0]]]])
    x.requires_grad = True

    y = torch.nn.functional.max_pool2d(x, kernel_size=2)
    assert type(y.grad_fn).__name__ == "MaxPool2dBackward0"
    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (1, 1, 2, 2)
    assert x.grad.tolist() == [[[[0.0, 1.0], [0.0, 0.0]]]]


def test_autograd_prod_propagates_grad_to_factors():
    x = torch.tensor([2.0, 3.0, 4.0])
    x.requires_grad = True

    y = torch.prod(x)
    assert type(y.grad_fn).__name__ == "ProdBackward0"
    y.backward()

    assert x.grad is not None
    assert x.grad.shape == (3,)
    assert x.grad.tolist() == [12.0, 8.0, 6.0]


def test_autograd_repeat_reduces_grad_to_input_shape():
    x = torch.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True

    y = x.repeat(2)
    assert type(y.grad_fn).__name__ == "RepeatBackward0"
    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (3,)
    assert x.grad.tolist() == [2.0, 2.0, 2.0]


def test_autograd_sort_propagates_grad_to_original_positions():
    x = torch.tensor([3.0, 1.0, 2.0])
    x.requires_grad = True

    values, indices = torch.sort(x)
    assert type(values.grad_fn).__name__ == "SortBackward0"
    assert indices.requires_grad is False
    (values * torch.tensor([10.0, 20.0, 30.0])).sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (3,)
    assert x.grad.tolist() == [30.0, 10.0, 20.0]


def test_autograd_topk_propagates_grad_to_selected_positions():
    x = torch.tensor([1.0, 4.0, 2.0, 3.0])
    x.requires_grad = True

    values, indices = torch.topk(x, 2)
    assert type(values.grad_fn).__name__ == "TopkBackward0"
    assert indices.requires_grad is False
    values.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (4,)
    assert x.grad.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_autograd_fmod_tensor_routes_compiled_tensor_overload():
    x = torch.tensor([5.5, -5.5])
    y = torch.tensor([2.0, 2.0])
    x.requires_grad = True
    y.requires_grad = True

    out = torch.fmod(x, y)
    assert type(out.grad_fn).__name__ == "FmodTensorBackward0"
    out.sum().backward()

    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.array([-2.0, 2.0], dtype=np.float32))

    scalar_x = torch.tensor([5.5, -5.5])
    scalar_x.requires_grad = True
    scalar_out = torch.fmod(scalar_x, 2.0)
    assert type(scalar_out.grad_fn).__name__ == "FmodScalarBackward0"
    scalar_out.sum().backward()
    assert scalar_x.grad is not None
    np.testing.assert_allclose(scalar_x.grad.numpy(), np.array([1.0, 1.0], dtype=np.float32))


def test_autograd_remainder_tensor_routes_compiled_tensor_overload():
    x = torch.tensor([5.5, -5.5])
    y = torch.tensor([2.0, 2.0])
    x.requires_grad = True
    y.requires_grad = True

    out = torch.remainder(x, y)
    assert type(out.grad_fn).__name__ == "RemainderTensorBackward0"
    out.sum().backward()

    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.array([-2.0, 3.0], dtype=np.float32))

    scalar_x = torch.tensor([5.5, -5.5])
    scalar_x.requires_grad = True
    scalar_out = torch.remainder(scalar_x, 2.0)
    assert type(scalar_out.grad_fn).__name__ == "RemainderScalarBackward0"
    scalar_out.sum().backward()
    assert scalar_x.grad is not None
    np.testing.assert_allclose(scalar_x.grad.numpy(), np.array([1.0, 1.0], dtype=np.float32))


def test_autograd_norm_dim_routes_compiled_dim_overload():
    x = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    x.requires_grad = True

    out = torch.norm(x, 2, dim=1)
    assert type(out.grad_fn).__name__ == "NormScalarOptDimBackward0"
    out.sum().backward()

    assert x.grad is not None
    expected = np.array([[3.0 / 5.0, 4.0 / 5.0], [6.0 / 10.0, 8.0 / 10.0]], dtype=np.float32)
    np.testing.assert_allclose(x.grad.numpy(), expected)


def test_autograd_pow_tensor_scalar_and_tensor_tensor_backward():
    x = torch.tensor([2.0, 3.0])
    x.requires_grad = True

    scalar_out = torch.pow(x, 3.0)
    assert type(scalar_out.grad_fn).__name__ == "PowTensorScalarBackward0"
    scalar_out.sum().backward()

    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([12.0, 27.0], dtype=np.float32))

    base = torch.tensor([2.0, 4.0])
    exponent = torch.tensor([3.0, 0.5])
    base.requires_grad = True
    exponent.requires_grad = True

    tensor_out = torch.pow(base, exponent)
    assert type(tensor_out.grad_fn).__name__ == "PowTensorTensorBackward0"
    tensor_out.sum().backward()

    assert base.grad is not None
    assert exponent.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), np.array([12.0, 0.25], dtype=np.float32))
    expected_exponent = np.array([8.0 * np.log(2.0), 2.0 * np.log(4.0)], dtype=np.float32)
    np.testing.assert_allclose(exponent.grad.numpy(), expected_exponent)

    reflected_exponent = torch.tensor([2.0, 3.0])
    reflected_exponent.requires_grad = True
    reflected_out = 2.0 ** reflected_exponent
    assert type(reflected_out.grad_fn).__name__ == "PowScalarBackward0"
    reflected_out.sum().backward()

    assert reflected_exponent.grad is not None
    expected_reflected = np.array([4.0 * np.log(2.0), 8.0 * np.log(2.0)], dtype=np.float32)
    np.testing.assert_allclose(reflected_exponent.grad.numpy(), expected_reflected)



def test_autograd_square_routes_compiled_backward():
    x = torch.tensor([2.0, -3.0])
    x.requires_grad = True

    out = torch.square(x)
    assert type(out.grad_fn).__name__ == "SquareBackward0"
    out.sum().backward()

    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([4.0, -6.0], dtype=np.float32))


def test_autograd_outer_routes_compiled_backward():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0, 5.0])
    x.requires_grad = True
    y.requires_grad = True

    out = torch.outer(x, y)
    assert type(out.grad_fn).__name__ == "OuterBackward0"
    out.sum().backward()

    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([12.0, 12.0], dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.array([3.0, 3.0, 3.0], dtype=np.float32))


def test_autograd_inner_routes_compiled_backward():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    x.requires_grad = True
    y.requires_grad = True

    out = torch.inner(x, y)
    assert type(out.grad_fn).__name__ == "InnerBackward0"
    out.backward()

    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([4.0, 5.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_autograd_selu_routes_compiled_backward():
    x = torch.tensor([-1.0, 0.0, 2.0])
    x.requires_grad = True

    out = torch.nn.functional.selu(x)
    assert type(out.grad_fn).__name__ == "SeluBackward0"
    out.sum().backward()

    assert x.grad is not None
    expected = np.array([1.0507009873554805 * 1.6732632423543772 * np.exp(-1.0),
                         1.0507009873554805,
                         1.0507009873554805], dtype=np.float32)
    np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-6)


def test_autograd_softsign_routes_compiled_backward():
    x = torch.tensor([-2.0, 0.0, 3.0])
    x.requires_grad = True

    out = torch.nn.functional.softsign(x)
    assert type(out.grad_fn).__name__ == "SoftsignBackward0"
    out.sum().backward()

    assert x.grad is not None
    expected = 1.0 / np.array([3.0, 1.0, 4.0], dtype=np.float32) ** 2
    np.testing.assert_allclose(x.grad.numpy(), expected)


def test_autograd_signbit_is_non_differentiable():
    x = torch.tensor([-1.0, 0.0, 2.0])
    x.requires_grad = True

    out = torch.signbit(x)

    assert out.requires_grad is False
    assert out.grad_fn is None


def test_autograd_heaviside_routes_compiled_zero_backward():
    x = torch.tensor([-1.0, 0.0, 2.0])
    y = torch.tensor([0.5, 0.5, 0.5])
    x.requires_grad = True
    y.requires_grad = True

    out = torch.heaviside(x, y)
    assert type(out.grad_fn).__name__ == "HeavisideBackward0"
    out.sum().backward()

    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.zeros(3, dtype=np.float32))


def test_autograd_floor_divide_routes_compiled_zero_backward():
    x = torch.tensor([5.0, -5.0])
    y = torch.tensor([2.0, 2.0])
    x.requires_grad = True
    y.requires_grad = True

    out = torch.floor_divide(x, y)
    assert type(out.grad_fn).__name__ == "Floor_divideBackward0"
    out.sum().backward()

    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.zeros(2, dtype=np.float32))


def test_autograd_true_divide_preserves_public_div_backward():
    x = torch.tensor([6.0, -8.0])
    y = torch.tensor([2.0, 4.0])
    x.requires_grad = True
    y.requires_grad = True

    out = torch.true_divide(x, y)
    assert type(out.grad_fn).__name__ == "DivTensorBackward0"
    out.sum().backward()

    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([0.5, 0.25], dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.array([-1.5, 0.5], dtype=np.float32))



def test_autograd_hstack_vstack_row_stack_dstack_route_compiled_backward():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    for fn, grad_name in [
        (torch.hstack, "HstackBackward0"),
        (torch.vstack, "VstackBackward0"),
        (torch.row_stack, "Row_stackBackward0"),
        (torch.dstack, "DstackBackward0"),
    ]:
        x = a.clone()
        y = b.clone()
        x.requires_grad = True
        y.requires_grad = True

        out = fn((x, y))
        assert type(out.grad_fn).__name__ == grad_name
        out.sum().backward()

        assert x.grad is not None
        assert y.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), np.ones_like(x.numpy()))
        np.testing.assert_allclose(y.grad.numpy(), np.ones_like(y.numpy()))


def test_autograd_column_stack_routes_compiled_backward():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    x.requires_grad = True
    y.requires_grad = True

    out = torch.column_stack((x, y))
    assert type(out.grad_fn).__name__ == "Column_stackBackward0"
    out.sum().backward()

    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.ones(2, dtype=np.float32))


def test_autograd_concat_public_alias_preserves_cat_backward():
    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[3.0, 4.0]])
    x.requires_grad = True
    y.requires_grad = True

    out = torch.concat((x, y), dim=0)
    assert type(out.grad_fn).__name__ == "CatBackward0"
    (out * torch.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()

    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.array([[3.0, 4.0]], dtype=np.float32))


def test_autograd_concatenate_routes_compiled_backward():
    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[3.0, 4.0]])
    x.requires_grad = True
    y.requires_grad = True

    out = torch.concatenate((x, y), dim=0)
    assert type(out.grad_fn).__name__ == "ConcatenateBackward0"
    (out * torch.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()

    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.array([[3.0, 4.0]], dtype=np.float32))


def test_autograd_pad_sequence_routes_compiled_backward():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0])
    x.requires_grad = True
    y.requires_grad = True

    out = torch.pad_sequence((x, y), batch_first=True, padding_value=-1.0)
    assert type(out.grad_fn).__name__ == "Pad_sequenceBackward0"
    out.sum().backward()

    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.ones(1, dtype=np.float32))



def test_autograd_relu6_softmax_log_softmax_route_compiled_backward():
    x = torch.tensor([-1.0, 0.5, 7.0])
    x.requires_grad = True

    relu_out = torch.relu6(x)
    assert type(relu_out.grad_fn).__name__ == "Relu6Backward0"
    relu_out.sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([0.0, 1.0, 0.0], dtype=np.float32))

    y = torch.tensor([[1.0, 2.0, 3.0]])
    y.requires_grad = True
    softmax_out = torch.softmax(y, dim=1)
    assert type(softmax_out.grad_fn).__name__ == "SoftmaxBackward0"
    softmax_out[:, 0].sum().backward()
    assert y.grad is not None
    expected_softmax = softmax_out.numpy()
    expected_grad = -expected_softmax[0, 0] * expected_softmax
    expected_grad[0, 0] = expected_softmax[0, 0] * (1.0 - expected_softmax[0, 0])
    np.testing.assert_allclose(y.grad.numpy(), expected_grad, rtol=1e-6, atol=1e-6)

    z = torch.tensor([[1.0, 2.0, 3.0]])
    z.requires_grad = True
    log_softmax_out = torch.log_softmax(z, dim=1)
    assert type(log_softmax_out.grad_fn).__name__ == "Log_softmaxBackward0"
    log_softmax_out[:, 0].sum().backward()
    assert z.grad is not None
    expected = -np.exp(log_softmax_out.numpy())
    expected[0, 0] += 1.0
    np.testing.assert_allclose(z.grad.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_autograd_prelu_normalize_nanmean_special_logit_route_compiled_backward():
    x = torch.tensor([-2.0, 3.0])
    weight = torch.tensor([0.25])
    x.requires_grad = True
    weight.requires_grad = True
    prelu_out = F.prelu(x, weight)
    assert type(prelu_out.grad_fn).__name__ == "PreluBackward0"
    prelu_out.sum().backward()
    assert x.grad is not None
    assert weight.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([0.25, 1.0], dtype=np.float32))
    np.testing.assert_allclose(weight.grad.numpy(), np.array([-2.0], dtype=np.float32))

    y = torch.tensor([[3.0, 4.0]])
    y.requires_grad = True
    normalize_out = F.normalize(y, p=2.0, dim=1, eps=1e-12)
    assert type(normalize_out.grad_fn).__name__ == "NormalizeBackward0"
    normalize_out.sum().backward()
    assert y.grad is not None
    np.testing.assert_allclose(y.grad.numpy(), np.array([[0.032, -0.024]], dtype=np.float32), rtol=1e-5, atol=1e-6)

    n = torch.tensor([1.0, np.nan, 3.0])
    n.requires_grad = True
    nanmean_out = torch.nanmean(n)
    assert type(nanmean_out.grad_fn).__name__ == "NanmeanBackward0"
    nanmean_out.backward()
    assert n.grad is not None
    np.testing.assert_allclose(n.grad.numpy(), np.array([0.5, 0.0, 0.5], dtype=np.float32))

    l = torch.tensor([0.25, 0.75])
    l.requires_grad = True
    logit_out = torch.special.logit(l)
    assert type(logit_out.grad_fn).__name__ == "Special_logitBackward0"
    logit_out.sum().backward()
    assert l.grad is not None
    np.testing.assert_allclose(l.grad.numpy(), np.array([5.3333335, 5.3333335], dtype=np.float32), rtol=1e-6)


def test_autograd_diff_cross_route_compiled_backward():
    x = torch.tensor([1.0, 3.0, 6.0])
    x.requires_grad = True
    diff_out = torch.diff(x)
    assert type(diff_out.grad_fn).__name__ == "DiffBackward0"
    diff_out.sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.array([-1.0, 0.0, 1.0], dtype=np.float32))

    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    a.requires_grad = True
    b.requires_grad = True
    cross_out = torch.cross(a, b, dim=0)
    assert type(cross_out.grad_fn).__name__ == "CrossBackward0"
    cross_out.sum().backward()
    assert a.grad is not None
    assert b.grad is not None
    np.testing.assert_allclose(a.grad.numpy(), np.array([1.0, 0.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(b.grad.numpy(), np.array([0.0, 1.0, -1.0], dtype=np.float32))

def test_autograd_special_unary_batch_routes_compiled_backward():
    x = torch.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True
    digamma_out = torch.special.digamma(x)
    assert type(digamma_out.grad_fn).__name__ == "Special_digammaBackward0"
    digamma_out.sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(
        x.grad.numpy(),
        torch.special.polygamma(1, torch.tensor([1.0, 2.0, 3.0])).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )

    y = torch.tensor([1.0, 2.0, 3.0])
    y.requires_grad = True
    gammaln_out = torch.special.gammaln(y)
    assert type(gammaln_out.grad_fn).__name__ == "Special_gammalnBackward0"
    gammaln_out.sum().backward()
    assert y.grad is not None
    np.testing.assert_allclose(
        y.grad.numpy(), torch.special.digamma(torch.tensor([1.0, 2.0, 3.0])).numpy(), rtol=1e-6, atol=1e-6
    )

    z = torch.tensor([0.0, 1.0, 2.0])
    z.requires_grad = True
    i0_out = torch.special.i0(z)
    assert type(i0_out.grad_fn).__name__ == "Special_i0Backward0"
    i0_out.sum().backward()
    assert z.grad is not None
    np.testing.assert_allclose(
        z.grad.numpy(), torch.special.i1(torch.tensor([0.0, 1.0, 2.0])).numpy(), rtol=1e-6, atol=1e-6
    )

    e = torch.tensor([-0.5, 0.0, 0.5])
    e.requires_grad = True
    erfinv_out = torch.special.erfinv(e)
    assert type(erfinv_out.grad_fn).__name__ == "Special_erfinvBackward0"
    erfinv_out.sum().backward()
    assert e.grad is not None
    erfinv_np = erfinv_out.numpy()
    expected_erfinv_grad = (np.sqrt(np.pi) / 2.0) * np.exp(erfinv_np * erfinv_np)
    np.testing.assert_allclose(e.grad.numpy(), expected_erfinv_grad, rtol=1e-6, atol=1e-6)

    n = torch.tensor([-1.0, 0.0, 1.0])
    n.requires_grad = True
    ndtr_out = torch.special.ndtr(n)
    assert type(ndtr_out.grad_fn).__name__ == "Special_ndtrBackward0"
    ndtr_out.sum().backward()
    assert n.grad is not None
    expected_ndtr_grad = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * n.numpy() * n.numpy())
    np.testing.assert_allclose(n.grad.numpy(), expected_ndtr_grad, rtol=1e-6, atol=1e-6)

    s = torch.tensor([0.5, 1.0, 2.0])
    s.requires_grad = True
    sinc_out = torch.special.sinc(s)
    assert type(sinc_out.grad_fn).__name__ == "Special_sincBackward0"
    sinc_out.sum().backward()
    assert s.grad is not None
    s_np = s.numpy()
    expected_sinc_grad = (np.cos(np.pi * s_np) * np.pi * s_np - np.sin(np.pi * s_np)) / (np.pi * s_np * s_np)
    np.testing.assert_allclose(s.grad.numpy(), expected_sinc_grad, rtol=1e-6, atol=1e-6)


def test_autograd_diag_and_special_binary_batch_routes_compiled_backward():
    x = torch.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True
    diag_out = torch.diag(x)
    assert type(diag_out.grad_fn).__name__ == "DiagBackward0"
    diag_out.sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.ones(3, dtype=np.float32))

    p = torch.tensor([1.5, 2.5, 3.5])
    p.requires_grad = True
    poly_out = torch.special.polygamma(1, p)
    assert type(poly_out.grad_fn).__name__ == "Special_polygammaBackward0"
    poly_out.sum().backward()
    assert p.grad is not None
    np.testing.assert_allclose(
        p.grad.numpy(),
        torch.special.polygamma(2, torch.tensor([1.5, 2.5, 3.5])).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )

    m = torch.tensor([2.0, 3.0, 4.0])
    m.requires_grad = True
    multi_out = torch.special.multigammaln(m, 2)
    assert type(multi_out.grad_fn).__name__ == "Special_multigammalnBackward0"
    multi_out.sum().backward()
    assert m.grad is not None
    m_np = m.numpy()
    expected_multi_grad = torch.special.digamma(torch.tensor(m_np)).numpy() + \
        torch.special.digamma(torch.tensor(m_np - 0.5)).numpy()
    np.testing.assert_allclose(m.grad.numpy(), expected_multi_grad, rtol=1e-5, atol=1e-5)

    a = torch.tensor([0.5, 1.0, 2.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    a.requires_grad = True
    b.requires_grad = True
    xl_out = torch.special.xlogy(a, b)
    assert type(xl_out.grad_fn).__name__ == "Special_xlogyBackward0"
    xl_out.sum().backward()
    assert a.grad is not None
    assert b.grad is not None
    np.testing.assert_allclose(a.grad.numpy(), np.log(b.numpy()), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(b.grad.numpy(), a.numpy() / b.numpy(), rtol=1e-6, atol=1e-6)

    g_a = torch.tensor([1.0, 2.0, 3.0])
    g_x = torch.tensor([0.5, 1.5, 2.5])
    g_x.requires_grad = True
    gi_out = torch.special.gammainc(g_a, g_x)
    assert type(gi_out.grad_fn).__name__ == "Special_gammaincBackward0"
    gi_out.sum().backward()
    assert g_x.grad is not None

    h_a = torch.tensor([1.0, 2.0, 3.0])
    h_x = torch.tensor([0.5, 1.5, 2.5])
    h_x.requires_grad = True
    gic_out = torch.special.gammaincc(h_a, h_x)
    assert type(gic_out.grad_fn).__name__ == "Special_gammainccBackward0"
    gic_out.sum().backward()
    assert h_x.grad is not None


def test_autograd_linalg_and_view_batch_routes_compiled_backward():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x.requires_grad = True
    contig_out = x.t().contiguous()
    assert type(contig_out.grad_fn).__name__ == "ContiguousBackward0"
    contig_out.sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.ones((2, 2), dtype=np.float32))

    p = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    p.requires_grad = True
    pad_out = torch.nn.functional.pad(p, [1, 1, 0, 0])
    assert type(pad_out.grad_fn).__name__ == "PadBackward0"
    pad_out.sum().backward()
    assert p.grad is not None
    np.testing.assert_allclose(p.grad.numpy(), np.ones((2, 2), dtype=np.float32))

    d = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    d.requires_grad = True
    det_out = torch.det(d)
    assert type(det_out.grad_fn).__name__ == "DetBackward0"
    det_out.backward()
    assert d.grad is not None

    mp = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    mp.requires_grad = True
    mp_out = torch.matrix_power(mp, 2)
    assert type(mp_out.grad_fn).__name__ == "Matrix_powerBackward0"

    lmp = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    lmp.requires_grad = True
    lmp_out = torch.linalg.matrix_power(lmp, 2)
    assert type(lmp_out.grad_fn).__name__ == "Linalg_matrix_powerBackward0"

    inv = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    inv.requires_grad = True
    inv_out = torch.linalg.inv(inv)
    assert type(inv_out.grad_fn).__name__ == "Linalg_invBackward0"
    inv_out.sum().backward()
    assert inv.grad is not None

    g = torch.tensor([1.0, 2.0, 3.0, 4.0])
    g.requires_grad = True
    g_out = g[1:3]
    assert type(g_out.grad_fn).__name__ == "GetitemBackward0"
    g_out.sum().backward()
    assert g.grad is not None
    np.testing.assert_allclose(g.grad.numpy(), np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32))


def test_autograd_fft_batch_routes_compiled_backward():
    grad_fn_classes = {
        torch.fft.fft: "Fft_fftBackward0",
        torch.fft.ifft: "Fft_ifftBackward0",
        torch.fft.fft2: "Fft_fft2Backward0",
        torch.fft.ifft2: "Fft_ifft2Backward0",
        torch.fft.rfft: "Fft_rfftBackward0",
        torch.fft.rfft2: "Fft_rfft2Backward0",
    }
    for fn, expected in grad_fn_classes.items():
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        x.requires_grad = True
        out = fn(x)
        assert type(out.grad_fn).__name__ == expected, f"{fn.__name__}: {type(out.grad_fn).__name__}"

    # irfft / irfft2 take complex input
    cx = torch.tensor([1.0, 2.0, 3.0]).to(dtype=torch.complex64)
    cx.requires_grad = True
    irfft_out = torch.fft.irfft(cx)
    assert type(irfft_out.grad_fn).__name__ == "Fft_irfftBackward0"

    cx2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to(dtype=torch.complex64)
    cx2.requires_grad = True
    irfft2_out = torch.fft.irfft2(cx2)
    assert type(irfft2_out.grad_fn).__name__ == "Fft_irfft2Backward0"

    for fn, expected in (
        (torch.fft.fftn, "Fft_fftnBackward0"),
        (torch.fft.ifftn, "Fft_ifftnBackward0"),
        (torch.fft.rfftn, "Fft_rfftnBackward0"),
    ):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        x.requires_grad = True
        out = fn(x)
        assert type(out.grad_fn).__name__ == expected, f"{fn.__name__}: {type(out.grad_fn).__name__}"

    cxn = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to(dtype=torch.complex64)
    cxn.requires_grad = True
    irfftn_out = torch.fft.irfftn(cxn)
    assert type(irfftn_out.grad_fn).__name__ == "Fft_irfftnBackward0"


def test_autograd_fft_residual_batch_routes_compiled_backward():
    for fn, expected in (
        (torch.fft.fftshift, "Fft_fftshiftBackward0"),
        (torch.fft.ifftshift, "Fft_ifftshiftBackward0"),
    ):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x.requires_grad = True
        out = fn(x)
        assert type(out.grad_fn).__name__ == expected, f"{fn.__name__}: {type(out.grad_fn).__name__}"

    h = torch.tensor([1.0, 2.0, 3.0]).to(dtype=torch.complex64)
    h.requires_grad = True
    hfft_out = torch.fft.hfft(h)
    assert type(hfft_out.grad_fn).__name__ == "Fft_hfftBackward0"

    ih = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ih.requires_grad = True
    ihfft_out = torch.fft.ihfft(ih)
    assert type(ihfft_out.grad_fn).__name__ == "Fft_ihfftBackward0"


def _assert_linalg_svd_grad_matches_pytorch(input_data, loss_selector, atol=1e-5, rtol=1e-5):
    import torch as real_torch

    candle_a = torch.tensor(input_data)
    candle_a.requires_grad = True
    candle_out = torch.linalg.svd(candle_a, full_matrices=False)
    loss_selector(candle_out).sum().backward()

    torch_a = real_torch.tensor(input_data, dtype=real_torch.float64, requires_grad=True)
    torch_out = real_torch.linalg.svd(torch_a, full_matrices=False)
    loss_selector(torch_out).sum().backward()

    np.testing.assert_allclose(
        np.asarray(candle_a.grad.tolist()),
        torch_a.grad.detach().numpy(),
        atol=atol,
        rtol=rtol,
    )


def test_linalg_svd_u_backward_matches_pytorch():
    _assert_linalg_svd_grad_matches_pytorch(
        [[3.0, 1.0], [0.5, 2.0], [1.0, -0.25]],
        lambda svd_out: svd_out[0],
    )


def test_linalg_svd_vh_backward_matches_pytorch():
    _assert_linalg_svd_grad_matches_pytorch(
        [[3.0, 1.0], [0.5, 2.0], [1.0, -0.25]],
        lambda svd_out: svd_out[2],
    )


def test_autograd_linalg_batch_routes_compiled_backward():
    a = torch.eye(3) * 2.0
    a.requires_grad = True
    out = torch.linalg.cholesky(a)
    assert type(out.grad_fn).__name__ == "Linalg_choleskyBackward0", (
        f"linalg.cholesky: {type(out.grad_fn).__name__}"
    )

    a = torch.eye(3) * 2.0
    a.requires_grad = True
    out = torch.linalg.cond(a)
    assert type(out.grad_fn).__name__ == "Linalg_condBackward0", (
        f"linalg.cond: {type(out.grad_fn).__name__}"
    )

    a = torch.eye(3) * 2.0
    a.requires_grad = True
    out = torch.linalg.det(a)
    assert type(out.grad_fn).__name__ == "Linalg_detBackward0", (
        f"linalg.det: {type(out.grad_fn).__name__}"
    )

    a = torch.eye(3) * 2.0
    a.requires_grad = True
    out = torch.linalg.eigvals(a)
    assert type(out.grad_fn).__name__ == "Linalg_eigvalsBackward0", (
        f"linalg.eigvals: {type(out.grad_fn).__name__}"
    )

    a = torch.eye(3) * 2.0
    a.requires_grad = True
    out = torch.linalg.eigvalsh(a)
    assert type(out.grad_fn).__name__ == "Linalg_eigvalshBackward0", (
        f"linalg.eigvalsh: {type(out.grad_fn).__name__}"
    )

    a = torch.eye(3) * 2.0
    a.requires_grad = True
    out = torch.linalg.matrix_norm(a)
    assert type(out.grad_fn).__name__ == "Linalg_matrix_normBackward0", (
        f"linalg.matrix_norm: {type(out.grad_fn).__name__}"
    )

    a = torch.eye(3) * 2.0
    a.requires_grad = True
    out = torch.linalg.matrix_rank(a)
    assert type(out.grad_fn).__name__ == "Linalg_matrix_rankBackward0", (
        f"linalg.matrix_rank: {type(out.grad_fn).__name__}"
    )

    a = torch.eye(3) * 2.0
    a.requires_grad = True
    out = torch.linalg.norm(a)
    assert type(out.grad_fn).__name__ == "Linalg_normBackward0", (
        f"linalg.norm: {type(out.grad_fn).__name__}"
    )

    a = torch.eye(3) * 2.0
    a.requires_grad = True
    b = torch.tensor([1.0, 2.0, 3.0])
    out = torch.linalg.solve(a, b)
    assert type(out.grad_fn).__name__ == "Linalg_solveBackward0", (
        f"linalg.solve: {type(out.grad_fn).__name__}"
    )

    a = torch.eye(3) * 2.0
    a.requires_grad = True
    out = torch.linalg.svdvals(a)
    assert type(out.grad_fn).__name__ == "Linalg_svdvalsBackward0", (
        f"linalg.svdvals: {type(out.grad_fn).__name__}"
    )

    a = torch.eye(6).reshape(2, 3, 2, 3) * 2.0
    a.requires_grad = True
    out = torch.linalg.tensorinv(a)
    assert type(out.grad_fn).__name__ == "Linalg_tensorinvBackward0", (
        f"linalg.tensorinv: {type(out.grad_fn).__name__}"
    )

    a = torch.eye(6).reshape(2, 3, 2, 3) * 2.0
    a.requires_grad = True
    b = torch.zeros(2, 3)
    out = torch.linalg.tensorsolve(a, b)
    assert type(out.grad_fn).__name__ == "Linalg_tensorsolveBackward0", (
        f"linalg.tensorsolve: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True
    out = torch.linalg.vander(x)
    assert type(out.grad_fn).__name__ == "Linalg_vanderBackward0", (
        f"linalg.vander: {type(out.grad_fn).__name__}"
    )


def test_linalg_svd_singular_values_backward_matches_svdvals():
    x = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    x.requires_grad = True
    _u, s, _vh = torch.linalg.svd(x)

    expected_x = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    expected_x.requires_grad = True
    expected = torch.linalg.svdvals(expected_x)

    s.sum().backward()
    expected.sum().backward()

    assert x.grad is not None
    assert np.allclose(x.grad.numpy(), expected_x.grad.numpy(), atol=1e-6, rtol=1e-6)


def test_linalg_eigh_eigenvectors_backward_matches_torch():
    x = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    x.requires_grad = True
    _values, vectors = torch.linalg.eigh(x)
    vectors.sum().backward()

    import torch as torch_ref

    x_ref = torch_ref.tensor([[3.0, 1.0], [1.0, 3.0]], requires_grad=True)
    _values_ref, vectors_ref = torch_ref.linalg.eigh(x_ref)
    vectors_ref.sum().backward()

    assert x.grad is not None
    assert np.allclose(x.grad.numpy(), x_ref.grad.numpy(), atol=1e-6, rtol=1e-6)


def test_linalg_multi_dot_three_matrix_backward_matches_torch():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    c = torch.tensor([[1.0], [2.0]])
    a.requires_grad = True
    b.requires_grad = True
    c.requires_grad = True
    out = torch.linalg.multi_dot([a, b, c])
    out.sum().backward()

    import torch as torch_ref

    a_ref = torch_ref.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b_ref = torch_ref.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    c_ref = torch_ref.tensor([[1.0], [2.0]], requires_grad=True)
    ref_out = torch_ref.linalg.multi_dot([a_ref, b_ref, c_ref])
    ref_out.sum().backward()

    assert np.allclose(a.grad.numpy(), a_ref.grad.numpy(), atol=1e-6, rtol=1e-6)
    assert np.allclose(b.grad.numpy(), b_ref.grad.numpy(), atol=1e-6, rtol=1e-6)
    assert np.allclose(c.grad.numpy(), c_ref.grad.numpy(), atol=1e-6, rtol=1e-6)


def test_max_unpool1d_backward_matches_torch():
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    x.requires_grad = True
    indices = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.int64)
    out = F.max_unpool1d(x, indices, kernel_size=2, stride=2, output_size=(1, 2, 4))
    weights = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    (out * weights).sum().backward()

    import torch as torch_ref

    x_ref = torch_ref.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    indices_ref = torch_ref.tensor([[[0, 1], [0, 1]]])
    ref_out = torch_ref.nn.functional.max_unpool1d(
        x_ref,
        indices_ref,
        kernel_size=2,
        stride=2,
        output_size=(1, 2, 4),
    )
    weights_ref = torch_ref.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    (ref_out * weights_ref).sum().backward()

    assert x.grad is not None
    assert np.allclose(x.grad.numpy(), x_ref.grad.numpy(), atol=1e-6, rtol=1e-6)


def test_autograd_normalization_batch_routes_compiled_backward():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
    x.requires_grad = True
    out = F.layer_norm(x, [4])
    assert type(out.grad_fn).__name__ == "Layer_normBackward0", (
        f"layer_norm: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])
    x.requires_grad = True
    out = F.batch_norm(x, None, None, training=True)
    assert type(out.grad_fn).__name__ == "Batch_normBackward0", (
        f"batch_norm: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]])
    x.requires_grad = True
    out = F.group_norm(x, 2)
    assert type(out.grad_fn).__name__ == "Group_normBackward0", (
        f"group_norm: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    x.requires_grad = True
    out = F.rms_norm(x, [3])
    assert type(out.grad_fn).__name__ == "Rms_normBackward0", (
        f"rms_norm: {type(out.grad_fn).__name__}"
    )


def test_autograd_pool_batch_routes_compiled_backward():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    x.requires_grad = True
    out = F.max_pool1d(x, 2)
    assert type(out.grad_fn).__name__ == "Max_pool1dBackward0", (
        f"max_pool1d: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])
    x.requires_grad = True
    out = F.max_pool3d(x, 2)
    assert type(out.grad_fn).__name__ == "Max_pool3dBackward0", (
        f"max_pool3d: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    x.requires_grad = True
    out = F.avg_pool1d(x, 2)
    assert type(out.grad_fn).__name__ == "Avg_pool1dBackward0", (
        f"avg_pool1d: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    x.requires_grad = True
    out = F.adaptive_avg_pool1d(x, 2)
    assert type(out.grad_fn).__name__ == "Adaptive_avg_pool1dBackward0", (
        f"adaptive_avg_pool1d: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    x.requires_grad = True
    out = F.adaptive_avg_pool2d(x, (1, 1))
    assert type(out.grad_fn).__name__ == "Adaptive_avg_pool2dBackward0", (
        f"adaptive_avg_pool2d: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])
    x.requires_grad = True
    out = F.adaptive_avg_pool3d(x, (1, 1, 1))
    assert type(out.grad_fn).__name__ == "Adaptive_avg_pool3dBackward0", (
        f"adaptive_avg_pool3d: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    x.requires_grad = True
    out = F.adaptive_max_pool1d(x, 2)
    assert type(out.grad_fn).__name__ == "Adaptive_max_pool1dBackward0", (
        f"adaptive_max_pool1d: {type(out.grad_fn).__name__}"
    )


def test_autograd_conv_batch_routes_compiled_backward():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    x.requires_grad = True
    w = torch.tensor([[[1.0, 0.0, -1.0]]])
    w.requires_grad = True
    out = F.conv1d(x, w)
    assert type(out.grad_fn).__name__ == "Conv1dBackward0", (
        f"conv1d: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    x.requires_grad = True
    w = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]])
    w.requires_grad = True
    out = F.conv2d(x, w)
    assert type(out.grad_fn).__name__ == "Conv2dBackward0", (
        f"conv2d: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])
    x.requires_grad = True
    w = torch.tensor([[[[[1.0, 0.0], [0.0, -1.0]], [[0.0, 1.0], [-1.0, 0.0]]]]])
    w.requires_grad = True
    out = F.conv3d(x, w, None, (1, 1, 1), (0, 0, 0), (1, 1, 1))
    assert type(out.grad_fn).__name__ == "Conv3dBackward0", (
        f"conv3d: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    x.requires_grad = True
    w = torch.tensor([[[1.0, 0.0]]])
    w.requires_grad = True
    out = F.conv_transpose1d(x, w)
    assert type(out.grad_fn).__name__ == "Conv_transpose1dBackward0", (
        f"conv_transpose1d: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    x.requires_grad = True
    w = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]])
    w.requires_grad = True
    out = F.conv_transpose2d(x, w)
    assert type(out.grad_fn).__name__ == "Conv_transpose2dBackward0", (
        f"conv_transpose2d: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])
    x.requires_grad = True
    w = torch.tensor([[[[[1.0, 0.0], [0.0, -1.0]], [[0.0, 1.0], [-1.0, 0.0]]]]])
    w.requires_grad = True
    out = F.conv_transpose3d(x, w)
    assert type(out.grad_fn).__name__ == "Conv_transpose3dBackward0", (
        f"conv_transpose3d: {type(out.grad_fn).__name__}"
    )


def test_autograd_misc_batch_routes_compiled_backward():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[0.5, -1.0], [2.0, 0.25]])
    x.requires_grad = True
    y.requires_grad = True
    out = torch.tensordot(x, y, dims=1)
    assert type(out.grad_fn).__name__ == "TensordotBackward0", (
        f"tensordot: {type(out.grad_fn).__name__}"
    )

    x1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    x2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    x1.requires_grad = True
    x2.requires_grad = True
    out = torch.cdist(x1, x2)
    assert type(out.grad_fn).__name__ == "CdistBackward0", (
        f"cdist: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([1.0, 3.0, 2.0])
    x.requires_grad = True
    out = torch.quantile(x, 0.5)
    assert type(out.grad_fn).__name__ == "QuantileBackward0", (
        f"quantile: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([1.0, np.nan, 3.0])
    x.requires_grad = True
    out = torch.nanquantile(x, 0.5)
    assert type(out.grad_fn).__name__ == "NanquantileBackward0", (
        f"nanquantile: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    grid = torch.tensor([[[[-1.0, -1.0]]]])
    x.requires_grad = True
    grid.requires_grad = True
    out = F.grid_sample(x, grid, align_corners=True)
    assert type(out.grad_fn).__name__ == "Grid_sampleBackward0", (
        f"grid_sample: {type(out.grad_fn).__name__}"
    )

    theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    theta.requires_grad = True
    out = F.affine_grid(theta, (1, 1, 2, 2), align_corners=False)
    assert type(out.grad_fn).__name__ == "Affine_gridBackward0", (
        f"affine_grid: {type(out.grad_fn).__name__}"
    )

    log_probs = torch.tensor([[[-0.1, -2.3]], [[-2.3, -0.1]]])
    targets = torch.tensor([1.0])
    log_probs.requires_grad = True
    out = F.ctc_loss(log_probs, targets, [2], [1], reduction="sum")
    assert type(out.grad_fn).__name__ == "Ctc_lossBackward0", (
        f"ctc_loss: {type(out.grad_fn).__name__}"
    )


def test_autograd_final_special_batch_routes_compiled_backward():
    from candle._dispatch.dispatcher import dispatch

    x = torch.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True
    out = dispatch("uniform", x.device.type, x)
    assert type(out.grad_fn).__name__ == "UniformBackward0", (
        f"uniform: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    x.requires_grad = True
    out = F.instance_norm(x)
    assert type(out.grad_fn).__name__ == "Instance_normBackward0", (
        f"instance_norm: {type(out.grad_fn).__name__}"
    )

    x = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    x.requires_grad = True
    sign, logabsdet = torch.linalg.slogdet(x)
    assert sign.requires_grad is False
    assert type(logabsdet.grad_fn).__name__ == "Linalg_slogdetBackward0", (
        f"linalg_slogdet: {type(logabsdet.grad_fn).__name__}"
    )

    x = torch.tensor([[1.0, 2.0, 3.0]])
    x.requires_grad = True
    out = x.sum_to_size((1, 1))
    assert type(out.grad_fn).__name__ == "SumToSizeBackward0", (
        f"sum_to_size: {type(out.grad_fn).__name__}"
    )
