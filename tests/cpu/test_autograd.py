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
