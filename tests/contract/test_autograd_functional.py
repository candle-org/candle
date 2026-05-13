import numpy as np
import candle as torch
import torch as pt

from .helpers import _to_numpy, assert_torch_error


def _assert_nested_allclose(actual, expected, *, rtol=1e-5, atol=1e-6):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_allclose(actual_item, expected_item, rtol=rtol, atol=atol)
        return
    np.testing.assert_allclose(_to_numpy(actual), _to_numpy(expected), rtol=rtol, atol=atol)


def test_autograd_functional_jacobian_single_tensor_matches_torch():
    def mt_func(x):
        return x * x

    def th_func(x):
        return x * x

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)

    mt_result = torch.autograd.functional.jacobian(mt_func, mt_x)
    th_result = pt.autograd.functional.jacobian(th_func, th_x)

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_jacobian_tuple_inputs_matches_torch():
    def mt_func(x, y):
        return x * x + 3.0 * y

    def th_func(x, y):
        return x * x + 3.0 * y

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    mt_y = torch.tensor([4.0, 5.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)
    th_y = pt.tensor([4.0, 5.0], requires_grad=True)

    mt_result = torch.autograd.functional.jacobian(mt_func, (mt_x, mt_y))
    th_result = pt.autograd.functional.jacobian(th_func, (th_x, th_y))

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_vjp_matches_torch():
    def mt_func(x):
        return x * x

    def th_func(x):
        return x * x

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)
    mt_v = torch.tensor([3.0, 4.0])
    th_v = pt.tensor([3.0, 4.0])

    mt_result = torch.autograd.functional.vjp(mt_func, mt_x, v=mt_v)
    th_result = pt.autograd.functional.vjp(th_func, th_x, v=th_v)

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_jvp_matches_torch():
    def mt_func(x):
        return x * x

    def th_func(x):
        return x * x

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)
    mt_v = torch.tensor([3.0, 4.0])
    th_v = pt.tensor([3.0, 4.0])

    mt_result = torch.autograd.functional.jvp(mt_func, mt_x, v=mt_v)
    th_result = pt.autograd.functional.jvp(th_func, th_x, v=th_v)

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_hessian_single_tensor_matches_torch():
    def mt_func(x):
        return (x * x).sum()

    def th_func(x):
        return (x * x).sum()

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)

    mt_result = torch.autograd.functional.hessian(mt_func, mt_x)
    th_result = pt.autograd.functional.hessian(th_func, th_x)

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_jacobian_tuple_outputs_matches_torch():
    def mt_func(x):
        return x * x, x.sum()

    def th_func(x):
        return x * x, x.sum()

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)

    mt_result = torch.autograd.functional.jacobian(mt_func, mt_x)
    th_result = pt.autograd.functional.jacobian(th_func, th_x)

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_jacobian_independent_input_matches_torch():
    def mt_func(x, y):
        del y
        return x * x

    def th_func(x, y):
        del y
        return x * x

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    mt_y = torch.tensor([4.0, 5.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)
    th_y = pt.tensor([4.0, 5.0], requires_grad=True)

    mt_result = torch.autograd.functional.jacobian(mt_func, (mt_x, mt_y))
    th_result = pt.autograd.functional.jacobian(th_func, (th_x, th_y))

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_vjp_tuple_inputs_matches_torch():
    def mt_func(x, y):
        return x * x + 3.0 * y

    def th_func(x, y):
        return x * x + 3.0 * y

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    mt_y = torch.tensor([4.0, 5.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)
    th_y = pt.tensor([4.0, 5.0], requires_grad=True)
    mt_v = torch.tensor([1.0, 1.0])
    th_v = pt.tensor([1.0, 1.0])

    mt_result = torch.autograd.functional.vjp(mt_func, (mt_x, mt_y), v=mt_v)
    th_result = pt.autograd.functional.vjp(th_func, (th_x, th_y), v=th_v)

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_vjp_scalar_output_default_vector_matches_torch():
    def mt_func(x):
        return (x * x).sum()

    def th_func(x):
        return (x * x).sum()

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)

    mt_result = torch.autograd.functional.vjp(mt_func, mt_x)
    th_result = pt.autograd.functional.vjp(th_func, th_x)

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_jvp_tuple_inputs_matches_torch():
    def mt_func(x, y):
        return x * x + 3.0 * y

    def th_func(x, y):
        return x * x + 3.0 * y

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    mt_y = torch.tensor([4.0, 5.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)
    th_y = pt.tensor([4.0, 5.0], requires_grad=True)
    mt_v = (torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0]))
    th_v = (pt.tensor([1.0, 1.0]), pt.tensor([2.0, 2.0]))

    mt_result = torch.autograd.functional.jvp(mt_func, (mt_x, mt_y), v=mt_v)
    th_result = pt.autograd.functional.jvp(th_func, (th_x, th_y), v=th_v)

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_jvp_scalar_input_default_vector_matches_torch():
    def mt_func(x):
        return x * x

    def th_func(x):
        return x * x

    mt_x = torch.tensor(2.0, requires_grad=True)
    th_x = pt.tensor(2.0, requires_grad=True)

    mt_result = torch.autograd.functional.jvp(mt_func, mt_x)
    th_result = pt.autograd.functional.jvp(th_func, th_x)

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_jacobian_vectorize_matches_torch():
    def mt_func(x):
        return x * x

    def th_func(x):
        return x * x

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)

    mt_result = torch.autograd.functional.jacobian(mt_func, mt_x, vectorize=True)
    th_result = pt.autograd.functional.jacobian(th_func, th_x, vectorize=True)

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_jacobian_forward_mode_vectorize_matches_torch():
    def mt_func(x):
        return x * x

    def th_func(x):
        return x * x

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)

    mt_result = torch.autograd.functional.jacobian(
        mt_func, mt_x, vectorize=True, strategy="forward-mode"
    )
    th_result = pt.autograd.functional.jacobian(
        th_func, th_x, vectorize=True, strategy="forward-mode"
    )

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_hessian_vectorize_matches_torch():
    def mt_func(x):
        return (x * x).sum()

    def th_func(x):
        return (x * x).sum()

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)

    mt_result = torch.autograd.functional.hessian(mt_func, mt_x, vectorize=True)
    th_result = pt.autograd.functional.hessian(th_func, th_x, vectorize=True)

    _assert_nested_allclose(mt_result, th_result)


def test_autograd_functional_hessian_forward_mode_vectorize_matches_torch():
    def mt_func(x):
        return (x * x).sum()

    def th_func(x):
        return (x * x).sum()

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)

    mt_result = torch.autograd.functional.hessian(
        mt_func, mt_x, vectorize=True, outer_jacobian_strategy="forward-mode"
    )
    th_result = pt.autograd.functional.hessian(
        th_func, th_x, vectorize=True, outer_jacobian_strategy="forward-mode"
    )

    _assert_nested_allclose(mt_result, th_result)

def test_autograd_functional_hessian_tuple_inputs_matches_torch():
    def mt_func(x, y):
        return (x * x + 3.0 * y * y).sum()

    def th_func(x, y):
        return (x * x + 3.0 * y * y).sum()

    mt_x = torch.tensor([1.0, 2.0], requires_grad=True)
    mt_y = torch.tensor([4.0, 5.0], requires_grad=True)
    th_x = pt.tensor([1.0, 2.0], requires_grad=True)
    th_y = pt.tensor([4.0, 5.0], requires_grad=True)

    mt_result = torch.autograd.functional.hessian(mt_func, (mt_x, mt_y))
    th_result = pt.autograd.functional.hessian(th_func, (th_x, th_y))

    _assert_nested_allclose(mt_result, th_result)

def test_autograd_functional_jacobian_empty_output_error_matches_torch():
    def mt():
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        torch.autograd.functional.jacobian(lambda value: value[:0], x)

    def th():
        x = pt.tensor([1.0, 2.0], requires_grad=True)
        pt.autograd.functional.jacobian(lambda value: value[:0], x)

    assert_torch_error(mt, th)


def test_autograd_functional_jacobian_forward_mode_without_vectorize_error_matches_torch():
    def mt():
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        torch.autograd.functional.jacobian(lambda value: value * value, x, strategy="forward-mode")

    def th():
        x = pt.tensor([1.0, 2.0], requires_grad=True)
        pt.autograd.functional.jacobian(lambda value: value * value, x, strategy="forward-mode")

    assert_torch_error(mt, th)


def test_autograd_functional_jacobian_forward_mode_create_graph_error_matches_torch():
    def mt():
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        torch.autograd.functional.jacobian(
            lambda value: value * value, x, vectorize=True, strategy="forward-mode", create_graph=True,
        )

    def th():
        x = pt.tensor([1.0, 2.0], requires_grad=True)
        pt.autograd.functional.jacobian(
            lambda value: value * value, x, vectorize=True, strategy="forward-mode", create_graph=True,
        )

    assert_torch_error(mt, th)


def test_autograd_functional_jacobian_forward_mode_strict_error_matches_torch():
    def mt():
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        torch.autograd.functional.jacobian(
            lambda value: value * value, x, vectorize=True, strategy="forward-mode", strict=True,
        )

    def th():
        x = pt.tensor([1.0, 2.0], requires_grad=True)
        pt.autograd.functional.jacobian(
            lambda value: value * value, x, vectorize=True, strategy="forward-mode", strict=True,
        )

    assert_torch_error(mt, th)


def test_autograd_functional_hessian_forward_mode_without_vectorize_error_matches_torch():
    def mt():
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        torch.autograd.functional.hessian(
            lambda value: (value * value).sum(), x, outer_jacobian_strategy="forward-mode",
        )

    def th():
        x = pt.tensor([1.0, 2.0], requires_grad=True)
        pt.autograd.functional.hessian(
            lambda value: (value * value).sum(), x, outer_jacobian_strategy="forward-mode",
        )

    assert_torch_error(mt, th)


