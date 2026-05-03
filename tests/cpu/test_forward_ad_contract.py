import numpy as np
import pytest

import candle
from candle.autograd import forward_ad


@pytest.mark.parametrize(
    "op_name",
    ["add", "mul", "sum", "mean", "view", "reshape", "neg", "exp", "sin", "cos", "tanh"],
)
def test_forward_ad_default_jvp_rules_live_in_compiled_c_boundary(op_name):
    rule = forward_ad.get_jvp(op_name)
    assert rule is not None
    assert rule.__module__ == "candle._C._forward_ad"


@pytest.mark.parametrize(
    ("op_name", "expected_fn"),
    [
        ("neg", lambda x, tx: -tx),
        ("exp", lambda x, tx: candle.exp(x) * tx),
        ("sin", lambda x, tx: candle.cos(x) * tx),
        ("cos", lambda x, tx: -candle.sin(x) * tx),
        (
            "tanh",
            lambda x, tx: (candle.ones_like(x) - candle.tanh(x) * candle.tanh(x)) * tx,
        ),
    ],
)
def test_forward_ad_unary_pointwise_jvp_contract(op_name, expected_fn):
    x = candle.tensor([0.2, -0.4, 0.7], dtype=candle.float32)
    tx = candle.tensor([1.0, 2.0, -3.0], dtype=candle.float32)

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(x, tx)
        out = getattr(candle, op_name)(dual)
        primal, tangent = forward_ad.unpack_dual(out)

    expected_primal = getattr(candle, op_name)(x)
    expected_tangent = expected_fn(x, tx)
    np.testing.assert_allclose(primal.numpy(), expected_primal.numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(tangent.numpy(), expected_tangent.numpy(), atol=1e-6, rtol=1e-6)


def test_forward_ad_mean_jvp_contract():
    x = candle.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=candle.float32)
    tx = candle.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=candle.float32)

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(x, tx)
        out = candle.mean(dual, dim=1, keepdim=True)
        primal, tangent = forward_ad.unpack_dual(out)

    expected_primal = candle.mean(x, dim=1, keepdim=True)
    expected_tangent = candle.mean(tx, dim=1, keepdim=True)
    np.testing.assert_allclose(primal.numpy(), expected_primal.numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(tangent.numpy(), expected_tangent.numpy(), atol=1e-6, rtol=1e-6)


def test_forward_ad_view_propagates_tangent_as_view():
    x = candle.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=candle.float32)
    tx = candle.tensor([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=candle.float32)

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(x, tx)
        out = dual.view((3, 2))
        primal, tangent = forward_ad.unpack_dual(out)

    expected_primal = x.view((3, 2))
    expected_tangent = tx.view((3, 2))
    np.testing.assert_allclose(primal.numpy(), expected_primal.numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(tangent.numpy(), expected_tangent.numpy(), atol=1e-6, rtol=1e-6)
    assert tangent._base is tx


def test_forward_ad_reshape_propagates_tangent_as_view_when_possible():
    x = candle.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=candle.float32)
    tx = candle.tensor([[20.0, 21.0, 22.0], [23.0, 24.0, 25.0]], dtype=candle.float32)

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(x, tx)
        out = dual.reshape((3, 2))
        primal, tangent = forward_ad.unpack_dual(out)

    expected_primal = x.reshape((3, 2))
    expected_tangent = tx.reshape((3, 2))
    np.testing.assert_allclose(primal.numpy(), expected_primal.numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(tangent.numpy(), expected_tangent.numpy(), atol=1e-6, rtol=1e-6)
    assert tangent._base is tx


def test_forward_ad_missing_rule_raises_explicit_error():
    x = candle.tensor([1.0, 2.0, 3.0], dtype=candle.float32)
    tx = candle.ones_like(x)

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(x, tx)
        with pytest.raises(RuntimeError, match=r"no forward-mode rule registered for op sqrt"):
            candle.sqrt(dual)
