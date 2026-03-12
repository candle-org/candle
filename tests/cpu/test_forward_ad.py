import pytest
import candle
from candle.autograd import forward_ad


def test_forward_ad_level_stack_and_make_unpack():
    x = candle.rand(2)
    with forward_ad.dual_level():
        tangent = candle.ones_like(x)
        dual = forward_ad.make_dual(x, tangent)
        primal, t = forward_ad.unpack_dual(dual)
        assert primal is x
        assert t is tangent


def test_forward_ad_nested_levels_isolated():
    x = candle.rand(2)
    with forward_ad.dual_level():
        t1 = candle.ones_like(x)
        d1 = forward_ad.make_dual(x, t1)
        with forward_ad.dual_level():
            t2 = candle.full_like(x, 2.0)
            d2 = forward_ad.make_dual(x, t2)
            _, t = forward_ad.unpack_dual(d2)
            assert t is t2
        _, t = forward_ad.unpack_dual(d1)
        assert t is t1


def test_forward_ad_requires_active_level():
    x = candle.rand(2)
    t = candle.ones_like(x)
    with pytest.raises(RuntimeError):
        forward_ad.make_dual(x, t)


def test_forward_ad_exit_requires_lifo():
    with forward_ad.dual_level() as lvl1:
        with forward_ad.dual_level() as lvl2:
            with pytest.raises(RuntimeError):
                forward_ad.exit_dual_level(level=lvl1)


def test_forward_ad_add_jvp():
    x = candle.rand(2)
    y = candle.rand(2)
    with forward_ad.dual_level():
        tx = candle.ones_like(x)
        ty = candle.full_like(y, 2.0)
        x = forward_ad.make_dual(x, tx)
        y = forward_ad.make_dual(y, ty)
        z = candle.add(x, y)
        _, tz = forward_ad.unpack_dual(z)
        assert tz is not None
        assert (tz == tx + ty).all()


def test_gradient_edge_roundtrip():
    x = candle.rand(2, requires_grad=True)
    out = x.clone()
    edge = candle.autograd.graph.get_gradient_edge(x)
    assert edge.node is x.grad_fn
    assert edge.output_nr == x.output_nr


def test_calculate_shape_util_basic():
    out = candle.randn(10, 5, requires_grad=True)
    grad = candle.randn(5, 10, requires_grad=True)
    out_shape, grad_shape = candle.autograd._calculate_shape(out, grad, False)
    assert out_shape == (10, 5)
    assert grad_shape == (5, 10)


def test_forward_ad_mul_jvp():
    x = candle.rand(2)
    y = candle.rand(2)
    with forward_ad.dual_level():
        tx = candle.ones_like(x)
        ty = candle.full_like(y, 2.0)
        x = forward_ad.make_dual(x, tx)
        y = forward_ad.make_dual(y, ty)
        z = candle.mul(x, y)
        _, tz = forward_ad.unpack_dual(z)
        assert tz is not None
        expected = tx * y + x * ty
        tz_arr = tz._numpy_view().ravel()
        exp_arr = expected._numpy_view().ravel()
        assert tz_arr == pytest.approx(exp_arr)


def test_forward_ad_sum_jvp():
    x = candle.rand(2, 3)
    with forward_ad.dual_level():
        tx = candle.ones_like(x)
        x = forward_ad.make_dual(x, tx)
        z = candle.sum(x)
        _, tz = forward_ad.unpack_dual(z)
        assert tz is not None
        assert tz.shape == candle.sum(tx).shape
        assert float(tz) == pytest.approx(float(candle.sum(tx)))
