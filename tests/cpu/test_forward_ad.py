import importlib.machinery
import threading

import pytest
import candle
from candle._C import _forward_ad
from candle.autograd import forward_ad


def test_forward_ad_state_lives_in_compiled_c_boundary():
    assert isinstance(_forward_ad.__loader__, importlib.machinery.ExtensionFileLoader)
    assert forward_ad.enter_dual_level.__module__ == "candle._C._forward_ad"
    assert forward_ad.exit_dual_level.__module__ == "candle._C._forward_ad"
    assert forward_ad.get_jvp.__module__ == "candle._C._forward_ad"
    assert forward_ad.register_jvp.__module__ == "candle._C._forward_ad"
    assert forward_ad.temporarily_disable.__module__ == "candle._C._forward_ad"


def test_forward_ad_dual_helpers_live_in_compiled_c_boundary():
    assert forward_ad.make_dual.__module__ == "candle._C._forward_ad"
    assert forward_ad.unpack_dual.__module__ == "candle._C._forward_ad"


def test_forward_ad_unpack_dual_requires_registered_public_type():
    x = candle.rand(2)
    _forward_ad.set_unpacked_dual_type(None)
    try:
        with pytest.raises(RuntimeError, match="UnpackedDualTensor"):
            _forward_ad.unpack_dual(x)
    finally:
        _forward_ad.set_unpacked_dual_type(forward_ad.UnpackedDualTensor)


def test_forward_ad_level_stack_is_thread_local():
    parent_states = []
    child_states = []

    with forward_ad.dual_level() as parent_level:
        parent_states.append((parent_level, forward_ad._current_level()))

        def _worker():
            child_states.append(forward_ad._current_level())
            with forward_ad.dual_level() as child_level:
                child_states.append((child_level, forward_ad._current_level()))
            child_states.append(forward_ad._current_level())

        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        parent_states.append(forward_ad._current_level())

    assert parent_states == [(0, 0), 0]
    assert child_states == [-1, (0, 0), -1]


def test_forward_ad_disabled_levels_are_thread_local():
    child_states = []

    with forward_ad.dual_level() as level:
        with forward_ad.temporarily_disable(level):
            assert forward_ad.is_level_disabled(level) is True

            def _worker():
                child_states.append(forward_ad.is_level_disabled(level))

            t = threading.Thread(target=_worker)
            t.start()
            t.join()

    assert child_states == [False]


def test_forward_ad_jvp_registry_is_shared_with_dispatcher():
    sentinel = object()
    forward_ad.register_jvp("__test_forward_ad_registry__", sentinel)
    assert _forward_ad.get_jvp("__test_forward_ad_registry__") is sentinel
    assert forward_ad.get_jvp("__test_forward_ad_registry__") is sentinel


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
