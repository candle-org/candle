from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_functions import _gen_one_node


def _node_src(func_name: str) -> str:
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    info = next(info for info in infos if info.func_name == func_name)
    return _gen_one_node(info)


def test_sigmoid_backward_uses_local_helper():
    src = _node_src('sigmoid')
    assert 'redispatch("sigmoid_backward"' not in src
    assert '_sigmoid_backward_helper(' in src


def test_tanh_backward_uses_local_helper():
    src = _node_src('tanh')
    assert 'redispatch("tanh_backward"' not in src
    assert '_tanh_backward_helper(' in src


def test_relu_backward_uses_local_helper():
    src = _node_src('relu')
    assert 'redispatch("threshold_backward"' not in src
    assert '_threshold_backward_helper(' in src


def test_hardtanh_backward_uses_local_helper():
    src = _node_src('hardtanh')
    assert 'redispatch("hardtanh_backward"' not in src
    assert '_hardtanh_backward_helper(' in src


def test_hardswish_backward_uses_local_helper():
    src = _node_src('hardswish')
    assert 'redispatch("hardswish_backward"' not in src
    assert '_hardswish_backward_helper(' in src


def test_hardsigmoid_backward_uses_local_helper():
    src = _node_src('hardsigmoid')
    assert 'redispatch("hardsigmoid_backward"' not in src
    assert '_hardsigmoid_backward_helper(' in src


def test_softplus_backward_uses_local_helper():
    src = _node_src('softplus')
    assert 'redispatch("softplus_backward"' not in src
    assert '_softplus_backward_helper(' in src


def test_sqrt_backward_does_not_emit_raw_python_scalar_mul():
    src = _node_src('sqrt')
    assert 'redispatch("mul", keyset, 2, result)' not in src
    assert '_sqrt_backward_helper(' in src


def test_rsqrt_backward_uses_local_helper():
    src = _node_src('rsqrt')
    assert 'redispatch("neg", keyset, 0.5)' not in src
    assert '_rsqrt_backward_helper(' in src


def test_exp2_backward_uses_local_helper():
    src = _node_src('exp2')
    assert 'M_LN2' not in src
    assert '_exp2_backward_helper(' in src


def test_mul_tensor_backward_uses_local_helper():
    src = _node_src('mul.Tensor')
    assert 'redispatch("mul_tensor_backward"' not in src
    assert '_mul_tensor_backward_helper(' in src


def test_div_tensor_backward_uses_local_helper():
    src = _node_src('div.Tensor')
    assert 'redispatch("div_tensor_self_backward"' not in src
    assert '_div_tensor_self_backward_helper(' in src
    assert 'redispatch("div_tensor_other_backward"' not in src
    assert '_div_tensor_other_backward_helper(' in src
