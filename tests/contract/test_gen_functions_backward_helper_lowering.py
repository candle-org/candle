from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_functions import _gen_one_node


def _node_src(func_name: str) -> str:
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    info = next(info for info in infos if info.func_name == func_name)
    return _gen_one_node(info)


def test_sum_dim_backward_uses_local_helper_not_missing_runtime_op():
    src = _node_src('sum.dim_IntList')
    assert 'redispatch("sum_backward"' not in src
    assert '_sum_backward_helper(' in src


def test_mean_dim_backward_uses_local_helper_not_missing_runtime_op():
    src = _node_src('mean.dim')
    assert 'redispatch("mean_backward"' not in src
    assert '_mean_backward_helper(' in src
